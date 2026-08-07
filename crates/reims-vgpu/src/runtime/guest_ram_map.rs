//! The process's imports of guest RAM: built once from the shim's spans, and
//! the only place a guest physical address becomes a bindable reference.
//!
//! # What this replaces
//!
//! The dma-buf rail had a cache here, and it had to: `UDMABUF_CREATE_LIST`
//! walked every page, took a kernel reference on each, and cost enough that a
//! digest-bucketed LRU bounded by pinned bytes was worth its own module.
//!
//! Under the host-pointer model there is nothing left to cache.
//! [`crate::runtime::guest_ram::GuestRamImport::slice`] is a range check.
//! What *is* worth holding is the small thing the cache was built around: the
//! imports themselves, one per RAMBlock, made once at first use and held for
//! the VM's lifetime. This module is that, and it is a `Vec` of at most a
//! handful of entries rather than a cache with an eviction policy.
//!
//! # Why the imports are built here and not at device create
//!
//! The backend measures the granularity; the runtime holds the
//! [`HostOps`](crate::runtime::HostOps) that can say where guest RAM lives.
//! Neither side has both, and the device context deliberately does not take a
//! host — see the module doc on [`crate::qemu::host_ops`] for why the runtime
//! keeps it. So the granularity is published by the backend through
//! [`crate::runtime::guest_ram::latch_granularity`] and the spans are fetched
//! here, on the first guest-memory reference of a boot.
//!
//! Building lazily rather than eagerly also gets the ordering right for free:
//! the device exists before any guest command is decoded, so the granularity is
//! always published by the time the first reference is asked for.
//!
//! # What a refusal means
//!
//! Every refusal here puts the whole boot on the copying rails for the
//! addresses it covers, so none of them is a slow path and none may be silent.
//! The one *expected* refusal is [`MapRefusal::NoBackendImport`](crate::runtime::guest_ram_map::MapRefusal::NoBackendImport): a host without
//! the extension, or an operator who set
//! [`crate::env::GUEST_IMPORT`](crate::env::GUEST_IMPORT) off. That one is a
//! statement about the host rather than a loss, so it is reported once on the
//! off channel rather than as a failure per reference.

use crate::runtime::guest_ram::{granularity, GuestRamError, GuestRamImport, GuestRef};
use crate::runtime::host::{GuestRamRegionsError, HostOps};
use std::sync::Arc;

/// Why a guest physical address did not become a bindable reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapRefusal {
    /// No backend published an import granularity: this host cannot import
    /// guest RAM, or an operator asked it not to. Expected, and the state every
    /// copying rail exists for.
    NoBackendImport,
    /// The shim could not say where guest RAM lives. Carries the check that
    /// refused.
    HostRefused(GuestRamRegionsError),
    /// The shim answered, but no span survived being bounded to the granularity
    /// — every region was empty, unmapped, malformed, or shorter than one
    /// granule. Distinct from [`Self::HostRefused`] because the host answered
    /// fine and it is our own bound that rejected every span.
    NoUsableRegion { spans: usize },
    /// The address is not inside any imported span. Guest RAM the GPU can reach
    /// exists, and this address is not in it — a device MMIO address, a hole,
    /// or a page the guest named that this machine does not back.
    GpaNotInAnyImport { gpa: u64 },
    /// The address is in a span, and the length asked for leaves it. Carries the
    /// bound's own reason so the check that refused keeps its name.
    OutsideImport(GuestRamError),
    /// A page list that is not one GPA-contiguous stretch.
    ///
    /// Not a statement that the pages are un-importable — they are all inside
    /// one RAMBlock and each is nameable. It is a statement about the *bind*: a
    /// `VkBuffer` range and a Metal buffer offset are each one offset and one
    /// length, so a surface assembled from four stretches is four of them, and
    /// no consumer takes several yet. Named and counted because how often it
    /// fires is what says whether widening them is worth doing.
    Scattered { pages: usize, first: u64 },
}

impl crate::observe::Decline for MapRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoBackendImport => "guest_ram_map_no_backend_import",
            Self::HostRefused(_) => "guest_ram_map_host_refused",
            Self::NoUsableRegion { .. } => "guest_ram_map_no_usable_region",
            Self::GpaNotInAnyImport { .. } => "guest_ram_map_gpa_not_in_any_import",
            Self::Scattered { .. } => "guest_ram_map_scattered",
            // The inner reason is the diagnosis; this wrapper only says where
            // it happened, so it forwards rather than adding a slug of its own.
            Self::OutsideImport(inner) => inner.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoBackendImport => Vec::new(),
            Self::HostRefused(inner) => {
                let mut f = vec![("host_reason", inner.slug().to_string())];
                f.extend(crate::observe::Decline::fields(inner));
                f
            }
            Self::NoUsableRegion { spans } => vec![("spans", spans.to_string())],
            Self::GpaNotInAnyImport { gpa } => vec![("gpa", format!("{gpa:#x}"))],
            Self::Scattered { pages, first } => vec![
                ("pages", pages.to_string()),
                ("first", format!("{first:#x}")),
            ],
            Self::OutsideImport(inner) => inner.fields(),
        }
    }
}

crate::observe::decline_display!(MapRefusal);

/// The greppable event class for this module's refusals.
const EVENT: &str = "guest_ram_map";

/// The imports this process holds, or the refusal that stopped it building any.
///
/// Resolved once and then read. A `Mutex` rather than a `OnceLock` because a
/// device recreate must be able to drop the imports: the backend's handles die
/// with the device, and an import whose identity outlived them would let a
/// stale [`crate::runtime::guest_ram::GuestSlice`] resolve against a
/// `VkDeviceMemory` that no longer exists.
static MAP: std::sync::Mutex<Option<Resolved>> = std::sync::Mutex::new(None);

#[derive(Debug)]
struct Resolved {
    /// One per usable RAMBlock span, in the order the shim reported them.
    /// Ordinary machines have one or two.
    imports: Vec<Arc<GuestRamImport>>,
    /// Set when the resolution refused, so the next reference does not re-ask
    /// the shim for an answer that will not change. A refusal here is about the
    /// host and the granularity, both of which are fixed for the device's life.
    refusal: Option<MapRefusal>,
}

/// Forget every import.
///
/// Called when the backend tears its device down. The next reference rebuilds,
/// against fresh identities, so nothing made before the teardown resolves after
/// it — see [`crate::runtime::guest_ram::ImportId`] for why that matters.
pub fn reset() {
    *MAP.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// Every import this process holds, for a backend that needs to create or
/// release its device-side handles.
///
/// Empty before the first reference of a boot and on a host that cannot import.
pub fn imports() -> Vec<Arc<GuestRamImport>> {
    MAP.lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|r| r.imports.clone())
        .unwrap_or_default()
}

/// Turn a guest physical address and a length into a bindable reference.
///
/// The whole guest-memory rail goes through here. Building the imports on the
/// first call is why `host` is taken: after that it is not touched.
pub fn reference<H: HostOps + ?Sized>(
    host: &mut H,
    gpa: u64,
    len: u64,
) -> Result<GuestRef, MapRefusal> {
    let mut guard = MAP.lock().unwrap_or_else(|p| p.into_inner());
    let resolved = match guard.as_mut() {
        Some(resolved) => resolved,
        None => {
            *guard = Some(resolve(host));
            guard.as_mut().expect("just resolved")
        }
    };
    if let Some(refusal) = resolved.refusal {
        return Err(report_once(refusal));
    }
    let import = resolved
        .imports
        .iter()
        .find(|i| i.contains_gpa(gpa))
        .ok_or(MapRefusal::GpaNotInAnyImport { gpa })
        .map_err(report_once)?;
    // `slice_for_gpa` emits its own named refusal on the fail channel, so the
    // wrapper forwards the reason rather than adding a second line for it.
    let slice = import
        .slice_for_gpa(gpa, len)
        .map_err(MapRefusal::OutsideImport)?;
    GuestRef::new(Arc::clone(import), slice).map_err(MapRefusal::OutsideImport)
}

/// [`reference`] for a decoded page list: `len` bytes starting `in_page` bytes
/// into `gpas[0]`.
///
/// The one implementation of the contiguity rule, so the sampled, buffer and
/// writeback rails cannot disagree about what a bindable page list is.
pub fn reference_for_pages<H: HostOps + ?Sized>(
    host: &mut H,
    gpas: &[u64],
    page_size: u64,
    in_page: u64,
    len: u64,
) -> Result<GuestRef, MapRefusal> {
    let Some(&first) = gpas.first() else {
        return Err(report_once(MapRefusal::Scattered {
            pages: 0,
            first: 0,
        }));
    };
    let contiguous = gpas
        .iter()
        .enumerate()
        .all(|(i, gpa)| *gpa == first + (i as u64) * page_size);
    if !contiguous {
        return Err(report_once(MapRefusal::Scattered {
            pages: gpas.len(),
            first,
        }));
    }
    reference(host, first + in_page, len)
}

/// Ask the host where guest RAM lives and bound every span to the backend's
/// granularity.
fn resolve<H: HostOps + ?Sized>(host: &mut H) -> Resolved {
    let Some(align) = granularity() else {
        return Resolved {
            imports: Vec::new(),
            refusal: Some(MapRefusal::NoBackendImport),
        };
    };
    let spans = match host.guest_ram_regions() {
        Ok(spans) => spans,
        Err(why) => {
            return Resolved {
                imports: Vec::new(),
                refusal: Some(MapRefusal::HostRefused(why)),
            }
        }
    };
    let count = spans.len();
    // A span this device cannot bound is skipped rather than fatal: a machine
    // with one ordinary RAMBlock and one odd sliver should import the RAMBlock.
    // `GuestRamImport::new` names the check that rejected each skipped one on
    // the fail channel, so a partial import is never silent.
    let imports: Vec<Arc<GuestRamImport>> = spans
        .into_iter()
        .filter_map(|span| GuestRamImport::new(span, align).ok().map(Arc::new))
        .collect();
    let refusal = imports
        .is_empty()
        .then_some(MapRefusal::NoUsableRegion { spans: count });
    Resolved { imports, refusal }
}

/// Emit `refusal` and hand it back.
///
/// Deduped by slug: these are per-reference and a decode path that names an
/// unbacked address once will name it every frame.
/// [`MapRefusal::NoBackendImport`] goes to the off channel — it is the host
/// saying what it is, not a loss of guest work — and everything else to the
/// fail channel.
fn report_once(refusal: MapRefusal) -> MapRefusal {
    let line = crate::observe::Emit::decline(EVENT, &refusal);
    match refusal {
        MapRefusal::NoBackendImport => {
            if crate::observe::first_sight("guest_ram_map_no_backend_import", 0) {
                line.off();
            }
        }
        _ => line.fail_once(0),
    }
    refusal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::guest_ram::{forget_granularity, latch_granularity, GuestRamRegion};

    /// The whole module is process-global, and so is the granularity latch.
    /// Every test here takes this and restores both.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct Spans(Vec<GuestRamRegion>);

    impl HostOps for Spans {
        fn mono_ns(&self) -> u64 {
            0
        }
        fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}
        fn schedule_bh(&mut self) {}
        fn guest_ram_regions(&mut self) -> Result<Vec<GuestRamRegion>, GuestRamRegionsError> {
            Ok(self.0.clone())
        }
    }

    struct Refusing;

    impl HostOps for Refusing {
        fn mono_ns(&self) -> u64 {
            0
        }
        fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}
        fn schedule_bh(&mut self) {}
        fn guest_ram_regions(&mut self) -> Result<Vec<GuestRamRegion>, GuestRamRegionsError> {
            Err(GuestRamRegionsError::NoRam)
        }
    }

    /// Two spans with a hole between them, which is the shape of an x86 machine
    /// with a PCI hole and the reason the lookup is a search rather than a
    /// single subtraction.
    fn two_spans() -> Spans {
        Spans(vec![
            GuestRamRegion {
                gpa_base: 0,
                host_va: 0x7f00_0000_0000,
                len: 0x8000_0000,
            },
            GuestRamRegion {
                gpa_base: 0x1_0000_0000,
                host_va: 0x7f80_0000_0000,
                len: 0x8000_0000,
            },
        ])
    }

    fn with_granularity<R>(align: Option<u64>, body: impl FnOnce() -> R) -> R {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset();
        match align {
            Some(a) => latch_granularity(a),
            None => forget_granularity(),
        }
        let out = body();
        reset();
        forget_granularity();
        out
    }

    /// An address in either span resolves, and the offset it resolves to is the
    /// one inside *that* span — not a distance from the first one. Getting this
    /// wrong on a machine with a PCI hole binds the GPU 4 GiB away from the
    /// bytes the guest named, inside a live import, where no bound would catch
    /// it.
    #[test]
    fn an_address_resolves_against_the_span_that_backs_it() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let low = reference(&mut host, 0x2000, 0x100).expect("in the first span");
            assert_eq!(low.bound().expect("checked").offset, 0x2000);

            let high = reference(&mut host, 0x1_0000_2000, 0x100).expect("in the second span");
            assert_eq!(
                high.bound().expect("checked").offset,
                0x2000,
                "the offset is into the second import, not from the first span's base"
            );
            assert_ne!(low.import().id(), high.import().id());
        });
    }

    /// The imports are built once. The shim's answer does not change, and
    /// re-asking would be an address-space walk per guest reference.
    #[test]
    fn the_host_is_asked_once_however_many_references_follow() {
        with_granularity(Some(0x1000), || {
            struct Counting(std::cell::Cell<usize>);
            impl HostOps for Counting {
                fn mono_ns(&self) -> u64 {
                    0
                }
                fn enqueue(&mut self, _a: crate::runtime::host::HostAction) {}
                fn schedule_bh(&mut self) {}
                fn guest_ram_regions(
                    &mut self,
                ) -> Result<Vec<GuestRamRegion>, GuestRamRegionsError> {
                    self.0.set(self.0.get() + 1);
                    Ok(vec![GuestRamRegion {
                        gpa_base: 0,
                        host_va: 0x7f00_0000_0000,
                        len: 0x8000,
                    }])
                }
            }
            let mut host = Counting(std::cell::Cell::new(0));
            for _ in 0..5 {
                reference(&mut host, 0x1000, 8).expect("inside");
            }
            assert_eq!(host.0.get(), 1);
            assert_eq!(imports().len(), 1);
        });
    }

    /// A host with no import capability refuses every reference by name, and
    /// never asks the shim at all. This is the ordinary state on a host without
    /// the extension, so it must not read as a failure of the guest's work.
    #[test]
    fn without_a_published_granularity_nothing_is_asked_and_nothing_resolves() {
        with_granularity(None, || {
            let mut host = two_spans();
            assert_eq!(
                reference(&mut host, 0x2000, 8).err(),
                Some(MapRefusal::NoBackendImport)
            );
            assert!(imports().is_empty());
        });
    }

    /// The shim's own refusal is carried through with its name, rather than
    /// collapsing into "no imports". "This machine has no RAM span" and "this
    /// build's shim is too old" are different things to go fix.
    #[test]
    fn the_hosts_own_refusal_keeps_its_name() {
        with_granularity(Some(0x1000), || {
            assert_eq!(
                reference(&mut Refusing, 0x2000, 8).err(),
                Some(MapRefusal::HostRefused(GuestRamRegionsError::NoRam))
            );
        });
    }

    /// An address in the hole between two spans is refused by name. It is not a
    /// bound violation — no import claims it — and reporting it as one would
    /// send a reader looking for arithmetic that is not there.
    #[test]
    fn an_address_no_span_backs_is_refused_by_name() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let hole = 0x8000_0000;
            assert_eq!(
                reference(&mut host, hole, 8).err(),
                Some(MapRefusal::GpaNotInAnyImport { gpa: hole })
            );
        });
    }

    /// A length that leaves the span it started in is refused by the bound, with
    /// the bound's own reason. The next span's bytes are elsewhere in host
    /// memory, so running off the end of one import is exactly the stray this
    /// device is bounded against.
    #[test]
    fn a_length_that_runs_off_the_end_of_a_span_is_refused_by_the_bound() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let last_page = 0x8000_0000 - 0x1000;
            assert!(reference(&mut host, last_page, 0x1000).is_ok());
            assert!(matches!(
                reference(&mut host, last_page, 0x2000),
                Err(MapRefusal::OutsideImport(
                    GuestRamError::SliceEndPastImport { .. }
                ))
            ));
        });
    }

    /// A span nothing can bound is skipped, and the ones that can be are still
    /// imported. Failing the whole map on one odd sliver would put a machine
    /// with an ordinary RAMBlock beside it on the copying rails for no reason.
    #[test]
    fn one_unusable_span_does_not_cost_the_usable_ones() {
        with_granularity(Some(0x1000), || {
            let mut host = Spans(vec![
                // Shorter than one granule: nothing to import.
                GuestRamRegion {
                    gpa_base: 0,
                    host_va: 0x7f00_0000_0000,
                    len: 0x400,
                },
                GuestRamRegion {
                    gpa_base: 0x1_0000_0000,
                    host_va: 0x7f80_0000_0000,
                    len: 0x8000,
                },
            ]);
            assert!(reference(&mut host, 0x1_0000_0000, 8).is_ok());
            assert_eq!(imports().len(), 1);
        });
    }

    /// Every span unusable is its own refusal, distinct from the host refusing:
    /// the host answered fine and our own bound rejected all of it.
    #[test]
    fn every_span_unusable_is_a_refusal_of_its_own() {
        with_granularity(Some(0x1000), || {
            let mut host = Spans(vec![GuestRamRegion {
                gpa_base: 0,
                host_va: 0x7f00_0000_0000,
                len: 0x400,
            }]);
            assert_eq!(
                reference(&mut host, 0, 8).err(),
                Some(MapRefusal::NoUsableRegion { spans: 1 })
            );
        });
    }

    /// A device recreate drops the imports, and the rebuilt ones carry new
    /// identities. A reference taken before the teardown must not resolve
    /// against the replacement, because the backend handle it named is gone.
    #[test]
    fn a_reset_rebuilds_against_fresh_identities() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let before = reference(&mut host, 0x2000, 8).expect("inside");
            reset();
            let after = reference(&mut host, 0x2000, 8).expect("inside");
            assert_ne!(before.import().id(), after.import().id());
            assert!(matches!(
                after.import().resolve(&before.import().slice(0, 8).unwrap()),
                Err(GuestRamError::SliceForeignImport { .. })
            ));
        });
    }

    /// The refusals reach the always-on log, and the expected one does not
    /// reach the fail channel.
    #[test]
    fn refusals_are_visible_and_the_expected_one_is_not_a_failure() {
        with_granularity(Some(0x1000), || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();
            let _ = reference(&mut host, 0x8000_0000, 8);
            let line = capture.one(EVENT);
            assert!(
                line.contains("reason=guest_ram_map_gpa_not_in_any_import"),
                "{line}"
            );
            assert!(line.contains("gpa=0x80000000"), "{line}");
            assert!(!line.starts_with("OFF "), "a lost reference is a failure");
        });
        with_granularity(None, || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();
            let _ = reference(&mut host, 0x2000, 8);
            // Not `capture.one(EVENT)`: an off-channel line's first token is
            // the literal `OFF`, which is the same thing the fail-log reading
            // notes in `AGENTS.md` warn about when ranking `reason=`.
            let lines = capture.lines();
            assert_eq!(
                lines,
                vec!["OFF guest_ram_map reason=guest_ram_map_no_backend_import"],
                "a host without the extension has not lost guest work"
            );
        });
    }
}
