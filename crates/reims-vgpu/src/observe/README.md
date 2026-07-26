# Crate-wide observability

`observe/` owns the vocabulary and delivery path for every genuine refusal in
`reims-vgpu`:

- `sink.rs` is the always-on `/tmp/reims-vgpu-fail.log` writer, background queue, and
  flood self-detector.
- `decline.rs` defines `Decline` and `Refusal` and contains the crate-wide
  `REGISTRY` of reason slugs, defining sites, and emission sites.
- `emit.rs` is the only reason-bearing line builder. `Emit::decline` requires a
  typed decline; `Emit::refusal` makes the successful status of a mixed status
  enum unrepresentable as a failure line.
- `gate.rs` proves the registry is unique and log-safe, that its vocabulary
  matches the defining types, that each reason reaches an always-on sink, and
  that error-shaped types or string-error results cannot bypass the registry.
- `mod.rs` exposes the shared API and owns small integration tests for the
  module boundary.

Pure layers may return a typed decline without logging it. The product boundary
that decides the command really failed owns emission through `Emit`; expected
speculative control flow remains silent. Re-attempted failures use
`Emit::fail_once` with a discriminant that distinguishes independent events.

The authoritative policy and the exceptions that require judgement are in
`AGENTS.md`; this file describes ownership, not a second copy of that policy.
