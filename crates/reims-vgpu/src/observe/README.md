# Crate-wide observability

`observe/` owns the vocabulary and delivery path for every genuine refusal in
`reims-vgpu`:

- `sink.rs` is the always-on `/tmp/reims-vgpu-fail.log` writer, background queue, and
  flood self-detector.
- `decline.rs` defines `Decline` and `Refusal`. The vocabulary lives in the
  `slug()` arms and nowhere else — the 2 700-line `REGISTRY` that used to
  restate every type's file, emission site and slug list was removed, because a
  copy of the arms can only ever agree or disagree with them.
- `emit.rs` is the only reason-bearing line builder. `Emit::decline` requires a
  typed decline; `Emit::refusal` makes the successful status of a mixed status
  enum unrepresentable as a failure line.
- `mod.rs` exposes the shared API and owns small integration tests for the
  module boundary.

Crate-wide slug uniqueness — the one property no single `impl` can see, and the
one that decides whether `Emit::fail_once`'s latch silences a second check — is
enforced by `tests/decline_slugs_are_unique.rs`. That every slug actually
reaches a sink at some call site remains the author's obligation; nothing checks
it.

Pure layers may return a typed decline without logging it. The product boundary
that decides the command really failed owns emission through `Emit`; expected
speculative control flow remains silent. Re-attempted failures use
`Emit::fail_once` with a discriminant that distinguishes independent events.

The authoritative policy and the exceptions that require judgement are in
`AGENTS.md`; this file describes ownership, not a second copy of that policy.
