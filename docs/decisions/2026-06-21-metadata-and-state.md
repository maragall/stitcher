# Decision record: reader metadata representation & core pipeline state

**Date:** 2026-06-21
**Status:** Decided (implementation of Call 1 deferred to the next reader change; Call 2 closed)
**Context:** Two design calls left open by the maintainability refactor (see `docs/refactor-overview.md`).

## How this was decided

Static analysis (blast-radius + consumer mapping) plus a parallel prototype sweep: five
throwaway implementations in isolated git worktrees, one per option, each run against the test
suite and measured for churn, dependency cost, and how it handled reader-private state.

**Caveat on the prototype run:** the isolation worktrees branched from a pre-refactor commit
(`c067227`, suite 110 passed) rather than current HEAD (199 passed), so the *churn magnitudes*
are not HEAD-accurate and the dataclass variants are inflated by recreating `io/base.py` (which
already exists at HEAD). The *qualitative* findings below are baseline-independent and converge
with the static analysis; the decision rests on those, not on the raw line counts.

## Call 1 — reader metadata representation

`Reader.load_metadata()` returns a `dict` mixing a ~15-field cross-module contract (what `core`
consumes) with reader-private bookkeeping (open `TiffFile` handles, tensorstore objects, file
maps, glob patterns).

**Decision: adopt a stdlib `@dataclass Metadata` for the cross-module contract (Option 1a),
implemented when the next reader change lands. Reject pydantic. Keep `TypedDict` (Option 1c) as
the accepted zero-churn fallback if typing is wanted sooner.**

Evidence:
- **pydantic (rejected):** prototype confirmed it requires adding a *declared dependency*
  (not currently in `pyproject.toml`), `ConfigDict(frozen=False)` to allow the region-filter
  mutation, and its validation is — in the prototype's own words — *"structural insurance, not
  a runtime guard that fires on real inputs"* (no first-party reader produces bad data). A new
  dependency on a lean scientific stack (numpy/tifffile/tensorstore/numba) for validation that
  never fires at a first-party boundary is not justified.
- **stdlib `@dataclass` (chosen):** gives the typed, documented contract a new reader can follow
  as a checklist; reader-private state stays on the reader (cached `self._raw`), out of the model.
  Mild but real legibility gain; no new dependency.
- **TypedDict (fallback):** zero runtime change, no dependency, no test churn; needs a small
  `io/_types.py` split to avoid a circular import. Lower payoff (no construction-time guard, weaker
  "checklist") but the cheapest way to type the contract if 1a is not yet warranted.
- **extra-hatch (rejected):** workable but keeps a soft grab-bag field and touches reader read
  methods for no contract benefit.

Why deferred: the payoff lands when a reader is written/updated (see the pending Squid reader
work). It is a maintainability/extensibility change, not a quality or memory change, so it is not
part of the upcoming quality+memory effort.

## Call 2 — core pipeline state (`pairwise_metrics`, `global_offsets`, fused-space geometry)

**Decision: keep pipeline data as instance state (Option 2a). The broad mutable-state reduction
(2c) is rejected. The `FusedSpace` value-object grouping (2b) is optional and not planned.**

Evidence:
- `pairwise_metrics` and `global_offsets` are a *public, inspectable, serializable* surface: read
  by the GUI (`gui/app.py`), asserted by tests, baked into the golden, and `pairwise_metrics` has
  a custom JSON `save`/`load` contract. Threading them as params (2c) would break the GUI and the
  serialization/golden compatibility — not behaviour-preserving.
- `FusedSpace` (2b) prototype: small and works, but only `padded_shape` is read externally (one
  property shim); the rest is internal indirection. Net clarity gain does not justify the churn
  now that `core` is already decomposed.

## Net

- Call 1: stdlib `@dataclass Metadata` (Option 1a), timed with reader work; pydantic rejected.
- Call 2: closed — keep instance state; no reduction.
- Neither is part of the upcoming quality + memory optimization effort.
