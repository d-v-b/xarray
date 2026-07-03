# Design: represent zarr storage layout in `encoding` as a spec metadata fragment

Date: 2026-07-02
Status: approved (brainstorming), pending implementation plan
Scope: xarray Zarr backend (`xarray/backends/zarr.py`)

## Problem

The Zarr backend flattens a zarr array's metadata into individual `.encoding`
keys (`chunks`, `compressor`/`compressors`, `filters`, `serializer`, `shards`,
`fill_value`) with xarray-chosen names, then reconstructs `zarr.create(**kwargs)`
from them on write. This projection is lossy and format-blind:

- The read path branches on the _installed_ zarr-python version (`_zarr_v3()`),
  not the _store_ format, so reading a v2 store with zarr-python 3 labels its
  numcodecs codecs with the v3-style key `compressors`.
- On write, xarray passes those codec objects straight to `zarr.create()` with no
  format awareness. A v2→v3 round-trip of any compressed array therefore fails
  deep inside zarr:

  ```
  TypeError: Expected a BytesBytesCodec. Got <class 'numcodecs.blosc.Blosc'> instead.
  ```

  (Verified on zarr-python 3.2.1; fails with or without an explicit `encoding=`.)

- `fill_value` handling is split across attrs and encoding depending on format
  and `use_zarr_fill_value_as_mask` (xarray issues #10269, #10646; mirrors
  zarr-python #2322).

The root cause is that `.encoding` never carries the array's actual, self-
describing metadata document — only a renamed, format-ambiguous subset.

## Goal

Carry the storage-layout portion of a zarr array's metadata in `.encoding` as a
single, spec-shaped, self-describing metadata fragment, and drive both read and
write through zarr-python's own `to_dict()`/`from_dict()`. Make v2⇄v3 round-trips
work, localize all format-specific logic, and keep existing user code working
during a deprecation period.

## Design decisions (from brainstorming)

1. **Single self-describing key.** One `encoding["zarr_array_metadata"]` holding
   a v2-or-v3 metadata dict. The dict carries its own `zarr_format`, so it is a
   discriminated union; there is never a second copy to disagree with it. Format
   conversion is an explicit write-time function, not stored state.
2. **Variable wins on overlaps.** The fragment may contain `shape`, `data_type`,
   `attributes`, `dimension_names`, but on write those are regenerated from the
   `Variable`/`attrs`/`dims` and overwrite whatever the fragment holds. The
   fragment is authoritative only for storage layout.
3. **Variable/resolved value wins (authoritative overwrite).** As implemented,
   the write path overwrites the storage-layout fields that matter —
   `shape`, `dimension_names`, chunk grid, `data_type`, and `fill_value` — from
   the write itself (via `_set_chunk_shape`/`_set_dtype`/`_set_fill_value`), so a
   stale fragment value is superseded rather than treated as a conflict (e.g.
   opening then rechunking, then writing, uses the new chunks). `merge_flat_aliases`
   retains a defensive `chunks` conflict check at the seam boundary (unit-tested),
   but because the fast path resolves chunks authoritatively it does not surface as
   an error through `to_zarr`. NOTE: this supersedes the original "alias conflict is
   an error" intent — the matured design favors authoritative overwrite, which is
   simpler and avoids false conflicts on legitimate rechunking.
4. **Read populates both; flat keys stay fully supported for now.** Read sets
   both the fragment and the derived flat aliases so existing code keeps working.
   Emitting a `DeprecationWarning` on flat-key use is DEFERRED to a separate
   follow-up PR (in trial it turned ~296 existing tests red, since setting
   `chunks`/`compressors` is ubiquitous — deprecating the common keys needs its
   own careful scoping). Hard removal is a later change beyond that. This PR adds
   the fragment as the preferred path without deprecating anything.
5. **Read via `to_dict`; create by persisting metadata buffers.** Read uses
   `zarr_array.metadata.to_dict()`. Write builds
   `ArrayV{2,3}Metadata.from_dict(canonical_dict)` and persists its
   `to_buffer_dict(prototype)` output to the array's store path. No kwarg
   translation. VERIFIED (zarr-python 3.2.1): `from_dict` alone constructs an
   in-memory handle but does NOT persist — a persist step is required. Migrate
   this single create call to a persisting upstream create-from-metadata API when
   zarr-python offers one (the "C later" path; the seam is the only call site).
6. **`zarr-metadata` typing only (near-term).** Import the v2/v3 metadata
   `TypedDict`s from the `zarr-metadata` package under `if TYPE_CHECKING:` to
   annotate the fragment (`ArrayMetadataV2 | ArrayMetadataV3`). No runtime
   dependency for now. This flips to a runtime (optional, zarr-extra) dependency
   if/when the dataclass direction below lands.
7. **Seam now, dataclass later.** Build the internal representation seam against
   plain dicts first so xarray is not blocked on a `zarr-metadata` release; swap
   the seam's backing to `zarr-metadata` dataclasses in a follow-up. The backend
   call sites do not change when that swap happens.

## Architecture

CF/decode-reversal keys (`_FillValue`, `scale_factor`, `add_offset`, `dtype`,
`units`, `calendar`, `_Unsigned`, ...) are unchanged and remain flat, produced by
the CF coding layer. `preferred_chunks` remains a flat, read-only hint. The new
fragment covers only storage layout: codecs/compressor/filters/serializer, chunk
grid, chunk-key encoding, and array-level `fill_value`.

### Internal representation seam

All interaction with the fragment goes through one small internal module that
owns: reading a store array into the representation, applying Variable-owned
field overrides, reconciling chunks, converting between formats, and writing back
out (`to_dict`/`from_dict`). The rest of the backend never touches fragment
internals directly. The backing of this representation is deliberately swappable:
it starts as a plain dict (see decision 6) and can later become a
`zarr-metadata` dataclass without changing the backend call sites. This seam is
what keeps the dict-vs-dataclass choice an implementation detail.

### Read path (`ZarrStore.open_store_variable`)

1. `encoding["zarr_array_metadata"] = zarr_array.metadata.to_dict()`.
2. Derive legacy flat aliases from the fragment for back-compat.
3. Because the fragment comes from `metadata.to_dict()`, it is always format-
   correct — this removes the current "installed-version, not store-format"
   labeling bug at the fragment level.

### Write path

Replaces `_create_new_array`. The append/region/existing-array paths (which
_open_ rather than create) are unchanged.

1. **Normalize** to one canonical metadata dict: start from
   `zarr_array_metadata` if present; fold in flat aliases; a disagreeing flat
   alias raises `ValueError` naming the field.
2. **Variable wins:** regenerate `shape`, `data_type`, `attributes`,
   `dimension_names` from the Variable/attrs/dims and overwrite them in the dict.
3. **Reconcile chunks** with dask/sharding via the existing chunk logic
   (`_determine_zarr_chunks`, alignment checks); write the resolved chunk grid
   into the dict.
4. **Convert format if needed** (see below).
5. **Persist** the array: build `ArrayV{2,3}Metadata.from_dict(canonical_dict)`
   and write its `to_buffer_dict(prototype)` output to the array's store path.
   (`from_dict` constructs but does not persist; the seam owns this persist step.)

### Format conversion

A single `convert_zarr_metadata(metadata_dict, target_format)` function produces a
target-format metadata dict when the target differs from the fragment's
`zarr_format`, or raises when a codec has no cross-format equivalent. This is the
one home for v2⇄v3 logic and is what makes the previously-failing v2→v3 round-trip
succeed.

### Typing

The fragment is annotated as `ArrayMetadataV2 | ArrayMetadataV3` using
`zarr-metadata` `TypedDict`s imported under `TYPE_CHECKING`. Runtime values are
plain dicts from zarr-python's `to_dict`/`from_dict`.

### Future direction: dataclass-based representation

The seam is designed so its backing can later become a dataclass-based metadata
representation from `zarr-metadata` (which the user maintains) instead of plain
dicts. Dataclasses fit the four operations the seam performs: `to_dict`/`from_dict`
round-trip, field override via `dataclasses.replace` (Variable-wins),
`isinstance`-dispatched format conversion, and clean equality/copy/pickle for the
append and dask paths. That would flip decision #6 (making `zarr-metadata` an
optional _runtime_ dependency in the zarr extra) and pin a minimum version.

**Sequencing (decided):** build the seam against plain dicts now so xarray work is
not gated on a `zarr-metadata` release; swap the seam's backing to dataclasses in
a follow-up once those ship upstream. The seam is the single place that change
touches.

## Back-compat & deprecation

- Read continues to emit the flat keys (derived from the fragment).
- Users may still set flat keys; they remain fully supported in this PR (no
  warning). Emitting a `DeprecationWarning` is deferred to a follow-up PR
  (see decision 4), and hard removal is later still.

## Testing

- Round-trip fidelity: v2→v2, v3→v3, and v2→v3 / v3→v2 (currently broken).
- `merge_flat_aliases` defensive `chunks` conflict check (unit-level); end-to-end
  a stale fragment + rechunk resolves to the new chunks (resolved value wins, no error).
- Variable-owned fields (shape/dims/chunk-grid/dtype/fill_value) always win over stale
  fragment fields, end-to-end through `to_zarr`.
- `fill_value` matrix across formats and `use_zarr_fill_value_as_mask`.
- (Deferred to follow-up PR) `DeprecationWarning` on flat-key use.

## Open implementation questions (resolve in the plan)

1. RESOLVED: `from_dict` does not persist; write path persists
   `ArrayV{2,3}Metadata.to_buffer_dict(prototype)` to the store (mechanism "A").
   Migrate to an upstream persisting create-from-metadata API when available.
2. How runtime-only store behaviors absent from the metadata document
   (`write_empty_chunks`/`config`, `overwrite`) are applied alongside `from_dict`.
3. Exact derivation of legacy flat aliases on read — match today's output vs.
   make them strictly format-correct. Leaning: match today's output to avoid a
   second behavior change, since the flat keys are deprecated anyway.

## Future direction: dataclass-based representation

The maintainer of `zarr-metadata` is open to evolving it, so a natural follow-up
is a dataclass-based metadata representation (frozen dataclasses for v2/v3 array
metadata plus a format-`convert` API). That would back the representation seam
instead of dicts and simplify exactly the operations this design leans on:

- field override → `dataclasses.replace(...)` rather than dict-key mutation;
- format conversion → a method/function returning the other dataclass, dispatched
  by type rather than by inspecting a `zarr_format` key;
- equality/copy/pickle for the append-compare and dask paths, for free;
- typing with real classes and no `Any`.

It would flip decision 6 to a runtime (optional) dependency and pin a minimum
`zarr-metadata` version. It is deliberately sequenced after the dict-backed seam
(decision 7) so xarray work is not gated on a cross-repo release. Not in scope
for the first implementation.

## Out of scope

- Removing the flat encoding keys.
- Non-zarr backends (this fragment is zarr-specific; netCDF backends ignore it).
- Changes to the CF coding layer.

## Relation to prior work

Builds on the landed refactor PR (branch `claude/zarr-encoding-refactor`): the
consolidated encoding-key constants and the split validator/create-kwargs
separation. This design subsumes the previously-planned Phase 3 (v2⇄v3 codec
translation layer) and Phase 4 (unifying `fill_value` handling).
