# Zarr encoding metadata fragment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry a zarr array's storage-layout metadata in `.encoding` as a single spec-shaped fragment (`zarr_array_metadata`), driven by zarr-python `to_dict`/metadata-buffer persistence, so v2⇄v3 round-trips work and all format-specific logic lives in one seam.

**Architecture:** A new seam module (`xarray/backends/zarr_array_metadata.py`) owns reading the fragment from a zarr array, deriving legacy flat aliases, normalizing flat-key + fragment inputs into one canonical metadata dict (conflict → error), applying Variable-owned field overrides, converting between formats, and persisting a new array by writing `ArrayV{2,3}Metadata.to_buffer_dict()` to the store. `xarray/backends/zarr.py` calls only this seam. The whole mechanism is gated behind `_zarr_v3()`; the zarr-python-2 code path is unchanged.

**Tech Stack:** Python, zarr-python ≥3 (`zarr.core.metadata.ArrayV2Metadata`/`ArrayV3Metadata`, `zarr.core.buffer.default_buffer_prototype`, `zarr.core.sync.sync`, `StorePath`), numcodecs, pytest.

## Global Constraints

- **Gate on zarr-python 3.** All new behavior applies only when `_zarr_v3()` is true. When zarr-python 2 is installed, the existing code path is used unchanged. Copy this guard into every read/write wiring task.
- **Behavior-preserving for same-format round-trips.** v2→v2 and v3→v3 output must be unchanged by this work; only v2⇄v3 gains new capability.
- **No `typing.Any`.** Use precise types; `object`/unions/`TypedDict` where values are heterogeneous. (Standing user preference.)
- **`zarr-metadata` is TYPE_CHECKING-only.** Import its metadata `TypedDict`s only under `if TYPE_CHECKING:`; no runtime dependency.
- **Deprecate, don't remove.** Legacy flat keys keep working; setting a storage-layout flat key warns. Removing flat keys is out of scope.
- **Run tests with:** `uv run pytest <path> -q` from the repo root.
- **Commit trailer:** every commit ends with `Assisted-by: ClaudeCode:claude-opus-4.8` and `Co-authored-by: Claude <noreply@anthropic.com>`.

The canonical metadata dict shape is whatever `zarr_array.metadata.to_dict()` returns. For zarr-python 3.2.1 a v3 dict has keys: `zarr_format`, `node_type`, `shape`, `data_type`, `chunk_grid`, `chunk_key_encoding`, `codecs`, `fill_value`, `attributes`, `storage_transformers`. A v2 dict has: `zarr_format`, `shape`, `chunks`, `dtype`, `compressor`, `filters`, `fill_value`, `order`, `dimension_separator`, `attributes`.

---

### Task 1: Seam module + `read_metadata_fragment`

**Files:**

- Create: `xarray/backends/zarr_array_metadata.py`
- Test: `xarray/tests/test_zarr_array_metadata.py`

**Interfaces:**

- Produces: `read_metadata_fragment(zarr_array) -> dict[str, object]` — returns `zarr_array.metadata.to_dict()`. Sole reason this is a named function (not an inline call): it is the one place the read representation is produced, so the dict→dataclass swap later touches only here.

- [ ] **Step 1: Write the failing test**

```python
# xarray/tests/test_zarr_array_metadata.py
import pytest

from xarray.tests import requires_zarr


@requires_zarr
def test_read_metadata_fragment_v3(tmp_path):
    import zarr

    from xarray.backends.zarr_array_metadata import read_metadata_fragment

    g = zarr.open_group(tmp_path / "g.zarr", mode="w", zarr_format=3)
    a = g.create_array("a", shape=(10,), chunks=(5,), dtype="f8")

    frag = read_metadata_fragment(a)
    assert frag["zarr_format"] == 3
    assert frag["shape"] == (10,) or frag["shape"] == [10]
    assert "codecs" in frag
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_read_metadata_fragment_v3 -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'xarray.backends.zarr_array_metadata'`.

- [ ] **Step 3: Write minimal implementation**

```python
# xarray/backends/zarr_array_metadata.py
"""Seam between xarray's variable `.encoding` and a zarr array's spec metadata.

All handling of the ``zarr_array_metadata`` encoding fragment lives here so the
backing representation (currently a plain dict from zarr-python's ``to_dict``)
can later be swapped for a ``zarr-metadata`` dataclass without touching the
backend call sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray.core.types import ZarrArray


def read_metadata_fragment(zarr_array: ZarrArray) -> dict[str, object]:
    """Return the spec metadata document for a zarr array as a plain dict."""
    return dict(zarr_array.metadata.to_dict())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_read_metadata_fragment_v3 -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add xarray/backends/zarr_array_metadata.py xarray/tests/test_zarr_array_metadata.py
git commit -m "feat(zarr): add metadata-fragment seam with read_metadata_fragment

Assisted-by: ClaudeCode:claude-opus-4.8
Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 2: `derive_flat_aliases`

Derive the legacy flat encoding keys from a fragment so read output stays backward-compatible. Per spec open-question #3 (leaning "match today's output"), derive the keys today's `open_store_variable` emits: `chunks`, `filters`, and — because that path currently keys on the installed library — `compressors`/`shards`/`serializer` for a zarr-python-3 runtime regardless of store format, plus `compressor` only for genuinely-v2 arrays read under zarr-python 2 (unchanged legacy path, not produced here).

**Files:**

- Modify: `xarray/backends/zarr_array_metadata.py`
- Test: `xarray/tests/test_zarr_array_metadata.py`

**Interfaces:**

- Consumes: a fragment dict from `read_metadata_fragment`, plus the live `zarr_array` (the codec/shard objects xarray currently stores are the array's live attributes, not the JSON forms).
- Produces: `derive_flat_aliases(zarr_array, dimensions: tuple[str, ...]) -> dict[str, object]` returning the flat keys `chunks`, `preferred_chunks`, `compressors`, `filters`, `shards`, and `serializer` (v3-format arrays only) exactly as `open_store_variable` builds them today.

- [ ] **Step 1: Write the failing test**

```python
@requires_zarr
def test_derive_flat_aliases_matches_live_attrs(tmp_path):
    import zarr

    from xarray.backends.zarr_array_metadata import derive_flat_aliases

    g = zarr.open_group(tmp_path / "g.zarr", mode="w", zarr_format=3)
    a = g.create_array("a", shape=(10,), chunks=(5,), dtype="f8")

    aliases = derive_flat_aliases(a, ("x",))
    assert aliases["chunks"] == a.chunks
    assert aliases["preferred_chunks"] == {"x": a.chunks[0]}
    assert aliases["compressors"] == a.compressors
    assert aliases["filters"] == a.filters
    assert aliases["shards"] == a.shards
    assert aliases["serializer"] == a.serializer
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_derive_flat_aliases_matches_live_attrs -q`
Expected: FAIL — `ImportError: cannot import name 'derive_flat_aliases'`.

- [ ] **Step 3: Write minimal implementation**

Mirror exactly the keys `ZarrStore.open_store_variable` sets today (see `xarray/backends/zarr.py` read path), so this becomes the single source for those aliases.

```python
def derive_flat_aliases(
    zarr_array: ZarrArray, dimensions: tuple[str, ...]
) -> dict[str, object]:
    """Build the legacy flat encoding keys from a live zarr (v3) array.

    Reproduces exactly what ``ZarrStore.open_store_variable`` emitted before the
    metadata fragment existed, for backward compatibility.
    """
    aliases: dict[str, object] = {
        "chunks": zarr_array.chunks,
        "preferred_chunks": dict(zip(dimensions, zarr_array.chunks, strict=True)),
        "compressors": zarr_array.compressors,
        "filters": zarr_array.filters,
        "shards": zarr_array.shards,
    }
    if zarr_array.metadata.zarr_format == 3:
        aliases["serializer"] = zarr_array.serializer
    return aliases
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_derive_flat_aliases_matches_live_attrs -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add xarray/backends/zarr_array_metadata.py xarray/tests/test_zarr_array_metadata.py
git commit -m "feat(zarr): derive legacy flat aliases from a zarr array in the seam

Assisted-by: ClaudeCode:claude-opus-4.8
Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 3: Wire the read path to populate the fragment

Add `encoding["zarr_array_metadata"]` in `open_store_variable`, keeping the flat aliases (now sourced from the seam). Gated on `_zarr_v3()`; the zarr-python-2 branch is untouched.

**Files:**

- Modify: `xarray/backends/zarr.py` — `ZarrStore.open_store_variable` (the `encoding = {...}` block and the `_zarr_v3()` branch that follows).
- Test: `xarray/tests/test_backends.py`

**Interfaces:**

- Consumes: `read_metadata_fragment`, `derive_flat_aliases` from Task 1–2.
- Produces: variables opened from a v3-runtime store carry `encoding["zarr_array_metadata"]` (a dict) in addition to the existing flat keys.

- [ ] **Step 1: Write the failing test**

```python
# xarray/tests/test_backends.py (near the other zarr encoding tests)
@requires_zarr
def test_open_populates_zarr_array_metadata(tmp_path):
    import xarray as xr
    from xarray.backends.zarr import _zarr_v3

    if not _zarr_v3():
        pytest.skip("requires zarr-python 3")

    ds = xr.Dataset({"a": ("x", [1.0, 2.0, 3.0, 4.0])}).chunk({"x": 2})
    ds.to_zarr(tmp_path / "s.zarr", zarr_format=3, mode="w")

    opened = xr.open_zarr(tmp_path / "s.zarr")
    frag = opened["a"].encoding["zarr_array_metadata"]
    assert frag["zarr_format"] == 3
    # legacy flat keys still present
    assert "chunks" in opened["a"].encoding
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest "xarray/tests/test_backends.py::test_open_populates_zarr_array_metadata" -q`
Expected: FAIL — `KeyError: 'zarr_array_metadata'`.

- [ ] **Step 3: Write minimal implementation**

In `xarray/backends/zarr.py`, add the import near the top:

```python
from xarray.backends.zarr_array_metadata import (
    read_metadata_fragment,
)
```

In `open_store_variable`, after the existing `encoding = {...}` and `_zarr_v3()` alias block, add (inside the `if _zarr_v3():` branch that already exists):

```python
encoding["zarr_array_metadata"] = read_metadata_fragment(zarr_array)
```

Leave the existing flat-key assignments in place for this task (they already match `derive_flat_aliases`; a later cleanup can route them through the seam, but that is not required for behavior).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest "xarray/tests/test_backends.py::test_open_populates_zarr_array_metadata" -q`
Expected: PASS.

- [ ] **Step 5: Run the zarr encoding regression slice**

Run: `uv run pytest xarray/tests/test_backends.py -k "zarr and encoding" -q`
Expected: PASS (no regressions; count ≥ the pre-change 156).

- [ ] **Step 6: Commit**

```bash
git add xarray/backends/zarr.py xarray/tests/test_backends.py
git commit -m "feat(zarr): populate encoding['zarr_array_metadata'] on read

Assisted-by: ClaudeCode:claude-opus-4.8
Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 4: `merge_flat_aliases` — normalize with conflict detection

Fold any legacy flat storage-layout keys present in a variable's `.encoding` into the fragment dict; if a flat key disagrees with the fragment's corresponding field, raise `ValueError` naming the field.

**Files:**

- Modify: `xarray/backends/zarr_array_metadata.py`
- Test: `xarray/tests/test_zarr_array_metadata.py`

**Interfaces:**

- Produces: `merge_flat_aliases(fragment: dict[str, object], encoding: Mapping[str, object]) -> dict[str, object]`. Returns a new fragment dict. Mapping of flat key → fragment field is fixed: `chunks`→(`chunk_grid` for v3 / `chunks` for v2), `fill_value`→`fill_value`. Codec flat keys (`compressors`/`compressor`/`filters`/`serializer`/`shards`) are compared against the fragment's `codecs` (v3) or `compressor`/`filters` (v2) only for presence/None; a non-None codec flat key that differs from the fragment raises (they are objects, not JSON — compare via the fragment already carrying them). For the first cut, implement `chunks` and `fill_value` reconciliation (the numeric/tuple fields where equality is well-defined) and pass codec keys through untouched (the fragment is authoritative for codecs on read).

- [ ] **Step 1: Write the failing test**

```python
@requires_zarr
def test_merge_flat_aliases_conflict_raises():
    from xarray.backends.zarr_array_metadata import merge_flat_aliases

    fragment = {
        "zarr_format": 3,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": (5,)},
        },
    }
    # agreeing chunks: no error, returns fragment unchanged for that field
    out = merge_flat_aliases(fragment, {"chunks": (5,)})
    assert out["chunk_grid"]["configuration"]["chunk_shape"] == (5,)

    # disagreeing chunks: raise, naming the field
    with pytest.raises(ValueError, match=r"chunks"):
        merge_flat_aliases(fragment, {"chunks": (10,)})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_merge_flat_aliases_conflict_raises -q`
Expected: FAIL — `ImportError: cannot import name 'merge_flat_aliases'`.

- [ ] **Step 3: Write minimal implementation**

```python
from collections.abc import Mapping


def _fragment_chunk_shape(fragment: Mapping[str, object]) -> tuple[int, ...] | None:
    if fragment.get("zarr_format") == 3:
        grid = fragment.get("chunk_grid")
        if isinstance(grid, Mapping):
            config = grid.get("configuration")
            if isinstance(config, Mapping):
                shape = config.get("chunk_shape")
                return tuple(shape) if shape is not None else None
        return None
    chunks = fragment.get("chunks")
    return tuple(chunks) if chunks is not None else None


def merge_flat_aliases(
    fragment: dict[str, object], encoding: Mapping[str, object]
) -> dict[str, object]:
    """Fold legacy flat keys into ``fragment``; raise on disagreement."""
    result = dict(fragment)

    if "chunks" in encoding and encoding["chunks"] is not None:
        flat = tuple(encoding["chunks"])  # type: ignore[arg-type]
        frag_chunks = _fragment_chunk_shape(result)
        if frag_chunks is not None and frag_chunks != flat:
            raise ValueError(
                "conflicting 'chunks': encoding has "
                f"{flat!r} but zarr_array_metadata has {frag_chunks!r}"
            )

    if "fill_value" in encoding and "fill_value" in result:
        if encoding["fill_value"] != result["fill_value"]:
            raise ValueError(
                "conflicting 'fill_value': encoding has "
                f"{encoding['fill_value']!r} but zarr_array_metadata has "
                f"{result['fill_value']!r}"
            )

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_merge_flat_aliases_conflict_raises -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add xarray/backends/zarr_array_metadata.py xarray/tests/test_zarr_array_metadata.py
git commit -m "feat(zarr): merge flat aliases into fragment with conflict detection

Assisted-by: ClaudeCode:claude-opus-4.8
Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 5: `apply_variable_fields` — Variable wins

Overwrite the fragment's xarray-owned fields (`shape`, `data_type`/`dtype`, `attributes`, `dimension_names`) from the Variable so a stale fragment can never contradict the data being written.

**Files:**

- Modify: `xarray/backends/zarr_array_metadata.py`
- Test: `xarray/tests/test_zarr_array_metadata.py`

**Interfaces:**

- Produces: `apply_variable_fields(fragment: dict[str, object], *, shape: tuple[int, ...], dims: tuple[str, ...]) -> dict[str, object]`. Sets `shape` in the dict; for v3 sets `dimension_names = dims`; leaves `data_type`/`attributes` to be handled by the create call (dtype is passed separately by the write path, attributes via xarray's attrs). Returns a new dict.

- [ ] **Step 1: Write the failing test**

```python
@requires_zarr
def test_apply_variable_fields_overrides_shape_and_dims():
    from xarray.backends.zarr_array_metadata import apply_variable_fields

    fragment = {"zarr_format": 3, "shape": (99,), "dimension_names": ("stale",)}
    out = apply_variable_fields(fragment, shape=(4,), dims=("x",))
    assert out["shape"] == (4,)
    assert out["dimension_names"] == ("x",)
    # input not mutated
    assert fragment["shape"] == (99,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_apply_variable_fields_overrides_shape_and_dims -q`
Expected: FAIL — `ImportError: cannot import name 'apply_variable_fields'`.

- [ ] **Step 3: Write minimal implementation**

```python
def apply_variable_fields(
    fragment: dict[str, object],
    *,
    shape: tuple[int, ...],
    dims: tuple[str, ...],
) -> dict[str, object]:
    """Overwrite xarray-owned fields in the fragment from the Variable."""
    result = dict(fragment)
    result["shape"] = shape
    if result.get("zarr_format") == 3:
        result["dimension_names"] = dims
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_apply_variable_fields_overrides_shape_and_dims -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add xarray/backends/zarr_array_metadata.py xarray/tests/test_zarr_array_metadata.py
git commit -m "feat(zarr): apply Variable-owned fields over the fragment

Assisted-by: ClaudeCode:claude-opus-4.8
Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 6: `convert_zarr_metadata` — v2⇄v3 (RISKIEST; includes a verification step)

Convert a fragment between formats. This is the one place codec representations are translated, and the exact zarr-python conversion helpers must be confirmed before coding — do not guess.

**Files:**

- Modify: `xarray/backends/zarr_array_metadata.py`
- Test: `xarray/tests/test_zarr_array_metadata.py`

**Interfaces:**

- Produces: `convert_zarr_metadata(fragment: dict[str, object], target_format: Literal[2, 3]) -> dict[str, object]`. If `fragment["zarr_format"] == target_format`, return it unchanged. Otherwise produce a target-format dict, raising `NotImplementedError` (message naming the codec) when a codec has no cross-format equivalent.

- [ ] **Step 1: Verification spike — find the supported conversion path**

Run this and record the output in the task notes; it determines the implementation:

```bash
uv run python - <<'PY'
import zarr, numcodecs, inspect
# v3 codec classes available for mapping
import zarr.codecs as c3
print("zarr.codecs:", [n for n in dir(c3) if n.endswith("Codec")])
# numcodecs<->v3 bridge?
try:
    import numcodecs.zarr3 as nz3
    print("numcodecs.zarr3:", [n for n in dir(nz3) if not n.startswith("_")][:20])
except Exception as e:
    print("no numcodecs.zarr3:", e)
# Does ArrayV2Metadata expose a to-v3 helper?
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
print("V2 methods:", [m for m in dir(ArrayV2Metadata) if "v3" in m.lower() or "convert" in m.lower()])
PY
```

Expected: a list of v3 codec classes (e.g. `BloscCodec`, `BytesCodec`, `GzipCodec`, `ZstdCodec`) and whether `numcodecs.zarr3` provides wrapper codecs. Use the discovered mapping in Step 3. If `numcodecs.zarr3` exists, prefer wrapping v2 numcodecs via it; otherwise map by codec name for blosc/gzip/zstd/blosc and raise `NotImplementedError` for anything else.

- [ ] **Step 2: Write the failing test**

```python
@requires_zarr
def test_convert_metadata_same_format_is_identity():
    from xarray.backends.zarr_array_metadata import convert_zarr_metadata

    frag = {"zarr_format": 3, "codecs": []}
    assert (
        convert_zarr_metadata(frag, 3) is frag or convert_zarr_metadata(frag, 3) == frag
    )


@requires_zarr
def test_convert_metadata_v2_to_v3_roundtrips_chunks(tmp_path):
    import zarr

    from xarray.backends.zarr_array_metadata import (
        convert_zarr_metadata,
        read_metadata_fragment,
    )

    g = zarr.open_group(tmp_path / "v2.zarr", mode="w", zarr_format=2)
    a = g.create_array("a", shape=(10,), chunks=(5,), dtype="f8", compressors=None)
    v2 = read_metadata_fragment(a)

    v3 = convert_zarr_metadata(v2, 3)
    assert v3["zarr_format"] == 3
    # chunk shape preserved across the conversion
    assert tuple(v3["chunk_grid"]["configuration"]["chunk_shape"]) == (5,)
```

- [ ] **Step 3: Write implementation using the verified mapping**

Implement `convert_zarr_metadata` per Step 1's findings. Structure: identity fast-path; build the target dict field-by-field (`shape`, `data_type`/`dtype`, chunk shape → `chunk_grid`/`chunks`, `fill_value`, `attributes`); translate codecs via the verified mechanism; `raise NotImplementedError(f"no zarr v{target} equivalent for codec {name!r}")` on unmapped codecs. (Code omitted here intentionally — it is written against the Step 1 output, not from memory, to avoid an incorrect codec mapping.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py -k convert -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add xarray/backends/zarr_array_metadata.py xarray/tests/test_zarr_array_metadata.py
git commit -m "feat(zarr): convert array metadata between zarr v2 and v3

Assisted-by: ClaudeCode:claude-opus-4.8
Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 7: `persist_array` — write a new array from a canonical metadata dict

**Files:**

- Modify: `xarray/backends/zarr_array_metadata.py`
- Test: `xarray/tests/test_zarr_array_metadata.py`

**Interfaces:**

- Consumes: a canonical, target-format fragment dict.
- Produces: `persist_array(store_path, fragment: dict[str, object]) -> None` — builds `ArrayV2Metadata`/`ArrayV3Metadata.from_dict(fragment)` (dispatched on `fragment["zarr_format"]`) and writes each `to_buffer_dict(default_buffer_prototype())` entry to `store_path / key`. Verified working on zarr-python 3.2.1.

- [ ] **Step 1: Write the failing test**

```python
@requires_zarr
def test_persist_array_roundtrips(tmp_path):
    import zarr
    from zarr.storage import LocalStore, StorePath

    from xarray.backends.zarr_array_metadata import (
        persist_array,
        read_metadata_fragment,
    )

    g = zarr.open_group(tmp_path / "src.zarr", mode="w", zarr_format=3)
    src = g.create_array("a", shape=(10,), chunks=(5,), dtype="f8")
    frag = read_metadata_fragment(src)

    store = LocalStore(str(tmp_path / "dst.zarr"))
    persist_array(StorePath(store, "a"), frag)

    reopened = zarr.open_array(str(tmp_path / "dst.zarr"), path="a", mode="r")
    assert reopened.shape == (10,)
    assert reopened.chunks == (5,)
    assert reopened.metadata.to_dict()["codecs"] == frag["codecs"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_persist_array_roundtrips -q`
Expected: FAIL — `ImportError: cannot import name 'persist_array'`.

- [ ] **Step 3: Write minimal implementation**

```python
def persist_array(store_path, fragment: dict[str, object]) -> None:
    """Persist a new zarr array from a canonical metadata dict.

    ``ArrayV{2,3}Metadata.from_dict`` builds an in-memory object that does NOT
    write to the store, so we serialize its buffers explicitly.
    """
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
    from zarr.core.sync import sync

    if fragment.get("zarr_format") == 2:
        meta = ArrayV2Metadata.from_dict(fragment)
    else:
        meta = ArrayV3Metadata.from_dict(fragment)

    async def _write() -> None:
        buffers = meta.to_buffer_dict(default_buffer_prototype())
        for key, buffer in buffers.items():
            await (store_path / key).set(buffer)

    sync(_write())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_persist_array_roundtrips -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add xarray/backends/zarr_array_metadata.py xarray/tests/test_zarr_array_metadata.py
git commit -m "feat(zarr): persist a new array from a canonical metadata dict

Assisted-by: ClaudeCode:claude-opus-4.8
Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 8: `build_canonical_metadata` — orchestrator

Tie Tasks 4–6 together: start from `encoding["zarr_array_metadata"]` if present (else from live-array aliases via a minimal synthesized fragment), merge flat aliases (conflict → error), apply Variable fields, convert to the target format, and inject the resolved chunk grid.

**Files:**

- Modify: `xarray/backends/zarr_array_metadata.py`
- Test: `xarray/tests/test_zarr_array_metadata.py`

**Interfaces:**

- Consumes: `merge_flat_aliases`, `apply_variable_fields`, `convert_zarr_metadata`.
- Produces: `build_canonical_metadata(encoding: Mapping[str, object], *, shape, dims, target_format, resolved_chunks: tuple[int, ...]) -> dict[str, object]`. Requires `encoding["zarr_array_metadata"]` to be present (write path guarantees it for round-tripped data; for freshly-created in-memory data with no fragment, the write path falls back to the legacy create path — see Task 9). Sets the chunk grid to `resolved_chunks`.

- [ ] **Step 1: Write the failing test**

```python
@requires_zarr
def test_build_canonical_metadata_v3(tmp_path):
    import zarr

    from xarray.backends.zarr_array_metadata import (
        build_canonical_metadata,
        read_metadata_fragment,
    )

    g = zarr.open_group(tmp_path / "g.zarr", mode="w", zarr_format=3)
    a = g.create_array("a", shape=(10,), chunks=(5,), dtype="f8")
    encoding = {"zarr_array_metadata": read_metadata_fragment(a)}

    out = build_canonical_metadata(
        encoding, shape=(8,), dims=("x",), target_format=3, resolved_chunks=(4,)
    )
    assert out["zarr_format"] == 3
    assert out["shape"] == (8,)
    assert out["dimension_names"] == ("x",)
    assert tuple(out["chunk_grid"]["configuration"]["chunk_shape"]) == (4,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_build_canonical_metadata_v3 -q`
Expected: FAIL — `ImportError: cannot import name 'build_canonical_metadata'`.

- [ ] **Step 3: Write minimal implementation**

```python
def _set_chunk_shape(fragment: dict[str, object], chunks: tuple[int, ...]) -> None:
    if fragment.get("zarr_format") == 3:
        fragment["chunk_grid"] = {
            "name": "regular",
            "configuration": {"chunk_shape": tuple(chunks)},
        }
    else:
        fragment["chunks"] = tuple(chunks)


def build_canonical_metadata(
    encoding: Mapping[str, object],
    *,
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    target_format: int,
    resolved_chunks: tuple[int, ...],
) -> dict[str, object]:
    """Produce the canonical, target-format metadata dict for a write."""
    fragment = encoding["zarr_array_metadata"]
    if not isinstance(fragment, dict):
        raise TypeError("encoding['zarr_array_metadata'] must be a dict")

    fragment = merge_flat_aliases(fragment, encoding)
    fragment = convert_zarr_metadata(fragment, target_format)
    fragment = apply_variable_fields(fragment, shape=shape, dims=dims)
    _set_chunk_shape(fragment, resolved_chunks)
    return fragment
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest xarray/tests/test_zarr_array_metadata.py::test_build_canonical_metadata_v3 -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add xarray/backends/zarr_array_metadata.py xarray/tests/test_zarr_array_metadata.py
git commit -m "feat(zarr): orchestrate canonical metadata assembly in the seam

Assisted-by: ClaudeCode:claude-opus-4.8
Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 9: Wire the write path through the seam

Route new-array creation in `ZarrStore.set_variables`/`_create_new_array` through `build_canonical_metadata` + `persist_array` **when** the variable carries a `zarr_array_metadata` fragment and `_zarr_v3()`; otherwise use the existing create path unchanged (covers fresh in-memory data and zarr-python 2). Preserve `write_empty_chunks`/`overwrite` semantics: after persisting, open the array and apply the store-level write-empty config as the legacy path does.

**Files:**

- Modify: `xarray/backends/zarr.py` — `_create_new_array` (branch on fragment presence) and its caller in `set_variables`.
- Test: `xarray/tests/test_backends.py`

**Interfaces:**

- Consumes: `build_canonical_metadata`, `persist_array` and `self.zarr_group.store_path`.
- Produces: v2⇄v3 round-trips succeed; same-format round-trips unchanged.

- [ ] **Step 1: Write the failing test (the currently-broken v2→v3 case)**

```python
@requires_zarr
def test_v2_to_v3_roundtrip_with_compression(tmp_path):
    import numpy as np

    import xarray as xr
    from xarray.backends.zarr import _zarr_v3

    if not _zarr_v3():
        pytest.skip("requires zarr-python 3")

    ds = xr.Dataset({"a": ("x", np.arange(10.0))}).chunk({"x": 5})
    ds.to_zarr(tmp_path / "v2.zarr", zarr_format=2, mode="w")
    opened = xr.open_zarr(tmp_path / "v2.zarr")

    # Previously raised: TypeError: Expected a BytesBytesCodec ...
    opened.to_zarr(tmp_path / "v3.zarr", zarr_format=3, mode="w")

    back = xr.open_zarr(tmp_path / "v3.zarr")
    xr.testing.assert_identical(back.compute(), ds.compute())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest "xarray/tests/test_backends.py::test_v2_to_v3_roundtrip_with_compression" -q`
Expected: FAIL — `TypeError: Expected a BytesBytesCodec. Got <class 'numcodecs.blosc.Blosc'>`.

- [ ] **Step 3: Write implementation**

In `_create_new_array`, before the existing legacy body, add the fragment path:

```python
fragment = encoding.get("zarr_array_metadata")
if _zarr_v3() and isinstance(fragment, dict):
    from xarray.backends.zarr_array_metadata import (
        build_canonical_metadata,
        persist_array,
    )

    target_format = 3 if self.zarr_group.metadata.zarr_format == 3 else 2
    canonical = build_canonical_metadata(
        encoding,
        shape=shape,
        dims=dims,
        target_format=target_format,
        resolved_chunks=encoding["chunks"],
    )
    store_path = self.zarr_group.store_path / name
    persist_array(store_path, canonical)
    zarr_array = self._open_existing_array(name=name)
    return _put_attrs(zarr_array, attrs)
```

Pass `dims` into `_create_new_array` (Task from the refactor PR already added `dims`). Keep the existing legacy body below as the fallback for `fragment is None` or zarr-python 2. Note: `resolved_chunks` must be the tuple already computed by `extract_zarr_variable_encoding`/`_determine_zarr_chunks` in `set_variables` (`encoding["chunks"]`); if it is the string `"auto"`, fall through to the legacy path (no explicit chunk grid).

- [ ] **Step 4: Run the new test and the same-format regressions**

Run: `uv run pytest "xarray/tests/test_backends.py::test_v2_to_v3_roundtrip_with_compression" -q`
Expected: PASS.

Run: `uv run pytest xarray/tests/test_backends.py -k "zarr and (encoding or roundtrip or write_empty or shard or append or region)" -q`
Expected: PASS (no regressions).

- [ ] **Step 5: Commit**

```bash
git add xarray/backends/zarr.py xarray/tests/test_backends.py
git commit -m "feat(zarr): create arrays from the metadata fragment, enabling v2<->v3

Assisted-by: ClaudeCode:claude-opus-4.8
Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 10: Deprecation warning for legacy flat storage-layout keys — DEFERRED

**DEFERRED to a separate follow-up PR (not implemented in this branch).** A trial
implementation turned ~296 existing tests red because setting `chunks`/`compressors`
via `encoding=` is ubiquitous and the suite treats the warning as an error.
Deprecating the common flat keys needs its own scoped PR (decide which keys to
deprecate, and reconcile the test suite). This branch adds `zarr_array_metadata`
as the preferred path without deprecating anything. The original task text is
retained below for that future PR.

Emit a `DeprecationWarning` when a user explicitly sets a flat storage-layout key (`chunks`, `compressor`, `compressors`, `filters`, `serializer`, `shards`, `fill_value`) via the `to_zarr(encoding=...)` argument, pointing them at `zarr_array_metadata`. Only warn for user-supplied encoding (the `check_encoding_set` variables), not for round-tripped `.encoding`.

**Files:**

- Modify: `xarray/backends/zarr.py` — in `set_variables`, where `raise_on_invalid=vn in check_encoding_set` is known.
- Test: `xarray/tests/test_backends.py`

**Interfaces:**

- Consumes: `check_encoding_set`, the constant `ZARR_ENCODING_KEYS` (from the refactor PR).
- Produces: a `DeprecationWarning` on user-set flat keys; no warning on round-trip.

- [ ] **Step 1: Write the failing test**

```python
@requires_zarr
def test_flat_encoding_key_deprecation_warns(tmp_path):
    import numpy as np

    import xarray as xr
    from xarray.backends.zarr import _zarr_v3

    if not _zarr_v3():
        pytest.skip("requires zarr-python 3")

    ds = xr.Dataset({"a": ("x", np.arange(4.0))})
    with pytest.warns(DeprecationWarning, match="zarr_array_metadata"):
        ds.to_zarr(
            tmp_path / "s.zarr",
            zarr_format=3,
            mode="w",
            encoding={"a": {"chunks": (2,)}},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest "xarray/tests/test_backends.py::test_flat_encoding_key_deprecation_warns" -q`
Expected: FAIL — no warning raised.

- [ ] **Step 3: Write implementation**

In `set_variables`, when `vn in check_encoding_set`, before extracting encoding:

```python
if vn in check_encoding_set:
    _deprecated = ZARR_ENCODING_KEYS & set(v.encoding)
    if _deprecated:
        emit_user_level_warning(
            f"Setting zarr storage-layout encoding keys "
            f"{sorted(_deprecated)!r} is deprecated; set "
            "encoding['zarr_array_metadata'] instead.",
            DeprecationWarning,
        )
```

(`emit_user_level_warning` is already imported in `zarr.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest "xarray/tests/test_backends.py::test_flat_encoding_key_deprecation_warns" -q`
Expected: PASS.

- [ ] **Step 5: Guard existing tests against the new warning**

Some existing tests set flat encoding keys and assert no warnings. Run the encoding slice and add `@pytest.mark.filterwarnings("ignore::DeprecationWarning")` only where a test legitimately exercises the legacy keys:

Run: `uv run pytest xarray/tests/test_backends.py -k "zarr and encoding" -q`
Expected: PASS (fix any newly-failing warning assertions by ignoring the deprecation in those specific legacy tests).

- [ ] **Step 6: Commit**

```bash
git add xarray/backends/zarr.py xarray/tests/test_backends.py
git commit -m "feat(zarr): deprecate flat storage-layout encoding keys

Assisted-by: ClaudeCode:claude-opus-4.8
Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 11: Typing + docs

Add the TYPE_CHECKING-only `zarr-metadata` annotation for the fragment and document the new encoding key.

**Files:**

- Modify: `xarray/backends/zarr_array_metadata.py` (typed alias), `pyproject.toml` (add `zarr-metadata` to the typing/test extra only), `doc/user-guide/io.rst` (zarr encoding section) or the zarr-encoding internals doc.
- Test: `uv run dmypy run` and doc build are the checks.

**Interfaces:**

- Produces: `ZarrArrayMetadata` type alias = `ArrayMetadataV2 | ArrayMetadataV3` (TYPE_CHECKING only), used to annotate fragment params/returns in the seam.

- [ ] **Step 1: Add the typing import and alias**

```python
# top of xarray/backends/zarr_array_metadata.py
if TYPE_CHECKING:
    from zarr_metadata import (
        ArrayMetadataV2,
        ArrayMetadataV3,
    )  # exact names per package

    ZarrArrayMetadata = ArrayMetadataV2 | ArrayMetadataV3
```

Replace `dict[str, object]` fragment annotations in the seam's signatures with `ZarrArrayMetadata` where they represent a full fragment (keep `dict[str, object]` for partially-built dicts). Confirm exact exported names by running:

```bash
uv run python -c "import zarr_metadata; print([n for n in dir(zarr_metadata) if 'Array' in n])"
```

(Install into the typing env first if needed: it is a dev/typing-only dependency.)

- [ ] **Step 2: Add the typing/test dependency**

In `pyproject.toml`, add `zarr-metadata` to the existing typing/test dependency group (NOT the runtime deps, NOT the `zarr` extra). Match the file's existing formatting.

- [ ] **Step 3: Type-check**

Run: `uv run dmypy run`
Expected: no new errors in `xarray/backends/zarr_array_metadata.py` or `zarr.py`.

- [ ] **Step 4: Document the encoding key**

Add a short subsection to the zarr I/O docs describing `encoding["zarr_array_metadata"]`: what it holds, that it is the preferred way to control zarr storage layout, that the flat keys (`chunks`, `compressors`, …) remain fully supported as aliases (no deprecation in this PR), and that it enables v2⇄v3 conversion.

- [ ] **Step 5: Commit**

```bash
git add xarray/backends/zarr_array_metadata.py pyproject.toml doc/
git commit -m "docs(zarr): type the metadata fragment and document zarr_array_metadata

Assisted-by: ClaudeCode:claude-opus-4.8
Co-authored-by: Claude <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the seam unit tests: `uv run pytest xarray/tests/test_zarr_array_metadata.py -q` — all pass.
- [ ] Run the full zarr backend slice: `uv run pytest xarray/tests/test_backends.py -k zarr -q` — no regressions.
- [ ] Type check: `uv run dmypy run` — clean for changed files.
- [ ] `pre-commit run --all-files` — clean.

## Notes on sequencing / future work (not tasks here)

- The seam (`zarr_array_metadata.py`) is the single place to later (a) swap the dict backing for a `zarr-metadata` dataclass and (b) replace `persist_array`'s buffer-writing with an upstream persisting create-from-metadata API. Neither changes `zarr.py` call sites.
- Codec conversion (Task 6) is the only task with an unresolved external API detail; its Step 1 spike must run before Step 3.

### Candidate upstream (zarr-python / zarr-metadata) improvements

The user maintains these packages, so improving them is preferred over
xarray-side workarounds where they belong upstream. Build the seam against the
current APIs now (unblocked); migrate to these when they land — each touches only
the seam, not `zarr.py`:

1. **Persisting create-from-metadata (zarr-python).** A public entry point that
   writes a new array from an `ArrayV{2,3}Metadata` (or metadata dict) to a store
   path — e.g. `Array.create_from_dict` / a persisting `from_dict`. Replaces
   `persist_array`'s reliance on `to_buffer_dict` + manual buffer writes (Task 7).
2. **v2⇄v3 codec/metadata conversion (zarr-python or zarr-metadata).** A
   spec-level `convert(metadata, target_format)` (or `ArrayV2Metadata.to_v3()`)
   owning the numcodecs⇄v3-codec mapping. Replaces xarray's in-seam
   `convert_zarr_metadata` (Task 6), which is the riskiest xarray-side code and is
   really a spec concern that belongs upstream. If pursued first, Task 6 becomes a
   thin call to that API instead of a bespoke mapping.
