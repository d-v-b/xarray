"""Seam between xarray's variable `.encoding` and a zarr array's spec metadata.

All handling of the ``zarr_array_metadata`` encoding fragment lives here so the
backing representation (currently a plain dict from zarr-python's ``to_dict``)
can later be swapped for a ``zarr-metadata`` dataclass without touching the
backend call sites.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray.core.types import ZarrArray


def read_metadata_fragment(zarr_array: ZarrArray) -> dict[str, object]:
    """Return the spec metadata document for a zarr array as a plain dict."""
    return dict(zarr_array.metadata.to_dict())


def derive_flat_aliases(
    zarr_array: ZarrArray, dimensions: tuple[str, ...]
) -> dict[str, object]:
    """Build the legacy flat encoding keys from a live zarr (v3) array.

    Reproduces exactly what ``ZarrStore.open_store_variable`` emits on the
    zarr-python-3 runtime, for backward compatibility.
    This function is only invoked under ``_zarr_v3()``; the zarr-python-2
    legacy path is handled in ``open_store_variable`` itself.
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
