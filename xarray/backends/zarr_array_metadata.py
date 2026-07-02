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
