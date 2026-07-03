"""Seam between xarray's variable `.encoding` and a zarr array's spec metadata.

All handling of the ``zarr_array_metadata`` encoding fragment lives here so the
backing representation (currently a plain dict from zarr-python's ``to_dict``)
can later be swapped for a ``zarr-metadata`` dataclass without touching the
backend call sites.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from zarr.storage import StorePath
    from zarr_metadata import ArrayMetadataV2, ArrayMetadataV3

    from xarray.core.types import ZarrArray

    ZarrArrayMetadata = ArrayMetadataV2 | ArrayMetadataV3

#: v2 compressor/filter ``id`` values that have a codec with a directly
#: equivalent name in ``zarr.codecs`` (native zarr-python v3 codecs, not the
#: ``zarr.codecs.numcodecs`` wrapper codecs). Verified against zarr-python
#: 3.2.1: ``ArrayV2Metadata`` exposes no v2<->v3 conversion helper, and
#: ``numcodecs.zarr3`` only re-exports numcodecs-as-v3-codec wrappers (e.g.
#: ``numcodecs.blosc``) rather than translating a v2 metadata dict into v3
#: metadata. Native v3 codecs with the same on-disk format as their v2
#: numcodecs counterpart exist only for gzip/blosc/zstd; this mapping is
#: intentionally limited to those.
_V2_TO_V3_COMPRESSOR_NAMES = frozenset({"blosc", "gzip", "zstd"})

# v2 blosc "shuffle" is a small int; v3 BloscCodec wants the string name.
_BLOSC_SHUFFLE_V2_TO_V3: dict[object, object] = {
    0: "noshuffle",
    1: "shuffle",
    2: "bitshuffle",
}
_BLOSC_SHUFFLE_V3_TO_V2: dict[object, object] = {
    v: k for k, v in _BLOSC_SHUFFLE_V2_TO_V3.items()
}


def read_metadata_fragment(zarr_array: ZarrArray) -> ZarrArrayMetadata:
    """Return the spec metadata document for a zarr array as a plain dict."""
    return dict(zarr_array.metadata.to_dict())  # type: ignore[return-value]


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


def _as_int_tuple(value: object) -> tuple[int, ...]:
    """Narrow an ``object`` known to be an iterable of ints to ``tuple[int, ...]``."""
    if not isinstance(value, Iterable):
        raise TypeError(f"expected an iterable of ints, got {value!r}")
    return tuple(int(v) for v in value)


def _fragment_chunk_shape(fragment: Mapping[str, object]) -> tuple[int, ...] | None:
    if fragment.get("zarr_format") == 3:
        grid = fragment.get("chunk_grid")
        if isinstance(grid, Mapping):
            config = grid.get("configuration")
            if isinstance(config, Mapping):
                shape = config.get("chunk_shape")
                return _as_int_tuple(shape) if shape is not None else None
        return None
    chunks = fragment.get("chunks")
    return _as_int_tuple(chunks) if chunks is not None else None


def merge_flat_aliases(
    fragment: ZarrArrayMetadata, encoding: Mapping[str, object]
) -> ZarrArrayMetadata:
    """Fold legacy flat keys into ``fragment``; raise on disagreement.

    This only defensively checks ``chunks`` at the seam boundary. There is no
    analogous ``fill_value`` check: ``build_canonical_metadata`` (the sole
    caller) always overwrites ``fragment["fill_value"]`` authoritatively via
    ``_set_fill_value`` immediately afterward, so any comparison here would be
    both discarded and unreliable -- ``encoding["fill_value"]`` is a raw
    Python/numpy scalar while ``result["fill_value"]`` is the fragment's
    JSON-serialized form (e.g. the string ``"NaN"`` for a v3 NaN fill value),
    and a naive ``!=`` between those representations can spuriously disagree
    on values that actually agree (e.g. ``nan != nan``).
    """
    result: dict[str, object] = dict(fragment)

    if "chunks" in encoding and encoding["chunks"] is not None:
        flat: tuple[int, ...] = _as_int_tuple(encoding["chunks"])
        frag_chunks = _fragment_chunk_shape(result)
        if frag_chunks is not None and frag_chunks != flat:
            raise ValueError(
                "conflicting 'chunks': encoding has "
                f"{flat!r} but zarr_array_metadata has {frag_chunks!r}"
            )

    # `result` is a faithful copy of `fragment`, whose shape mypy cannot infer
    # through `dict(...)`; narrow it back to the TypedDict union it started as.
    return cast("ZarrArrayMetadata", result)


def apply_variable_fields(
    fragment: ZarrArrayMetadata,
    *,
    shape: tuple[int, ...],
    dims: tuple[str, ...],
) -> ZarrArrayMetadata:
    """Overwrite xarray-owned fields in the fragment from the Variable."""
    result: dict[str, object] = dict(fragment)
    result["shape"] = shape
    if result.get("zarr_format") == 3:
        result["dimension_names"] = dims
    # See `merge_flat_aliases` for why this cast is needed.
    return cast("ZarrArrayMetadata", result)


def _v2_compressor_to_v3_codec(compressor: Mapping[str, object]) -> dict[str, object]:
    """Translate a v2 ``compressor``/``filters`` entry into a v3 codec dict.

    Only compressors with a native zarr-python v3 codec of the same name
    (blosc, gzip, zstd) are supported; anything else raises
    ``NotImplementedError`` naming the codec, per the module's scope
    decision (see module docstring / task report).
    """
    codec_id = compressor.get("id")
    if codec_id not in _V2_TO_V3_COMPRESSOR_NAMES:
        raise NotImplementedError(f"no zarr v3 equivalent for codec {codec_id!r}")
    if codec_id == "blosc":
        shuffle = compressor.get("shuffle")
        configuration: dict[str, object] = {
            "typesize": compressor.get("typesize", 1),
            "cname": compressor.get("cname"),
            "clevel": compressor.get("clevel"),
            "shuffle": _BLOSC_SHUFFLE_V2_TO_V3.get(shuffle, shuffle),
            "blocksize": compressor.get("blocksize", 0),
        }
        return {"name": "blosc", "configuration": configuration}
    if codec_id == "gzip":
        return {"name": "gzip", "configuration": {"level": compressor.get("level")}}
    # zstd
    return {
        "name": "zstd",
        "configuration": {
            "level": compressor.get("level", 0),
            "checksum": bool(compressor.get("checksum", False)),
        },
    }


def _v3_codec_to_v2_compressor(codec: Mapping[str, object]) -> dict[str, object]:
    """Translate a v3 codec dict (blosc/gzip/zstd) into a v2 compressor dict."""
    name = codec.get("name")
    if name not in _V2_TO_V3_COMPRESSOR_NAMES:
        raise NotImplementedError(f"no zarr v2 equivalent for codec {name!r}")
    config = codec.get("configuration")
    config = config if isinstance(config, Mapping) else {}
    if name == "blosc":
        shuffle = config.get("shuffle")
        return {
            "id": "blosc",
            "cname": config.get("cname"),
            "clevel": config.get("clevel"),
            "shuffle": _BLOSC_SHUFFLE_V3_TO_V2.get(shuffle, shuffle),
            "blocksize": config.get("blocksize", 0),
        }
    if name == "gzip":
        return {"id": "gzip", "level": config.get("level")}
    # zstd
    return {"id": "zstd", "level": config.get("level", 0)}


def _convert_dtype(dtype_str: object, *, target_format: Literal[2, 3]) -> object:
    """Round-trip a dtype string/name through ``zarr.dtype`` for the target format.

    Mirrors ``ArrayV2Metadata.to_dict``/``ArrayV3Metadata.to_dict``: the v2
    ``dtype`` field is the bare ``"name"`` pulled out of the dtype's v2 JSON
    spec, while the v3 ``data_type`` field is the dtype's v3 JSON spec
    (itself already a bare string for non-structured dtypes).
    """
    from zarr.dtype import parse_dtype

    source_format: Literal[2, 3] = 3 if target_format == 2 else 2
    zdtype = parse_dtype(dtype_str, zarr_format=source_format)  # type: ignore[arg-type]
    target_json = zdtype.to_json(zarr_format=target_format)
    if target_format == 2 and isinstance(target_json, Mapping):
        return target_json["name"]
    return target_json


def convert_zarr_metadata(
    fragment: ZarrArrayMetadata, target_format: Literal[2, 3]
) -> ZarrArrayMetadata:
    """Convert a zarr array metadata fragment between zarr formats v2 and v3.

    ``fragment`` is a plain dict as returned by ``read_metadata_fragment``
    (i.e. ``zarr_array.metadata.to_dict()``). Returns a new dict; ``fragment``
    is never mutated.

    Only the structural fields plus a minimal, verified compressor mapping
    (no-compressor, and blosc/gzip/zstd with no array-array filters) are
    supported. Anything else raises ``NotImplementedError`` naming the
    unsupported codec -- see the module's design notes / task report for why
    full arbitrary-codec conversion is out of scope here.
    """
    source_format = fragment["zarr_format"]
    if source_format == target_format:
        return fragment

    # The `_convert_*` helpers build the target-format dict field-by-field as
    # a plain `dict[str, object]` (some fields go through further conversion
    # helpers with their own narrow `type: ignore`s); cast the fully-built
    # result back to the TypedDict union it structurally matches.
    if target_format == 3:
        return cast("ZarrArrayMetadata", _convert_v2_to_v3(fragment))
    return cast("ZarrArrayMetadata", _convert_v3_to_v2(fragment))


def _convert_v2_to_v3(fragment: Mapping[str, object]) -> dict[str, object]:
    filters = fragment.get("filters")
    if filters:
        raise NotImplementedError(
            f"no zarr v3 equivalent for codec {filters!r} (array-array filters "
            "are not supported by convert_zarr_metadata)"
        )

    chunks = fragment.get("chunks")
    codecs: list[dict[str, object]] = [
        {"name": "bytes", "configuration": {"endian": "little"}}
    ]
    compressor = fragment.get("compressor")
    if isinstance(compressor, Mapping):
        codecs.append(_v2_compressor_to_v3_codec(compressor))

    separator = fragment.get("dimension_separator", ".")

    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": tuple(fragment["shape"]),  # type: ignore[arg-type]
        "data_type": _convert_dtype(fragment.get("dtype"), target_format=3),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": tuple(chunks)},  # type: ignore[arg-type]
        },
        "chunk_key_encoding": {
            "name": "v2",
            "configuration": {"separator": separator},
        },
        "codecs": codecs,
        "fill_value": fragment.get("fill_value"),
        "attributes": dict(fragment.get("attributes") or {}),  # type: ignore[call-overload]
        "storage_transformers": [],
    }


def _convert_v3_to_v2(fragment: Mapping[str, object]) -> dict[str, object]:
    codecs = fragment.get("codecs")
    codecs = list(codecs) if isinstance(codecs, (list, tuple)) else []

    compressor: dict[str, object] | None = None
    for codec in codecs:
        if not isinstance(codec, Mapping):
            continue
        name = codec.get("name")
        if name == "bytes":
            continue
        compressor = _v3_codec_to_v2_compressor(codec)

    chunk_grid = fragment.get("chunk_grid")
    chunk_shape: object = None
    if isinstance(chunk_grid, Mapping):
        config = chunk_grid.get("configuration")
        if isinstance(config, Mapping):
            chunk_shape = config.get("chunk_shape")

    chunk_key_encoding = fragment.get("chunk_key_encoding")
    separator = "."
    if isinstance(chunk_key_encoding, Mapping):
        config = chunk_key_encoding.get("configuration")
        if isinstance(config, Mapping) and "separator" in config:
            separator = config["separator"]
        elif chunk_key_encoding.get("name") == "default":
            separator = "/"

    return {
        "zarr_format": 2,
        "shape": tuple(fragment["shape"]),  # type: ignore[arg-type]
        "chunks": tuple(chunk_shape) if chunk_shape is not None else None,  # type: ignore[arg-type]
        "dtype": _convert_dtype(fragment.get("data_type"), target_format=2),
        "compressor": compressor,
        "filters": None,
        "fill_value": fragment.get("fill_value"),
        "order": "C",
        "dimension_separator": separator,
        "attributes": dict(fragment.get("attributes") or {}),  # type: ignore[call-overload]
    }


def _set_chunk_shape(fragment: dict[str, object], chunks: tuple[int, ...]) -> None:
    if fragment.get("zarr_format") == 3:
        fragment["chunk_grid"] = {
            "name": "regular",
            "configuration": {"chunk_shape": tuple(chunks)},
        }
    else:
        fragment["chunks"] = tuple(chunks)


#: ``numpy`` dtype ``.kind`` codes for which the write ``dtype`` argument
#: unambiguously determines a single zarr dtype, so ``_set_dtype`` can safely
#: stamp it into the fragment. Excludes object/string/bytes/unicode kinds
#: (``O``, ``U``, ``S``, ``T``): those cover vlen-string and other
#: object-backed encodings where ``zarr.dtype.parse_dtype`` either raises
#: (bare ``object``, verified empirically) or where the fragment's existing,
#: already-correct dtype/codec pairing (e.g. a vlen-utf8 codec) must not be
#: second-guessed from the numpy dtype alone.
_CONCRETE_DTYPE_KINDS = frozenset({"b", "i", "u", "f", "c", "M", "m"})


def _set_dtype(
    fragment: dict[str, object],
    dtype: object,
    *,
    zarr_format: Literal[2, 3],
) -> None:
    """Overwrite the fragment's dtype field with the write ``dtype``.

    ``dtype`` here is the ``dtype`` argument ``_create_new_array`` receives,
    i.e. the dtype the write actually encodes to (e.g. ``int16`` for a
    CF-packed ``scale_factor``/``add_offset`` variable) -- not the fragment's
    own ``data_type``/``dtype`` field, which reflects whatever the *source*
    array had and can disagree (e.g. an unpacked float64 array's fragment,
    reused to write packed int16 data). Left unfixed, the fast path would
    persist a fragment whose on-disk dtype disagrees with the dtype the
    writer actually streams into the array.

    Only stamped for concrete numpy numeric/bool/datetime/timedelta dtypes
    (see ``_CONCRETE_DTYPE_KINDS``): for those, ``zarr.dtype.parse_dtype``
    unambiguously resolves a single zarr dtype from the numpy dtype alone.
    Object-backed dtypes (vlen strings, etc.) are left untouched -- verified
    empirically that ``parse_dtype`` raises ``ValueError`` on a bare
    ``object`` dtype ("ambiguous... multiple zarr data types can be
    represented by the numpy Object data type"), and more generally the
    fragment's existing dtype/codec pairing for those encodings must not be
    second-guessed from the numpy dtype alone.
    """
    np_dtype = np.dtype(dtype)  # type: ignore[call-overload]
    if np_dtype.kind not in _CONCRETE_DTYPE_KINDS:
        return

    from zarr.dtype import parse_dtype

    zdtype = parse_dtype(np_dtype, zarr_format=zarr_format)
    target_json: object = zdtype.to_json(zarr_format=zarr_format)
    if zarr_format == 2 and isinstance(target_json, Mapping):
        target_json = target_json["name"]
    dtype_field = "data_type" if zarr_format == 3 else "dtype"
    fragment[dtype_field] = target_json


def _set_fill_value(
    fragment: dict[str, object],
    fill_value: object,
    *,
    zarr_format: Literal[2, 3],
) -> None:
    """Overwrite the fragment's ``fill_value`` with the resolved value.

    ``fill_value`` here is the raw Python/numpy scalar computed by
    ``set_variables`` (float default -> NaN, ``_FillValue``-attr driven,
    ``use_zarr_fill_value_as_mask`` handling, etc.) -- the same value the
    legacy path would hand to ``zarr_group.create()``. This function must
    reproduce what ``zarr_group.create(fill_value=fill_value)`` would have
    written for that same value, which (verified against zarr-python 3.2.1)
    is format-dependent for the ``fill_value=None`` case:

    - For a **v3** fragment, ``create_array(fill_value=None)`` resolves the
      dtype's own default scalar (``0`` for ints, ``False`` for bool, ``0.0``
      for floats) via ``dtype.default_scalar()`` and stores *that*, not a
      null. We reproduce the same resolution here on the fragment's own
      (already target-format) dtype field, rather than falling through to
      the legacy path for this case -- falling through would also throw away
      this fragment's already-converted codecs/compressor, reintroducing the
      v2->v3 compressor-translation problem the fast path exists to avoid.
    - For a **v2** fragment, ``create_array(fill_value=None)`` writes a
      literal ``null``/``None`` -- *not* a dtype default. This is
      semantically load-bearing: v2 is the format where
      ``use_zarr_fill_value_as_mask`` defaults to ``True``, i.e. the on-disk
      ``fill_value`` doubles as the "this value marks missing data" sentinel;
      resolving it to e.g. ``0`` would make ``0``/``False`` entries decode as
      missing on the next read. So for v2, ``None`` is passed through as-is.

    Whichever scalar we end up with is then converted to the exact
    JSON-shaped representation ``ArrayV{2,3}Metadata.from_dict`` expects for
    the ``fill_value`` field -- verified against zarr-python 3.2.1:
    ``from_dict`` parses this field with ``dtype.from_json_scalar``, which is
    strict (e.g. it rejects a bare numpy scalar like ``np.int32(0)``,
    requiring a plain Python ``int``, and represents float NaN as the string
    ``"NaN"``). ``dtype.to_json_scalar`` is the exact inverse of that parsing
    (it's what ``to_dict()`` itself uses to serialize ``fill_value``), so
    routing non-``None`` values through it here guarantees the fragment
    matches what a fresh ``to_dict()`` of an equivalently-created array would
    contain. ``to_json_scalar`` itself has no sensible ``None`` input (there
    is no dtype-compatible JSON form of "null"), so the v2 ``None`` case
    bypasses it and is stored directly.
    """
    if fill_value is None and zarr_format != 3:
        fragment["fill_value"] = None
        return

    from zarr.dtype import parse_dtype

    dtype_field = "data_type" if zarr_format == 3 else "dtype"
    zdtype = parse_dtype(fragment[dtype_field], zarr_format=zarr_format)  # type: ignore[arg-type]
    if fill_value is None:
        fill_value = zdtype.default_scalar()
    fragment["fill_value"] = zdtype.to_json_scalar(fill_value, zarr_format=zarr_format)


def build_canonical_metadata(
    encoding: Mapping[str, object],
    *,
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    target_format: Literal[2, 3],
    resolved_chunks: tuple[int, ...],
    resolved_fill_value: object,
    resolved_dtype: object,
) -> ZarrArrayMetadata:
    """Produce the canonical, target-format metadata dict for a write."""
    raw_fragment = encoding["zarr_array_metadata"]
    if not isinstance(raw_fragment, dict):
        raise TypeError("encoding['zarr_array_metadata'] must be a dict")
    # The runtime value is a plain dict shaped like the fragment TypedDicts
    # (built by `read_metadata_fragment`/`convert_zarr_metadata`); narrow it
    # here since mypy cannot verify a `dict[str, object]`'s shape statically.
    fragment = cast("ZarrArrayMetadata", raw_fragment)

    fragment = merge_flat_aliases(fragment, encoding)
    fragment = convert_zarr_metadata(fragment, target_format)
    fragment = apply_variable_fields(fragment, shape=shape, dims=dims)
    mutable_fragment: dict[str, object] = dict(fragment)
    _set_chunk_shape(mutable_fragment, resolved_chunks)
    # Must run before `_set_fill_value`: the fill-value default resolution
    # (the `fill_value is None` branch, e.g. `0` for ints/`False` for bool)
    # reads the fragment's own dtype field, so it needs to already be the
    # write dtype rather than the fragment's stale one.
    _set_dtype(mutable_fragment, resolved_dtype, zarr_format=target_format)
    _set_fill_value(mutable_fragment, resolved_fill_value, zarr_format=target_format)
    return cast("ZarrArrayMetadata", mutable_fragment)


def persist_array(store_path: StorePath, fragment: ZarrArrayMetadata) -> None:
    """Persist a new zarr array from a canonical metadata dict.

    ``ArrayV{2,3}Metadata.from_dict`` builds an in-memory object that does NOT
    write to the store, so we serialize its buffers explicitly.
    """
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
    from zarr.core.sync import sync

    # `from_dict` wants a plain, unstructured dict (it's what `to_dict()`
    # itself round-trips through); the TypedDict param exists for callers.
    raw_fragment: dict[str, object] = dict(fragment)

    meta: ArrayV2Metadata | ArrayV3Metadata
    if raw_fragment.get("zarr_format") == 2:
        meta = ArrayV2Metadata.from_dict(raw_fragment)
    else:
        meta = ArrayV3Metadata.from_dict(raw_fragment)  # type: ignore[arg-type]

    async def _write() -> None:
        buffers = meta.to_buffer_dict(default_buffer_prototype())
        for key, buffer in buffers.items():
            await (store_path / key).set(buffer)

    sync(_write())
