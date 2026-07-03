"""Seam between xarray's variable `.encoding` and a zarr array's spec metadata.

All handling of the ``zarr_array_metadata`` encoding fragment lives here so the
backing representation (currently a plain dict from zarr-python's ``to_dict``)
can later be swapped for a ``zarr-metadata`` dataclass without touching the
backend call sites.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
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


def _codec_to_dict(codec: object) -> dict[str, object]:
    """Normalize a single v3 codec-like value to its ``to_dict()`` JSON form.

    Accepts either a live ``zarr.abc.codec.Codec`` (verified empirically:
    ``.to_dict()`` returns exactly ``{"name": ..., "configuration": ...}``,
    the same shape used in a v3 fragment's ``codecs`` list) or an
    already-JSON ``Mapping`` (e.g. a value round-tripped through a dict),
    which is returned as-is (copied).
    """
    if isinstance(codec, Mapping):
        return dict(codec)
    to_dict = getattr(codec, "to_dict", None)
    if callable(to_dict):
        return cast("dict[str, object]", to_dict())
    raise TypeError(
        f"expected a zarr v3 codec object or a JSON-shaped dict, got {codec!r}"
    )


def _codec_to_config(codec: object) -> dict[str, object]:
    """Normalize a single v2 codec-like value to its ``get_config()`` form.

    Accepts either a live ``numcodecs`` codec (verified empirically:
    ``.get_config()`` returns exactly ``{"id": ..., ...}``, the same shape
    used in a v2 fragment's ``compressor``/``filters`` fields) or an
    already-JSON ``Mapping``, which is returned as-is (copied).
    """
    if isinstance(codec, Mapping):
        return dict(codec)
    get_config = getattr(codec, "get_config", None)
    if callable(get_config):
        return cast("dict[str, object]", get_config())
    raise TypeError(
        f"expected a numcodecs codec object or a JSON-shaped dict, got {codec!r}"
    )


def _as_codec_list(
    value: object, *, to_dict: Callable[[object], dict[str, object]]
) -> list[dict[str, object]]:
    """Normalize a flat-alias codec value to a list of JSON-shaped dicts.

    The flat encoding keys (``compressors``, ``filters``, ``serializer``,
    ``compressor``) accept, per zarr-python's own ``*Like`` type aliases
    (``CompressorsLike``/``FiltersLike``/``SerializerLike`` in
    ``zarr.core.array``, verified against zarr-python 3.2.1): a single codec
    object, a single JSON dict, an iterable of either, or ``None``. Codec
    objects (both v3 ``Codec`` and ``numcodecs`` codecs) are not
    ``Iterable``, so a bare codec and an iterable-of-codecs are
    distinguishable by ``isinstance(value, Iterable)`` alone; a bare dict
    *is* ``Iterable`` (over its keys), so ``Mapping`` must be checked first
    to avoid misreading a single JSON codec dict as "an iterable of its
    keys".
    """
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [to_dict(value)]
    if isinstance(value, Iterable):
        return [to_dict(v) for v in value]
    return [to_dict(value)]


def _fold_flat_codecs(
    fragment: ZarrArrayMetadata, encoding: Mapping[str, object]
) -> ZarrArrayMetadata:
    """Overwrite the fragment's codec fields from the flat encoding aliases.

    Makes the flat codec keys (``compressors``, ``filters``, ``serializer``
    for a v3-shaped fragment; ``compressor``, ``filters`` for a v2-shaped
    fragment) authoritative over whatever the fragment already carries, the
    same way ``_set_dtype``/``_set_fill_value``/``_set_chunk_shape`` make
    their corresponding flat values authoritative. This runs on the
    fragment's *source* format, before ``convert_zarr_metadata`` translates
    it to the target format, so the folded codecs are written in the same
    representation the rest of the (still-source-format) fragment is in.

    Only keys actually present in ``encoding`` are folded; any codec
    sub-group not present is left as whatever the fragment already had, so
    e.g. mutating only ``encoding["compressors"]`` preserves the existing
    filters/serializer untouched.
    """
    result: dict[str, object] = dict(fragment)

    if result.get("zarr_format") == 3:
        raw_codecs = result.get("codecs")
        codecs: list[Mapping[str, object]] = (
            [c for c in raw_codecs if isinstance(c, Mapping)]
            if isinstance(raw_codecs, Iterable)
            else []
        )

        # Split the existing codecs list into its three sub-groups so any
        # sub-group absent from `encoding` can be preserved unchanged.
        # Verified empirically (zarr-python 3.2.1): a v3 `codecs` list is
        # ordered array-array filters, then exactly one array->bytes
        # serializer (`bytes` for numeric arrays, or a vlen codec such as
        # `vlen-utf8` for strings -- always present), then bytes->bytes
        # compressors.
        serializer_names = {"bytes", *_V3_VLEN_SERIALIZER_NAMES}
        existing_filters: list[dict[str, object]] = []
        existing_serializer: dict[str, object] | None = None
        existing_compressors: list[dict[str, object]] = []
        for codec in codecs:
            name = codec.get("name")
            if existing_serializer is None and name not in serializer_names:
                existing_filters.append(dict(codec))
            elif existing_serializer is None:
                existing_serializer = dict(codec)
            else:
                existing_compressors.append(dict(codec))

        if "filters" in encoding:
            new_filters = _as_codec_list(encoding["filters"], to_dict=_codec_to_dict)
        else:
            new_filters = existing_filters

        if "serializer" in encoding and encoding["serializer"] is not None:
            new_serializer = _codec_to_dict(encoding["serializer"])
        elif existing_serializer is not None:
            new_serializer = existing_serializer
        else:
            new_serializer = {"name": "bytes", "configuration": {"endian": "little"}}

        if "compressors" in encoding:
            new_compressors = _as_codec_list(
                encoding["compressors"], to_dict=_codec_to_dict
            )
        else:
            new_compressors = existing_compressors

        result["codecs"] = tuple([*new_filters, new_serializer, *new_compressors])
        return cast("ZarrArrayMetadata", result)

    # v2-shaped fragment: `compressor` is a single (or absent/None) codec,
    # `filters` is a list (or absent/None -- verified empirically: an
    # ArrayV2Metadata with no filters serializes `"filters": null`, not
    # `"filters": []`).
    if "compressor" in encoding:
        compressor_value = encoding["compressor"]
        result["compressor"] = (
            None if compressor_value is None else _codec_to_config(compressor_value)
        )
    elif "compressors" in encoding:
        # `derive_flat_aliases`/user code may instead set the v3-style
        # plural `compressors` key even against a v2 fragment (e.g. it was
        # copied wholesale from a v3-opened variable); accept a single
        # codec or a length<=1 iterable, since v2 supports only one
        # compressor.
        compressors_list = _as_codec_list(
            encoding["compressors"], to_dict=_codec_to_config
        )
        if len(compressors_list) > 1:
            raise NotImplementedError(
                "zarr v2 arrays support at most one compressor, got "
                f"{len(compressors_list)} in encoding['compressors']"
            )
        result["compressor"] = compressors_list[0] if compressors_list else None

    if "filters" in encoding:
        filters_value = encoding["filters"]
        if filters_value is None or (
            isinstance(filters_value, tuple) and len(filters_value) == 0
        ):
            result["filters"] = None
        else:
            filters_list = _as_codec_list(filters_value, to_dict=_codec_to_config)
            result["filters"] = filters_list or None

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


def _convert_dtype(
    dtype_str: object,
    *,
    target_format: Literal[2, 3],
    endian: Literal["little", "big"] | None = None,
) -> object:
    """Round-trip a dtype string/name through ``zarr.dtype`` for the target format.

    Mirrors ``ArrayV2Metadata.to_dict``/``ArrayV3Metadata.to_dict``: the v2
    ``dtype`` field is the bare ``"name"`` pulled out of the dtype's v2 JSON
    spec, while the v3 ``data_type`` field is the dtype's v3 JSON spec
    (itself already a bare string for non-structured dtypes).

    ``endian``, when given (only meaningful for ``target_format=2``), is
    stamped onto the parsed dtype before serializing to v2. This is needed
    because a v3 ``data_type`` string carries no byte-order information --
    it is always parsed back as native/little by ``parse_dtype`` -- while the
    real byte order for a v3 array lives in the ``bytes`` codec's ``endian``
    field (verified empirically: ``parse_dtype("float64", zarr_format=3)``
    always yields ``endianness='little'`` regardless of how the array was
    actually written). Dtypes with no byte-order concept (e.g. single-byte
    ints, bool) have no ``endianness`` attribute at all, so this is a no-op
    for them.
    """
    from zarr.dtype import parse_dtype

    source_format: Literal[2, 3] = 3 if target_format == 2 else 2
    zdtype = parse_dtype(dtype_str, zarr_format=source_format)  # type: ignore[arg-type]
    if endian is not None and hasattr(zdtype, "endianness"):
        # `ZDType.replace`'s signature is `**changes: object`, so mypy cannot
        # verify `endianness` is a valid field for this particular dtype
        # subclass; verified empirically (see docstring) that it is, for
        # every dtype where `hasattr(zdtype, "endianness")` holds.
        zdtype = dataclasses.replace(zdtype, endianness=endian)  # type: ignore[call-arg]
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


def _v2_dtype_endian(dtype_str: object) -> Literal["little", "big"]:
    """Derive the v3 bytes-codec ``endian`` value from a v2 ``dtype`` string.

    Mirrors ``numpy.dtype(...).byteorder``: ``"<"`` -> little, ``">"`` -> big,
    ``"="`` -> native (treated as little, matching the little-endian default
    this converter already hardcoded), ``"|"`` -> not-applicable (single-byte
    dtypes have no byte order; little is an arbitrary but harmless choice
    since the bytes codec's endian is a no-op for them).
    """
    byteorder = np.dtype(dtype_str).byteorder  # type: ignore[call-overload]
    if byteorder == ">":
        return "big"
    return "little"


def _convert_v2_to_v3(fragment: Mapping[str, object]) -> dict[str, object]:
    filters = fragment.get("filters")
    if filters:
        raise NotImplementedError(
            f"no zarr v3 equivalent for codec {filters!r} (array-array filters "
            "are not supported by convert_zarr_metadata)"
        )

    order = fragment.get("order", "C")
    if order is not None and order != "C":
        raise NotImplementedError(
            f"cannot convert zarr v2 array with order={order!r} to v3: "
            "non-C memory order is not supported by the metadata-fragment "
            "converter (write via the flat encoding path instead)"
        )

    chunks = fragment.get("chunks")
    endian = _v2_dtype_endian(fragment.get("dtype"))
    codecs: list[dict[str, object]] = [
        {"name": "bytes", "configuration": {"endian": endian}}
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


#: v3 array-to-bytes *serializer* codec names other than the plain ``bytes``
#: codec. These encode variable-length data (strings/bytes) with no v2
#: equivalent understood by this converter; they must never be misrouted
#: into the compressor-detection loop below (verified empirically: a v3
#: string array serializes with codecs ``({"name": "vlen-utf8", ...},)`` and
#: *no* ``bytes`` codec at all, and a v3 bytes/vlen array uses
#: ``vlen-bytes``).
_V3_VLEN_SERIALIZER_NAMES = frozenset({"vlen-utf8", "vlen-bytes"})


def _convert_v3_to_v2(fragment: Mapping[str, object]) -> dict[str, object]:
    codecs = fragment.get("codecs")
    codecs = list(codecs) if isinstance(codecs, (list, tuple)) else []

    compressor: dict[str, object] | None = None
    endian: Literal["little", "big"] = "little"
    for codec in codecs:
        if not isinstance(codec, Mapping):
            continue
        name = codec.get("name")
        if name == "bytes":
            config = codec.get("configuration")
            if isinstance(config, Mapping) and config.get("endian") == "big":
                endian = "big"
            continue
        if name == "transpose":
            raise NotImplementedError(
                "cannot convert zarr v3 array using a 'transpose' codec to "
                "v2: non-C memory order is not supported by the "
                "metadata-fragment converter (write via the flat encoding "
                "path instead)"
            )
        if name in _V3_VLEN_SERIALIZER_NAMES:
            raise NotImplementedError(
                f"cannot convert zarr v3 array using the {name!r} serializer "
                "to v2: string/vlen conversion is not supported by the "
                "metadata-fragment converter (write via the flat encoding "
                "path instead)"
            )
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
        "dtype": _convert_dtype(
            fragment.get("data_type"), target_format=2, endian=endian
        ),
        "compressor": compressor,
        "filters": None,
        "fill_value": fragment.get("fill_value"),
        # Safe to hardcode: any 'transpose' codec (the only source of
        # non-C order in v3) is caught and raises above.
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
    # Make the flat codec keys authoritative over the fragment's own,
    # possibly-stale codecs -- e.g. a user setting
    # `encoding["compressors"] = GzipCodec(level=9)` after opening must be
    # honored, not silently dropped in favor of the source array's original
    # codecs. This must run in the fragment's *source* format (before
    # `convert_zarr_metadata` below translates the whole fragment to
    # `target_format`), so the folded-in codec values -- which are
    # source-format codec objects/dicts as populated by `open_store_variable`
    # or set by the user -- land in the representation `convert_zarr_metadata`
    # expects to translate.
    fragment = _fold_flat_codecs(fragment, encoding)
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
