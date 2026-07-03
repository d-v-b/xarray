import numpy as np
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
    assert frag["shape"] == (10,)
    assert isinstance(frag["shape"], tuple)
    assert "codecs" in frag


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


@requires_zarr
def test_apply_variable_fields_overrides_shape_and_dims():
    from xarray.backends.zarr_array_metadata import apply_variable_fields

    fragment = {"zarr_format": 3, "shape": (99,), "dimension_names": ("stale",)}
    out = apply_variable_fields(fragment, shape=(4,), dims=("x",))
    assert out["shape"] == (4,)
    assert out["dimension_names"] == ("x",)
    # input not mutated
    assert fragment["shape"] == (99,)


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


@requires_zarr
def test_convert_metadata_v2_to_v3_does_not_mutate_input(tmp_path):
    import zarr

    from xarray.backends.zarr_array_metadata import (
        convert_zarr_metadata,
        read_metadata_fragment,
    )

    g = zarr.open_group(tmp_path / "v2.zarr", mode="w", zarr_format=2)
    a = g.create_array("a", shape=(10,), chunks=(5,), dtype="f8", compressors=None)
    v2 = read_metadata_fragment(a)
    v2_copy = dict(v2)

    convert_zarr_metadata(v2, 3)
    assert v2 == v2_copy


@requires_zarr
def test_convert_metadata_v3_to_v2_roundtrips_chunks(tmp_path):
    import zarr

    from xarray.backends.zarr_array_metadata import (
        convert_zarr_metadata,
        read_metadata_fragment,
    )

    g = zarr.open_group(tmp_path / "v3.zarr", mode="w", zarr_format=3)
    a = g.create_array("a", shape=(10,), chunks=(5,), dtype="f8", compressors=None)
    v3 = read_metadata_fragment(a)

    v2 = convert_zarr_metadata(v3, 2)
    assert v2["zarr_format"] == 2
    assert tuple(v2["chunks"]) == (5,)
    assert v2["compressor"] is None


@requires_zarr
def test_convert_metadata_v3_to_v2_preserves_compressor(tmp_path):
    import zarr
    import zarr.codecs

    from xarray.backends.zarr_array_metadata import (
        convert_zarr_metadata,
        read_metadata_fragment,
    )

    g = zarr.open_group(tmp_path / "v3.zarr", mode="w", zarr_format=3)
    a = g.create_array(
        "a",
        shape=(10,),
        chunks=(5,),
        dtype="f8",
        compressors=zarr.codecs.GzipCodec(level=4),
    )
    v3 = read_metadata_fragment(a)
    # zarr-python returns codecs as a tuple, not a list
    assert isinstance(v3["codecs"], tuple)

    v2 = convert_zarr_metadata(v3, 2)
    assert v2["compressor"] is not None
    assert v2["compressor"]["id"] == "gzip"
    assert v2["compressor"]["level"] == 4


@requires_zarr
def test_convert_metadata_v2_to_v3_maps_known_compressor(tmp_path):
    import numcodecs
    import zarr

    from xarray.backends.zarr_array_metadata import (
        convert_zarr_metadata,
        read_metadata_fragment,
    )

    g = zarr.open_group(tmp_path / "v2.zarr", mode="w", zarr_format=2)
    a = g.create_array(
        "a",
        shape=(10,),
        chunks=(5,),
        dtype="f8",
        compressors=numcodecs.GZip(level=4),
    )
    v2 = read_metadata_fragment(a)

    v3 = convert_zarr_metadata(v2, 3)
    codec_names = [c["name"] for c in v3["codecs"]]
    assert "gzip" in codec_names


@requires_zarr
def test_convert_metadata_v2_to_v3_raises_for_unmapped_filter(tmp_path):
    import numcodecs
    import zarr

    from xarray.backends.zarr_array_metadata import (
        convert_zarr_metadata,
        read_metadata_fragment,
    )

    g = zarr.open_group(tmp_path / "v2.zarr", mode="w", zarr_format=2)
    a = g.create_array(
        "a",
        shape=(10,),
        chunks=(5,),
        dtype="i4",
        compressors=None,
        filters=[numcodecs.Delta(dtype="i4")],
    )
    v2 = read_metadata_fragment(a)

    with pytest.raises(NotImplementedError, match="delta"):
        convert_zarr_metadata(v2, 3)


@requires_zarr
def test_convert_metadata_v2_to_v3_raises_for_fortran_order(tmp_path):
    import zarr

    from xarray.backends.zarr_array_metadata import (
        convert_zarr_metadata,
        read_metadata_fragment,
    )

    g = zarr.open_group(tmp_path / "v2.zarr", mode="w", zarr_format=2)
    a = g.create_array(
        "a", shape=(4, 4), chunks=(2, 2), dtype="f8", order="F", compressors=None
    )
    v2 = read_metadata_fragment(a)
    assert v2["order"] == "F"

    with pytest.raises(NotImplementedError, match="order"):
        convert_zarr_metadata(v2, 3)


@requires_zarr
def test_convert_metadata_v3_to_v2_raises_for_transpose_codec():
    from xarray.backends.zarr_array_metadata import convert_zarr_metadata

    v3 = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4, 4),
        "data_type": "float64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": (2, 2)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "codecs": (
            {"name": "transpose", "configuration": {"order": (1, 0)}},
            {"name": "bytes", "configuration": {"endian": "little"}},
        ),
        "fill_value": 0.0,
        "attributes": {},
        "storage_transformers": (),
    }

    with pytest.raises(NotImplementedError, match="transpose"):
        convert_zarr_metadata(v3, 2)


@requires_zarr
def test_convert_metadata_v2_to_v3_preserves_big_endian(tmp_path):
    import zarr

    from xarray.backends.zarr_array_metadata import (
        convert_zarr_metadata,
        read_metadata_fragment,
    )

    g = zarr.open_group(tmp_path / "v2.zarr", mode="w", zarr_format=2)
    a = g.create_array("a", shape=(4,), chunks=(2,), dtype=">f8", compressors=None)
    v2 = read_metadata_fragment(a)
    assert v2["dtype"] == ">f8"

    v3 = convert_zarr_metadata(v2, 3)
    bytes_codec = next(c for c in v3["codecs"] if c["name"] == "bytes")
    assert bytes_codec["configuration"]["endian"] == "big"


@requires_zarr
def test_convert_metadata_v3_to_v2_preserves_big_endian():
    from xarray.backends.zarr_array_metadata import convert_zarr_metadata

    v3 = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4,),
        "data_type": "float64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": (2,)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "codecs": ({"name": "bytes", "configuration": {"endian": "big"}},),
        "fill_value": 0.0,
        "attributes": {},
        "storage_transformers": (),
    }

    v2 = convert_zarr_metadata(v3, 2)
    assert v2["dtype"] == ">f8"


@requires_zarr
def test_convert_metadata_v3_to_v2_raises_for_vlen_utf8_serializer():
    from xarray.backends.zarr_array_metadata import convert_zarr_metadata

    v3 = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4,),
        "data_type": "string",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": (2,)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "codecs": ({"name": "vlen-utf8", "configuration": {}},),
        "fill_value": "",
        "attributes": {},
        "storage_transformers": (),
    }

    with pytest.raises(NotImplementedError, match="vlen-utf8"):
        convert_zarr_metadata(v3, 2)


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
        encoding,
        shape=(8,),
        dims=("x",),
        target_format=3,
        resolved_chunks=(4,),
        resolved_fill_value=0.0,
        resolved_dtype=np.dtype("f8"),
    )
    assert out["zarr_format"] == 3
    assert out["shape"] == (8,)
    assert out["dimension_names"] == ("x",)
    assert tuple(out["chunk_grid"]["configuration"]["chunk_shape"]) == (4,)
    assert out["fill_value"] == 0.0
    assert out["data_type"] == "float64"
