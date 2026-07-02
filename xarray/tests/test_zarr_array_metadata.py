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
