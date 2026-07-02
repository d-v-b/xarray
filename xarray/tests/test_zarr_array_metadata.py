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
