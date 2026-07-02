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
