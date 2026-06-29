"""Tests for the streaming OME-TIFF export (tilefusion.ome_tiff_export)."""

import json

import numpy as np
import tifffile

from tilefusion.ome_tiff_export import export_zarr_to_ome_tiff, write_ome_tiff


def test_roundtrip_cyx_with_edge_padding(tmp_path):
    # non-tile-multiple, non-square -> exercises edge-tile zero padding
    C, Y, X = 3, 1500, 1100
    arr = (np.random.default_rng(0).random((C, Y, X)) * 1000).astype(np.uint16)
    out = tmp_path / "m.ome.tif"
    write_ome_tiff(arr, str(out), pixel_size_um=(0.65, 0.65),
                   channel_names=["DAPI", "488", "561"], tile=(512, 512))
    with tifffile.TiffFile(out) as tf:
        s = tf.series[0]
        assert tf.is_bigtiff and tf.is_ome
        assert tuple(s.shape) == (C, Y, X)
        data = s.asarray().reshape(C, Y, X)
        xml = tf.ome_metadata
    assert np.array_equal(data, arr)            # exact: tiling/padding must not alter pixels
    assert 'PhysicalSizeX="0.65"' in xml and "DAPI" in xml and "µm" in xml


def test_2d_and_5d_shapes(tmp_path):
    # 2D (Y,X)
    a2 = (np.random.default_rng(1).random((300, 400)) * 100).astype(np.uint16)
    p2 = tmp_path / "a2.ome.tif"
    write_ome_tiff(a2, str(p2), pixel_size_um=(1.0, 1.0), tile=(256, 256))
    with tifffile.TiffFile(p2) as tf:
        assert tuple(tf.series[0].shape) == (300, 400) or tuple(tf.series[0].shape) == (1, 1, 1, 300, 400)

    # 5D input is TCZYX (the fused tensorstore's order); output is Squid TZCYX.
    a5 = (np.random.default_rng(2).random((1, 2, 1, 260, 260)) * 100).astype(np.uint16)
    p5 = tmp_path / "a5.ome.tif"
    write_ome_tiff(a5, str(p5), pixel_size_um=(0.5, 0.5), z_step_um=2.0,
                   channel_names=["c0", "c1"], tile=(128, 128))
    with tifffile.TiffFile(p5) as tf:
        data = tf.series[0].asarray().reshape(1, 1, 2, 260, 260)   # TZCYX
        assert np.array_equal(data, a5.transpose(0, 2, 1, 3, 4))   # TCZYX -> TZCYX
        assert 'PhysicalSizeZ="2.0"' in tf.ome_metadata
        assert tf.series[0].axes.replace("Q", "") in ("TZCYX", "ZCYX", "CYX")


def test_time_increment_for_squid_parity(tmp_path):
    # Squid's writer records the time-lapse interval as TimeIncrement (s), not PhysicalSizeT.
    a = (np.random.default_rng(7).random((300, 300)) * 100).astype(np.uint16)
    out = tmp_path / "t.ome.tif"
    write_ome_tiff(a, str(out), pixel_size_um=(0.5, 0.5), time_increment_s=1.5, tile=(256, 256))
    with tifffile.TiffFile(out) as tf:
        xml = tf.ome_metadata
    assert 'TimeIncrement="1.5"' in xml and 'TimeIncrementUnit="s"' in xml


def test_export_zarr_to_ome_tiff_reads_pixel_size_from_ngff(tmp_path):
    """The button path: open scale0/image and pick PhysicalSize up from the NGFF
    metadata, so the caller doesn't have to thread pixel size through."""
    import tensorstore as ts

    zdir = tmp_path / "m.ome.zarr"
    arr_path = zdir / "scale0" / "image"
    C, Y, X = 2, 400, 300
    data = (np.random.default_rng(3).random((1, C, 1, Y, X)) * 1000).astype(np.uint16)
    store = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(arr_path)},
            "metadata": {
                "shape": [1, C, 1, Y, X],
                "data_type": "uint16",
                "chunk_grid": {"name": "regular",
                               "configuration": {"chunk_shape": [1, 1, 1, 256, 256]}},
            },
            "create": True,
        }
    ).result()
    store.write(data).result()

    # NGFF group metadata carrying the pixel size in the scale0 transform.
    (zdir / "zarr.json").write_text(json.dumps({
        "attributes": {"ome": {"version": "0.5", "multiscales": [{
            "axes": [{"name": a, "type": t} for a, t in
                     [("t", "time"), ("c", "channel"), ("z", "space"),
                      ("y", "space"), ("x", "space")]],
            "datasets": [{"path": "scale0/image", "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0, 0.65, 0.65]}]}],
            "name": "image",
        }]}},
        "zarr_format": 3,
        "node_type": "group",
    }))

    tif = export_zarr_to_ome_tiff(str(zdir), channel_names=["DAPI", "488"])
    assert tif.endswith(".ome.tif")
    with tifffile.TiffFile(tif) as tf:
        xml = tf.ome_metadata
    assert 'PhysicalSizeX="0.65"' in xml and 'PhysicalSizeY="0.65"' in xml
    assert "DAPI" in xml
