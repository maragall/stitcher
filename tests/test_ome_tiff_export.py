"""Tests for the streaming OME-TIFF export (tilefusion.ome_tiff_export)."""

import numpy as np
import tifffile

from tilefusion.ome_tiff_export import write_ome_tiff


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
