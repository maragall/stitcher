"""Internal (numpy) flatfield estimation should recover a known shading field."""
import numpy as np
from tilefusion.flatfield import calculate_flatfield, estimate_flatfield_channel, apply_flatfield


def _shading(Y, X):
    yy, xx = np.mgrid[0:Y, 0:X]
    s = 1.0 + 0.5 * np.exp(-(((yy - Y / 2) ** 2 + (xx - X / 2) ** 2) / (2 * (Y / 3.0) ** 2)))
    return (s / s.mean()).astype(np.float32)


def test_recovers_shading_from_varied_tiles():
    rng = np.random.default_rng(0)
    Y = X = 256
    shading = _shading(Y, X)
    tiles = [((shading * rng.uniform(50, 200, (Y, X))).astype(np.float32))[None] for _ in range(40)]
    ff, df = calculate_flatfield(tiles, use_darkfield=False)
    assert ff.shape == (1, Y, X) and df is None
    rec = ff[0]
    assert abs(float(rec.mean()) - 1.0) < 0.05               # normalized to mean 1
    assert np.corrcoef(rec.ravel(), shading.ravel())[0, 1] > 0.95   # tracks true shading


def test_channel_helper_and_darkfield_shapes():
    rng = np.random.default_rng(1)
    Y = X = 128
    stack = (_shading(Y, X) * rng.uniform(40, 160, (30, Y, X))).astype(np.float32)
    ff, df = estimate_flatfield_channel(stack, use_darkfield=True, constant_darkfield=True)
    assert ff.shape == (Y, X) and df.shape == (Y, X)
    assert np.allclose(df, df.flat[0])                       # constant darkfield
    # applying the correction flattens the shading: corrected tile is ~uniform-content
    corrected = apply_flatfield((stack[0])[None], ff[None], None)
    assert corrected.shape == (1, Y, X)
