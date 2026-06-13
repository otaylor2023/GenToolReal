import numpy as np

from generative_str.mde_align import fit_scale_shift


def test_fit_scale_shift_recovers_affine():
    rng = np.random.default_rng(0)
    # Strictly positive GT so default mask (gt > 0) keeps all pixels.
    mde0 = rng.uniform(0.5, 2.0, size=(32, 48)).astype(np.float32)
    s_true, b_true = 2.5, 0.3
    gt0 = s_true * mde0 + b_true
    s, b, meta = fit_scale_shift(mde0, gt0, mask=None, robust=False)
    assert abs(s - s_true) < 1e-4
    assert abs(b - b_true) < 1e-4
    assert meta.num_pixels == mde0.size
