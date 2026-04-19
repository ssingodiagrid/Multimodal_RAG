"""Sanity check for ColPali-style MaxSim (late interaction) scoring (numpy only)."""


def test_maxsim_scalar_matches_einsum_pattern():
    import numpy as np

    rng = np.random.default_rng(0)
    q = rng.standard_normal((4, 8)).astype(np.float64)
    p = rng.standard_normal((10, 8)).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    manual = float((q @ p.T).max(axis=1).sum())
    batch_q = q[np.newaxis, ...]
    batch_p = p[np.newaxis, ...]
    via = np.einsum("bnd,csd->bcns", batch_q, batch_p).max(axis=3).sum(axis=2)
    assert via.shape == (1, 1)
    assert abs(float(via[0, 0]) - manual) < 1e-5
