"""Basic tests for topo flow generation."""

import numpy as np
import pytest
from topo import generate_direct_flows, generate_diffusion_flows, compute_flow_targets


def _make_sphere(shape, center, radius):
    """Create a binary sphere mask."""
    coords = np.mgrid[tuple(slice(0, s) for s in shape)]
    dist = sum((c - cx) ** 2 for c, cx in zip(coords, center))
    return (dist <= radius ** 2).astype(np.int32)


class TestDirectFlows:
    def test_unit_vectors(self):
        mask = _make_sphere((32, 32, 32), (16, 16, 16), 8)
        flows = generate_direct_flows(mask)
        assert flows.shape == (3, 32, 32, 32)

        fg = mask > 0
        mag = np.sqrt((flows[:, fg] ** 2).sum(axis=0))
        # All foreground vectors should be unit length (except at center)
        non_center = mag > 0.01
        np.testing.assert_allclose(mag[non_center], 1.0, atol=0.01)

    def test_points_toward_center(self):
        mask = _make_sphere((32, 32, 32), (16, 16, 16), 8)
        flows = generate_direct_flows(mask)
        # Voxel at (16, 16, 24) should have flow pointing in -x direction
        assert flows[2, 16, 16, 24] < -0.9  # dx should be negative

    def test_background_is_zero(self):
        mask = _make_sphere((32, 32, 32), (16, 16, 16), 8)
        flows = generate_direct_flows(mask)
        bg = mask == 0
        assert np.all(flows[:, bg] == 0.0)


class TestDiffusionFlows:
    def test_produces_valid_flows(self):
        mask = _make_sphere((32, 32, 32), (16, 16, 16), 8)
        flows = generate_diffusion_flows(mask, n_iter=50)
        assert flows.shape == (3, 32, 32, 32)

        fg = mask > 0
        mag = np.sqrt((flows[:, fg] ** 2).sum(axis=0))
        # Should produce non-zero flows
        assert mag.mean() > 0.1

    def test_background_is_zero(self):
        mask = _make_sphere((32, 32, 32), (16, 16, 16), 8)
        flows = generate_diffusion_flows(mask, n_iter=50)
        bg = mask == 0
        np.testing.assert_allclose(flows[:, bg], 0.0, atol=1e-6)


class TestComputeFlowTargets:
    def test_single_class(self):
        mask = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        instance_ids = mask[np.newaxis]  # [1, D, H, W]

        flows, class_fg = compute_flow_targets(
            instance_ids,
            class_names=["nuc"],
            class_config={"nuc": {"flow_type": "direct"}},
        )

        assert flows.shape == (3, 16, 16, 16)
        assert class_fg.shape == (1, 16, 16, 16)
        assert class_fg[0].sum() == (mask > 0).sum()

    def test_multiple_instances(self):
        mask = np.zeros((32, 32, 32), dtype=np.int32)
        mask += _make_sphere((32, 32, 32), (8, 8, 8), 5) * 1
        mask += _make_sphere((32, 32, 32), (24, 24, 24), 5) * 2

        flows, class_fg = compute_flow_targets(
            mask[np.newaxis],
            class_names=["ves"],
            class_config={"ves": {"flow_type": "direct"}},
        )

        fg = class_fg[0] > 0
        assert fg.sum() == (mask > 0).sum()
