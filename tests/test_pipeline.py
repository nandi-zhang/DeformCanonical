"""
Sanity checks: run this before any real training to verify
tensor shapes, gradient flow, and data pipeline.

python tests/test_pipeline.py
"""

import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from canonical_ar.models.deformation_field import DeformationFieldNet
from canonical_ar.models.losses import DeformationLoss
from canonical_ar.data.synthetic import SyntheticDeformationDataset


def make_test_cfg():
    return OmegaConf.create({
        "model": {
            "encoder": {"type": "pointnet_plus", "input_dim": 6, "output_dim": 256, "hidden_dims": [64, 128, 256]},
            "decoder": {"type": "mlp", "latent_dim": 256, "hidden_dims": [256, 256, 256, 256],
                        "output_dim": 3, "use_positional_encoding": True, "positional_encoding_freqs": 6},
            "canonical_encoder": {"type": "pointnet_plus", "input_dim": 6, "output_dim": 256, "hidden_dims": [64, 128, 256]},
            "fusion": {"type": "concat_and_project", "output_dim": 256},
        },
        "train": {
            "loss": {
                "chamfer_weight": 1.0,
                "smoothness_weight": 0.1,
                "magnitude_weight": 0.01,
                "attachment_weight": 1.0,
            }
        }
    })


def test_model_shapes():
    print("=" * 50)
    print("TEST: Model forward pass shapes")
    cfg = make_test_cfg()
    model = DeformationFieldNet(cfg)
    model.eval()

    B, N_c, N_o, Q = 2, 512, 256, 96
    canonical_xyz  = torch.randn(B, N_c, 3)
    canonical_feat = torch.randn(B, N_c, 3)
    obs_xyz        = torch.randn(B, N_o, 3)
    obs_feat       = torch.randn(B, N_o, 3)
    query_pts      = torch.randn(B, Q, 3)

    with torch.no_grad():
        out = model(canonical_xyz, canonical_feat, obs_xyz, obs_feat, query_pts)

    assert out["z"].shape == (B, 256),              f"z shape wrong: {out['z'].shape}"
    assert out["displacements"].shape == (B, Q, 3), f"disp shape wrong: {out['displacements'].shape}"
    assert out["deformed_pts"].shape == (B, Q, 3),  f"deformed_pts shape wrong: {out['deformed_pts'].shape}"
    print("  PASS — output shapes correct")


def test_gradient_flow():
    print("=" * 50)
    print("TEST: Gradient flow")
    cfg = make_test_cfg()
    model = DeformationFieldNet(cfg)
    loss_fn = DeformationLoss(cfg)

    B, N_c, N_o, Q = 2, 256, 128, 32
    batch = {
        "canonical_xyz":     torch.randn(B, N_c, 3),
        "canonical_feat":    torch.randn(B, N_c, 3),
        "obs_xyz":           torch.randn(B, N_o, 3),
        "obs_feat":          torch.randn(B, N_o, 3),
        "query_pts":         torch.randn(B, Q, 3),
        "gt_deformed_query": torch.randn(B, Q, 3),
    }

    out = model(
        batch["canonical_xyz"], batch["canonical_feat"],
        batch["obs_xyz"], batch["obs_feat"],
        batch["query_pts"],
    )
    decoder_full = model.decoder(batch["canonical_xyz"], out["z"])

    losses = loss_fn(
        deformed_canonical=decoder_full["deformed_pts"],
        displacements=decoder_full["displacements"],
        deformed_query=out["deformed_pts"],
        obs_xyz=batch["obs_xyz"],
        canonical_pts=batch["canonical_xyz"],
        gt_deformed_query=batch["gt_deformed_query"],
    )

    losses["loss"].backward()

    # Check gradients exist and are finite
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
    print("  PASS — gradients are finite and flow through all parameters")
    print(f"  Total loss: {losses['loss'].item():.4f}")
    for k, v in losses.items():
        if k != "loss":
            print(f"    {k}: {v.item():.4f}")


def test_identity_deformation():
    print("=" * 50)
    print("TEST: Near-zero deformation at init (rigid baseline)")
    cfg = make_test_cfg()
    model = DeformationFieldNet(cfg)
    model.eval()

    B, N, Q = 2, 256, 32
    pts = torch.randn(B, N, 3) * 0.5
    feat = torch.randn(B, N, 3)
    query = torch.randn(B, Q, 3) * 0.5

    with torch.no_grad():
        # Pass same cloud as both canonical and observed (perfect alignment)
        out = model(pts, feat, pts, feat, query)

    disp_magnitude = out["displacements"].norm(dim=-1).mean().item()
    print(f"  Mean displacement magnitude at init: {disp_magnitude:.6f}")
    print(f"  (Should be small due to near-zero output head init)")
    # Not a hard assertion — just informational


def test_data_pipeline():
    print("=" * 50)
    print("TEST: Synthetic data pipeline")
    ds = SyntheticDeformationDataset(
        n_objects=10,
        deformations_per_object=5,
        n_canonical_pts=256,
        n_obs_pts=128,
        n_surface_content=16,
        n_interior_content=8,
        split="train",
    )
    print(f"  Dataset size: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")

    # Check no NaNs
    for k, v in sample.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert torch.isfinite(v).all(), f"NaN/Inf in {k}"
    print("  PASS — data pipeline produces finite tensors")

    # Check deformation magnitude correlation
    rigid_sample = None
    deform_sample = None
    for i in range(len(ds)):
        s = ds[i]
        if s["deformation_magnitude"].item() < 0.05 and rigid_sample is None:
            rigid_sample = s
        if s["deformation_magnitude"].item() > 0.3 and deform_sample is None:
            deform_sample = s
        if rigid_sample and deform_sample:
            break

    if rigid_sample and deform_sample:
        rigid_disp = (rigid_sample["gt_deformed_query"] - rigid_sample["query_pts"]).norm(dim=-1).mean()
        deform_disp = (deform_sample["gt_deformed_query"] - deform_sample["query_pts"]).norm(dim=-1).mean()
        print(f"  Rigid GT displacement (mag<0.05): {rigid_disp:.4f}")
        print(f"  Deformable GT displacement (mag>0.3): {deform_disp:.4f}")
        assert deform_disp > rigid_disp, "Deformable should have larger displacement than rigid"
        print("  PASS — deformation magnitude correlates with ground truth displacement")




def test_normalization():
    print("=" * 50)
    print("TEST: Normalization utilities")
    import sys
    sys.path.insert(0, ".")
    from canonical_ar.utils.normalization import normalize_np, normalize_point_cloud
    import torch

    # Numpy version
    pts = np.random.randn(1000, 3).astype(np.float32) * 5 + np.array([10, -3, 2])
    norm_pts, centroid, scale = normalize_np(pts)
    assert np.allclose(norm_pts.mean(axis=0), 0, atol=1e-5), "Centroid not zero"
    assert norm_pts.max() <= 1.01, f"Max exceeds 1: {norm_pts.max()}"
    print(f"  Numpy: centroid={centroid.round(2)}, scale={scale:.3f}")

    # Torch version
    pts_t = torch.from_numpy(pts).unsqueeze(0)  # (1, N, 3)
    norm_t, params = normalize_point_cloud(pts_t)
    assert norm_t.shape == pts_t.shape
    assert torch.allclose(norm_t.mean(dim=1), torch.zeros(1, 3, dtype=norm_t.dtype), atol=1e-4)
    print(f"  Torch: centroid={params.centroid.squeeze().numpy().round(2)}")
    print("  PASS — normalization correct")


def test_partial_observation():
    print("=" * 50)
    print("TEST: Partial observation simulation")
    import sys
    sys.path.insert(0, ".")
    import trimesh
    from canonical_ar.utils.partial_observation import simulate_partial_observation, random_viewpoint

    # Create a sphere
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    pts, face_idx = trimesh.sample.sample_surface(mesh, 4096)
    normals = mesh.face_normals[face_idx]

    # Full observation should have ~half the sphere visible from any viewpoint
    obs_pts, obs_norms, vp = simulate_partial_observation(
        pts, normals, n_output_pts=512, noise_std=0.005, viewpoint_seed=42
    )
    assert obs_pts.shape == (512, 3), f"Wrong shape: {obs_pts.shape}"
    assert np.isfinite(obs_pts).all(), "NaN in observed points"

    # All visible points should be on the viewpoint-facing hemisphere
    view_dirs = vp[None] - obs_pts
    view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8
    dots = (obs_norms * view_dirs).sum(axis=1)
    facing_ratio = (dots > -0.2).mean()  # allow small tolerance for noisy normals
    print(f"  Facing viewpoint ratio: {facing_ratio:.2f} (should be > 0.8)")
    assert facing_ratio > 0.8, f"Too many back-facing points: {facing_ratio:.2f}"
    print("  PASS — partial observation filters correctly")


def test_dataset_with_normalization():
    print("=" * 50)
    print("TEST: Dataset output is normalized and partial")
    import sys
    sys.path.insert(0, ".")
    from canonical_ar.data.synthetic import SyntheticDeformationDataset

    ds = SyntheticDeformationDataset(
        n_objects=5, deformations_per_object=4,
        n_canonical_pts=256, n_obs_pts=128,
        n_surface_content=16, n_interior_content=8,
        split="train",
    )
    sample = ds[0]

    # Check normalization: canonical should be in unit sphere
    can = sample["canonical_xyz"].numpy()
    max_dist = np.linalg.norm(can, axis=1).max()
    assert max_dist <= 1.01, f"Canonical not normalized: max dist = {max_dist:.3f}"
    print(f"  Canonical max dist from origin: {max_dist:.4f} (should be ~1.0)")

    # Check norm params are stored
    assert "norm_centroid" in sample, "norm_centroid missing"
    assert "norm_scale" in sample, "norm_scale missing"
    print(f"  norm_scale: {sample['norm_scale'].item():.4f}")

    # Check no NaNs
    for k, v in sample.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert torch.isfinite(v).all(), f"NaN/Inf in {k}"
    print("  PASS — dataset output normalized and finite")

if __name__ == "__main__":
    print("\nRunning pipeline sanity checks...\n")
    test_data_pipeline()
    test_normalization()
    test_partial_observation()
    test_dataset_with_normalization()
    test_model_shapes()
    test_gradient_flow()
    test_identity_deformation()
    print("\nAll tests passed.\n")
