"""
Demo: Simulated Tracking Loop

Runs the full end-to-end pipeline on a synthetic object without
needing a real camera or trained model — uses random weights so
you can verify the pipeline structure is correct before training.

Visualizes:
  - Canonical point cloud (blue)
  - Simulated observed point cloud (grey, partial)
  - Predicted virtual content positions (red)
  - Ground truth virtual content positions (green)

Run:
    python scripts/demo_simulated.py

After training, pass a checkpoint to see real predictions:
    python scripts/demo_simulated.py checkpoint_path=checkpoints/checkpoint_best.pt
"""

import sys
import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_default_cfg():
    return OmegaConf.create({
        "model": {
            "encoder": {
                "type": "pointnet_plus", "input_dim": 6,
                "output_dim": 256, "hidden_dims": [64, 128, 256]
            },
            "decoder": {
                "type": "mlp", "latent_dim": 256,
                "hidden_dims": [256, 256, 256, 256],
                "output_dim": 3, "use_positional_encoding": True,
                "positional_encoding_freqs": 6,
            },
            "canonical_encoder": {
                "type": "pointnet_plus", "input_dim": 6,
                "output_dim": 256, "hidden_dims": [64, 128, 256]
            },
            "fusion": {"type": "concat_and_project", "output_dim": 256},
        },
    })


def generate_demo_scene(
    deformation_magnitude: float = 0.2,
    seed: int = 42,
):
    """Generate a synthetic canonical + deformed scene for demo."""
    from canonical_ar.data.synthetic import (
        generate_primitive_mesh,
        smooth_random_deformation,
        sample_surface_points,
        register_virtual_content,
    )
    from canonical_ar.utils.normalization import normalize_np
    from canonical_ar.utils.partial_observation import simulate_partial_observation
    from scipy.spatial import KDTree

    rng = np.random.default_rng(seed)
    mesh = generate_primitive_mesh("sphere", scale=1.0)

    # Canonical point cloud
    can_pts, can_norms = sample_surface_points(mesh, 512)
    can_norm, centroid, scale = normalize_np(can_pts)

    # Virtual content: surface + interior
    content_pts = register_virtual_content(mesh, n_surface=32, n_interior=16)
    query_norm = ((content_pts - centroid) / scale).astype(np.float32)

    # Deformed mesh
    deformed_verts = smooth_random_deformation(
        mesh.vertices.copy(), magnitude=deformation_magnitude, seed=seed
    )
    deformed_mesh = trimesh.Trimesh(
        vertices=deformed_verts, faces=mesh.faces, process=False
    )

    # Observed partial point cloud
    obs_full, obs_norms_full = sample_surface_points(deformed_mesh, 2048)
    obs_pts, obs_norms, viewpoint = simulate_partial_observation(
        obs_full, obs_norms_full, n_output_pts=256, noise_std=0.005, viewpoint_seed=seed
    )
    obs_centroid = obs_pts.mean(axis=0)
    obs_norm = ((obs_pts - obs_centroid) / scale).astype(np.float32)

    # Ground truth deformed content
    tree = KDTree(mesh.vertices)
    _, nn_idx = tree.query(content_pts)
    vertex_disp = deformed_verts[nn_idx] - mesh.vertices[nn_idx]
    gt_deformed_content = content_pts + vertex_disp
    gt_query_norm = ((gt_deformed_content - centroid) / scale).astype(np.float32)

    return {
        "canonical_xyz":  torch.from_numpy(can_norm).unsqueeze(0),      # (1, N, 3)
        "canonical_feat": torch.from_numpy(can_norms).unsqueeze(0),
        "obs_xyz":        torch.from_numpy(obs_norm).unsqueeze(0),       # (1, M, 3)
        "obs_feat":       torch.from_numpy(obs_norms).unsqueeze(0),
        "query_pts":      torch.from_numpy(query_norm).unsqueeze(0),     # (1, Q, 3)
        "gt_deformed_query": torch.from_numpy(gt_query_norm).unsqueeze(0),
        "n_surface": 32,
        "n_interior": 16,
    }


def visualize_scene(scene: dict, predicted: np.ndarray, title: str = ""):
    fig = plt.figure(figsize=(14, 6))

    # Canonical view
    ax1 = fig.add_subplot(121, projection='3d')
    can = scene["canonical_xyz"].squeeze(0).numpy()
    ax1.scatter(can[:, 0], can[:, 1], can[:, 2],
                c='steelblue', s=2, alpha=0.4, label='Canonical')
    q = scene["query_pts"].squeeze(0).numpy()
    ax1.scatter(q[:32, 0], q[:32, 1], q[:32, 2],
                c='red', s=30, marker='*', label='Surface content')
    ax1.scatter(q[32:, 0], q[32:, 1], q[32:, 2],
                c='orange', s=30, marker='^', label='Interior content')
    ax1.set_title("Canonical Space")
    ax1.legend(fontsize=7)
    ax1.set_xlim(-1.2, 1.2); ax1.set_ylim(-1.2, 1.2); ax1.set_zlim(-1.2, 1.2)

    # Observed + prediction view
    ax2 = fig.add_subplot(122, projection='3d')
    obs = scene["obs_xyz"].squeeze(0).numpy()
    gt  = scene["gt_deformed_query"].squeeze(0).numpy()

    ax2.scatter(obs[:, 0], obs[:, 1], obs[:, 2],
                c='grey', s=2, alpha=0.3, label='Observed (partial)')
    ax2.scatter(gt[:32, 0], gt[:32, 1], gt[:32, 2],
                c='green', s=30, marker='*', label='GT surface')
    ax2.scatter(gt[32:, 0], gt[32:, 1], gt[32:, 2],
                c='lime', s=30, marker='^', label='GT interior')
    ax2.scatter(predicted[:32, 0], predicted[:32, 1], predicted[:32, 2],
                c='red', s=30, marker='*', alpha=0.7, label='Pred surface')
    ax2.scatter(predicted[32:, 0], predicted[32:, 1], predicted[32:, 2],
                c='salmon', s=30, marker='^', alpha=0.7, label='Pred interior')

    ae = np.linalg.norm(predicted - gt, axis=-1).mean()
    ax2.set_title(f"Observation Space  |  AE={ae:.4f}")
    ax2.legend(fontsize=7)
    ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5); ax2.set_zlim(-1.5, 1.5)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig("demo_output.png", dpi=120, bbox_inches='tight')
    print(f"Saved visualization to demo_output.png")
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--deformation", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load or initialize model
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        print(f"Loading checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        cfg = OmegaConf.create(checkpoint["cfg"])
        model = DeformationFieldNet(cfg)
        model.load_state_dict(checkpoint["model_state"])
        title = f"Trained model (epoch {checkpoint['epoch']})"
    else:
        print("No checkpoint — using random weights (for pipeline verification only)")
        cfg = make_default_cfg()
        from canonical_ar.models.deformation_field import DeformationFieldNet
        model = DeformationFieldNet(cfg)
        title = "Random weights (pipeline check)"

    model.to(device)
    model.eval()

    # Generate scene
    scene = generate_demo_scene(deformation_magnitude=args.deformation)
    scene = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in scene.items()}

    # Run inference
    with torch.no_grad():
        out = model(
            canonical_xyz=scene["canonical_xyz"],
            canonical_feat=scene["canonical_feat"],
            obs_xyz=scene["obs_xyz"],
            obs_feat=scene["obs_feat"],
            query_pts=scene["query_pts"],
        )

    predicted = out["deformed_pts"].squeeze(0).cpu().numpy()
    gt = scene["gt_deformed_query"].squeeze(0).cpu().numpy()
    ae = np.linalg.norm(predicted - gt, axis=-1).mean()

    print(f"\nDeformation magnitude: {args.deformation}")
    print(f"Attachment Error: {ae:.4f} (normalized units)")
    print(f"  Surface AE:  {np.linalg.norm(predicted[:32] - gt[:32], axis=-1).mean():.4f}")
    print(f"  Interior AE: {np.linalg.norm(predicted[32:] - gt[32:], axis=-1).mean():.4f}")

    scene_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                 for k, v in scene.items()}
    visualize_scene(scene_cpu, predicted, title=title)


if __name__ == "__main__":
    from canonical_ar.models.deformation_field import DeformationFieldNet
    main()
