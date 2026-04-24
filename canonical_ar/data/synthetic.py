"""
Synthetic data generation for training.

We generate canonical shapes (meshes), apply random smooth deformations,
and sample point clouds from both the canonical and deformed shapes.
Virtual content points are registered in canonical space and their
ground-truth deformed positions are computed.

Deformation types we generate:
  - Zero deformation (rigid baseline — model should output near-zero displacement)
  - Global rigid transform (rotation + translation — handled as a limiting case)
  - Smooth elastic deformation (the interesting case)

All generated from simple primitive meshes initially (sphere, box, cylinder)
which we can replace with real scanned objects later without changing
anything downstream.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import trimesh
from scipy.spatial import KDTree
from omegaconf import DictConfig

from canonical_ar.utils.normalization import normalize_np
from canonical_ar.utils.partial_observation import simulate_partial_observation


def generate_primitive_mesh(shape: str = "sphere", scale: float = 1.0) -> trimesh.Trimesh:
    """Generate a simple primitive mesh."""
    if shape == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=scale)
    elif shape == "box":
        mesh = trimesh.creation.box(extents=[scale, scale * 0.6, scale * 0.4])
    elif shape == "cylinder":
        mesh = trimesh.creation.cylinder(radius=scale * 0.3, height=scale)
    elif shape == "torus":
        mesh = trimesh.creation.torus(major_radius=scale * 0.5, minor_radius=scale * 0.2)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return mesh


def smooth_random_deformation(
    vertices: np.ndarray,
    magnitude: float,
    n_control_points: int = 8,
    seed: int = None,
) -> np.ndarray:
    """
    Apply smooth elastic deformation to vertices using RBF interpolation
    from random control point displacements.

    Args:
        vertices: (N, 3)
        magnitude: maximum displacement as fraction of object scale
        n_control_points: number of random control points
        seed: random seed

    Returns:
        deformed_vertices: (N, 3)
    """
    rng = np.random.default_rng(seed)

    # Normalize vertices to unit sphere scale
    scale = np.linalg.norm(vertices, axis=1).max()
    if scale < 1e-6:
        return vertices.copy()

    # Random control points on/near the surface
    idx = rng.choice(len(vertices), size=n_control_points, replace=False)
    control_pts = vertices[idx]  # (K, 3)

    # Random displacements at control points
    displacements = rng.normal(0, magnitude * scale, size=(n_control_points, 3))

    # RBF interpolation: Gaussian kernel
    sigma = scale * 0.5
    diff = vertices[:, None, :] - control_pts[None, :, :]  # (N, K, 3)
    dist_sq = (diff ** 2).sum(axis=-1)                      # (N, K)
    weights = np.exp(-dist_sq / (2 * sigma ** 2))           # (N, K)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

    # Interpolated displacement at each vertex
    vertex_disp = (weights[:, :, None] * displacements[None, :, :]).sum(axis=1)
    # (N, 3)

    return vertices + vertex_disp


def sample_surface_points(
    mesh: trimesh.Trimesh,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample points uniformly from mesh surface with face normals.
    Noise is handled downstream by simulate_partial_observation.
    Returns:
        pts: (n_points, 3)
        normals: (n_points, 3)
    """
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_idx]
    return pts.astype(np.float32), normals.astype(np.float32)


def register_virtual_content(
    mesh: trimesh.Trimesh,
    n_surface: int = 64,
    n_interior: int = 32,
) -> np.ndarray:
    """
    Register virtual content points in canonical space.
    Returns array of (n_surface + n_interior, 3) canonical coordinates.

    Surface points: on the mesh surface (decoration use case)
    Interior points: inside the mesh volume (containment use case)
    """
    # Surface points
    surface_pts, _ = trimesh.sample.sample_surface(mesh, n_surface)

    # Interior points: rejection sample inside mesh bounding box
    bounds = mesh.bounds  # (2, 3)
    interior_pts = []
    while len(interior_pts) < n_interior:
        candidates = np.random.uniform(
            bounds[0], bounds[1],
            size=(n_interior * 4, 3)
        )
        inside = mesh.contains(candidates)
        interior_pts.extend(candidates[inside].tolist())

    interior_pts = np.array(interior_pts[:n_interior], dtype=np.float32)
    return np.concatenate([surface_pts.astype(np.float32), interior_pts], axis=0)


class SyntheticDeformationDataset(Dataset):
    """
    Dataset of (canonical, observed, query_pts, gt_deformed_query) tuples.

    Each sample:
      - A canonical point cloud sampled from a primitive mesh
      - An observed point cloud sampled from a deformed version of that mesh
      - Virtual content query points in canonical space
      - Ground truth positions of those query points after deformation
    """
    def __init__(
        self,
        n_objects: int = 500,
        deformations_per_object: int = 20,
        n_canonical_pts: int = 2048,
        n_obs_pts: int = 1024,
        n_surface_content: int = 64,
        n_interior_content: int = 32,
        deformation_mag_min: float = 0.0,
        deformation_mag_max: float = 0.4,
        noise_std: float = 0.005,
        split: str = "train",
        split_ratios: tuple = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        super().__init__()
        self.n_canonical_pts = n_canonical_pts
        self.n_obs_pts = n_obs_pts
        self.n_content = n_surface_content + n_interior_content
        self.n_surface_content = n_surface_content
        self.n_interior_content = n_interior_content
        self.deformation_mag_min = deformation_mag_min
        self.deformation_mag_max = deformation_mag_max
        self.noise_std = noise_std

        rng = np.random.default_rng(seed)
        shapes = ["sphere", "box", "cylinder", "torus"]

        # Generate all object indices for this split
        total_samples = n_objects * deformations_per_object
        all_idx = np.arange(total_samples)
        rng.shuffle(all_idx)

        n_train = int(total_samples * split_ratios[0])
        n_val = int(total_samples * split_ratios[1])

        if split == "train":
            self.indices = all_idx[:n_train]
        elif split == "val":
            self.indices = all_idx[n_train:n_train + n_val]
        else:  # test
            self.indices = all_idx[n_train + n_val:]

        # Precompute seeds and parameters for reproducibility
        self.object_shape = [shapes[rng.integers(0, len(shapes))] for _ in range(n_objects)]
        self.object_scale = rng.uniform(0.5, 1.5, size=n_objects).astype(np.float32)
        self.object_seed = rng.integers(0, 100000, size=n_objects)

        self.deformation_mag = rng.uniform(
            deformation_mag_min, deformation_mag_max,
            size=(n_objects, deformations_per_object)
        ).astype(np.float32)
        self.deformation_seed = rng.integers(
            0, 100000,
            size=(n_objects, deformations_per_object)
        )
        self.deformations_per_object = deformations_per_object

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        flat_idx = self.indices[idx]
        obj_idx = flat_idx // self.deformations_per_object
        def_idx = flat_idx % self.deformations_per_object

        # Generate canonical mesh
        mesh = generate_primitive_mesh(
            self.object_shape[obj_idx],
            scale=float(self.object_scale[obj_idx])
        )

        # Sample canonical point cloud
        can_pts, can_norms = sample_surface_points(mesh, self.n_canonical_pts)

        # Register virtual content in canonical space
        content_pts = register_virtual_content(
            mesh,
            n_surface=self.n_surface_content,
            n_interior=self.n_interior_content,
        )

        # Apply deformation
        mag = float(self.deformation_mag[obj_idx, def_idx])
        def_seed = int(self.deformation_seed[obj_idx, def_idx])

        deformed_verts = smooth_random_deformation(
            mesh.vertices.copy(),
            magnitude=mag,
            seed=def_seed,
        )
        deformed_mesh = trimesh.Trimesh(
            vertices=deformed_verts,
            faces=mesh.faces,
            process=False,
        )

        # Sample observed point cloud — oversample then cull to single
        # viewpoint to simulate a real depth camera (partial + noisy)
        obs_pts_full, obs_norms_full = sample_surface_points(
            deformed_mesh, self.n_obs_pts * 4
        )
        obs_pts, obs_norms, _ = simulate_partial_observation(
            obs_pts_full, obs_norms_full,
            n_output_pts=self.n_obs_pts,
            noise_std=self.noise_std,
            viewpoint_seed=def_seed,
        )

        # Ground truth: where do content points end up after deformation?
        # Use KDTree to find nearest canonical vertex, apply same offset.
        tree = KDTree(mesh.vertices)
        _, nn_idx = tree.query(content_pts)
        canonical_verts = mesh.vertices[nn_idx]           # (Q, 3)
        deformed_verts_nn = deformed_verts[nn_idx]        # (Q, 3)
        vertex_disp = deformed_verts_nn - canonical_verts # (Q, 3)
        gt_deformed_content = content_pts + vertex_disp   # (Q, 3)

        # ── Normalization ───────────────────────────────────────────
        # Canonical cloud sets reference centroid and scale.
        can_norm, can_centroid, can_scale = normalize_np(can_pts)

        # Observed: center on itself, scale with canonical scale
        obs_centroid = obs_pts.mean(axis=0)
        obs_norm = ((obs_pts - obs_centroid) / can_scale).astype(np.float32)

        # Query/GT: apply canonical normalization
        query_norm    = ((content_pts       - can_centroid) / can_scale).astype(np.float32)
        gt_query_norm = ((gt_deformed_content - can_centroid) / can_scale).astype(np.float32)

        return {
            # Canonical point cloud (normalized)
            "canonical_xyz":  torch.from_numpy(can_norm),       # (N_c, 3)
            "canonical_feat": torch.from_numpy(can_norms),      # (N_c, 3)
            # Observed point cloud (normalized, partial)
            "obs_xyz":        torch.from_numpy(obs_norm),       # (N_o, 3)
            "obs_feat":       torch.from_numpy(obs_norms),      # (N_o, 3)
            # Virtual content (normalized canonical coords)
            "query_pts":         torch.from_numpy(query_norm),    # (Q, 3)
            "gt_deformed_query": torch.from_numpy(gt_query_norm), # (Q, 3)
            # Normalization params — needed to recover world coords at inference
            "norm_centroid": torch.from_numpy(can_centroid),    # (3,)
            "norm_scale":    torch.tensor(can_scale),           # scalar
            # Metadata
            "deformation_magnitude": torch.tensor(mag),
            "object_idx": torch.tensor(obj_idx),
        }


def build_dataloaders(cfg: DictConfig):
    """Build train/val/test dataloaders from config."""
    from torch.utils.data import DataLoader

    data_cfg = cfg.data
    gen = data_cfg.generation
    pc = data_cfg.pointcloud
    vc = data_cfg.virtual_content
    split = data_cfg.split

    shared_kwargs = dict(
        n_objects=gen.num_objects,
        deformations_per_object=gen.deformations_per_object,
        n_canonical_pts=pc.num_points_canonical,
        n_obs_pts=pc.num_points_observed,
        n_surface_content=vc.num_surface_points,
        n_interior_content=vc.num_interior_points,
        deformation_mag_min=gen.deformation_magnitude_min,
        deformation_mag_max=gen.deformation_magnitude_max,
        noise_std=gen.noise_std,
        split_ratios=(split.train, split.val, split.test),
    )

    train_ds = SyntheticDeformationDataset(split="train", **shared_kwargs)
    val_ds   = SyntheticDeformationDataset(split="val",   **shared_kwargs)
    test_ds  = SyntheticDeformationDataset(split="test",  **shared_kwargs)

    loader_kwargs = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )
