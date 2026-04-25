"""
Synthetic data generation using ShapeNet meshes.

Pipeline per sample:
  1. Load a ShapeNet mesh with texture
  2. Normalize mesh to unit sphere
  3. Sample canonical point cloud: xyz + normals + RGB (9-dim, but features=normals+RGB=6-dim)
  4. Apply RBF deformation to get deformed mesh (same topology, same UV)
  5. Render observed point cloud from single viewpoint: xyz + normals + RGB
  6. Compute GT deformed positions for all canonical points (dense correspondence)
  7. Normalize everything into canonical coordinate frame

Key design decisions:
  - RGB carried through deformation via UV preservation
  - Dense GT correspondences for all 2048 canonical points
  - Visibility mask separates visible from occluded canonical points
  - Interior virtual content removed (unreliable for complex meshes)
  - Color augmentation for sim-to-real robustness
"""

from __future__ import annotations

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import KDTree
import trimesh
import trimesh.repair
from omegaconf import DictConfig

from canonical_ar.utils.normalization import normalize_np
from canonical_ar.utils.partial_observation import simulate_partial_observation


# ── ShapeNet mesh loading ──────────────────────────────────────────────────────

def load_shapenet_mesh(obj_path: str) -> tuple:
    """
    Load a ShapeNet .obj mesh with texture if available.
    Returns (mesh, has_texture).
    """
    try:
        loaded = trimesh.load(obj_path, force='mesh', process=False)
        if isinstance(loaded, trimesh.Scene):
            # Some ShapeNet objects load as scenes — merge geometries
            mesh = trimesh.util.concatenate([
                g for g in loaded.geometry.values()
                if isinstance(g, trimesh.Trimesh)
            ])
        else:
            mesh = loaded

        # Basic repair
        trimesh.repair.fix_normals(mesh)

        if len(mesh.vertices) < 100 or len(mesh.faces) < 50:
            return None, False

        has_texture = False
        try:
            color_mesh = mesh.visual.to_color()
            if color_mesh.vertex_colors is not None:
                has_texture = True
        except Exception:
            pass

        return mesh, has_texture

    except Exception:
        return None, False


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh to unit sphere centered at origin."""
    verts = mesh.vertices.copy()
    centroid = verts.mean(axis=0)
    verts -= centroid
    scale = np.linalg.norm(verts, axis=1).max()
    if scale < 1e-6:
        return mesh
    verts /= scale
    return trimesh.Trimesh(
        vertices=verts,
        faces=mesh.faces,
        visual=mesh.visual,
        process=False,
    )


def sample_surface_with_color(
    mesh: trimesh.Trimesh,
    n_points: int,
    has_texture: bool,
    obj_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample points from mesh surface with normals and RGB.
    Returns:
        pts:     (N, 3)
        normals: (N, 3)
        colors:  (N, 3) in [0,1] — zeros if no texture
    """
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_idx].copy()

    if has_texture:
        try:
            color_mesh = mesh.visual.to_color()
            vc = color_mesh.vertex_colors  # (V, 4) RGBA uint8
            # Use first vertex of each face as color approximation
            face_verts = mesh.faces[face_idx, 0]
            colors = vc[face_verts, :3].astype(np.float32) / 255.0
        except Exception:
            has_texture = False

    if not has_texture:
        # Deterministic pseudo-color per face — breaks symmetry without real texture
        rng = np.random.default_rng(seed=obj_seed)
        face_colors = rng.uniform(0.2, 0.9, (len(mesh.faces), 3)).astype(np.float32)
        colors = face_colors[face_idx]

    return (
        pts.astype(np.float32),
        normals.astype(np.float32),
        colors.astype(np.float32),
    )


def augment_colors(
    colors: np.ndarray,
    rng: np.random.Generator,
    brightness_range: float = 0.3,
    contrast_range: float = 0.3,
    saturation_range: float = 0.3,
) -> np.ndarray:
    """
    Random color augmentation for sim-to-real robustness.
    Brightness, contrast, saturation jitter.
    """
    # Brightness
    b = 1.0 + rng.uniform(-brightness_range, brightness_range)
    colors = colors * b

    # Contrast
    mean = colors.mean(axis=0, keepdims=True)
    c = 1.0 + rng.uniform(-contrast_range, contrast_range)
    colors = (colors - mean) * c + mean

    # Saturation
    gray = colors.mean(axis=1, keepdims=True)
    s = 1.0 + rng.uniform(-saturation_range, saturation_range)
    colors = gray + (colors - gray) * s

    return np.clip(colors, 0.0, 1.0).astype(np.float32)


# ── Deformation ───────────────────────────────────────────────────────────────

def smooth_random_deformation(
    vertices: np.ndarray,
    magnitude: float,
    n_control_points: int = 16,
    seed: int = None,
) -> np.ndarray:
    """
    RBF deformation from random control points.
    Uses 16 control points (vs 8 before) for more localized deformations.
    """
    rng = np.random.default_rng(seed)
    scale = np.linalg.norm(vertices, axis=1).max()
    if scale < 1e-6 or magnitude < 1e-6:
        return vertices.copy()

    idx = rng.choice(len(vertices), size=n_control_points, replace=False)
    control_pts = vertices[idx]
    displacements = rng.normal(0, magnitude * scale, size=(n_control_points, 3))

    # Smaller sigma for more localized deformation
    sigma = scale * 0.3
    diff = vertices[:, None, :] - control_pts[None, :, :]
    dist_sq = (diff ** 2).sum(axis=-1)
    weights = np.exp(-dist_sq / (2 * sigma ** 2))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    vertex_disp = (weights[:, :, None] * displacements[None, :, :]).sum(axis=1)
    return (vertices + vertex_disp).astype(np.float32)


# ── ShapeNet dataset loader ───────────────────────────────────────────────────

class ShapeNetMeshLoader:
    """
    Scans a ShapeNet directory for .obj files and provides random access.
    Expected structure:
        shapenet_root/
            <category_id>/
                <model_id>/
                    models/model_normalized.obj
    """
    def __init__(self, shapenet_root: str, max_meshes: int = 500, cache_dir: str = None):
        self.cache_dir = cache_dir
        self.paths = []
        pattern_obj = os.path.join(shapenet_root, '**', '*.obj')
        pattern_off = os.path.join(shapenet_root, '**', '*.off')
        all_paths = glob.glob(pattern_obj, recursive=True) + glob.glob(pattern_off, recursive=True)
        # Filter to model_normalized.obj preferentially
        normalized = [p for p in all_paths if 'model_normalized' in p]
        other = [p for p in all_paths if 'model_normalized' not in p]
        candidates = normalized + other
        if len(candidates) > max_meshes:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(candidates), size=max_meshes, replace=False)
            candidates = [candidates[i] for i in idx]
        self.paths = candidates
        if len(self.paths) == 0:
            raise ValueError(
                f"No .obj files found in {shapenet_root}. "
                "Check that ShapeNet is downloaded correctly."
            )
        print(f"ShapeNetMeshLoader: found {len(self.paths)} meshes")

    def __len__(self):
        return len(self.paths)

    def get(self, idx: int):
        path = self.paths[idx % len(self.paths)]
        
        if self.cache_dir:
            cache_key = path.replace("/", "_").replace(".", "_")
            cache_path = f"{self.cache_dir}/{cache_key}.npz"
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path)
                    mesh = trimesh.Trimesh(
                        vertices=data["vertices"].astype(np.float64),
                        faces=data["faces"],
                        process=False,  # skip all repair — already clean in cache
                    )
                    return mesh, False, obj_seed, \
                        data["pts"], data["norms"], data["rgb"]
                except Exception:
                    pass
        
        result = load_shapenet_mesh(path)
        if result[0] is None:
            return None
        mesh, has_texture = result
        mesh = normalize_mesh(mesh)
        return mesh, has_texture, idx, None, None, None


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShapeNetDeformationDataset(Dataset):
    """
    Dataset of (canonical, observed, gt_deformed_canonical) tuples
    using ShapeNet meshes.

    Each sample:
      - Canonical point cloud (xyz + normals + RGB) from a ShapeNet mesh
      - Observed point cloud (partial, noisy, same features) from deformed mesh
      - GT deformed positions for ALL canonical points (dense correspondence)
      - Visibility mask: which canonical points are visible in the observation
    """

    def __init__(
        self,
        shapenet_root: str,
        cache_dir: str = None,
        n_objects: int = 400,
        deformations_per_object: int = 20,
        n_canonical_pts: int = 2048,
        n_obs_pts: int = 1024,
        n_surface_content: int = 64,
        deformation_mag_min: float = 0.0,
        deformation_mag_max: float = 0.4,
        noise_std: float = 0.005,
        visibility_threshold: float = 0.15,
        split: str = "train",
        split_ratios: tuple = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        super().__init__()
        self.n_canonical_pts = n_canonical_pts
        self.n_obs_pts = n_obs_pts
        self.n_surface_content = n_surface_content
        self.deformation_mag_min = deformation_mag_min
        self.deformation_mag_max = deformation_mag_max
        self.noise_std = noise_std
        self.visibility_threshold = visibility_threshold

        self.mesh_loader = ShapeNetMeshLoader(shapenet_root, max_meshes=n_objects, cache_dir=cache_dir)
        n_meshes = len(self.mesh_loader)

        rng = np.random.default_rng(seed)
        total = n_meshes * deformations_per_object
        all_idx = np.arange(total)
        rng.shuffle(all_idx)

        n_train = int(total * split_ratios[0])
        n_val = int(total * split_ratios[1])
        if split == "train":
            self.indices = all_idx[:n_train]
        elif split == "val":
            self.indices = all_idx[n_train:n_train + n_val]
        else:
            self.indices = all_idx[n_train + n_val:]

        self.deformations_per_object = deformations_per_object
        self.deformation_mag = rng.uniform(
            deformation_mag_min, deformation_mag_max,
            size=(n_meshes, deformations_per_object)
        ).astype(np.float32)
        self.deformation_seed = rng.integers(
            0, 100000, size=(n_meshes, deformations_per_object)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        flat_idx = self.indices[idx]
        obj_idx = flat_idx // self.deformations_per_object
        def_idx = flat_idx % self.deformations_per_object

        result = self.mesh_loader.get(obj_idx)
        if result is None:
            return self.__getitem__((idx + 1) % len(self))
        mesh, has_texture, obj_seed, cached_pts, cached_norms, cached_rgb = result

        # Use cached point cloud if available, else sample fresh
        if cached_pts is not None:
            # Subsample from cached 4096 points to n_canonical_pts
            rng_sub = np.random.default_rng(seed=obj_seed + def_idx)
            idx_sub = rng_sub.choice(len(cached_pts), size=self.n_canonical_pts, replace=False)
            can_pts = cached_pts[idx_sub]
            can_norms = cached_norms[idx_sub]
            can_rgb = cached_rgb[idx_sub]
        else:
            can_pts, can_norms, can_rgb = sample_surface_with_color(
                mesh, self.n_canonical_pts, has_texture, obj_seed
            )
        # Color augmentation for canonical (mild)
        rng_aug = np.random.default_rng(seed=obj_seed + def_idx)
        can_rgb = augment_colors(can_rgb, rng_aug,
                                 brightness_range=0.1,
                                 contrast_range=0.1,
                                 saturation_range=0.1)

        # ── Virtual content registration (surface only) ───────────────────
        content_idx = np.random.default_rng(seed=obj_seed).choice(
            self.n_canonical_pts, size=self.n_surface_content, replace=False
        )
        content_pts = can_pts[content_idx]  # (Q, 3) in canonical space

        # ── Apply deformation ─────────────────────────────────────────────
        mag = float(self.deformation_mag[obj_idx, def_idx])
        def_seed = int(self.deformation_seed[obj_idx, def_idx])

        deformed_verts = smooth_random_deformation(
            mesh.vertices.copy(), magnitude=mag, seed=def_seed
        )
        deformed_mesh = trimesh.Trimesh(
            vertices=deformed_verts,
            faces=mesh.faces,
            visual=mesh.visual,   # UV texture carried over unchanged
            process=False,
        )

        # ── Observed point cloud (partial, noisy, with color) ─────────────
        obs_pts_full, obs_norms_full, obs_rgb_full = sample_surface_with_color(
            deformed_mesh, self.n_obs_pts * 4, has_texture, obj_seed
        )
        obs_pts, obs_norms, viewpoint = simulate_partial_observation(
            obs_pts_full, obs_norms_full,
            n_output_pts=self.n_obs_pts,
            noise_std=self.noise_std,
            viewpoint_seed=def_seed,
        )
        # Match colors to surviving observed points via KDTree
        tree_obs_full = KDTree(obs_pts_full)
        _, obs_color_idx = tree_obs_full.query(obs_pts)
        obs_rgb = obs_rgb_full[obs_color_idx]
        # Stronger color augmentation for observed (simulates lighting variation)
        obs_rgb = augment_colors(obs_rgb, rng_aug,
                                 brightness_range=0.3,
                                 contrast_range=0.2,
                                 saturation_range=0.2)

        # ── GT dense correspondences ───────────────────────────────────────
        # Map each canonical point to its nearest mesh vertex,
        # then look up where that vertex went after deformation.
        tree_mesh = KDTree(mesh.vertices)

        # GT for full canonical cloud
        _, nn_idx_can = tree_mesh.query(can_pts)
        vertex_disp_can = deformed_verts[nn_idx_can] - mesh.vertices[nn_idx_can]
        gt_deformed_canonical = (can_pts + vertex_disp_can).astype(np.float32)

        # GT for virtual content
        _, nn_idx_q = tree_mesh.query(content_pts)
        vertex_disp_q = deformed_verts[nn_idx_q] - mesh.vertices[nn_idx_q]
        gt_deformed_content = (content_pts + vertex_disp_q).astype(np.float32)

        # ── Visibility mask ────────────────────────────────────────────────
        # Which canonical points have their GT deformed position
        # within threshold of the observed cloud?
        tree_obs = KDTree(obs_pts)
        dist_to_obs, _ = tree_obs.query(gt_deformed_canonical)
        visible_mask = (dist_to_obs < self.visibility_threshold).astype(np.float32)

        # ── Normalization ──────────────────────────────────────────────────
        can_norm, can_centroid, can_scale = normalize_np(can_pts)

        obs_centroid = obs_pts.mean(axis=0)
        obs_norm = ((obs_pts - obs_centroid) / can_scale).astype(np.float32)

        gt_can_norm = ((gt_deformed_canonical - can_centroid) / can_scale).astype(np.float32)
        query_norm = ((content_pts - can_centroid) / can_scale).astype(np.float32)
        gt_query_norm = ((gt_deformed_content - can_centroid) / can_scale).astype(np.float32)

        # Features: normals + RGB (6-dim) — xyz handled separately
        can_feat = np.concatenate([can_norms, can_rgb], axis=1).astype(np.float32)
        obs_feat = np.concatenate([obs_norms, obs_rgb], axis=1).astype(np.float32)

        return {
            "canonical_xyz":          torch.from_numpy(can_norm),       # (N_c, 3)
            "canonical_feat":         torch.from_numpy(can_feat),       # (N_c, 6)
            "obs_xyz":                torch.from_numpy(obs_norm),       # (N_o, 3)
            "obs_feat":               torch.from_numpy(obs_feat),       # (N_o, 6)
            "query_pts":              torch.from_numpy(query_norm),     # (Q, 3)
            "gt_deformed_query":      torch.from_numpy(gt_query_norm),  # (Q, 3)
            "gt_deformed_canonical":  torch.from_numpy(gt_can_norm),   # (N_c, 3)
            "visible_mask":           torch.from_numpy(visible_mask),   # (N_c,) float
            "norm_centroid":          torch.from_numpy(can_centroid),   # (3,)
            "norm_scale":             torch.tensor(can_scale),          # scalar
            "deformation_magnitude":  torch.tensor(mag),
            "object_idx":             torch.tensor(obj_idx),
        }


def build_dataloaders(cfg: DictConfig):
    """Build train/val/test dataloaders from config."""
    dc = cfg.data
    gen = dc.generation
    pc = dc.pointcloud
    vc = dc.virtual_content
    sp = dc.split

    shared = dict(
        shapenet_root=dc.shapenet_root,
        cache_dir=dc.get("cache_dir", None),
        n_objects=gen.num_objects,
        deformations_per_object=gen.deformations_per_object,
        n_canonical_pts=pc.num_points_canonical,
        n_obs_pts=pc.num_points_observed,
        n_surface_content=vc.num_surface_points,
        deformation_mag_min=gen.deformation_magnitude_min,
        deformation_mag_max=gen.deformation_magnitude_max,
        noise_std=gen.noise_std,
        visibility_threshold=dc.get("visibility_threshold", 0.15),
        split_ratios=(sp.train, sp.val, sp.test),
    )

    train_ds = ShapeNetDeformationDataset(split="train", **shared)
    val_ds   = ShapeNetDeformationDataset(split="val",   **shared)
    test_ds  = ShapeNetDeformationDataset(split="test",  **shared)

    loader_kw = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kw),
        DataLoader(val_ds,   shuffle=False, **loader_kw),
        DataLoader(test_ds,  shuffle=False, **loader_kw),
    )

# Backward compatibility alias
SyntheticDeformationDataset = ShapeNetDeformationDataset