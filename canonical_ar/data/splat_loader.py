"""
Gaussian Splat → Canonical Point Cloud

This module bridges nerfstudio-trained Gaussian splats and our
deformation field model. The splat serves as the canonical
representation of the physical object — replacing the ground-truth
mesh we use during synthetic training.

Pipeline:
  1. Scan object with phone/depth camera (30-60 second video)
  2. Run nerfstudio `ns-train splatfacto` to get a .ply splat
  3. Call load_splat_as_pointcloud() to get (xyz, normals, colors)
  4. This becomes the canonical_xyz / canonical_feat for the model

The splat gives us a much richer canonical representation than a
simple mesh — it captures appearance, handles thin structures, and
doesn't require manual mesh cleanup.

Nerfstudio installation (on RunPod):
  pip install nerfstudio
  ns-train splatfacto --data /path/to/video_frames

To export splat as .ply:
  ns-export gaussian-splat --load-config outputs/.../config.yml \
      --output-dir exports/splat/
"""

import numpy as np
import torch
from pathlib import Path

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def load_splat_ply(ply_path: str | Path) -> dict[str, np.ndarray]:
    """
    Load a Gaussian splat .ply file exported from nerfstudio.

    Gaussian splat PLYs store per-Gaussian properties as vertex attributes:
      - x, y, z: Gaussian center positions
      - nx, ny, nz: (sometimes) normals
      - f_dc_0, f_dc_1, f_dc_2: DC color coefficients (SH degree 0)
      - opacity: raw opacity (before sigmoid)
      - scale_0, scale_1, scale_2: log scale of Gaussian ellipsoid
      - rot_0..3: rotation quaternion

    Returns dict with numpy arrays.
    """
    # We read the PLY manually to avoid heavy dependencies
    # Format: binary_little_endian or ascii
    try:
        import plyfile
        plydata = plyfile.PlyData.read(str(ply_path))
        vertex = plydata['vertex']

        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)

        # Colors from DC spherical harmonics coefficients
        # SH DC coefficient maps to RGB via: color = 0.5 + 0.2820948 * f_dc
        colors = None
        try:
            r = 0.5 + 0.2820948 * vertex['f_dc_0']
            g = 0.5 + 0.2820948 * vertex['f_dc_1']
            b = 0.5 + 0.2820948 * vertex['f_dc_2']
            colors = np.clip(np.stack([r, g, b], axis=1), 0, 1).astype(np.float32)
        except (ValueError, KeyError):
            pass

        # Opacity (sigmoid to get actual opacity)
        opacity = None
        try:
            raw_opacity = np.array(vertex['opacity']).astype(np.float32)
            opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
        except (ValueError, KeyError):
            pass

        # Scales (log-space, exponentiate to get actual scales)
        scales = None
        try:
            scales = np.exp(np.stack([
                vertex['scale_0'], vertex['scale_1'], vertex['scale_2']
            ], axis=1)).astype(np.float32)
        except (ValueError, KeyError):
            pass

        return {
            "xyz": xyz,
            "colors": colors,
            "opacity": opacity,
            "scales": scales,
        }

    except ImportError:
        raise ImportError(
            "plyfile not installed. Run: pip install plyfile"
        )


def splat_to_pointcloud(
    splat: dict[str, np.ndarray],
    n_points: int = 2048,
    opacity_threshold: float = 0.1,
    use_colors_as_features: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Gaussian splat to a point cloud suitable for the encoder.

    We:
      1. Filter low-opacity Gaussians (background/floaters)
      2. Sample Gaussians weighted by opacity
      3. Estimate normals via PCA on local neighborhoods

    Args:
        splat: output of load_splat_ply
        n_points: number of output points
        opacity_threshold: discard Gaussians below this opacity
        use_colors_as_features: if True, return RGB as features instead of normals

    Returns:
        xyz:      (n_points, 3) point positions
        features: (n_points, 3) normals or colors
    """
    xyz = splat["xyz"]
    opacity = splat.get("opacity")

    # Filter by opacity
    if opacity is not None:
        mask = opacity > opacity_threshold
        xyz = xyz[mask]
        if splat.get("colors") is not None:
            colors_filtered = splat["colors"][mask]
        opacity_filtered = opacity[mask]
    else:
        opacity_filtered = np.ones(len(xyz))
        colors_filtered = splat.get("colors")

    if len(xyz) == 0:
        raise ValueError(
            f"No Gaussians survived opacity threshold {opacity_threshold}. "
            "Try lowering opacity_threshold."
        )

    # Sample weighted by opacity
    weights = opacity_filtered / opacity_filtered.sum()
    if len(xyz) >= n_points:
        idx = np.random.choice(len(xyz), size=n_points, replace=False, p=weights)
    else:
        idx = np.random.choice(len(xyz), size=n_points, replace=True, p=weights)
    xyz_sampled = xyz[idx]

    # Features: normals (default) or colors
    if use_colors_as_features and colors_filtered is not None:
        features = colors_filtered[idx]
    else:
        # Estimate normals via Open3D PCA
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_sampled.astype(np.float64))
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)
            features = np.asarray(pcd.normals).astype(np.float32)
        else:
            # Fallback: zero normals (model will still work, just loses normal signal)
            print("Warning: open3d not available, using zero normals. "
                  "Install with: pip install open3d")
            features = np.zeros_like(xyz_sampled)

    return xyz_sampled.astype(np.float32), features.astype(np.float32)


def load_splat_as_pointcloud(
    ply_path: str | Path,
    n_points: int = 2048,
    opacity_threshold: float = 0.1,
    normalize: bool = True,
) -> dict[str, torch.Tensor | np.ndarray]:
    """
    Full pipeline: .ply file → normalized point cloud ready for the encoder.

    Args:
        ply_path: path to nerfstudio-exported splat .ply
        n_points: number of points to sample
        opacity_threshold: filter low-opacity Gaussians
        normalize: if True, center and scale to unit sphere

    Returns dict with:
        canonical_xyz:  (n_points, 3) torch.Tensor, normalized
        canonical_feat: (n_points, 3) torch.Tensor, normals
        norm_centroid:  (3,) np.ndarray — for inverting normalization
        norm_scale:     float — for inverting normalization
    """
    from canonical_ar.utils.normalization import normalize_np

    splat = load_splat_ply(ply_path)
    xyz, features = splat_to_pointcloud(splat, n_points, opacity_threshold)

    if normalize:
        xyz_norm, centroid, scale = normalize_np(xyz)
    else:
        xyz_norm, centroid, scale = xyz, np.zeros(3), 1.0

    return {
        "canonical_xyz":  torch.from_numpy(xyz_norm).unsqueeze(0),   # (1, N, 3)
        "canonical_feat": torch.from_numpy(features).unsqueeze(0),   # (1, N, 3)
        "norm_centroid":  centroid,
        "norm_scale":     scale,
    }


def register_content_on_splat(
    splat_data: dict,
    content_type: str = "surface",
    n_points: int = 64,
    depth: float = 0.0,
) -> torch.Tensor:
    """
    Interactively register virtual content in canonical splat space.
    Returns query_pts (1, n_points, 3) in normalized canonical coords.

    For now this is a programmatic version — in the full system this
    would be driven by a UI where the user paints on the object.

    Args:
        splat_data: output of load_splat_as_pointcloud
        content_type: "surface" — points on the point cloud surface
                      "interior" — points offset inward from surface
        n_points: number of virtual content points
        depth: for "interior", how far inward from surface (normalized units)
    """
    can_xyz = splat_data["canonical_xyz"].squeeze(0).numpy()  # (N, 3)

    if content_type == "surface":
        # Random subset of canonical points
        idx = np.random.choice(len(can_xyz), size=n_points, replace=False)
        query_pts = can_xyz[idx]

    elif content_type == "interior":
        # Push surface points inward along approximate inward normal
        idx = np.random.choice(len(can_xyz), size=n_points, replace=False)
        pts = can_xyz[idx]
        # Inward direction: toward centroid
        centroid = can_xyz.mean(axis=0)
        inward = centroid - pts
        inward /= np.linalg.norm(inward, axis=1, keepdims=True) + 1e-8
        query_pts = pts + inward * depth

    else:
        raise ValueError(f"Unknown content_type: {content_type}")

    return torch.from_numpy(query_pts.astype(np.float32)).unsqueeze(0)  # (1, Q, 3)
