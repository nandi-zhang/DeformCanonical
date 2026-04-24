"""
RGB-D Frame Processor

Converts a raw RGB-D frame (depth image + color image + camera intrinsics)
into the normalized point cloud format the deformation field model expects.

At runtime the pipeline per frame is:
  1. Back-project depth image to 3D point cloud
  2. Segment the target object (mask out background)
  3. Estimate normals from the depth cloud
  4. Normalize using the canonical object's stored scale/centroid
  5. Subsample to fixed n_obs_pts

This module handles steps 1-4.
Segmentation (step 2) is intentionally kept simple for the course
project — we use a bounding box or a SAM2 mask passed in externally.
For CHI/CVPR we'll plug in a proper segmentation model.
"""

import numpy as np
import torch
from dataclasses import dataclass

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_realsense_d435(cls):
        """Default RealSense D435 at 640x480."""
        return cls(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)

    @classmethod
    def from_iphone(cls):
        """Approximate iPhone LiDAR at 256x192 (ARKit depth resolution)."""
        return cls(fx=210.0, fy=210.0, cx=128.0, cy=96.0, width=256, height=192)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    color: np.ndarray = None,
    depth_scale: float = 1000.0,
    depth_min: float = 0.1,
    depth_max: float = 3.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Back-project depth image to 3D point cloud.

    Args:
        depth:       (H, W) uint16 or float32 depth image
        intrinsics:  camera intrinsics
        color:       (H, W, 3) uint8 RGB image, optional
        depth_scale: depth units per meter (1000 for mm, 1 for m)
        depth_min/max: valid depth range in meters

    Returns:
        pts:    (N, 3) valid 3D points in camera frame
        colors: (N, 3) float32 RGB if color provided, else None
    """
    depth_m = depth.astype(np.float32) / depth_scale

    H, W = depth_m.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    # Back-project
    x = (uu - intrinsics.cx) * depth_m / intrinsics.fx
    y = (vv - intrinsics.cy) * depth_m / intrinsics.fy
    z = depth_m

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Valid depth mask
    valid = (z.reshape(-1) > depth_min) & (z.reshape(-1) < depth_max)
    pts = pts[valid]

    colors = None
    if color is not None:
        colors_flat = color.reshape(-1, 3).astype(np.float32) / 255.0
        colors = colors_flat[valid]

    return pts.astype(np.float32), colors


def apply_mask(
    pts: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float = 1000.0,
    depth_min: float = 0.1,
    depth_max: float = 3.0,
) -> np.ndarray:
    """
    Filter point cloud to only points within a 2D segmentation mask.

    Args:
        pts:    (N, 3) full point cloud in camera frame
        depth:  (H, W) original depth image
        mask:   (H, W) bool segmentation mask
        ...

    Returns:
        masked_pts: (M, 3) points inside mask
    """
    H, W = depth.shape
    depth_m = depth.astype(np.float32) / depth_scale

    # Reconstruct which pixels were valid
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    z_flat = depth_m.reshape(-1)
    valid = (z_flat > depth_min) & (z_flat < depth_max)
    mask_flat = mask.reshape(-1)

    keep = valid & mask_flat
    pts_all = np.zeros((H * W, 3), dtype=np.float32)
    pts_all[valid] = pts  # re-expand (pts only has valid entries)

    return pts_all[keep]


def estimate_normals_from_depth(
    pts: np.ndarray,
    k_neighbors: int = 20,
    viewpoint: np.ndarray = None,
) -> np.ndarray:
    """
    Estimate surface normals via PCA on local neighborhoods.

    Args:
        pts:         (N, 3)
        k_neighbors: neighborhood size
        viewpoint:   (3,) if provided, orient normals toward this point
                     defaults to camera origin (0, 0, 0)

    Returns:
        normals: (N, 3)
    """
    if viewpoint is None:
        viewpoint = np.zeros(3)

    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )
        # Orient toward camera
        pcd.orient_normals_towards_camera_location(
            camera_location=viewpoint.astype(np.float64)
        )
        return np.asarray(pcd.normals).astype(np.float32)
    else:
        # Fallback: approximate normals from point cloud structure
        # Uses cross products of nearest neighbor differences
        from scipy.spatial import KDTree
        tree = KDTree(pts)
        _, idx = tree.query(pts, k=min(k_neighbors, len(pts)))

        normals = np.zeros_like(pts)
        for i in range(len(pts)):
            neighbors = pts[idx[i]]
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            _, vecs = np.linalg.eigh(cov)
            normal = vecs[:, 0]  # smallest eigenvector
            # Orient toward viewpoint
            if np.dot(normal, viewpoint - pts[i]) < 0:
                normal = -normal
            normals[i] = normal

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return (normals / (norms + 1e-8)).astype(np.float32)


def subsample_pointcloud(
    pts: np.ndarray,
    normals: np.ndarray,
    n_output: int,
    method: str = "fps",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample point cloud to fixed size.

    Args:
        pts:      (N, 3)
        normals:  (N, 3)
        n_output: target number of points
        method:   "fps" (farthest point — spatially uniform, slower)
                  "random" (random subsample — faster, fine for training)

    Returns:
        pts_out:     (n_output, 3)
        normals_out: (n_output, 3)
    """
    N = len(pts)
    if N == n_output:
        return pts, normals

    if N < n_output:
        # Upsample with repetition
        idx = np.random.choice(N, size=n_output, replace=True)
        return pts[idx], normals[idx]

    if method == "random":
        idx = np.random.choice(N, size=n_output, replace=False)
        return pts[idx], normals[idx]

    elif method == "fps":
        # Farthest point sampling in numpy
        idx = np.zeros(n_output, dtype=int)
        distances = np.full(N, np.inf)
        farthest = np.random.randint(0, N)
        for i in range(n_output):
            idx[i] = farthest
            centroid = pts[farthest]
            dist = ((pts - centroid) ** 2).sum(axis=1)
            distances = np.minimum(distances, dist)
            farthest = distances.argmax()
        return pts[idx], normals[idx]

    else:
        raise ValueError(f"Unknown method: {method}")


class FrameProcessor:
    """
    Stateful processor that converts raw RGB-D frames to model inputs.

    Holds the canonical object representation and normalization params
    so each frame can be processed consistently.

    Usage:
        processor = FrameProcessor.from_splat("path/to/splat.ply", intrinsics)
        # or
        processor = FrameProcessor.from_pointcloud(canonical_xyz, canonical_feat, intrinsics)

        for frame in camera_stream:
            model_input = processor.process_frame(frame.depth, frame.color, mask)
            deformed_pts = model.infer(**model_input)
            world_pts = processor.to_world(deformed_pts)
    """

    def __init__(
        self,
        canonical_xyz: np.ndarray,       # (N, 3) already normalized
        canonical_feat: np.ndarray,      # (N, 3)
        norm_centroid: np.ndarray,       # (3,)
        norm_scale: float,
        intrinsics: CameraIntrinsics,
        n_obs_pts: int = 1024,
        subsample_method: str = "fps",
        device: str = "cuda",
    ):
        self.norm_centroid = norm_centroid
        self.norm_scale = norm_scale
        self.intrinsics = intrinsics
        self.n_obs_pts = n_obs_pts
        self.subsample_method = subsample_method
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Pre-load canonical cloud as tensors on device
        self.canonical_xyz = torch.from_numpy(canonical_xyz).unsqueeze(0).to(self.device)
        self.canonical_feat = torch.from_numpy(canonical_feat).unsqueeze(0).to(self.device)

    @classmethod
    def from_splat(
        cls,
        ply_path: str,
        intrinsics: CameraIntrinsics,
        n_canonical_pts: int = 2048,
        **kwargs,
    ) -> "FrameProcessor":
        from canonical_ar.data.splat_loader import load_splat_as_pointcloud
        splat_data = load_splat_as_pointcloud(ply_path, n_points=n_canonical_pts)
        return cls(
            canonical_xyz=splat_data["canonical_xyz"].squeeze(0).numpy(),
            canonical_feat=splat_data["canonical_feat"].squeeze(0).numpy(),
            norm_centroid=splat_data["norm_centroid"],
            norm_scale=float(splat_data["norm_scale"]),
            intrinsics=intrinsics,
            **kwargs,
        )

    @classmethod
    def from_pointcloud(
        cls,
        canonical_xyz: np.ndarray,
        canonical_feat: np.ndarray,
        intrinsics: CameraIntrinsics,
        normalize: bool = True,
        **kwargs,
    ) -> "FrameProcessor":
        from canonical_ar.utils.normalization import normalize_np
        if normalize:
            xyz_norm, centroid, scale = normalize_np(canonical_xyz)
        else:
            xyz_norm = canonical_xyz
            centroid = np.zeros(3, dtype=np.float32)
            scale = 1.0
        return cls(
            canonical_xyz=xyz_norm,
            canonical_feat=canonical_feat,
            norm_centroid=centroid,
            norm_scale=scale,
            intrinsics=intrinsics,
            **kwargs,
        )

    def process_frame(
        self,
        depth: np.ndarray,
        color: np.ndarray = None,
        mask: np.ndarray = None,
        depth_scale: float = 1000.0,
    ) -> dict[str, torch.Tensor]:
        """
        Convert a single RGB-D frame into model-ready tensors.

        Args:
            depth: (H, W) uint16 or float32 depth image
            color: (H, W, 3) uint8 RGB, optional
            mask:  (H, W) bool segmentation mask for the target object.
                   If None, uses all valid depth points — only works
                   if the object is the only thing in the scene.
            depth_scale: depth units per meter

        Returns dict ready to unpack into model.infer():
            canonical_xyz:  (1, N_c, 3)
            canonical_feat: (1, N_c, 3)
            obs_xyz:        (1, N_o, 3)
            obs_feat:       (1, N_o, 3)
        """
        # 1. Back-project depth to point cloud
        pts, colors = depth_to_pointcloud(
            depth, self.intrinsics, color,
            depth_scale=depth_scale,
        )

        if len(pts) < 10:
            raise ValueError(
                "Too few valid depth points — check depth range or camera connection."
            )

        # 2. Apply segmentation mask if provided
        if mask is not None:
            pts = apply_mask(pts, depth, mask, self.intrinsics, depth_scale)
            if len(pts) < 10:
                raise ValueError(
                    "Too few points after masking — check segmentation mask."
                )

        # 3. Estimate normals
        normals = estimate_normals_from_depth(pts, k_neighbors=20)

        # 4. Subsample to fixed size
        pts, normals = subsample_pointcloud(
            pts, normals, self.n_obs_pts, method=self.subsample_method
        )

        # 5. Normalize: center on observed cloud, scale with canonical scale
        obs_centroid = pts.mean(axis=0)
        obs_norm = ((pts - obs_centroid) / self.norm_scale).astype(np.float32)

        obs_xyz  = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)
        obs_feat = torch.from_numpy(normals).unsqueeze(0).to(self.device)

        return {
            "canonical_xyz":  self.canonical_xyz,
            "canonical_feat": self.canonical_feat,
            "obs_xyz":        obs_xyz,
            "obs_feat":       obs_feat,
        }

    def to_world(self, normalized_pts: torch.Tensor) -> np.ndarray:
        """
        Convert model output (normalized canonical space) back to world coords.

        Args:
            normalized_pts: (1, Q, 3) or (Q, 3)

        Returns:
            world_pts: (Q, 3) numpy array in camera/world frame
        """
        pts = normalized_pts.squeeze(0).cpu().numpy()
        return pts * self.norm_scale + self.norm_centroid
