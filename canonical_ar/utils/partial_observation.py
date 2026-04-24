"""
Partial observation simulation.

Real depth cameras only see the side of an object facing them.
If we train on full 360° point clouds, the model never learns to
handle partial observations and will fail catastrophically on real data.

This module simulates a single-viewpoint depth capture:
  1. Choose a random viewpoint on a sphere around the object
  2. Cast rays from viewpoint through each point
  3. Keep only points visible from that viewpoint (front-facing, not occluded)
  4. Optionally simulate depth noise and missing regions

We also add a frustum simulator for more realistic RGB-D camera behavior.
"""

import torch
import numpy as np


def random_viewpoint(radius: float = 2.5, seed: int = None) -> np.ndarray:
    """
    Sample a random viewpoint on a sphere of given radius.
    Returns (3,) position.
    """
    rng = np.random.default_rng(seed)
    # Uniform sampling on sphere via Gaussian method
    v = rng.normal(0, 1, size=3)
    v /= np.linalg.norm(v)
    return (v * radius).astype(np.float32)


def visibility_filter(
    pts: np.ndarray,
    normals: np.ndarray,
    viewpoint: np.ndarray,
    back_face_cull: bool = True,
    occlusion_filter: bool = True,
    occlusion_resolution: int = 64,
) -> np.ndarray:
    """
    Return boolean mask of visible points from viewpoint.

    Args:
        pts:      (N, 3) point positions
        normals:  (N, 3) surface normals
        viewpoint: (3,) camera position
        back_face_cull: remove back-facing points (dot(normal, view_dir) < 0)
        occlusion_filter: approximate occlusion via depth buffer

    Returns:
        mask: (N,) bool, True = visible
    """
    # View direction per point
    view_dirs = viewpoint[None, :] - pts          # (N, 3)
    view_dirs_norm = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8)

    mask = np.ones(len(pts), dtype=bool)

    # Back-face culling
    if back_face_cull:
        dot = (normals * view_dirs_norm).sum(axis=1)  # (N,)
        mask &= dot > 0.05   # small threshold to avoid grazing surfaces

    if not occlusion_filter or mask.sum() == 0:
        return mask

    # Approximate occlusion via a depth buffer in spherical coordinates
    # Project points onto unit sphere centered at viewpoint
    visible_pts = pts[mask]
    view_relative = visible_pts - viewpoint[None, :]
    dists = np.linalg.norm(view_relative, axis=1)

    # Spherical coordinates
    dirs = view_relative / (dists[:, None] + 1e-8)
    theta = np.arctan2(dirs[:, 1], dirs[:, 0])   # azimuth in [-pi, pi]
    phi   = np.arcsin(np.clip(dirs[:, 2], -1, 1)) # elevation in [-pi/2, pi/2]

    # Discretize into depth buffer cells
    res = occlusion_resolution
    ti = ((theta + np.pi) / (2 * np.pi) * res).astype(int).clip(0, res - 1)
    pi_ = ((phi + np.pi / 2) / np.pi * res).astype(int).clip(0, res - 1)

    # Build depth buffer: minimum distance per cell
    depth_buffer = np.full((res, res), np.inf)
    for i, (t, p, d) in enumerate(zip(ti, pi_, dists)):
        if d < depth_buffer[t, p]:
            depth_buffer[t, p] = d

    # A point is visible if its depth is within tolerance of the minimum
    tolerance = 0.05  # normalized units
    occ_visible = np.array([
        dists[i] <= depth_buffer[ti[i], pi_[i]] + tolerance
        for i in range(len(visible_pts))
    ])

    # Map back to full mask
    visible_indices = np.where(mask)[0]
    mask[visible_indices[~occ_visible]] = False

    return mask


def simulate_partial_observation(
    pts: np.ndarray,
    normals: np.ndarray,
    n_output_pts: int,
    noise_std: float = 0.005,
    viewpoint_seed: int = None,
    viewpoint_radius: float = 2.5,
    min_visible_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a realistic single-viewpoint depth camera observation.

    Args:
        pts:            (N, 3) full surface point cloud
        normals:        (N, 3) surface normals
        n_output_pts:   desired number of output points
        noise_std:      depth noise std dev (normalized units)
        viewpoint_seed: seed for reproducibility
        viewpoint_radius: distance of simulated camera
        min_visible_ratio: if fewer visible points, fall back to random subsample

    Returns:
        obs_pts:      (n_output_pts, 3) partial noisy point cloud
        obs_normals:  (n_output_pts, 3)
        viewpoint:    (3,) camera position used
    """
    viewpoint = random_viewpoint(radius=viewpoint_radius, seed=viewpoint_seed)
    vis_mask = visibility_filter(pts, normals, viewpoint)

    vis_pts = pts[vis_mask]
    vis_normals = normals[vis_mask]

    # Fall back if too few visible points (degenerate viewpoint)
    if len(vis_pts) < max(10, int(len(pts) * min_visible_ratio)):
        vis_pts = pts
        vis_normals = normals

    # Subsample or upsample to n_output_pts
    rng = np.random.default_rng(viewpoint_seed)
    if len(vis_pts) >= n_output_pts:
        idx = rng.choice(len(vis_pts), size=n_output_pts, replace=False)
    else:
        # Upsample with replacement — common for small visible regions
        idx = rng.choice(len(vis_pts), size=n_output_pts, replace=True)

    obs_pts = vis_pts[idx].copy()
    obs_normals = vis_normals[idx].copy()

    # Add depth noise
    if noise_std > 0:
        # Noise along view ray (realistic depth camera noise model)
        view_dirs = viewpoint[None, :] - obs_pts
        ray_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8)
        depth_noise = rng.normal(0, noise_std, size=(n_output_pts, 1))
        obs_pts = obs_pts + depth_noise * ray_dirs

        # Small normal perturbation
        obs_normals += rng.normal(0, noise_std * 0.5, size=obs_normals.shape)
        obs_normals /= np.linalg.norm(obs_normals, axis=1, keepdims=True) + 1e-8

    return (
        obs_pts.astype(np.float32),
        obs_normals.astype(np.float32),
        viewpoint,
    )
