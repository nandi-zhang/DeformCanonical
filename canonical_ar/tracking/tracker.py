"""
CanonicalARTracker

The top-level runtime object. Wraps:
  - FrameProcessor: raw depth → normalized point cloud
  - DeformationFieldNet: normalized observation → deformation field
  - Smoother: stabilize noisy per-frame output

Usage (with a real camera):

    tracker = CanonicalARTracker.from_checkpoint(
        checkpoint_path="checkpoints/checkpoint_best.pt",
        splat_path="scans/bottle.ply",
        query_pts=my_content_canonical_coords,   # (Q, 3) registered at authoring time
        intrinsics=CameraIntrinsics.from_realsense_d435(),
    )

    for depth, color, mask in camera_stream():
        world_pts = tracker.step(depth, color, mask)
        render_virtual_content(world_pts)

Usage (offline on saved frames — for evaluation):

    tracker = CanonicalARTracker.from_checkpoint(...)
    results = tracker.run_sequence(depths, colors, masks)
"""

import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from canonical_ar.models.deformation_field import DeformationFieldNet
from canonical_ar.tracking.frame_processor import FrameProcessor, CameraIntrinsics
from canonical_ar.tracking.smoother import ExponentialSmoother, KalmanSmoother


class CanonicalARTracker:
    def __init__(
        self,
        model: DeformationFieldNet,
        frame_processor: FrameProcessor,
        query_pts: torch.Tensor,        # (1, Q, 3) in normalized canonical space
        smoother_type: str = "ema",     # "ema" | "kalman" | "none"
        ema_alpha: float = 0.5,
        kalman_process_noise: float = 1e-3,
        kalman_measurement_noise: float = 1e-2,
    ):
        self.model = model
        self.model.eval()
        self.frame_processor = frame_processor
        self.query_pts = query_pts.to(frame_processor.device)

        if smoother_type == "ema":
            self.smoother = ExponentialSmoother(alpha=ema_alpha)
        elif smoother_type == "kalman":
            self.smoother = KalmanSmoother(
                process_noise=kalman_process_noise,
                measurement_noise=kalman_measurement_noise,
            )
        else:
            self.smoother = None

        self._frame_count = 0
        self._last_world_pts: np.ndarray | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        splat_path: str | Path,
        query_pts: np.ndarray,          # (Q, 3) canonical coords at authoring time
        intrinsics: CameraIntrinsics,
        device: str = "cuda",
        **tracker_kwargs,
    ) -> "CanonicalARTracker":
        """
        Load a trained model from checkpoint and set up the full tracker.

        Args:
            checkpoint_path: path to .pt checkpoint from training
            splat_path: path to nerfstudio-exported .ply splat
            query_pts: virtual content positions in canonical space,
                       registered during authoring
            intrinsics: camera intrinsics for depth back-projection
            device: "cuda" or "cpu"
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg = OmegaConf.create(checkpoint["cfg"])

        model = DeformationFieldNet(cfg)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

        frame_processor = FrameProcessor.from_splat(
            splat_path, intrinsics, device=device
        )

        # Normalize query points with canonical scale
        query_norm = (
            (query_pts - frame_processor.norm_centroid) / frame_processor.norm_scale
        ).astype(np.float32)
        query_tensor = torch.from_numpy(query_norm).unsqueeze(0)

        return cls(
            model=model,
            frame_processor=frame_processor,
            query_pts=query_tensor,
            **tracker_kwargs,
        )

    @classmethod
    def from_pointcloud(
        cls,
        checkpoint_path: str | Path,
        canonical_xyz: np.ndarray,
        canonical_feat: np.ndarray,
        query_pts: np.ndarray,
        intrinsics: CameraIntrinsics,
        device: str = "cuda",
        **tracker_kwargs,
    ) -> "CanonicalARTracker":
        """Alternative constructor using a pre-built canonical point cloud."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg = OmegaConf.create(checkpoint["cfg"])

        model = DeformationFieldNet(cfg)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)

        frame_processor = FrameProcessor.from_pointcloud(
            canonical_xyz, canonical_feat, intrinsics, device=device
        )

        query_norm = (
            (query_pts - frame_processor.norm_centroid) / frame_processor.norm_scale
        ).astype(np.float32)
        query_tensor = torch.from_numpy(query_norm).unsqueeze(0)

        return cls(
            model=model,
            frame_processor=frame_processor,
            query_pts=query_tensor,
            **tracker_kwargs,
        )

    @torch.no_grad()
    def step(
        self,
        depth: np.ndarray,
        color: np.ndarray = None,
        mask: np.ndarray = None,
        depth_scale: float = 1000.0,
    ) -> np.ndarray:
        """
        Process one frame and return virtual content positions in world space.

        Args:
            depth:       (H, W) depth image
            color:       (H, W, 3) RGB image, optional
            mask:        (H, W) bool segmentation mask, optional
            depth_scale: depth units per meter

        Returns:
            world_pts: (Q, 3) virtual content positions in camera/world frame
        """
        # 1. Convert raw depth frame to model input
        try:
            model_input = self.frame_processor.process_frame(
                depth, color, mask, depth_scale
            )
        except ValueError as e:
            # Bad frame (too few points, occlusion, etc.)
            # Return last known position if available
            if self._last_world_pts is not None:
                return self._last_world_pts
            raise e

        # 2. Run model
        deformed_norm = self.model.infer(
            canonical_xyz=model_input["canonical_xyz"],
            canonical_feat=model_input["canonical_feat"],
            obs_xyz=model_input["obs_xyz"],
            obs_feat=model_input["obs_feat"],
            query_pts=self.query_pts,
        )  # (1, Q, 3) normalized

        # 3. Smooth
        if self.smoother is not None:
            deformed_norm = self.smoother.update_tensor(deformed_norm)

        # 4. Denormalize to world coordinates
        world_pts = self.frame_processor.to_world(deformed_norm)

        self._last_world_pts = world_pts
        self._frame_count += 1
        return world_pts

    def run_sequence(
        self,
        depths: list[np.ndarray],
        colors: list[np.ndarray] = None,
        masks: list[np.ndarray] = None,
        depth_scale: float = 1000.0,
        verbose: bool = False,
    ) -> dict[str, list]:
        """
        Run tracker on a saved sequence. Used for offline evaluation.

        Returns dict with:
            world_pts_per_frame: list of (Q, 3) arrays
            failed_frames: list of frame indices where tracking failed
        """
        if colors is None:
            colors = [None] * len(depths)
        if masks is None:
            masks = [None] * len(depths)

        self.reset()
        world_pts_per_frame = []
        failed_frames = []

        for i, (depth, color, mask) in enumerate(zip(depths, colors, masks)):
            try:
                pts = self.step(depth, color, mask, depth_scale)
                world_pts_per_frame.append(pts)
            except ValueError as e:
                if verbose:
                    print(f"Frame {i} failed: {e}")
                failed_frames.append(i)
                world_pts_per_frame.append(None)

        return {
            "world_pts_per_frame": world_pts_per_frame,
            "failed_frames": failed_frames,
            "total_frames": len(depths),
            "success_rate": 1.0 - len(failed_frames) / max(len(depths), 1),
        }

    def reset(self):
        """Reset tracker state (call when switching objects or scenes)."""
        if self.smoother is not None:
            self.smoother.reset()
        self._frame_count = 0
        self._last_world_pts = None
