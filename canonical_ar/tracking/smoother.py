"""
Temporal Smoothing for AR Stability

Per-frame model output is noisy — depth cameras have measurement noise,
the model has inference uncertainty, and sudden occlusions cause jumps.
Without smoothing, virtual content jitters visibly which breaks AR immersion.

We implement three strategies that can be combined:

1. Exponential moving average (EMA) — cheap, adds latency
2. Velocity-based prediction (constant velocity model) — handles fast motion
3. Outlier rejection — ignores frames where the model output jumps implausibly

For the course project, EMA is sufficient.
For CVPR, the Kalman filter variant is more principled and publishable.
"""

import numpy as np
import torch
from dataclasses import dataclass, field


@dataclass
class SmoothingState:
    """Per-query-point smoothing state."""
    position: np.ndarray        # (Q, 3) current smoothed position
    velocity: np.ndarray        # (Q, 3) estimated velocity
    initialized: bool = False
    frame_count: int = 0


class ExponentialSmoother:
    """
    Per-point EMA smoother with velocity estimation.

    alpha: smoothing factor.
        0.0 = never update (frozen)
        1.0 = no smoothing (raw model output)
        0.3 = heavy smoothing, more latency
        0.7 = light smoothing, less latency
    """

    def __init__(
        self,
        alpha: float = 0.5,
        velocity_alpha: float = 0.3,
        outlier_threshold: float = 0.15,
    ):
        """
        Args:
            alpha: position smoothing factor
            velocity_alpha: velocity smoothing factor
            outlier_threshold: max plausible per-frame displacement (normalized units).
                If model output jumps more than this, the frame is rejected.
        """
        self.alpha = alpha
        self.velocity_alpha = velocity_alpha
        self.outlier_threshold = outlier_threshold
        self.state: SmoothingState | None = None

    def reset(self):
        self.state = None

    def update(self, raw_pts: np.ndarray) -> np.ndarray:
        """
        Args:
            raw_pts: (Q, 3) raw model output for this frame

        Returns:
            smoothed_pts: (Q, 3)
        """
        if self.state is None or not self.state.initialized:
            self.state = SmoothingState(
                position=raw_pts.copy(),
                velocity=np.zeros_like(raw_pts),
                initialized=True,
                frame_count=1,
            )
            return raw_pts.copy()

        # Outlier rejection: if displacement is implausibly large, skip this frame
        displacement = np.linalg.norm(raw_pts - self.state.position, axis=-1)
        mean_disp = displacement.mean()
        if mean_disp > self.outlier_threshold and self.state.frame_count > 5:
            # Return velocity-predicted position instead of raw model output
            predicted = self.state.position + self.state.velocity
            return predicted

        # Velocity estimate: difference between current raw and previous smoothed
        raw_velocity = raw_pts - self.state.position

        # Update velocity with EMA
        self.state.velocity = (
            self.velocity_alpha * raw_velocity
            + (1 - self.velocity_alpha) * self.state.velocity
        )

        # Update position with EMA
        self.state.position = (
            self.alpha * raw_pts
            + (1 - self.alpha) * (self.state.position + self.state.velocity)
        )

        self.state.frame_count += 1
        return self.state.position.copy()

    def update_tensor(self, raw_pts: torch.Tensor) -> torch.Tensor:
        """Tensor wrapper around update()."""
        squeeze = raw_pts.dim() == 3
        if squeeze:
            raw_pts = raw_pts.squeeze(0)
        device = raw_pts.device
        smoothed = self.update(raw_pts.cpu().numpy())
        result = torch.from_numpy(smoothed).to(device)
        if squeeze:
            result = result.unsqueeze(0)
        return result


class KalmanSmoother:
    """
    Per-point Kalman filter with constant velocity model.
    More principled than EMA — handles variable frame rate,
    models measurement noise and process noise separately.

    State per point: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]

    This is the version to use for the CVPR paper —
    it's the standard approach in tracking literature and
    reviewers will recognise it.
    """

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-2,
    ):
        self.q = process_noise       # process noise variance
        self.r = measurement_noise   # measurement noise variance
        self.states: np.ndarray | None = None    # (Q, 6) [x,y,z,vx,vy,vz]
        self.covariances: np.ndarray | None = None  # (Q, 6, 6)

        # Transition matrix: constant velocity
        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 3] = 1.0  # x += vx
        self.F[1, 4] = 1.0  # y += vy
        self.F[2, 5] = 1.0  # z += vz

        # Measurement matrix: observe position only
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

    def reset(self):
        self.states = None
        self.covariances = None

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Args:
            measurement: (Q, 3) observed positions

        Returns:
            filtered: (Q, 3) Kalman-filtered positions
        """
        Q = len(measurement)

        # Initialize on first call
        if self.states is None:
            self.states = np.zeros((Q, 6), dtype=np.float32)
            self.states[:, :3] = measurement
            self.covariances = np.stack([np.eye(6, dtype=np.float32)] * Q)
            return measurement.copy()

        # Process noise covariance
        Q_noise = self.q * np.eye(6, dtype=np.float32)
        # Measurement noise covariance
        R_noise = self.r * np.eye(3, dtype=np.float32)

        filtered = np.zeros_like(measurement)

        for i in range(Q):
            # — Predict —
            x = self.F @ self.states[i]                          # (6,)
            P = self.F @ self.covariances[i] @ self.F.T + Q_noise  # (6,6)

            # — Update —
            S = self.H @ P @ self.H.T + R_noise                  # (3,3)
            K = P @ self.H.T @ np.linalg.inv(S)                  # (6,3) Kalman gain
            y = measurement[i] - self.H @ x                      # (3,) innovation
            x = x + K @ y
            P = (np.eye(6) - K @ self.H) @ P

            self.states[i] = x
            self.covariances[i] = P
            filtered[i] = x[:3]

        return filtered

    def update_tensor(self, raw_pts: torch.Tensor) -> torch.Tensor:
        squeeze = raw_pts.dim() == 3
        if squeeze:
            raw_pts = raw_pts.squeeze(0)
        device = raw_pts.device
        filtered = self.update(raw_pts.cpu().numpy())
        result = torch.from_numpy(filtered).to(device)
        if squeeze:
            result = result.unsqueeze(0)
        return result
