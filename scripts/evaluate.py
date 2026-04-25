"""
Evaluation Script

Computes quantitative metrics on the synthetic test set.
Run after training to get the numbers for your course project report
and the preliminary results section of the CHI paper.

Metrics:
  - Attachment Error (AE): mean L2 distance between predicted and GT
    virtual content positions. Primary metric — directly measures
    the task objective.
  - Surface AE / Interior AE: broken down by content registration type.
    Important for the paper: surface and interior should have similar
    error, validating the unified approach.
  - AE by Deformation Magnitude: shows the model handles the
    rigid-to-deformable spectrum. Should see near-zero error for
    rigid (mag=0) objects and graceful degradation for large deformations.

Launch:
    python scripts/evaluate.py checkpoint_path=checkpoints/checkpoint_best.pt
"""

import torch
import numpy as np
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from canonical_ar.models.deformation_field import DeformationFieldNet
from canonical_ar.data.synthetic import ShapeNetDeformationDataset
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def attachment_error(
    pred: torch.Tensor,      # (B, Q, 3)
    gt: torch.Tensor,        # (B, Q, 3)
) -> torch.Tensor:
    """Mean L2 distance per sample. Returns (B,)."""
    return (pred - gt).norm(dim=-1).mean(dim=-1)


def run_evaluation(
    model: DeformationFieldNet,
    test_loader: DataLoader,
    device: torch.device,
    n_surface: int,
    n_interior: int,
) -> dict[str, float | list]:
    """
    Run full evaluation over test set.
    Returns dict of metrics.
    """
    model.eval()

    all_ae = []
    all_ae_surface = []
    all_ae_interior = []
    deformation_bins = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            out = model(
                canonical_xyz=batch["canonical_xyz"],
                canonical_feat=batch["canonical_feat"],
                obs_xyz=batch["obs_xyz"],
                obs_feat=batch["obs_feat"],
                query_pts=batch["query_pts"],
            )

            pred = out["deformed_pts"]    # (B, Q, 3)
            gt   = batch["gt_deformed_query"]

            # Overall AE
            ae = attachment_error(pred, gt)   # (B,)
            all_ae.extend(ae.cpu().tolist())

            # Surface vs interior
            ae_per_point = (pred - gt).norm(dim=-1)  # (B, Q)
            ae_surf = ae_per_point[:, :n_surface].mean(dim=-1)
            ae_int  = ae_per_point[:, n_surface:].mean(dim=-1)
            all_ae_surface.extend(ae_surf.cpu().tolist())
            all_ae_interior.extend(ae_int.cpu().tolist())

            # Bin by deformation magnitude
            mags = batch["deformation_magnitude"].cpu().tolist()
            for mag, err in zip(mags, ae.cpu().tolist()):
                bin_key = round(mag * 10) / 10  # bin to nearest 0.1
                deformation_bins[bin_key].append(err)

    # Aggregate
    ae_by_magnitude = {
        mag: np.mean(errs)
        for mag, errs in sorted(deformation_bins.items())
    }

    return {
        "ae_mean":          np.mean(all_ae),
        "ae_median":        np.median(all_ae),
        "ae_std":           np.std(all_ae),
        "ae_surface_mean":  np.mean(all_ae_surface),
        "ae_interior_mean": np.mean(all_ae_interior),
        "ae_by_magnitude":  ae_by_magnitude,
        "n_samples":        len(all_ae),
    }


def print_results(metrics: dict):
    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Samples evaluated:     {metrics['n_samples']}")
    print(f"  Attachment Error (mean):   {metrics['ae_mean']:.4f}")
    print(f"  Attachment Error (median): {metrics['ae_median']:.4f}")
    print(f"  Attachment Error (std):    {metrics['ae_std']:.4f}")
    print(f"  AE surface points:         {metrics['ae_surface_mean']:.4f}")
    print(f"  AE interior points:        {metrics['ae_interior_mean']:.4f}")
    print()
    print("  AE by deformation magnitude:")
    for mag, ae in metrics["ae_by_magnitude"].items():
        bar = "█" * int(ae * 200)
        print(f"    mag={mag:.1f}  AE={ae:.4f}  {bar}")
    print("=" * 55)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # checkpoint_path must be passed: checkpoint_path=path/to/ckpt.pt
    ckpt_path = cfg.get("checkpoint_path")
    if ckpt_path is None:
        raise ValueError("Pass checkpoint_path=path/to/checkpoint.pt")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    log.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Use config from checkpoint so eval matches training setup exactly
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    model = DeformationFieldNet(train_cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    log.info(f"Loaded model from epoch {checkpoint['epoch']}")

    # Build test dataset
    dc = train_cfg.data
    test_ds = ShapeNetDeformationDataset(
        split="test",
        seed=42,
        n_objects=dc.generation.num_objects,
        deformations_per_object=dc.generation.deformations_per_object,
        n_canonical_pts=dc.pointcloud.num_points_canonical,
        n_obs_pts=dc.pointcloud.num_points_observed,
        n_surface_content=dc.virtual_content.num_surface_points,
        n_interior_content=dc.virtual_content.num_interior_points,
        deformation_mag_min=dc.generation.deformation_magnitude_min,
        deformation_mag_max=dc.generation.deformation_magnitude_max,
        noise_std=dc.generation.noise_std,
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    log.info(f"Test samples: {len(test_ds)}")

    # Run evaluation
    metrics = run_evaluation(
        model, test_loader, device,
        n_surface=dc.virtual_content.num_surface_points,
        n_interior=dc.virtual_content.num_interior_points,
    )

    print_results(metrics)

    # Save results
    out_path = Path(ckpt_path).parent / "eval_results.npy"
    np.save(out_path, metrics)
    log.info(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
