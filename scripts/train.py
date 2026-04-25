"""
Main training script — v3.

Key changes:
  - run_batch passes gt_deformed_canonical, visible_mask, and
    cross-attention intermediate features to the loss
  - GT for SA2 canonical points computed on-the-fly via KNN lookup
  - Logs correspondence (visible/occluded split) and contrastive loss
"""

from __future__ import annotations

import torch
import hydra
import wandb
import logging
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from pathlib import Path

from canonical_ar.models.deformation_field import DeformationFieldNet
from canonical_ar.models.losses import DeformationLoss
from canonical_ar.data.synthetic import build_dataloaders

log = logging.getLogger(__name__)


def build_optimizer(model, cfg):
    o = cfg.train.optimizer
    if o.type == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=o.lr,
            weight_decay=o.weight_decay, betas=o.betas,
        )
    raise ValueError(f"Unknown optimizer: {o.type}")


def build_scheduler(optimizer, cfg):
    s = cfg.train.scheduler
    if s.type == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=s.T_max, eta_min=s.eta_min
        )
    raise ValueError(f"Unknown scheduler: {s.type}")


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def get_sa2_gt(
    canonical_xyz: torch.Tensor,          # (B, N, 3)
    can_xyz_sa2: torch.Tensor,            # (B, 128, 3)
    gt_deformed_canonical: torch.Tensor,  # (B, N, 3)
) -> torch.Tensor:
    """
    For each SA2 canonical point, find its nearest full canonical point
    and return the corresponding GT deformed position.

    SA2 points are selected by FPS from canonical_xyz, so each SA2 point
    is (approximately) one of the canonical points. We find the nearest
    canonical point to get the GT.

    Returns: (B, 128, 3)
    """
    # (B, 128, N) distances
    diff = can_xyz_sa2.unsqueeze(2) - canonical_xyz.unsqueeze(1)
    dist = (diff ** 2).sum(dim=-1)
    nn_idx = dist.argmin(dim=-1)  # (B, 128)

    B, K = nn_idx.shape
    idx_exp = nn_idx.unsqueeze(-1).expand(B, K, 3)
    return gt_deformed_canonical.gather(1, idx_exp)  # (B, 128, 3)


def run_batch(model, loss_fn, batch):
    """
    Forward pass + loss.

    Runs model over full canonical cloud (not just query points)
    to get deformation field for correspondence loss.
    Also extracts cross-attention intermediates for contrastive loss.
    """
    # Forward pass over query points (gets all intermediates)
    out = model(
        canonical_xyz=batch["canonical_xyz"],
        canonical_feat=batch["canonical_feat"],
        obs_xyz=batch["obs_xyz"],
        obs_feat=batch["obs_feat"],
        query_pts=batch["query_pts"],
    )

    # Decode over full canonical cloud for correspondence loss
    inner = model.module if hasattr(model, "module") else model
    local_feat_full = inner.get_local_features(
        batch["canonical_xyz"],
        out["can_xyz_sa2"],
        out["enriched_can_feat"],
    )
    decoder_full = inner.decoder(
        batch["canonical_xyz"], out["z"], local_feat_full
    )

    # GT for SA2 points (needed for contrastive loss)
    gt_sa2 = get_sa2_gt(
        batch["canonical_xyz"],
        out["can_xyz_sa2"],
        batch["gt_deformed_canonical"],
    )

    losses = loss_fn(
        deformed_canonical=decoder_full["deformed_pts"],
        displacements=decoder_full["displacements"],
        canonical_pts=batch["canonical_xyz"],
        gt_deformed_canonical=batch["gt_deformed_canonical"],
        visible_mask=batch["visible_mask"],
        enriched_can_feat=out["enriched_can_feat"],
        obs_feat_sa2=out["obs_feat_sa2"],
        can_xyz_sa2=out["can_xyz_sa2"],
        obs_xyz_sa2=out["obs_xyz_sa2"],
        gt_deformed_can_sa2=gt_sa2,
    )
    return losses


@torch.no_grad()
def evaluate(model, loss_fn, loader, device, max_batches=None):
    model.eval()
    totals = {}
    count = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move_batch(batch, device)
        losses = run_batch(model, loss_fn, batch)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        count += 1
    return {k: v / max(count, 1) for k, v in totals.items()}


def save_checkpoint(model, optimizer, scheduler, epoch, cfg, tag="latest"):
    ckpt_dir = Path(cfg.train.checkpointing.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_{tag}.pt"
    torch.save({
        "epoch": epoch,
        "model_state": (model.module.state_dict()
                        if hasattr(model, "module") else model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "cfg": OmegaConf.to_container(cfg),
    }, path)
    log.info(f"Saved checkpoint: {path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.project_name,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    log.info("Building dataloaders...")
    train_loader, val_loader, _ = build_dataloaders(cfg)
    log.info(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}")

    model = DeformationFieldNet(cfg).to(device)
    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    loss_fn = DeformationLoss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Resume
    start_epoch = 0
    resume_path = cfg.get("resume", None)
    if resume_path:
        log.info(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        state = ckpt["model_state"]
        wrapped = hasattr(model, "module")
        ckpt_wrapped = any(k.startswith("module.") for k in state.keys())
        if wrapped and not ckpt_wrapped:
            state = {"module." + k: v for k, v in state.items()}
        elif not wrapped and ckpt_wrapped:
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        log.info(f"Resumed from epoch {ckpt['epoch']}, starting at {start_epoch}")

    global_step = 0
    best_val_loss = float("inf")
    ckpt_cfg = cfg.train.checkpointing

    for epoch in range(start_epoch, cfg.train.num_epochs):
        model.train()
        epoch_losses = {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs}")
        for batch in pbar:
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            losses = run_batch(model, loss_fn, batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            if global_step % cfg.logging.log_every_n_steps == 0:
                log_dict = {f"train/{k}": v.item() for k, v in losses.items()}
                log_dict["train/lr"] = scheduler.get_last_lr()[0]
                if cfg.logging.use_wandb:
                    wandb.log(log_dict, step=global_step)
                pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")

            global_step += 1

        scheduler.step()

        val_losses = evaluate(model, loss_fn, val_loader, device, max_batches=50)
        val_loss = val_losses["loss"]
        n = len(train_loader)

        log.info(
            f"Epoch {epoch+1} | "
            f"train={epoch_losses.get('loss',0)/n:.4f} | "
            f"val={val_loss:.4f} | "
            f"corr_vis={val_losses.get('loss_corr_visible',0):.4f} | "
            f"corr_occ={val_losses.get('loss_corr_occluded',0):.4f} | "
            f"nce={val_losses.get('loss_contrastive',0):.4f}"
        )

        if cfg.logging.use_wandb:
            log_dict = {f"train_epoch/{k}": v/n for k, v in epoch_losses.items()}
            log_dict.update({f"val/{k}": v for k, v in val_losses.items()})
            log_dict["epoch"] = epoch
            wandb.log(log_dict, step=global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, tag="best")

        if (epoch + 1) % ckpt_cfg.save_every_n_epochs == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, cfg,
                           tag=f"epoch{epoch+1}")

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
