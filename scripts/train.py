"""
Main training script.

Launch with:
    python scripts/train.py +run_name=exp001
    python scripts/train.py +run_name=exp002 train.batch_size=32
    python scripts/train.py +run_name=ablation_no_smooth train.loss.smoothness_weight=0.0
"""

import os
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


def build_optimizer(model: torch.nn.Module, cfg: DictConfig):
    opt_cfg = cfg.train.optimizer
    if opt_cfg.type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=opt_cfg.betas,
        )
    raise ValueError(f"Unknown optimizer: {opt_cfg.type}")


def build_scheduler(optimizer, cfg: DictConfig):
    sch_cfg = cfg.train.scheduler
    if sch_cfg.type == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sch_cfg.T_max, eta_min=sch_cfg.eta_min
        )
    raise ValueError(f"Unknown scheduler: {sch_cfg.type}")


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def run_batch(
    model: DeformationFieldNet,
    loss_fn: DeformationLoss,
    batch: dict,
) -> dict:
    """Forward pass + loss computation. Shared between train and val."""
    # Encode both point clouds and decode deformation for query points
    out = model(
        canonical_xyz=batch["canonical_xyz"],
        canonical_feat=batch["canonical_feat"],
        obs_xyz=batch["obs_xyz"],
        obs_feat=batch["obs_feat"],
        query_pts=batch["query_pts"],
    )

    # Also need deformed canonical points for Chamfer loss.
    # Run decoder over the full canonical point cloud.
    decoder_out_full = model.decoder(batch["canonical_xyz"], out["z"])
    deformed_canonical = decoder_out_full["deformed_pts"]

    losses = loss_fn(
        deformed_canonical=deformed_canonical,
        displacements=decoder_out_full["displacements"],
        deformed_query=out["deformed_pts"],
        obs_xyz=batch["obs_xyz"],
        canonical_pts=batch["canonical_xyz"],
        gt_deformed_query=batch["gt_deformed_query"],
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
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "cfg": OmegaConf.to_container(cfg),
    }, path)
    log.info(f"Saved checkpoint: {path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # ------------------------------------------------------------------ setup
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

    # ------------------------------------------------------------------ data
    log.info("Building dataloaders...")
    train_loader, val_loader, _ = build_dataloaders(cfg)
    log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ----------------------------------------------------------------- model
    model = DeformationFieldNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    loss_fn = DeformationLoss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # --------------------------------------------------------------- training
    global_step = 0
    best_val_loss = float("inf")
    ckpt_cfg = cfg.train.checkpointing

    for epoch in range(cfg.train.num_epochs):
        model.train()
        epoch_losses = {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs}")
        for batch in pbar:
            batch = move_batch(batch, device)
            optimizer.zero_grad()

            losses = run_batch(model, loss_fn, batch)
            losses["loss"].backward()

            # Gradient clipping — important for stability with FiLM layers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            # Step-level logging
            if global_step % cfg.logging.log_every_n_steps == 0:
                log_dict = {f"train/{k}": v.item() for k, v in losses.items()}
                log_dict["train/lr"] = scheduler.get_last_lr()[0]
                if cfg.logging.use_wandb:
                    wandb.log(log_dict, step=global_step)
                pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")

            global_step += 1

        scheduler.step()

        # --------------------------------------------------------- validation
        val_losses = evaluate(model, loss_fn, val_loader, device, max_batches=50)
        val_loss = val_losses["loss"]

        # Epoch-level logging
        n_batches = len(train_loader)
        log_dict = {f"train_epoch/{k}": v / n_batches for k, v in epoch_losses.items()}
        log_dict.update({f"val/{k}": v for k, v in val_losses.items()})
        log_dict["epoch"] = epoch

        log.info(
            f"Epoch {epoch+1} | "
            f"train_loss={epoch_losses.get('loss', 0)/n_batches:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_attach={val_losses.get('loss_attachment', 0):.4f}"
        )

        if cfg.logging.use_wandb:
            wandb.log(log_dict, step=global_step)

        # ---------------------------------------------------- checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, tag="best")

        if (epoch + 1) % ckpt_cfg.save_every_n_epochs == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, tag=f"epoch{epoch+1}")

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
