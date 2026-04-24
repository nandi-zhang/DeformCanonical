# canonical-ar

**Canonical-Space Virtual Content Registration for AR**

A unified deformation field model for attaching virtual content to physical
objects across the rigid-to-deformable spectrum. Rigid objects are a
degenerate case (near-zero deformation field). The same model handles both.

---

## Architecture

```
Canonical point cloud ──► PointNet++ encoder ──┐
                                                ├──► Fusion ──► Fused latent z
Observed point cloud  ──► PointNet++ encoder ──┘

Query points (virtual content in canonical space)
    + z
    ▼
FiLM-conditioned MLP decoder
    ▼
Displacement field Δp
    ▼
Deformed query positions = virtual content in observation space
```

**Key design decisions:**
- Rigid tracking is a special case (Δp ≈ 0 or globally consistent)
- Virtual content registered as canonical coordinates at authoring time
- Surface and interior registration both supported — just different canonical positions
- Output head initialized near-zero so training starts close to identity

---

## Setup

```bash
pip install -e .
# For cloud (RunPod/Colab), pytorch3d needs special install:
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Quick sanity check

```bash
python tests/test_pipeline.py
```
Should print PASS for all four tests in ~60 seconds on CPU.

## Training

```bash
# Basic run
python scripts/train.py +run_name=exp001

# Override any config
python scripts/train.py +run_name=exp002 train.batch_size=32 train.optimizer.lr=5e-5

# Ablation: no smoothness loss
python scripts/train.py +run_name=no_smooth train.loss.smoothness_weight=0.0

# With wandb logging
python scripts/train.py +run_name=exp001 logging.use_wandb=true
```

Checkpoints saved to `checkpoints/`. Best model saved as `checkpoint_best.pt`.

---

## Roadmap

### Course project (current)
- [x] Model architecture (PointNet++ encoder + FiLM decoder)
- [x] Synthetic data pipeline (primitive meshes + RBF deformation)
- [x] Training loop with Hydra configs
- [ ] Gaussian splat integration (scan-then-track pipeline)
- [ ] Preliminary quantitative results on synthetic benchmark

### CHI paper
- [ ] Real object scanning pipeline
- [ ] Demo scenes: surface decoration + interior containment
- [ ] User study on attachment semantic intuitiveness
- [ ] Taxonomy formalization

### CVPR paper
- [ ] Replace PointNet++ encoder with transformer (better for partial obs)
- [ ] Learned deformation prior per object (Taichi soft-body sim)
- [ ] Physical benchmark dataset (20+ objects, motion capture GT)
- [ ] Comparison to FoundationPose (rigid baseline) and SAM2+spline (deformable baseline)
- [ ] Ablation study (shared encoder vs. separate, FiLM vs. concat, etc.)

---

## File structure

```
canonical_ar/
├── canonical_ar/
│   ├── models/
│   │   ├── deformation_field.py   # top-level model
│   │   ├── encoder.py             # PointNet++ encoder
│   │   ├── decoder.py             # FiLM-conditioned MLP decoder
│   │   ├── losses.py              # Chamfer + smoothness + attachment losses
│   │   └── utils.py               # PositionalEncoding, MLP
│   └── data/
│       └── synthetic.py           # synthetic dataset + dataloader builder
├── configs/
│   ├── config.yaml                # root config
│   ├── model/deformation_field.yaml
│   ├── data/synthetic.yaml
│   └── train/base.yaml
├── scripts/
│   └── train.py                   # main training script
├── tests/
│   └── test_pipeline.py           # sanity checks
└── setup.py
```
