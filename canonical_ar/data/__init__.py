from canonical_ar.data.synthetic import SyntheticDeformationDataset, build_dataloaders
from canonical_ar.data.splat_loader import load_splat_as_pointcloud, register_content_on_splat

__all__ = [
    "SyntheticDeformationDataset",
    "build_dataloaders",
    "load_splat_as_pointcloud",
    "register_content_on_splat",
]
