from canonical_ar.data.synthetic import ShapeNetDeformationDataset, build_dataloaders
from canonical_ar.data.splat_loader import load_splat_as_pointcloud, register_content_on_splat

__all__ = [
    "ShapeNetDeformationDataset",
    "build_dataloaders",
    "load_splat_as_pointcloud",
    "register_content_on_splat",
]
