import torchvision
from torchvision.datasets import OxfordIIITPet

dataset = OxfordIIITPet(
    root="./dataset",
    download=True,
    target_types="segmentation"
)