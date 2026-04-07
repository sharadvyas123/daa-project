import torchvision
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader

class PetDataset(OxfordIIITPet):
    def __getitem__(self, idx):
        img , mask = super().__getitem__(idx)
        mask = (mask == 1).float()
        return img , mask
    

dataset = PetDataset(
    root="./dataset",
    target_types="segmentation",
    transform=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ]),
    target_transform=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
)

loader = DataLoader(dataset , batch_size=16, shuffle=True)
