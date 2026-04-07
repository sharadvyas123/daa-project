import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = OxfordIIITPet(
    root="./dataset",
    target_types="segmentation",
    transform=transform,
    target_transform=transform
)

image , mask = dataset[0]

plt.subplot(1,2,1)
plt.title("Image")
plt.imshow(image.permute(1,2,0),cmap="gray")

plt.savefig("output/image.png")
print(f"Image saved to output/image.png")