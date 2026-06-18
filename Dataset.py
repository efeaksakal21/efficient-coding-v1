import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])


dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


patches = []
patch_size = 8

for img, _ in dataset:
    img = img.squeeze().numpy()  

    for i in range(0, 32 - patch_size + 1, patch_size):
        for j in range(0, 32 - patch_size + 1, patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            patches.append(patch)


patches = np.array(patches)

print("Patches shape:", patches.shape)


patches = (patches - np.mean(patches)) / np.std(patches)


plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(patches[i], cmap="gray")
    plt.axis("off")

plt.tight_layout()

plt.savefig("outputs/Dataset.py.png", dpi=300, bbox_inches="tight")
plt.show()


patches = patches.reshape(patches.shape[0], -1)

print("Vectorized patches shape:", patches.shape)


