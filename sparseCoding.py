import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import MiniBatchDictionaryLearning

# -----------------------------
# 1. Dataset ve patch hazırlığı
# -----------------------------
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
    img = img.squeeze().numpy()  # (32, 32)

    for i in range(0, 32 - patch_size + 1, patch_size):
        for j in range(0, 32 - patch_size + 1, patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

patches = np.array(patches, dtype=np.float32)

print("Original patches shape:", patches.shape)

# normalize
patches = (patches - np.mean(patches)) / np.std(patches)

# vectorize
patches = patches.reshape(patches.shape[0], -1)

print("Vectorized patches shape:", patches.shape)

# ---------------------------------
# 2. Hesaplama için veri küçültme
# ---------------------------------
n_samples = 50000
patches_small = patches[:n_samples]

print("Training subset shape:", patches_small.shape)

# ---------------------------------
# 3. Sparse coding / dictionary learning
# ---------------------------------
n_components = 128

dict_learner = MiniBatchDictionaryLearning(
    n_components=n_components,
    alpha=0.8,
    max_iter=300,
    batch_size=256,
    random_state=42
)

codes = dict_learner.fit_transform(patches_small)
dictionary = dict_learner.components_

print("Codes shape:", codes.shape)
print("Dictionary shape:", dictionary.shape)

np.save("sparse_filters.npy", dictionary)
print("Sparse filters saved.")

plt.figure(figsize=(12, 16))

for i in range(n_components):
    plt.subplot(16, 8, i + 1)
    filt = dictionary[i].reshape(8, 8)
    plt.imshow(filt, cmap="gray")
    plt.axis("off")

plt.suptitle("Learned Sparse Coding Filters", fontsize=14)
plt.tight_layout()
plt.show()