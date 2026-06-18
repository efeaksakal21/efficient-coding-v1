import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import MiniBatchDictionaryLearning



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

patches = np.array(patches, dtype=np.float32)

print("Original patches shape:", patches.shape)


patches = (patches - np.mean(patches)) / np.std(patches)


patches = patches.reshape(patches.shape[0], -1)

print("Vectorized patches shape:", patches.shape)




n_samples = 50000
patches_small = patches[:n_samples]

print("Training subset shape:", patches_small.shape)



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

background_color = "#BFE8C2"

fig = plt.figure(figsize=(12, 16))
fig.patch.set_facecolor(background_color)

for i in range(n_components):
    ax = plt.subplot(16, 8, i + 1)

    filt = dictionary[i].reshape(8, 8)
    plt.imshow(filt, cmap="gray")
    plt.axis("off")

    ax.set_facecolor(background_color)

plt.suptitle(
    "Sparse Coding Learned Filters",
    fontsize=22,
    fontweight="bold",
    y=0.98
)

plt.figtext(
    0.5,
    0.02,
    "V1-like edge, orientation and contrast filters",
    ha="center",
    fontsize=12
)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

plt.savefig(
    "outputs/sparse_coding_filters_green.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.show()