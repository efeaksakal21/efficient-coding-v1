import os
import numpy as np
import matplotlib.pyplot as plt

print("comparison.py started")

print("Current folder:", os.getcwd())
print("Files in folder:", os.listdir("."))

sparse_file = "sparse_filters.npy"
cnn_file = "cnn_filters.npy"

print("Checking files...")
print("sparse_filters.npy exists:", os.path.exists(sparse_file))
print("cnn_filters.npy exists:", os.path.exists(cnn_file))

if not os.path.exists(sparse_file):
    raise FileNotFoundError("sparse_filters.npy bulunamadı. Önce sparseCoding.py çalıştırılmalı.")

if not os.path.exists(cnn_file):
    raise FileNotFoundError("cnn_filters.npy bulunamadı. Önce cnn_filters.py çalıştırılmalı.")

# Filtreleri yükle
sparse_filters = np.load(sparse_file)
cnn_filters = np.load(cnn_file)

print("Sparse filters shape:", sparse_filters.shape)
print("CNN filters shape:", cnn_filters.shape)

n_show = 64

plt.figure(figsize=(16, 8))

# Üst sıra: sparse coding
for i in range(n_show):
    plt.subplot(8, 16, i + 1)
    filt = sparse_filters[i].reshape(8, 8)
    plt.imshow(filt, cmap="gray")
    plt.axis("off")

# Alt sıra: CNN
for i in range(n_show):
    plt.subplot(8, 16, n_show + i + 1)
    filt = cnn_filters[i, 0]
    plt.imshow(filt, cmap="gray")
    plt.axis("off")

plt.suptitle("Top: Sparse Coding Filters | Bottom: CNN Filters", fontsize=14)
plt.tight_layout()
plt.show()

print("comparison.py finished")