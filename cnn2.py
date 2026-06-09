import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# outputs klasörü yoksa oluştur
os.makedirs("outputs", exist_ok=True)

# CPU / GPU seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Dataset
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 8x8 CNN filtreleri
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=8,
            stride=1
        )

        self.relu = nn.ReLU()

        
        self.fc = nn.Linear(64 * 25 * 25, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

# Training
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(trainloader)
    acc = 100 * correct / total

    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {avg_loss:.3f}, "
        f"Acc: {acc:.2f}%"
    )

# Modeli kaydet
torch.save(model.state_dict(), "cnn_model.pth")
print("CNN model saved.")

# İlk layer filtrelerini çıkar
filters = model.conv1.weight.data.cpu()
filters_np = filters.numpy()

np.save("cnn_filters.npy", filters_np)
print("CNN filters saved.")
print("CNN filters shape:", filters_np.shape)

# Filtreleri çiz
plt.figure(figsize=(10, 10))

for i in range(filters.shape[0]):
    filt = filters[i][0].detach().cpu().numpy()

    # Görselleştirme için normalize et
    filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)

    plt.subplot(8, 8, i + 1)
    plt.imshow(filt, cmap="gray")
    plt.axis("off")

plt.suptitle("CNN Learned Filters")
plt.tight_layout()

plt.savefig("outputs/cnn_filters.png", dpi=300, bbox_inches="tight")
plt.show()
