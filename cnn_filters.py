import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
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
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training (sadece birkaç epoch yeter)
epochs = 50

for epoch in range(epochs):

    running_loss = 0

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}")

# İlk layer filtrelerini çıkar
filters = model.conv1.weight.data.cpu()

filters_np = filters.numpy()

np.save("cnn_filters.npy", filters_np)
print("CNN filters saved.")

plt.figure(figsize=(10,10))

for i in range(64):

    filt = filters[i][0]

    plt.subplot(8,8,i+1)
    plt.imshow(filt, cmap="gray")
    plt.axis("off")

plt.suptitle("CNN Learned Filters")

plt.tight_layout()

plt.savefig("outputs/cnn_filters.py.png", dpi=300, bbox_inches="tight")

plt.show()