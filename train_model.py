import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score


# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------
# Dataset
# --------------------
class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in ["NORMAL", "PNEUMONIA"]:
            class_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(0 if label == "NORMAL" else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------
# Transforms
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------
# Datasets & Loaders
# --------------------
train_dataset = PneumoniaDataset("data/train", transform)
val_dataset   = PneumoniaDataset("data/val", transform)
test_dataset  = PneumoniaDataset("data/test", transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --------------------
# Model
# --------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
 # NORMAL / PNEUMONIA
model = model.to(device)


# --------------------
# Training Setup
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 3


# --------------------
# Training Loop
# --------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch + 1}, batch {i}/{len(train_loader)}")

    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Loss: {running_loss / len(train_loader):.4f}")


# --------------------
# Validation
# --------------------
model.eval()
val_labels = []
val_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        val_labels.extend(labels.cpu().numpy())
        val_preds.extend(preds.cpu().numpy())

val_accuracy = accuracy_score(val_labels, val_preds)
print("Validation accuracy:", val_accuracy)


# --------------------
# Testing
# --------------------
test_labels = []
test_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
print("Test accuracy:", test_accuracy)


# --------------------
# Save Model
# --------------------
torch.save(model.state_dict(), "pneumonia_classifier.pth")
print("Model saved as pneumonia_classifier.pth")

