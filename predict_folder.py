import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from torchvision import transforms, models
from PIL import Image

# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Load model
# --------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("pneumonia_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# --------------------
# Transforms (same as training)
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes = ["NORMAL", "PNEUMONIA"]

# --------------------
# File picker (USER selects image)
# --------------------
root = tk.Tk()
root.withdraw()  # hide tkinter window

img_path = filedialog.askopenfilename(
    title="Select a chest X-ray image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not img_path:
    print("No image selected. Exiting.")
    exit()

# --------------------
# Prediction
# --------------------
image = Image.open(img_path).convert("RGB")
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    _, prediction = torch.max(output, 1)

predicted_class = classes[prediction.item()]

# --------------------
# Output
# --------------------
print(f"{os.path.basename(img_path)} → {predicted_class}")

plt.imshow(image)
plt.title(f"{predicted_class}")
plt.axis("off")
plt.show()
