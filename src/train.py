import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

# Data
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

train_data = datasets.Food101(root='../data', split='train', transform=transform_train, download=True)
val_data = datasets.Food101(root='../data', split='test', transform=transform_val, download=True)

load_train = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
load_validation = DataLoader(val_data, batch_size=64, num_workers=16, pin_memory=True)

NUM_CLASSES = 101

# Model
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")
model = model.to(device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for images, labels in load_train:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in load_validation:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Epoch {epoch+1} | Val Acc: {correct/total:.3f}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), '../model/model.pth')
print("Model saved.")
