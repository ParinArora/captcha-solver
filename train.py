import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # Import tqdm for progress bars

# ----------------------------
# Configuration
# ----------------------------
train_dir = "/home/parin.arora_ug2023/CNN+/upscaler/google-recaptcha/data/train"
test_dir  = "/home/parin.arora_ug2023/CNN+/upscaler/google-recaptcha/data/test"
train_img_size = 120   # Training images resized to 120x120
val_img_size   = 100   # Validation images resized to 100x100
batch_size = 64
num_classes = 12
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cpu":
    num_cpu_cores = os.cpu_count() - 4
    torch.set_num_threads(num_cpu_cores)
    num_workers = num_cpu_cores
    print("Running on CPU. Using available cores:", num_cpu_cores)
else:
    # For GPU, you may stick with a lower number of DataLoader workers.
    num_workers = 4

# ----------------------------
# Transforms
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((train_img_size, train_img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize((val_img_size, val_img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ----------------------------
# Datasets & Loaders
# ----------------------------
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(test_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

print("Classes:", train_dataset.classes)

# ----------------------------
# Model Definition
# ----------------------------
class ImprovedCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNNModel, self).__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: Input -> 120x120 (or 100x100) images
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # output same spatial size
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves spatial dimensions (120->60, 100->50)
            nn.Dropout(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves spatial dimensions again (60->30, 50->25)
            nn.Dropout(0.3),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves spatial dimensions (30->15, 25->12)
            nn.Dropout(0.4),
        )
        # Global average pooling to reduce the spatial dimension to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

model = ImprovedCNNModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

# ----------------------------
# Training Loop with tqdm progress bars
# ----------------------------
for epoch in range(num_epochs):
    # Training Phase
    model.train()
    total_train = 0
    correct_train = 0
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct_train += torch.sum(preds == labels).item()
        total_train += labels.size(0)
        running_loss += loss.item() * labels.size(0)
        
        # Optionally update progress bar with current loss
        train_bar.set_postfix(loss=loss.item())
    
    train_loss = running_loss / total_train
    train_acc = correct_train / total_train
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_acc)
    
    # Validation Phase
    model.eval()
    total_val = 0
    correct_val = 0
    running_val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False)
    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct_val += torch.sum(preds == labels).item()
            total_val += labels.size(0)
            running_val_loss += loss.item() * labels.size(0)
            val_bar.set_postfix(loss=loss.item())
            
    val_loss = running_val_loss / total_val
    val_acc = correct_val / total_val
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")

# ----------------------------
# Save the Model
# ----------------------------
torch.save(model.state_dict(), 'improved_cnn_model_120train_100val.pth')
print("Model saved successfully.")

# ----------------------------
# Plot Training History (Optional)
# ----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history['train_accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
