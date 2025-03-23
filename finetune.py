import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------
# Helper: Split a grid image into cells
# ----------------------------
def split_grid_cells(image, grid_size=(3, 3)):
    """
    Splits a PIL image into grid cells.
    
    Parameters:
      image (PIL.Image): Input image.
      grid_size (tuple): (rows, cols) of the grid.
      
    Returns:
      List of PIL.Image objects, one per cell.
    """
    rows, cols = grid_size
    width, height = image.size
    cell_width = width // cols
    cell_height = height // rows
    cells = []
    for r in range(rows):
        for c in range(cols):
            left = c * cell_width
            upper = r * cell_height
            right = left + cell_width
            lower = upper + cell_height
            cell = image.crop((left, upper, right, lower))
            cells.append(cell)
    return cells

# ----------------------------
# Helper: Extract ground truth from filename
# ----------------------------
def extract_ground_truth(filename, num_cells=9):
    """
    Extracts grid cell indices for the target from the filename.
    
    Expected filename format example:
      anyprefix_1-2-3_anything.png
      
    Returns:
      A binary torch.Tensor of shape (num_cells,) where the cells indicated in the filename are set to 1.
    """
    m = re.search(r'_(\d+(?:-\d+)*)_', filename)
    target = torch.zeros(num_cells, dtype=torch.float32)
    if m:
        parts = m.group(1).split('-')
        try:
            indices = [int(x) for x in parts]
            for idx in indices:
                if 1 <= idx <= num_cells:
                    target[idx - 1] = 1.0
        except Exception as e:
            print(f"Error parsing ground truth in {filename}: {e}")
    return target

# ----------------------------
# Custom Dataset for Multi-Class Grid Images
# ----------------------------
class MultiClassGridImageDataset(Dataset):
    def __init__(self, root_dir, transform, grid_size=(3, 3), class_to_idx=None):
        """
        Expects images organized in subfolders (one per target class).
        Each image’s filename should encode the ground truth grid cell indices 
        (e.g. _1-2-3_ indicates that cells 1, 2, and 3 contain the object).
        
        Parameters:
          root_dir (str): Directory containing subfolders for each class.
          transform: Transformations to apply to each grid cell.
          grid_size (tuple): e.g. (3, 3).
          class_to_idx (dict, optional): Your original mapping from class names to indices.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.grid_size = grid_size
        self.samples = []  # list of (image_path, class_name)
        
        # Walk through each subfolder.
        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for file in os.listdir(class_folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_folder, file), class_name))
                    
        # Use provided mapping if available; otherwise build one from folder names.
        if class_to_idx is None:
            classes = sorted(list(set([label for _, label in self.samples])))
            self.class_to_idx = {cls.lower(): i for i, cls in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        
        self.num_cells = grid_size[0] * grid_size[1]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, class_name = self.samples[idx]
        label = self.class_to_idx[class_name.lower()]
        filename = os.path.basename(image_path)
        image = Image.open(image_path).convert('RGB')
        cells = split_grid_cells(image, self.grid_size)
        # Apply the transform to each cell
        cells_tensor = [self.transform(cell) for cell in cells]
        cells_tensor = torch.stack(cells_tensor)  # shape: (num_cells, C, H, W)
        target = extract_ground_truth(filename, num_cells=self.num_cells)
        return cells_tensor, label, target

# ----------------------------
# Model Definition (ImprovedCNNModel)
# ----------------------------
class ImprovedCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNNModel, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
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

# ----------------------------
# Fine-Tuning Loop for Multiple Classes with Train/Test Split
# ----------------------------
def finetune_multiclass_model(root_dir, num_epochs=10, lr=0.001, batch_size=8, grid_size=(3,3), class_to_idx=None):
    """
    Fine-tunes the model on a dataset with subfolders for each class.
    The dataset is split into training (80%) and testing (20%).
    
    For each image, only the output logits corresponding to the folder’s target class
    (as specified by your class_to_idx mapping) are used for loss.
    
    If CUDA is not available, the code uses as many CPU cores as possible.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        num_workers = os.cpu_count()
    else:
        num_workers = 4

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = MultiClassGridImageDataset(root_dir, transform, grid_size=grid_size, class_to_idx=class_to_idx)
    
    # Split dataset into training (80%) and testing (20%)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    num_classes = len(dataset.class_to_idx)
    model = ImprovedCNNModel(num_classes).to(device)
    # Optionally load pretrained weights:
    model.load_state_dict(torch.load('improved_cnn_model_120train_100val.pth', map_location=device))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.train()
    history_train_loss = []
    history_test_loss = []
    
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        for cells_tensor, labels, targets in pbar:
            # cells_tensor: shape (B, num_cells, C, H, W)
            # labels: (B,) target class index (from folder name)
            # targets: (B, num_cells) binary ground truth for that image
            B, num_cells, C, H, W = cells_tensor.shape
            cells_tensor = cells_tensor.view(B * num_cells, C, H, W).to(device)
            targets = targets.to(device)
            labels = labels.to(device)
            
            outputs = model(cells_tensor)  # shape: (B*num_cells, num_classes)
            outputs = outputs.view(B, num_cells, num_classes)  # shape: (B, num_cells, num_classes)
            # Select the logits corresponding to each sample's target class.
            target_logits = outputs[torch.arange(B).to(device), :, labels]  # shape: (B, num_cells)
            
            loss = criterion(target_logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Evaluate on test set
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for cells_tensor, labels, targets in test_loader:
                B, num_cells, C, H, W = cells_tensor.shape
                cells_tensor = cells_tensor.view(B * num_cells, C, H, W).to(device)
                targets = targets.to(device)
                labels = labels.to(device)
                
                outputs = model(cells_tensor)
                outputs = outputs.view(B, num_cells, num_classes)
                target_logits = outputs[torch.arange(B).to(device), :, labels]  # shape: (B, num_cells)
                loss = criterion(target_logits, targets)
                epoch_test_loss += loss.item()
                
        avg_test_loss = epoch_test_loss / len(test_loader)
        history_test_loss.append(avg_test_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Test Loss: {avg_test_loss:.4f}")
        model.train()
    
    torch.save(model.state_dict(), 'finetuned_multiclass_model_with_pretrained_weights.pth')
    print("Model saved.")
    
    # Plot training and test loss history
    plt.figure()
    plt.plot(history_train_loss, label='Train Loss')
    plt.plot(history_test_loss, label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Test Loss History")
    plt.legend()
    plt.show()

# ----------------------------
# Main entry point
# ----------------------------
if __name__ == '__main__':
    # Provide your original mapping (adjust as needed)
    original_class_to_idx = {
        "crosswalk": 0,
        "chimney": 1,
        "traffic light": 2,
        "stair": 3,
        "car": 4,
        "bus": 5,
        "palm": 6,
        "bicycle": 7,
        "hydrant": 8,
        "motorcycle": 9,
        "other": 10,
        "bridge": 11
    }
    # The root_dir should contain subfolders (e.g., car, bus, etc.) with grid images.
    root_dir = '/home/parin.arora_ug2023/CNN+/label2'
    finetune_multiclass_model(root_dir, num_epochs=10, lr=0.001, batch_size=8, grid_size=(3, 3), class_to_idx=original_class_to_idx)
