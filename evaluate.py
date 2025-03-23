import os
import re
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import subprocess

#--------------------------------------
# Utility functions
#--------------------------------------
def delete_elements_in_dir(dir_path):
    """Delete all files and folders in the given directory."""
    if not os.path.exists(dir_path):
        return
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            delete_elements_in_dir(file_path)
            os.rmdir(file_path)

#--------------------------------------
# Grid image splitting and denoising functions
#--------------------------------------
def split_grid_image(input_path, output_dir='output_images', rows=3, cols=3):
    """
    Splits a grid image into smaller images.
    
    Parameters:
    - input_path (str): Path to the input grid image.
    - output_dir (str): Directory to save the output cell images.
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    """
    img = Image.open(input_path)
    width, height = img.size
    cell_width = width // cols
    cell_height = height // rows

    os.makedirs(output_dir, exist_ok=True)
    count = 1
    for row in range(rows):
        for col in range(cols):
            left = col * cell_width
            upper = row * cell_height
            right = left + cell_width
            lower = upper + cell_height
            cropped_img = img.crop((left, upper, right, lower))
            cropped_img.save(os.path.join(output_dir, f'{count}.jpg'))
            count += 1

    print(f"Done! Saved {count-1} images to '{output_dir}'.")

def denoise(dir_path='upscaled_images', output_dir='denoised_images'):
    """
    Denoise images in the given directory using OpenCV's fastNlMeansDenoisingColored.
    """
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(dir_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Could not read {img_path}")
                continue
            # Two passes (as in your original code)
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)
            cv2.imwrite(os.path.join(output_dir, img_name), denoised)
            print(f"✅ Denoised {img_path}")

#--------------------------------------
# Ground truth extraction from filename
#--------------------------------------
def extract_ground_truth(filename):
    """
    Extract the ground truth grid cell indices from the filename.
    Expected pattern: someprefix_<cell1-cell2-...>_random.jpg
    For example, "image_1-2-3_xyz.jpg" -> [1, 2, 3]
    """
    m = re.search(r'_(\d+(?:-\d+)+)_', filename)
    if m:
        parts = m.group(1).split('-')
        try:
            return [int(p) for p in parts]
        except Exception as e:
            print(f"Error converting ground truth in {filename}: {e}")
            return []
    else:
        return []  # or return None if you prefer

#--------------------------------------
# Modified prediction function
#--------------------------------------
def predict_target_for_grid(image_path, target_class, model, transform, threshold, device, class_to_idx):
    """
    Process a grid image: split into cells, denoise and run prediction for each cell.
    For the target class, only keep cells with probability above the threshold. If more than
    4 cells exceed the threshold, keep the 4 with highest probabilities.
    
    Returns:
        A dictionary mapping grid cell index (int) -> predicted probability.
    """
    # Create temporary directories
    temp_split_dir = "split_images"
    temp_denoise_dir = "denoised_images"
    os.makedirs(temp_split_dir, exist_ok=True)
    os.makedirs(temp_denoise_dir, exist_ok=True)
    
    # Split the grid image (assumes a 3x3 grid)
    split_grid_image(image_path, output_dir=temp_split_dir, rows=3, cols=3)
    
    # Optionally, you might call an upscaler here via subprocess if needed.
    # e.g.: subprocess.run(["python", "upscaler.py"], check=True)
    
    # Denoise the split images
    denoise(dir_path=temp_split_dir, output_dir=temp_denoise_dir)
    
    predictions = {}
    for cell_file in os.listdir(temp_denoise_dir):
        if cell_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            cell_path = os.path.join(temp_denoise_dir, cell_file)
            try:
                image = Image.open(cell_path).convert('RGB')
            except Exception as e:
                print(f"Error opening {cell_path}: {e}")
                continue
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(image_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # shape: (num_classes,)
            target_idx = class_to_idx[target_class]
            prob = float(probs[target_idx])
            try:
                cell_index = int(os.path.splitext(cell_file)[0])
            except Exception as e:
                print(f"Error extracting cell index from {cell_file}: {e}")
                continue
            predictions[cell_index] = prob

    # Clean up temporary directories for this image (optional)
    delete_elements_in_dir(temp_split_dir)
    delete_elements_in_dir(temp_denoise_dir)
    
    # Filter cells above threshold
    selected = {k: v for k, v in predictions.items() if v >= threshold}
    # If more than 4 cells pass the threshold, take the 4 with the highest probabilities.
    if len(selected) > 4:
        sorted_selected = sorted(selected.items(), key=lambda x: x[1], reverse=True)[:4]
        selected = dict(sorted_selected)
    
    return selected

#--------------------------------------
# Model definition (ImprovedCNNModel)
#--------------------------------------
class ImprovedCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNNModel, self).__init__()
        # Block 1
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # output same spatial size
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves spatial dimensions
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
        # Global average pooling to reduce spatial dimensions to 1x1
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

#--------------------------------------
# Evaluation function
#--------------------------------------
import random

def evaluate_model(dataset_dir, class_to_idx, threshold=0.7, max_images=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(class_to_idx)
    model = ImprovedCNNModel(num_classes=num_classes)
    model_path = '/home/parin.arora_ug2023/CNN+/final_pipeline/finetuned_multiclass_model_with_pretrained_weights.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    val_img_size = 100
    transform = transforms.Compose([
        transforms.Resize((val_img_size, val_img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Collect all image paths and their class names
    image_list = []
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append((class_dir.lower(), os.path.join(class_path, file)))

    # Randomly select up to max_images
    selected_images = random.sample(image_list, min(max_images, len(image_list)))

    total_images = 0
    fully_correct = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for expected_class, image_path in selected_images:
        file = os.path.basename(image_path)
        total_images += 1
        gt = extract_ground_truth(file)
        if not gt:
            print(f"Could not extract ground truth from {file}")
            continue

        predictions = predict_target_for_grid(
            image_path, expected_class, model, transform, threshold, device, class_to_idx
        )
        predicted_indices = set(predictions.keys())
        gt_set = set(gt)
        tp = len(predicted_indices & gt_set)
        fp = len(predicted_indices - gt_set)
        fn = len(gt_set - predicted_indices)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if predicted_indices == gt_set:
            fully_correct += 1

        print(f"Image: {file}")
        print(f"  Ground truth: {gt_set}")
        print(f"  Predicted cells (with prob): { {k: round(v,3) for k,v in predictions.items()} }")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")

        # Clean up temporary directories before processing the next image
        delete_elements_in_dir('split_images')
        delete_elements_in_dir('denoised_images')

    # Compute metrics
    grid_image_accuracy = fully_correct / total_images if total_images > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\nEvaluation Results:")
    print(f"Total grid images processed: {total_images}")
    print(f"Fully solved grid images: {fully_correct}")
    print(f"Cell-level True Positives: {total_tp}")
    print(f"Cell-level False Positives: {total_fp}")
    print(f"Cell-level False Negatives: {total_fn}")
    print(f"Grid Image Accuracy: {grid_image_accuracy:.3f}")
    print(f"Cell-level Precision: {precision:.3f}")
    print(f"Cell-level Recall: {recall:.3f}")
    print(f"Cell-level F1 Score: {f1:.3f}")


#--------------------------------------
# Main entry point
#--------------------------------------
if __name__ == '__main__':
    dataset_dir = '/home/parin.arora_ug2023/CNN+/label2'
    class_to_idx = {
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
    evaluate_model(dataset_dir, class_to_idx, threshold=0.7)
