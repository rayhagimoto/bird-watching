# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# Bird Photo Anomaly Detection Pipeline
# 
# This notebook implements a convolutional autoencoder-based anomaly detection system
# specifically designed for finding bird photos in a dataset of background images.
# 
# Key Innovation: Uses localized reconstruction scoring + Otsu thresholding
# to achieve high precision with minimal training data (20% of dataset).
# %load_ext autoreload
# %autoreload 2

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import os
import yaml

# Define root path
ROOT_PATH = Path(os.getcwd()).resolve().parent.parent
print(f"Root path: {ROOT_PATH}")

# %% [markdown]
# # Bird Photo Anomaly Detection Pipeline
#
# ## Overview
# **Goal**: Find anomalous bird photos in a dataset so you can check them in Lightroom and edit if needed.
#
# **Method**: Train a convolutional autoencoder on normal background images, then flag photos that have high reconstruction error.
#
# **Key Innovation**: This pipeline is designed for **dataset-specific** anomaly detection, not generalization across different datasets. It's optimized for finding birds in a specific photo session.

# %% [markdown]
# ## Data Strategy: Cross-Set Training and Testing
#
# **New Approach**: Train on sets 0 and 2, test on set 3
# - This pipeline tests **generalization** across different shooting sessions
# - We train on background patterns from sets 0 and 2
# - We test on set 3 to see if the model can detect anomalies in a new session
# - This approach tests the model's ability to generalize beyond the training data
#
# **Data Split:**
# - **Training**: 30% random sample from sets 0 and 2 (learns background patterns)
# - **Validation**: 30% random sample from sets 0 and 2 (establishes baseline for Otsu thresholding)
# - **Test**: FULL SET 3 (tests generalization to new shooting session)

# %% [markdown]
# ## 1. Configuration and Model Setup
#
# Load experiment configuration and initialize the convolutional autoencoder.

# %%
# Simple Fully Connected Autoencoder with 2D Latent Space
import copy
from bird_detector.data.data_loaders import SimpleBirdDataset, MultiSetBirdDataset
from bird_detector.autoencoder import SimpleAutoencoder, ConvAutoencoder
import torch.nn as nn

# %%
# Load experiment configuration
with open('conv-ae-config.yaml') as f:
  config = yaml.safe_load(f)
model_config = config['model']
training_config = config['training']
data_config = config['data']
model_metadata = config['model_metadata']
model_config['image_size'] = data_config['image_size']

print("Configuration loaded:")
print(f"  Model: ConvAutoencoder with {model_config['latent_dim']}D latent space")
print(f"  Image size: {data_config['image_size']}x{data_config['image_size']}")
print(f"  Training: {data_config['train_frac']*100}% of data")
print(f"  Validation: {data_config['val_frac']*100}% of data")

# %% [markdown]
# ## 2. Data Loading and Splitting
#
# Create reproducible train/validation splits using a fixed seed.
# **NEW APPROACH**: Train on sets 0 and 2, test on set 3.
# The remaining data will be used for the final anomaly detection on the full dataset.

# %%
# Set fixed seeds for reproducible splits
SEED = config['seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGE_SIZE = data_config['image_size']

# Load training dataset (sets 0 and 2)
print("Loading training dataset (sets 0 and 2)...")
train_dataset_paths = [
    str(ROOT_PATH / "data" / "interim" / "jpg256_set0"),
    str(ROOT_PATH / "data" / "interim" / "jpg256_set2")
]

# Check if directories exist
for path in train_dataset_paths:
    if not Path(path).exists():
        print(f"❌ Error: Training dataset directory does not exist: {path}")
        raise FileNotFoundError(f"Training dataset directory not found: {path}")

train_full_dataset = MultiSetBirdDataset(train_dataset_paths, IMAGE_SIZE)
print(f"Training dataset size: {len(train_full_dataset)} images")

# Check if we have any images
if len(train_full_dataset) == 0:
    print("❌ Error: No images found in the training dataset directories!")
    raise ValueError("No images found in training dataset directories")

# Load test dataset (set 3)
print("Loading test dataset (set 3)...")
test_dataset_path = ROOT_PATH / "data" / "interim" / "jpg256_set3"
if not test_dataset_path.exists():
    print(f"❌ Error: Test dataset directory does not exist: {test_dataset_path}")
    raise FileNotFoundError(f"Test dataset directory not found: {test_dataset_path}")

test_full_dataset = SimpleBirdDataset(str(test_dataset_path), IMAGE_SIZE)
print(f"Test dataset size: {len(test_full_dataset)} images")

if len(test_full_dataset) == 0:
    print("❌ Error: No images found in the test dataset directory!")
    raise ValueError("No images found in test dataset directory")

# Create train/validation split from training data (sets 0 and 2)
total_train_size = len(train_full_dataset)
train_size = int(data_config['train_frac'] * total_train_size)  # 30% for training
val_size = int(data_config['val_frac'] * total_train_size)      # 30% for validation
remaining_size = total_train_size - train_size - val_size

# Create indices and shuffle them with fixed seed
train_indices = list(range(total_train_size))
random.shuffle(train_indices)

# Split indices for training data
train_indices_final = train_indices[:train_size]
val_indices = train_indices[train_size:train_size + val_size]
remaining_indices = train_indices[train_size + val_size:]

print(f"Training set: {len(train_indices_final)} images ({len(train_indices_final)/total_train_size*100:.1f}%)")
print(f"Validation set: {len(val_indices)} images ({len(val_indices)/total_train_size*100:.1f}%)")
print(f"Remaining training data: {len(remaining_indices)} images ({len(remaining_indices)/total_train_size*100:.1f}%)")
print(f"Test set (set 3): {len(test_full_dataset)} images")

# Create subsets
train_dataset = Subset(train_full_dataset, train_indices_final)
val_dataset = Subset(train_full_dataset, val_indices)
train_and_val_dataset = Subset(train_full_dataset, train_indices[:train_size + val_size])

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_full_dataset, batch_size=8, shuffle=False, num_workers=0)  # Test on set 3
train_and_val_dataloader = DataLoader(train_and_val_dataset, batch_size=8, shuffle=False, num_workers=0)

# %% [markdown]
# ## 3. Model Training
#
# Train the convolutional autoencoder to learn normal background patterns.
# The model will learn to compress and reconstruct background images efficiently.

# %%
from bird_detector.autoencoder import get_loss_function, get_optimizer

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ConvAutoencoder(**model_config).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

loss_fn = get_loss_function(training_config['loss_fn'])
optimizer = get_optimizer(training_config['optimizer'], model.parameters(), lr=training_config['lr'])

# %%
# Train the model with early stopping
from bird_detector.autoencoder import train_model

print("Starting model training...")
results = train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    training_config,
    metrics_to_track=['loss', 'mse'],
    device=device
)

# Extract training results
print("\nTraining completed!")
print("Available metrics:", "\n".join(results.keys()))

final_train_loss = results['final_train_loss']
best_train_loss = results['best_train_loss']
final_val_loss = results['final_val_loss']
best_val_loss = results['best_val_loss']
validation_losses = results['validation_loss_vals']
training_losses = results['training_loss_vals']
time_elapsed_sec = results['time_elapsed_sec']
training_mse_vals = results['training_mse_vals']
validation_mse_vals = results['validation_mse_vals']

print(f"\nTraining Summary:")
print(f"  Total Time: {int(time_elapsed_sec)} seconds")
print(f"  Epochs trained: {results.get('epochs_trained', 'N/A')}")
print(f"  Best validation loss: {best_val_loss:.6f}")
print(f"  Final validation loss: {final_val_loss:.6f}")

# %%
torch.save(model.state_dict(), './models/conv-ae-0/model.pth')
model_metadata = {
  "model": {
    "class" : "ConvAutoencoder",
    "latent_dim": 2,
  },
  "data": {
    "seed": config["seed"],
    "train_frac": 0.8,
    "val_frac": 0.2,
    "paths": ['data/interim/jpg256_set0', 'data/interim/jpg256_set2']
  },
  "train": {
    "lr": 5.0e-4,
    "loss_fn": "log_mse",
    "epochs": 50,
  }
}

with open ('./models/conv-ae-0/config.yaml', 'w') as f:
  yaml.dump(model_metadata, f)

# %% [markdown]
# ## 4. Training Progress Visualization
#
# Visualize how well the model is learning over time.

# %%
# %matplotlib inline
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Plot loss curves
ax[0].plot(training_losses, label='Training Loss', color='blue')
ax[0].plot(validation_losses, label='Validation Loss', color='red')
ax[0].set_title(f'Loss ({training_config["loss_fn"]})')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Plot MSE curves
ax[1].plot(training_mse_vals, label='Training MSE', color='blue')
ax[1].plot(validation_mse_vals, label='Validation MSE', color='red')
ax[1].set_title('Mean Squared Error')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('MSE')
ax[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Localized Anomaly Scoring
#
# **Key Innovation**: Instead of using global reconstruction error, we use a **localized scoring method**:
#
# 1. **Difference Map**: Compute pixel-wise difference between original and reconstructed images
# 2. **Gaussian Smoothing**: Apply Gaussian convolution to reduce noise
# 3. **Max Pooling**: Progressively downsample to 4x4, keeping the maximum value in each region
# 4. **Peak Detection**: Return the highest value as the anomaly score
#
# This method is more sensitive to localized anomalies (like birds) while being robust to global lighting variations.

# %%
import torch
import torch.nn as nn
import numpy as np
import math
from torchvision import transforms
from bird_detector.autoencoder.otsu_method import otsu_method
from bird_detector.autoencoder.scores import localized_reconstruction_score

print("Computing localized anomaly scores on train+validation dataset...")
print("This establishes the baseline distribution for Otsu thresholding.")

model.eval()  # Ensure model is in evaluation mode
anomaly_scores = []
val_filenames = []

with torch.no_grad():
    for batch_idx, (data, filenames) in enumerate(train_and_val_dataloader):
        data = data.to(device)
        output = model(data)
        diff_batch = output - data  # Shape: [N, C, H, W] (e.g., [16, 3, 64, 64])

        # Calculate localized anomaly scores
        batch_anomaly_scores = localized_reconstruction_score(
            diff_batch=diff_batch**2,
            original_image_size=data_config['image_size'],
            gaussian_kernel_size=10,
            gaussian_sigma=1.0,
            device=device
        )
        batch_anomaly_scores = torch.log(batch_anomaly_scores)
        anomaly_scores.extend(batch_anomaly_scores.cpu().numpy().tolist())
        val_filenames.extend(filenames)

        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx+1}/{len(train_and_val_dataloader)}")

anomaly_scores = np.array(anomaly_scores)

# Apply Otsu method to automatically determine optimal threshold
anomaly_threshold = otsu_method(anomaly_scores)

print(f"\nAnomaly Score Statistics (Train+Validation Set):")
print(f"  Mean: {anomaly_scores.mean():.6f}")
print(f"  Min:  {anomaly_scores.min():.6f}")
print(f"  Max:  {anomaly_scores.max():.6f}")
print(f"  Std:  {anomaly_scores.std():.6f}")
print(f"  Otsu Threshold: {anomaly_threshold:.6f}")

# %% [markdown]
# ## 5.5. Fine-tune Decoder on Set 3 Background Images
#
# **Transfer Learning Approach**: Use the first 200 images from set 3 to fine-tune the decoder
# while keeping the encoder frozen. This helps the model adapt to the new camera angle, 
# framing, and lighting conditions in set 3.
#
# **Rationale**: Sets 0 and 2 have similar conditions (same bird bath, tripod camera), 
# but set 3 has different angles/framing. Fine-tuning the decoder should help it 
# reconstruct the new background patterns better.

# %%
model.load_state_dict(torch.load('./models/conv-ae-0/weights.pth'))

# %% [markdown]
# ## 5.5. Initial Adaptation on First 200 Images
#
# Load the first 200 images from set 3 and train the model for 50 steps to establish
# initial adaptation to the new shooting conditions.

# %%
# %% [markdown]
# ## 5.5. Initial Adaptation on First 200 Images
#
# Load the first 200 images from set 3 and train the model for 50 steps to establish
# initial adaptation to the new shooting conditions.

# %%
import torch
import torch.nn as nn
import numpy as np
import math
from torchvision import transforms
from bird_detector.autoencoder.otsu_method import otsu_method
from bird_detector.autoencoder.scores import localized_reconstruction_score
from tqdm.notebook import tqdm
from collections import deque
import torch.optim as optim
from torch.optim import lr_scheduler

print("Initial adaptation on first 200 images from set 3...")

# Initialize rolling window storage
window_size = 200
image_window = deque(maxlen=window_size)
filename_window = deque(maxlen=window_size)

# Initialize storage for dynamic thresholding
all_anomaly_scores = []
dynamic_thresholds = []

# Initialize optimizer for progressive adaptation
progressive_optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.LinearLR(progressive_optimizer, start_factor=1, end_factor=5e-3, total_iters=200)

# Progressive adaptation configuration
adaptation_config = {
    'initial_steps': 5,     # Steps on first 200 images
    'steps_per_image': 2,    # Steps per new image
    'batch_size': 16,        # Batch size for training
    'freeze_encoder': True,  # Only update decoder for stability
}

# Function to calculate dynamic threshold
def calculate_dynamic_threshold(anomaly_scores):
    """Calculate dynamic threshold using 98th percentile or Otsu method, whichever is larger."""
    if len(anomaly_scores) < 10:  # Need minimum samples for meaningful threshold
        return anomaly_threshold  # Use original threshold if not enough data
    
    scores_array = np.array(anomaly_scores)
    percentile_98 = np.percentile(scores_array, 98)
    otsu_threshold = otsu_method(scores_array)
    
    dynamic_threshold = max(percentile_98, otsu_threshold)
    return dynamic_threshold

# Function to train on current window
def train_on_window(num_steps, description=""):
    """Train the model on the current rolling window for specified number of steps."""
    if len(image_window) < adaptation_config['batch_size']:
        return 0.0
    
    # Create dataset from current window
    window_tensors = torch.stack(list(image_window)).detach().requires_grad_(True)
    window_dataset = torch.utils.data.TensorDataset(window_tensors, window_tensors)
    window_dataloader = torch.utils.data.DataLoader(
        window_dataset, 
        batch_size=adaptation_config['batch_size'], 
        shuffle=True
    )
    
    # Freeze encoder if specified
    if adaptation_config['freeze_encoder']:
        for param in model.encoder_conv.parameters():
            param.requires_grad = False
        for param in model.fc_enc.parameters():
            param.requires_grad = False
        for param in model.fc_dec.parameters():
            param.requires_grad = True
        for param in model.decoder_conv.parameters():
            param.requires_grad = True
    
    model.train()
    total_loss = 0.0
    steps_completed = 0
    
    # Train for specified number of steps
    dataloader_iter = iter(window_dataloader)
    for step in range(num_steps):
        try:
            batch_data, _ = next(dataloader_iter)
        except StopIteration:
            # Restart iterator if we run out of batches
            dataloader_iter = iter(window_dataloader)
            batch_data, _ = next(dataloader_iter)
        
        batch_data = batch_data.to(device)
        
        progressive_optimizer.zero_grad()
        output = model(batch_data)
        loss = loss_fn(output, batch_data)
        loss.backward()
        progressive_optimizer.step()
        
        total_loss += loss.item()
        steps_completed += 1
        scheduler.step()
    
    avg_loss = total_loss / steps_completed if steps_completed > 0 else 0.0
    print(f"{description} - Steps: {steps_completed}, Avg Loss: {avg_loss:.6f}")
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    model.eval()
    return avg_loss

# Load first 200 images and train
print("Loading first 200 images and initial training...")
first_200_loaded = 0

for batch_idx, (imgs, filenames) in enumerate(test_dataloader):
    if first_200_loaded >= window_size:
        break
    
    imgs = imgs.to(device)
    
    # Add images to window
    for img, filename in zip(imgs, filenames):
        if first_200_loaded >= window_size:
            break
        image_window.append(img.detach().cpu())
        filename_window.append(filename)
        first_200_loaded += 1

print(f"Loaded {len(image_window)} images into initial window")

# Initial training on first 200 images
initial_loss = train_on_window(
    adaptation_config['initial_steps'], 
    f"Initial training on {len(image_window)} images"
)

print(f"Initial adaptation completed with loss: {initial_loss:.6f}")

# %% [markdown]
# ## 5.6. Progressive Adaptation with Dynamic Thresholding
#
# Process the remaining images in set 3 one by one, maintaining a rolling window of 200 images
# and using dynamic thresholding based on the 98th percentile or Otsu method.

# %%
# Initialize storage for results
anomalous_filenames = []
anomalous_images = []
adaptation_losses = [initial_loss]  # Start with initial loss
processed_images = 0

# Create a new dataloader for the remaining images
remaining_indices = list(range(window_size, len(test_full_dataset)))
remaining_dataset = Subset(test_full_dataset, remaining_indices)
remaining_dataloader = DataLoader(remaining_dataset, batch_size=1, shuffle=False, num_workers=0)

print("Starting progressive adaptation with dynamic thresholding...")

for batch_idx, (img, filename) in tqdm(enumerate(remaining_dataloader), desc="Progressive adaptation"):
    img = img.squeeze(0).to(device)  # Remove batch dimension
    
    # Add new image to window (oldest will be automatically dropped)
    image_window.append(img.detach().cpu())
    filename_window.append(filename)
    
    # Train for 5 steps on current window
    step_loss = train_on_window(
        adaptation_config['steps_per_image'], 
        f"Image {batch_idx + window_size + 1}/{len(test_full_dataset)}"
    )
    adaptation_losses.append(step_loss)
    
    # Perform anomaly detection on this image
    model.eval()
    with torch.no_grad():
        output = model(img.unsqueeze(0))  # Add batch dimension back
        diff = output - img.unsqueeze(0)
        
        # Calculate localized anomaly score
        anomaly_score = localized_reconstruction_score(
            diff_batch=diff**2,
            original_image_size=data_config['image_size'],
            gaussian_kernel_size=10,
            gaussian_sigma=1.0,
            device=device
        )
        anomaly_score = torch.log(anomaly_score)
        
        # Store the anomaly score
        all_anomaly_scores.append(anomaly_score.item())
        
        # Calculate dynamic threshold
        current_threshold = calculate_dynamic_threshold(all_anomaly_scores)
        dynamic_thresholds.append(current_threshold)
        
        
        # Flag if anomalous using dynamic threshold
        if anomaly_score > current_threshold:
            print("ANOMALY DETECTED with {anomaly_score} > {current_threshold}")
            anomalous_filenames.append(filename)
            anomalous_images.append(transforms.ToPILImage()(img.cpu()))
    
    processed_images += 1
    
    # Print progress every 100 images
    if (batch_idx + 1) % 500 == 0:
        current_anomaly_rate = len(anomalous_filenames) / (batch_idx + 1) * 100
        print(f"Processed {batch_idx + 1} images, Anomaly rate: {current_anomaly_rate:.2f}%, "
              f"Dynamic threshold: {current_threshold:.6f}")

print(f"\nProgressive Adaptation Results:")
print(f"  Total images processed: {len(test_full_dataset)}")
print(f"  Anomalies found: {len(anomalous_filenames)}")
print(f"  Anomaly rate: {len(anomalous_filenames)/len(test_full_dataset)*100:.2f}%")
print(f"  Adaptation steps completed: {len(adaptation_losses)}")
print(f"  Final dynamic threshold: {dynamic_thresholds[-1]:.6f}")
print(f"  Original Otsu threshold: {anomaly_threshold:.6f}")

# %%

# %%
from pathlib import Path

# Load known anomalies (bird photos) from set 3
known_anomalies = [x.stem for x in Path("D:/BIRDS/set_3/RAW/ANOMALIES").glob("*.ARW")]
print(f"Known anomalies (bird photos): {len(known_anomalies)}")

# Clean up anomalous_filenames to match format
_anomalous_filenames = [x[0].split(".")[0] for x in anomalous_filenames]
print(f"Predicted anomalies: {len(_anomalous_filenames)}")

def calculate_metrics(predicted_anomalies, true_anomalies, total_images):
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        predicted_anomalies: List of filenames predicted as anomalies (without extension)
        true_anomalies: List of filenames that are actually anomalies (without extension)
        total_images: Total number of images in the dataset
    """
    # Convert to sets for easier comparison
    pred_set = set(predicted_anomalies)
    true_set = set(true_anomalies)
    
    # Filter true anomalies to only include those after image 200 (since you didn't do inference on first 200)
    true_set_filtered = {name for name in true_set if int(name) >= 200}
    
    # Calculate basic metrics
    true_positives = len(pred_set.intersection(true_set_filtered))
    false_positives = len(pred_set - true_set_filtered)  # Predicted but not true
    false_negatives = len(true_set_filtered - pred_set)  # True but not predicted
    true_negatives = (total_images - 200) - true_positives - false_positives  # Total after 200 minus anomalies
    
    # Calculate rates
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate rates
    fp_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    fn_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'total_analyzed': total_images - 200
    }

# Calculate metrics
metrics = calculate_metrics(_anomalous_filenames, known_anomalies, len(test_full_dataset))

print(f"\n=== Anomaly Detection Performance ===")
print(f"Total images in set 3: {len(test_full_dataset)}")
print(f"Images analyzed (after first 200): {metrics['total_analyzed']}")
print(f"Known anomalies (after image 200): {len([x for x in known_anomalies if int(x) >= 200])}")
print(f"Predicted anomalies: {len(_anomalous_filenames)}")
print(f"\nTrue Positives: {metrics['true_positives']}")
print(f"False Positives: {metrics['false_positives']}")
print(f"True Negatives: {metrics['true_negatives']}")
print(f"False Negatives: {metrics['false_negatives']}")
print(f"\nPrecision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
print(f"Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print(f"False Positive Rate: {metrics['fp_rate']:.4f} ({metrics['fp_rate']*100:.2f}%)")
print(f"False Negative Rate: {metrics['fn_rate']:.4f} ({metrics['fn_rate']*100:.2f}%)")

# Show some examples
print(f"\n=== Examples ===")
pred_set = set(_anomalous_filenames)
true_set = set(known_anomalies)
true_set_filtered = {name for name in true_set if int(name) >= 200}

true_positives_examples = list(pred_set.intersection(true_set_filtered))[:5]
false_positives_examples = list(pred_set - true_set_filtered)[:5]
false_negatives_examples = list(true_set_filtered - pred_set)[:5]

print(f"True Positives (correctly identified birds): {true_positives_examples}")
print(f"False Positives (background flagged as bird): {false_positives_examples}")
print(f"False Negatives (missed birds): {false_negatives_examples}")

# %%

# %% [markdown]
# ## 5.7. Dynamic Thresholding Analysis and Visualization
#
# Analyze the dynamic thresholding behavior and plot the results.

# %%
# Plot adaptation progress
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Adaptation loss
ax1.plot(adaptation_losses, label='Adaptation Loss', color='purple', marker='o', markersize=2)
ax1.set_title('Progressive Model Adaptation Progress')
ax1.set_xlabel('Adaptation Step')
ax1.set_ylabel('Average Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Dynamic thresholding
ax2.plot(dynamic_thresholds, label='Dynamic Threshold', color='red', linewidth=2)
ax2.axhline(y=anomaly_threshold, color='blue', linestyle='--', label='Original Otsu Threshold')
ax2.set_title('Dynamic Threshold Evolution')
ax2.set_xlabel('Image Number')
ax2.set_ylabel('Threshold Value')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Show adaptation trend
if len(adaptation_losses) > 1:
    initial_avg = np.mean(adaptation_losses[:10])  # First 10 steps
    final_avg = np.mean(adaptation_losses[-10:])   # Last 10 steps
    improvement = (initial_avg - final_avg) / initial_avg * 100
    print(f"Adaptation improvement: {improvement:.2f}%")
    print(f"Initial average loss: {initial_avg:.6f}")
    print(f"Final average loss: {final_avg:.6f}")

# Analyze dynamic thresholding
print(f"\nDynamic Thresholding Analysis:")
print(f"  Initial dynamic threshold: {dynamic_thresholds[0]:.6f}")
print(f"  Final dynamic threshold: {dynamic_thresholds[-1]:.6f}")
print(f"  Threshold change: {dynamic_thresholds[-1] - dynamic_thresholds[0]:.6f}")
print(f"  Original Otsu threshold: {anomaly_threshold:.6f}")

# Analyze anomaly score distribution
anomaly_scores_array = np.array(all_anomaly_scores)
print(f"\nAnomaly Score Statistics:")
print(f"  Mean: {anomaly_scores_array.mean():.6f}")
print(f"  Std: {anomaly_scores_array.std():.6f}")
print(f"  Min: {anomaly_scores_array.min():.6f}")
print(f"  Max: {anomaly_scores_array.max():.6f}")
print(f"  98th percentile: {np.percentile(anomaly_scores_array, 98):.6f}")

# Plot anomaly score distribution
plt.figure(figsize=(10, 4))
plt.hist(anomaly_scores_array, bins=50, alpha=0.7, color='green', edgecolor='black')
plt.axvline(x=anomaly_threshold, color='blue', linestyle='--', label='Original Otsu Threshold')
plt.axvline(x=dynamic_thresholds[-1], color='red', linestyle='--', label='Final Dynamic Threshold')
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Additional analysis
print(f"\nDetailed Results:")
print(f"  Initial window training steps: {adaptation_config['initial_steps']}")
print(f"  Steps per new image: {adaptation_config['steps_per_image']}")
print(f"  Rolling window size: {window_size}")
print(f"  Total adaptation steps: {len(adaptation_losses)}")
print(f"  Images with anomalies: {len(anomalous_filenames)}")

# %%

# %% [markdown]
# ## 7. Visualize Detected Anomalies
#
# Display all the anomalous images that were flagged by our system.
# These should be the bird photos in your dataset.

# %%
# Plot all anomalous images
_anomalous_images = [Image.open(ROOT_PATH / "data" / "interim" / "jpg256_set3" / filename[0]) for filename in anomalous_filenames]

if len(_anomalous_images) > 0:
    print(f"Visualizing {len(_anomalous_images)} anomalous images...")
    
    # Calculate grid dimensions
    n_images = len(_anomalous_images)
    cols = min(8, n_images)  # Max 8 columns
    rows = (n_images + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), dpi=75)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (image, filename) in enumerate(zip(_anomalous_images, anomalous_filenames)):
        row = i // cols
        col = i % cols
        
        img_display = image
        
        axes[row, col].imshow(img_display)
        axes[row, col].set_title(f'{filename}', fontsize=8)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed {len(_anomalous_images)} anomalous images")
    print("These should be the bird photos in set 3!")
else:
    print("No anomalies to display!")

# %% [markdown]
# ## 8. Latent Space Analysis
#
# Visualize how the model organizes images in its 2D latent space.
# This helps understand how the autoencoder separates normal background images from anomalies.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm

# Set model to evaluation mode
model.eval()

# Initialize lists to store latent space representations
normal_latent_vecs = []
anomaly_latent_vecs = []
anomaly_imgs = []
anomaly_scores = []

print("Mapping images to latent space and classifying by anomaly score...")

with torch.no_grad():
    for images, _ in tqdm(test_dataloader, desc="Processing batches"):
        images = images.to(device)

        # Encode images to latent space
        encoded_batch = model.encoder(images).cpu().numpy()

        # Decode images for reconstruction error calculation
        reconstructions = model(images)

        # Calculate the reconstruction difference
        diff_batch = reconstructions - images

        # Calculate anomaly scores for the batch
        scores = localized_reconstruction_score(
            diff_batch=diff_batch**2,
            original_image_size=data_config['image_size'],
            gaussian_kernel_size=10,
            gaussian_sigma=1.0,
            device=device
        )
        scores = torch.log(scores).cpu().numpy()
        anomaly_scores.extend(list(scores))

        # Categorize latent vectors based on anomaly score threshold
        for i, (score, encoded_vec) in enumerate(zip(scores, encoded_batch)):
            if score <= anomaly_threshold:
                normal_latent_vecs.append(encoded_vec)
            else:
                anomaly_latent_vecs.append(encoded_vec)
                anomaly_imgs.append(images[i].permute(1,2,0).cpu().numpy())

anomaly_scores = np.array(anomaly_scores)

# Convert lists to NumPy arrays
normal_latent_vecs = np.array(normal_latent_vecs)
anomaly_latent_vecs = np.array(anomaly_latent_vecs)
latent_space_vals = np.concatenate([anomaly_latent_vecs, normal_latent_vecs])

print(f"Normal samples in latent space: {normal_latent_vecs.shape[0]}")
print(f"Anomaly samples in latent space: {anomaly_latent_vecs.shape[0]}")

# %%
anomaly_latent_vecs


# %%
# %matplotlib inline

def pil_to_tensor(img):
  return transforms.ToTensor()(img).to(device).unsqueeze(0)

def tensor_to_pil(img_tensor):
  return img_tensor.squeeze().permute(1,2,0).cpu().detach().numpy()

tensor_img = pil_to_tensor(anomaly_imgs[11])

fig, ax = plt.subplots(1, 3)
ax[0].imshow(anomaly_imgs[11])
ax[1].imshow(tensor_to_pil(model(tensor_img)))
ax[2].imshow(tensor_to_pil(model(tensor_img) - tensor_img))
plt.show()

# %% [markdown]
# ## 9. Latent Space Visualization
#
# Plot the distribution of images in the 2D latent space, colored by anomaly status.

# %%
# Plot Latent Space Distribution
fig, ax = plt.subplots(figsize=(8, 6))

if model_config['latent_dim'] == 1:
    # Flatten latent vectors for 1D histogram
    normal_vals = normal_latent_vecs.flatten()
    anomaly_vals = anomaly_latent_vecs.flatten()

    if normal_vals.size > 0:
        ax.hist(normal_vals, bins=50, alpha=0.7, color='blue', label='Normal Samples')
    if anomaly_vals.size > 0:
        ax.hist(anomaly_vals, bins=50, alpha=0.7, color='red', label='Anomaly Samples')

    ax.set_xlabel('Latent Space Value')
    ax.set_ylabel('Count')
    ax.set_title('1D Latent Space Distribution by Anomaly Score')
    ax.legend()

elif model_config['latent_dim'] == 2:
    # Reshape latent vectors for 2D scatter plot
    normal_2d_vecs = normal_latent_vecs.reshape(-1, 2)
    anomaly_2d_vecs = anomaly_latent_vecs.reshape(-1, 2)

    # Plot only if arrays are not empty
    if normal_2d_vecs.shape[0] > 0:
        ax.scatter(normal_2d_vecs[:, 0], normal_2d_vecs[:, 1], s=5, alpha=0.1, c='blue', label='Normal Samples')
    if anomaly_2d_vecs.shape[0] > 0:
        ax.scatter(anomaly_2d_vecs[:, 0], anomaly_2d_vecs[:, 1], s=50, alpha=0.8, c='red', label='Anomaly Samples')

    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('2D Latent Space Distribution by Anomaly Score')
    ax.legend()

else:
    print(f"Warning: Latent space dimension {model_config['latent_dim']} not supported for direct visualization.")
    print("This script currently supports 1D or 2D latent space visualizations.")

plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

print("\nLatent space visualization complete.")

# %% [markdown]
# ## 10. Interactive Latent Space Exploration
#
# Interactive tools to explore the latent space and understand how the model organizes different types of images.

# %%
# %matplotlib widget
# %autoreload 2
from bird_detector.autoencoder.interactive_utils import interactive_1d_latent_space, interactive_2d_latent_space

# Interactive exploration
if model_config['latent_dim'] == 1:
    z_min = float(np.min(latent_space_vals))
    z_max = float(np.max(latent_space_vals))
    z_initial = 0.5 * (z_min + z_max)
    interactive_1d_latent_space(model, model_config, z_min=z_min, z_max=z_max, z_initial=z_initial)
elif model_config['latent_dim'] == 2:
    latent_space_vals = anomaly_latent_vecs
    model_config['input_size'] = data_config['image_size']
    x_min, x_max = float(np.min(latent_space_vals[:, 0])), float(np.max(latent_space_vals[:, 0]))
    y_min, y_max = float(np.min(latent_space_vals[:, 1])), float(np.max(latent_space_vals[:, 1]))
    boundaries = [x_min, x_max, y_min, y_max]
    interactive_2d_latent_space(model, model_config, boundaries, latent_space_vals=latent_space_vals, hist_kwargs={'cmap': 'magma', 'bins': 80})
    fig = plt.gcf()
    ax = fig.axes[1]
    ax.vlines([x_min, x_max], ymin=y_min, ymax=y_max, colors='r')
    ax.hlines([y_min, y_max], xmin=x_min, xmax=x_max, colors='r')

# %% [markdown]
# ## 11. Experiment Logging with MLflow
#
# Log the experiment results, model, and key findings for reproducibility.

# %%
# -- IMPORTANT --
# Summary of key findings from this cross-set experiment
RUN_NAME = 'conv-ae-cross-set'

NOTE = r"""
Key Findings from Cross-Set ConvAE Anomaly Detection Experiment:

1. **Cross-Set Generalization Test**
   - Trained on sets 0 and 2, tested on set 3
   - This tests the model's ability to generalize across different shooting sessions
   - Results show whether the model can detect anomalies in new, unseen data

2. **Localized Scoring + Otsu Thresholding for Cross-Set Detection**
   - Using localized reconstruction scoring instead of global MSE for better generalization
   - Otsu method automatically finds optimal threshold from training data
   - Tests whether the threshold generalizes to new shooting conditions

3. **Training Strategy**
   - 30% of sets 0 and 2 used for training (learns background patterns)
   - 30% of sets 0 and 2 used for validation (establishes Otsu threshold baseline)
   - Full set 3 testing validates cross-set generalization

4. **Cross-Set Performance Analysis**
   - This experiment tests if the model can generalize beyond the training data
   - Results indicate whether the approach works across different shooting sessions
   - Important for practical deployment across multiple photo sessions

This approach tests the robustness of the anomaly detection system across different
shooting conditions and datasets.
"""

# %%
# import mlflow
# from bird_detector.mlflow_utils import get_input_example, get_output_example
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("conv-ae-cross-set")

# with mlflow.start_run(run_name=RUN_NAME) as run:
#     # Log configuration
#     mlflow.log_dict(config, 'config.json')
    
#     # Log key metrics
#     mlflow.log_metric(training_config['loss_fn'], results['final_val_loss'])
#     mlflow.log_metric('mse', validation_mse_vals[-1])
#     mlflow.log_metric('anomalies_found', len(anomalous_filenames))
#     mlflow.log_metric('anomaly_rate_percent', len(anomalous_filenames)/len(test_full_dataset)*100)

#     # Create input and output examples for model signature
#     device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')

#     # Get a sample from the dataloader and immediately convert to numpy
#     sample_batch = next(iter(train_dataloader))
#     if isinstance(sample_batch, (list, tuple)):
#         sample_images, _ = sample_batch
#     else:
#         sample_images = sample_batch

#     # Take just the first image and convert to numpy immediately
#     input_example = sample_images[0:1].cpu().numpy()  # Shape: (1, 3, H, W)

#     # Generate output example
#     with torch.no_grad():
#         output_example = model(sample_images[0:1].to(device)).cpu().numpy()

#     signature = mlflow.models.infer_signature(input_example, output_example)
#     mlflow.pytorch.log_model(model, name='autoencoder_model', signature=signature, input_example=input_example)

#     # Store the run_id for later reference
#     run_id = run.info.run_id

#     # Log the model parameters
#     mlflow.log_params(model_config)
    
#     # Log data configuration
#     mlflow.log_params({
#         'train_sets': 'set0,set2',
#         'test_set': 'set3',
#         'train_data_paths': str(train_dataset_paths),
#         'test_data_path': str(test_dataset_path)
#     })
#     _data_config = {k: v for k,v in data_config.items() if k != 'path'}
#     mlflow.log_params(_data_config)

#     # Set experiment tags
#     mlflow.set_tags({
#         "architecture": "convolutional",
#         "latent_dim": model_config['latent_dim'],
#         "latent_norm": model_config.get('softmax', 'none'),
#         "activation": "relu",
#         "experiment_group": "cross_set_generalization",
#         "scoring_method": "localized_reconstruction",
#         "threshold_method": "otsu",
#         "training_data_fraction": data_config['train_frac'],
#         "train_sets": "set0,set2",
#         "test_set": "set3"
#     })

#     if NOTE:
#         mlflow.set_tags({"mlflow.note.content": NOTE})

# print(f"Experiment logged with run_id: {run_id}")

# %%
