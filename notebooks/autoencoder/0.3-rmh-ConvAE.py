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

import sys
sys.path.insert(0, str(ROOT_PATH))

# %% [markdown]
# # Bird Photo Anomaly Detection Pipeline
#
# ## Overview
# **Goal**: Find anomalous bird photos in a dataset so I can check them in Lightroom and edit if needed.
#
# **Method**: Train a convolutional autoencoder on a random subject of images, then flag photos that have high reconstruction error. This subset might contain pictures of birds but they are so few and far between that I will assume that it won't affect what the model will learn.
#
# **Key Innovation**: This pipeline is designed for **dataset-specific** anomaly detection, not generalization across different datasets. It's optimized for finding birds in a specific photo session.

# %% [markdown]
# ## Data Strategy: Minimal Training Data Approach
#
# **Why only 20% training data?**
# - This pipeline is designed for **post-shooting analysis** of a specific dataset
# - We don't need to generalize to other datasets - just find birds in this one
# - 20% of normal background images is sufficient to learn the background patterns
# - This approach reduces training time and computational requirements
#
# **Data Split:**
# - **Training**: 20% random sample (learns normal background patterns)
# - **Validation**: 30% random sample (establishes baseline for Otsu thresholding)
# - **Field Test**: FULL DATASET (because this autoencoder is only supposed to work on this dataset)

# %% [markdown]
# ## 1. Configuration and Model Setup
#
# Load experiment configuration and initialize the convolutional autoencoder.

# %%
# Simple Fully Connected Autoencoder with 2D Latent Space
import copy
from bird_detector.data.data_loaders import SimpleBirdDataset
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
# The remaining data will be used for the final anomaly detection on the full dataset.

# %%
# Set fixed seeds for reproducible splits
SEED = config['seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGE_SIZE = data_config['image_size']

# Load full dataset
print("Loading dataset...")
dataset_path = ROOT_PATH / data_config['path']
print(f"Dataset path: {dataset_path}")

# Check if directory exists
if not dataset_path.exists():
    print(f"❌ Error: Dataset directory does not exist: {dataset_path}")
    print("Please check the path and ensure the data directory exists.")
    raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

full_dataset = SimpleBirdDataset(str(dataset_path), IMAGE_SIZE)
print(f"Full dataset size: {len(full_dataset)} images")

# Check if we have any images
if len(full_dataset) == 0:
    print("❌ Error: No images found in the dataset directory!")
    print(f"Please check that {dataset_path} contains .jpg files.")
    raise ValueError("No images found in dataset directory")

# Create three-way split: 20% train, 30% validation
total_size = len(full_dataset)
train_size = int(data_config['train_frac'] * total_size)  # 20% for training
val_size = int(data_config['val_frac'] * total_size)      # 30% for validation
test_size = total_size - train_size - val_size

# Create indices and shuffle them with fixed seed
indices = list(range(total_size))
random.shuffle(indices)

# Split indices
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

print(f"Training set: {len(train_indices)} images ({len(train_indices)/total_size*100:.1f}%)")
print(f"Validation set: {len(val_indices)} images ({len(val_indices)/total_size*100:.1f}%)")
print(f"Remaining: {len(test_indices)} images ({len(test_indices)/total_size*100:.1f}%)")

# Create subsets
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)
train_and_val_dataset = Subset(full_dataset, indices[:train_size + val_size])

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
full_dataloader = DataLoader(full_dataset, batch_size=8, shuffle=False, num_workers=0)  # Batch size 8 for field test
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)  # Batch size 8 for field test
train_and_val_dataloader = DataLoader(train_and_val_dataset, batch_size=8, shuffle=False, num_workers=0)  # Batch size 8 for field test

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
import numpy as np
from bird_detector.autoencoder.otsu_method import otsu_method
from bird_detector.autoencoder.scores import localized_reconstruction_score
from tqdm.notebook import tqdm

print("Computing localized anomaly scores on train+validation dataset...")
print("This establishes the baseline distribution for Otsu thresholding.")

model.eval()  # Ensure model is in evaluation mode
anomaly_scores = []
val_filenames = []

with torch.no_grad():
    for batch_idx, (data, filenames) in tqdm(enumerate(train_and_val_dataloader)):
        data = data.to(device)
        output = model(data)
        diff_batch = output - data  # Shape: [N, C, H, W] (e.g., [16, 3, 64, 64])

        # Calculate localized anomaly scores
        batch_anomaly_scores = localized_reconstruction_score(
            diff_batch=torch.abs(diff_batch),
            original_image_size=data_config['image_size'],
            gaussian_kernel_size=10,
            gaussian_sigma=1.0,
            device=device
        )
        batch_anomaly_scores = torch.log(batch_anomaly_scores)
        anomaly_scores.extend(batch_anomaly_scores.cpu().numpy().tolist())
        val_filenames.extend(filenames)

anomaly_scores = np.array(anomaly_scores)

# Apply Otsu method to automatically determine optimal threshold
anomaly_threshold = otsu_method(anomaly_scores)

print(f"\nAnomaly Score Statistics (Train+Validation Set):")
print(f"  Mean: {anomaly_scores.mean():.6f}")
print(f"  Min:  {anomaly_scores.min():.6f}")
print(f"  Max:  {anomaly_scores.max():.6f}")
print(f"  Std:  {anomaly_scores.std():.6f}")
print(f"  Otsu Threshold: {anomaly_threshold:.6f}")

plt.hist(anomaly_scores, 20)
plt.title('Anomaly Scores')

# %% [markdown]
# ## 6. Full Dataset Anomaly Detection
#
# Now apply the trained model and Otsu threshold to the **entire dataset** to find bird photos.
# This is the key step - we're testing on the full dataset, not just a test split.

# %%
import torch
import torch.nn as nn
import numpy as np
import math
from torchvision import transforms
from bird_detector.autoencoder.otsu_method import otsu_method
from bird_detector.autoencoder.scores import localized_reconstruction_score
from tqdm.notebook import tqdm

print("Finding anomalies in the FULL dataset...")
print("This is the field test - applying our model to all images.")

model.eval()  # Ensure model is in evaluation mode
anomalous_filenames = []
anomalous_images = []

with torch.no_grad():
    for batch_idx, (imgs, filenames) in tqdm(enumerate(full_dataloader), desc="Processing full dataset"):
        imgs = imgs.to(device)
        output = model(imgs)
        diff_batch = output - imgs  # Shape: [N, C, H, W] (e.g., [16, 3, 64, 64])

        # Calculate localized anomaly scores
        batch_anomaly_scores = localized_reconstruction_score(
            diff_batch=torch.abs(diff_batch),
            original_image_size=data_config['image_size'],
            gaussian_kernel_size=10,
            gaussian_sigma=1.0,
            device=device
        )
        batch_anomaly_scores = torch.log(batch_anomaly_scores)

        # Flag images above the Otsu threshold
        for (img, score, filename) in zip(imgs, batch_anomaly_scores, filenames):
            if score > anomaly_threshold:
                anomalous_filenames.append(filename)
                anomalous_images.append(transforms.ToPILImage()(img.cpu()))

print(f"\nAnomaly Detection Results:")
print(f"  Total images processed: {len(full_dataset)}")
print(f"  Anomalies found: {len(anomalous_filenames)}")
print(f"  Anomaly rate: {len(anomalous_filenames)/len(full_dataset)*100:.2f}%")

# %% [markdown]
# ## 7. Visualize Detected Anomalies
#
# Display all the anomalous images that were flagged by our system.
# These should be the bird photos in your dataset.

# %%
# Plot all anomalous images
_anomalous_images = [Image.open(ROOT_PATH / data_config['path'] / filename) for filename in anomalous_filenames]

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
    print("These should be the bird photos in your dataset!")
else:
    print("No anomalies to display!")

# %%
from pathlib import Path
img_path = 'D:/BIRDS/set_3/JPG/0009.jpg'
with open(img_path, 'rb') as f:
  img_pil = Image.open(f).convert('RGB').resize((64,64))
  img_np = np.asarray(img_pil).astype(np.float32) / 255.0
img_path = 'D:/BIRDS/set_3/JPG/0010.jpg'
with open(img_path, 'rb') as f:
  img_pil = Image.open(f).convert('RGB').resize((64,64))
  img_bg_true = np.asarray(img_pil).astype(np.float32) / 255.0

with torch.no_grad():
  img_tensor = transforms.ToTensor()(img_np).unsqueeze(0).to(device)
  img_bg = model(img_tensor).squeeze(0).permute(1,2,0).cpu().numpy()

fig, ax = plt.subplots(1,3, figsize=(10,3))
ax[0].imshow(img_np)
ax[1].imshow(img_bg)
ax[2].imshow(img_bg_true)
ax[0].set_title("Original Image")
ax[1].set_title("Pure Background (Autoencoder)")
ax[2].set_title("Pure Background (True)")

output_path = Path('C:/Users/ray/Desktop/bird_blog/bird_removal_example.svg')
plt.savefig(output_path, bbox_inches='tight')

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

print("Mapping images to latent space and classifying by anomaly score...")

with torch.no_grad():
    for images, _ in tqdm(full_dataloader, desc="Processing batches"):
        images = images.to(device)

        # Encode images to latent space
        encoded_batch = model.encoder(images).cpu().numpy()

        # Decode images for reconstruction error calculation
        reconstructions = model(images)

        # Calculate the reconstruction difference
        diff_batch = reconstructions - images

        # Calculate anomaly scores for the batch
        scores = localized_reconstruction_score(
            diff_batch=torch.abs(diff_batch),
            original_image_size=data_config['image_size'],
            gaussian_kernel_size=10,
            gaussian_sigma=1.0,
            device=device
        )
        scores = torch.log(scores).cpu().numpy()

        # Categorize latent vectors based on anomaly score threshold
        for score, encoded_vec in zip(scores, encoded_batch):
            if score <= anomaly_threshold:
                normal_latent_vecs.append(encoded_vec)
            else:
                anomaly_latent_vecs.append(encoded_vec)

# Convert lists to NumPy arrays
normal_latent_vecs = np.array(normal_latent_vecs)
anomaly_latent_vecs = np.array(anomaly_latent_vecs)
latent_space_vals = np.concatenate([anomaly_latent_vecs, normal_latent_vecs])

print(f"Normal samples in latent space: {normal_latent_vecs.shape[0]}")
print(f"Anomaly samples in latent space: {anomaly_latent_vecs.shape[0]}")

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
    model_config['input_size'] = data_config['image_size']
    x_min, x_max = float(np.min(latent_space_vals[:, 0])), float(np.max(latent_space_vals[:, 0]))
    y_min, y_max = float(np.min(latent_space_vals[:, 1])), float(np.max(latent_space_vals[:, 1]))
    boundaries = [x_min - 1, x_max + 1, y_min - 1, y_max + 1]
    interactive_2d_latent_space(model, model_config, boundaries, latent_space_vals=latent_space_vals, hist_kwargs={'cmap': 'magma', 'bins': 80})
    fig = plt.gcf()
    ax = fig.axes[1]
    ax.vlines([x_min, x_max], ymin=y_min, ymax=y_max, colors='r')
    ax.hlines([y_min, y_max], xmin=x_min, xmax=x_max, colors='r')

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
    model_config['input_size'] = data_config['image_size']
    x_min, x_max = float(np.min(latent_space_vals[:, 0])), float(np.max(latent_space_vals[:, 0]))
    y_min, y_max = float(np.min(latent_space_vals[:, 1])), float(np.max(latent_space_vals[:, 1]))
    boundaries = [x_min - 3, x_max + 3, y_min - 3, y_max + 3]
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
# Summary of key findings from this experiment
RUN_NAME = 'conv-ae'

NOTE = r"""
Key Findings from ConvAE Anomaly Detection Experiment:

1. **Localized Scoring + Otsu Thresholding Works Exceptionally Well**
   - Using localized reconstruction scoring instead of global MSE significantly improves precision
   - Otsu method automatically finds optimal threshold without manual tuning
   - Achieved 93 bird detections with zero false positives

2. **Minimal Training Data is Sufficient**
   - Only 20% of dataset used for training (learns background patterns)
   - 30% used for validation (establishes Otsu threshold baseline)
   - Full dataset testing validates the approach works end-to-end

3. **Dataset-Specific Optimization is Effective**
   - This pipeline is designed for post-shooting analysis of specific datasets
   - No need for generalization across different shooting conditions
   - Focus on precision over recall for practical photo editing workflow

4. **Convolutional Architecture Benefits**
   - ConvAE captures spatial patterns better than fully-connected autoencoder
   - More efficient parameter usage for image data
   - Better separation of normal vs anomalous images in latent space

This approach provides a practical solution for bird photographers to quickly identify 
images requiring attention in their editing workflow.
"""

# %%
import mlflow
from bird_detector.utils.mlflow_utils import get_input_example, get_output_example
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("conv-ae")

with mlflow.start_run(run_name=RUN_NAME) as run:
    # Log configuration
    mlflow.log_dict(config, 'config.json')
    
    # Log key metrics
    mlflow.log_metric(training_config['loss_fn'], results['final_val_loss'])
    mlflow.log_metric('mse', validation_mse_vals[-1])
    mlflow.log_metric('anomalies_found', len(anomalous_filenames))
    mlflow.log_metric('anomaly_rate_percent', len(anomalous_filenames)/len(full_dataset)*100)

    # Create input and output examples for model signature
    device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')

    # Get a sample from the dataloader and immediately convert to numpy
    sample_batch = next(iter(train_dataloader))
    if isinstance(sample_batch, (list, tuple)):
        sample_images, _ = sample_batch
    else:
        sample_images = sample_batch

    # Take just the first image and convert to numpy immediately
    input_example = sample_images[0:1].cpu().numpy()  # Shape: (1, 3, H, W)

    # Generate output example
    with torch.no_grad():
        output_example = model(sample_images[0:1].to(device)).cpu().numpy()

    signature = mlflow.models.infer_signature(input_example, output_example)
    mlflow.pytorch.log_model(model, name='autoencoder_model', signature=signature, input_example=input_example)

    # Store the run_id for later reference
    run_id = run.info.run_id

    # Log the model parameters
    mlflow.log_params(model_config)
    
    # Log data configuration
    mlflow.log_params({'data_path': data_config['path']})
    _data_config = {k: v for k,v in data_config.items() if k != 'path'}
    mlflow.log_params(_data_config)

    # Set experiment tags
    mlflow.set_tags({
        "architecture": "convolutional",
        "latent_dim": model_config['latent_dim'],
        "latent_norm": model_config.get('softmax', 'none'),
        "activation": "relu",
        "experiment_group": "architecture_comparison",
        "scoring_method": "localized_reconstruction",
        "threshold_method": "otsu",
        "training_data_fraction": data_config['train_frac']
    })

    if NOTE:
        mlflow.set_tags({"mlflow.note.content": NOTE})

print(f"Experiment logged with run_id: {run_id}")

# %%
