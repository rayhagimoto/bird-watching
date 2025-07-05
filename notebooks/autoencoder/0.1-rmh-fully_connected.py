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
# Bird Anomaly Detection with Autoencoder
# This notebook implements anomaly detection using a simple autoencoder
# with filename tracking and 95th percentile threshold
# %load_ext autoreload
# %autoreload 2


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
# # Bird Photo Anomaly Detection
#
# **Goal**: Find anomalous bird photos by filename so you can check them in Lightroom and edit if needed.
#
# **Method**: Train an autoencoder on normal background images, then flag photos that have high reconstruction error.

# %% [markdown]
# ## 1. Autoencoder Model
#
# Simple fully connected autoencoder that compresses images to 2D latent space and reconstructs them.

# %%
# Simple Fully Connected Autoencoder with 2D Latent Space
import copy
from bird_watching.data.data_loaders import SimpleBirdDataset
from bird_watching.autoencoder import SimpleAutoencoder, SimpleAutoencoderDetector

class SigmoidLatentSpaceAutoencoder(SimpleAutoencoder):

    def _encoder_from_layer_sizes(self) -> nn.Sequential:
        """Build the encoder network from layer sizes.
        
        Creates a sequence of fully connected layers with ReLU activations,
        ending with a linear layer to the latent space (no activation).
        
        Architecture:
        Flatten → FC+ReLU → FC+ReLU → ... → FC (no activation) → Latent
        
        Returns:
            nn.Sequential: Encoder network
        """
        layers = []
        
        # Add flatten layer first
        layers.append(nn.Flatten())
        
        # Build intermediate layers
        current_size = self.flattened_size
        for size in self.layer_sizes:
            layers.append(nn.Linear(current_size, size))
            layers.append(nn.ReLU())
            current_size = size
        
        # Add final layer to latent space (no activation)
        layers.append(nn.Linear(current_size, self.latent_dim))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)

# %%
with open('fc-ae-config.yaml') as f:
  config = yaml.safe_load(f)
model_config = config['model']
training_config = config['training']
data_config = config['data']
model_metadata = config['model_metadata']

# %% [markdown]
# ## 2. Data Splitting Strategy
#
# **Training**: 20% random sample (learns normal background patterns)
# **Validation**: 20% random sample (establishes baseline MSE distribution)
# **Field Test**: FULL DATASET (because this autoencoder is only supposed to work on this dataset)

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

# Create three-way split: 20% train, 20% validation
total_size = len(full_dataset)
train_size = int(data_config['train_frac'] * total_size)  # 20% for each of train and validation
val_size = int(data_config['val_frac'] * total_size)  # 20% for each of train and validation

# Create indices and shuffle them with fixed seed
indices = list(range(total_size))
random.shuffle(indices)

# Split indices
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]

print(f"Training set: {len(train_indices)} images ({len(train_indices)/total_size*100:.1f}%)")
print(f"Validation set: {len(val_indices)} images ({len(val_indices)/total_size*100:.1f}%)")

# Create subsets
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
full_dataloader = DataLoader(full_dataset, batch_size=8, shuffle=False, num_workers=0)  # Batch size 8 for field test

# %% [markdown]
# ## 3. Model Training
#
# Train the autoencoder to learn normal background patterns.

# %%
from bird_watching.autoencoder import get_loss_function, get_optimizer

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SimpleAutoencoder(metric=training_config['loss_fn'], **model_config).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

loss_fn = get_loss_function(training_config['loss_fn'])
optimizer = get_optimizer(training_config['optimizer'], model.parameters(), lr=training_config['lr'])

# %%
from bird_watching.autoencoder import train_model

results = train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    training_config,
    metrics_to_track=['loss', 'mse'],
    device = device
)

print("\n".join(results.keys()))
final_train_loss = results['final_train_loss']
best_train_loss = results['best_train_loss']
final_val_loss = results['final_val_loss']
best_val_loss = results['best_val_loss']
validation_losses = results['validation_loss_vals']
training_losses = results['training_loss_vals']
time_elapsed_sec = results['time_elapsed_sec']
training_mse_vals = results['training_mse_vals']
validation_mse_vals = results['validation_mse_vals']

print(f"\nTotal Time elapsed:\t{int(time_elapsed_sec)} sec")

# %% [markdown]
# # 5. Training Progress
#
# Visualize how well the model is learning.

# %%
# %matplotlib inline
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(training_losses, label='Training Loss', color='blue')
ax[0].plot(validation_losses, label='Validation Loss', color='red')
ax[0].set_title(f'Loss ({training_config["loss_fn"]})')
ax[0].legend()

ax[1].plot(training_mse_vals, label='Training MSE', color='blue')
ax[1].plot(validation_mse_vals, label='Validation MSE', color='red')
ax[1].set_title('MSE')
# ax[1].set_yscale('log')
ax[1].legend()

plt.show()


# %%
def log_run(run_name, experiment_name='fc-autoenconder', tracking_uri="http://127.0.0.1:5000", note=None):

  import mlflow
  from bird_watching.mlflow_utils import get_input_example, get_output_example
  mlflow.set_tracking_uri(tracking_uri)
  mlflow.set_experiment(experiment_name)

  input_example = get_input_example(train_dataloader)
  output_example = get_output_example(model, train_dataloader)

  with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_dict(config, 'config.json')
    mlflow.log_metric("MSE", results['final_val_loss'])

    # Store the run_id for later in case I want to update
    run_id = run.info.run_id

    signature = mlflow.models.infer_signature(input_example, output_example)
    mlflow.pytorch.log_model(model, name='autoencoder_model')

    # Log the model params
    mlflow.log_params(model_config)
    
    # Log 'path' separate from the rest of the data params
    mlflow.log_params({'data_path': data_config['path']})
    _data_config = {k: v for k,v in data_config.items() if k != 'path'}
    mlflow.log_params(_data_config)

    mlflow.set_tags({
      "architecture": "fully_connected",
      "encoder_layers": "-".join([str(x) for x in model_config['layer_sizes']]),
      "latent_dim": model_config['latent_dim'],
      "latent_norm": model_config.get('softmax', 'none'),
      "activation": "relu",
      "experiment_group": "architecture_comparison"
    })

    if note:
      mlflow.set_tags({"mlflow.note.content": note})


# %%
# run_name = f'fc-autoenc-softmax-lr-{training_config["lr"]:.1e}'

# %% [markdown]
# # 6. Compute MSE Distribution on Full Dataset
#
# Calculate MSE for each image independently and establish the anomaly threshold.

# %%
PERCENTILE = 98.0

print("Computing MSE distribution on the full dataset...")
model.eval()
full_mse_scores = []
full_filenames = []

full_dataloader = DataLoader(full_dataset, batch_size=16, shuffle=False, num_workers=0)

with torch.no_grad():
    for batch_idx, (data, filenames) in enumerate(full_dataloader):
        data = data.to(device)
        output = model(data)
        
        # Calculate MSE for each image in the batch independently
        # Shape: (batch_size, 3, 64, 64) -> (batch_size,)
        mse_per_image = torch.mean((output - data) ** 2, dim=[1, 2, 3])
        batch_scores = mse_per_image.cpu().numpy().tolist()
        
        full_mse_scores.extend(batch_scores)
        full_filenames.extend(filenames)
        
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx+1}/{len(full_dataloader)}")

full_mse_scores = np.array(full_mse_scores)
anomaly_threshold = np.percentile(full_mse_scores, PERCENTILE)
print(f"Full dataset log-MSE statistics:")
print(f"  Mean: {full_mse_scores.mean():.6f}")
print(f"  Std:  {full_mse_scores.std():.6f}")
print(f"  Min:  {full_mse_scores.min():.6f}")
print(f"  Max:  {full_mse_scores.max():.6f}")
print(f"  Anomaly threshold ({float(PERCENTILE):.1f}th percentile): {anomaly_threshold:.6f}")

# %%
from bird_watching.autoencoder.plot_utils import plot_mse_scores_hist
plot_mse_scores_hist(full_mse_scores, anomaly_threshold, PERCENTILE)

# %% [markdown]
# # 7. Anomaly Detection on Full Dataset
#
# Process all images and flag anomalies with filenames.

# %%
print("Starting anomaly detection on full dataset...")
model.eval()
anomalies = []
anomalous_image_names = []
ANOMALY_THRESHOLD = np.percentile(np.log10(full_mse_scores), PERCENTILE)

print(f"Processing {len(full_dataset)} images in batches...")
print(f"Anomaly threshold: {ANOMALY_THRESHOLD:.6f}")
print("-" * 60)

with torch.no_grad():
    for batch_idx, (data, filenames) in enumerate(full_dataloader):
        data = data.to(device)
        output = model(data)
        
        # Calculate MSE for each image in the batch independently
        mse_per_image = torch.mean((output - data) ** 2, dim=[1, 2, 3])
        log_mse_per_image = torch.log10(mse_per_image)
        batch_scores = log_mse_per_image.cpu().numpy().tolist()
        
        # Check for anomalies in this batch
        batch_anomalies = []
        for i, score in enumerate(batch_scores):
            if score > ANOMALY_THRESHOLD:
                global_idx = batch_idx * full_dataloader.batch_size + i
                anomalies.append((global_idx, score, filenames[i]))
                batch_anomalies.append((filenames[i], score))
        
        # Report anomalies in this batch
        if batch_anomalies:
            for filename, score in batch_anomalies:
                anomalous_image_names.append(filename)
                print(f"  🚨 {filename} ({training_config['loss_fn']}: {score:.4f})")

print("-" * 60)
print("Anomaly detection complete!")

print(f"\nAnomaly Detection Results:")
print(f"  Total images processed: {len(full_mse_scores)}")
print(f"  Anomaly threshold (95th percentile): {ANOMALY_THRESHOLD:.6f}")
print(f"  Anomalies detected: {len(anomalies)} ({len(anomalies)/len(full_mse_scores)*100:.1f}%)")
print(f"  Dataset log10 mse mean: {np.log10(full_mse_scores).mean():.6f}")
print(f"  Dataset log10 mse std:  {np.log10(full_mse_scores).std():.6f}")

# %%
from PIL import Image

with torch.no_grad():

  img_path = ROOT_PATH / data_config['path'] / "0000.jpg"

  img = Image.open(img_path)

  transform = transforms.Compose([
    transforms.Resize((model_config['input_size'], model_config['input_size'])),
    transforms.ToTensor(),
  ])

  img_tensor = transform(img.convert('RGB')).unsqueeze(0).to(torch.device('cuda'))
  decoded = model(img_tensor)
  score = torch.log10(torch.mean((decoded - img_tensor)**2, axis=[1,2,3])).detach()
  print(score > ANOMALY_THRESHOLD)

# %%
# Get a list of the images corresponding to those filenames
anomalous_images = []
for name in anomalous_image_names:

  img_path = str(ROOT_PATH / data_config['path'] / name)
  img = np.array(Image.open(img_path).convert('RGB'))
  anomalous_images.append(img)

# %%
with torch.no_grad():
  img_tensor = torch.tensor(img) / 255.0
  img_tensor = img_tensor.permute(2, 0, 1).to(device)
  img_tensor = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(img_tensor).unsqueeze(0)

  img_in = img_tensor[0].permute(1, 2, 0).cpu().numpy() 
  img_out = model(img_tensor)[0].permute(1, 2, 0).cpu().numpy()

log_diff_img = - np.log10(np.maximum(1e-8, img_out - img_in)).mean(axis=-1)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(img_in)
ax[1].imshow(img_out)
ax[2].imshow(log_diff_img / np.max(log_diff_img))

# %%
# Plot all anomalous images
if len(anomalous_images) > 0:
    print(f"Visualizing {len(anomalous_images)} anomalous images...")
    
    # Calculate grid dimensions
    n_images = len(anomalous_images)
    cols = min(8, n_images)  # Max 8 columns
    rows = (n_images + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), dpi=75)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (image, (idx, score, filename)) in enumerate(zip(anomalous_images, anomalies)):
        row = i // cols
        col = i % cols
        
        # Your image is already a numpy array in HWC format
        img_display = image  # No need for permute or numpy conversion
        
        axes[row, col].imshow(img_display)
        axes[row, col].set_title(f'{filename}\nMSE: {score:.4f}', fontsize=8)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed {len(anomalous_images)} anomalous images")
else:
    print("No anomalies to display!")

# %% [markdown]
# From this run there are 7 false positives out of 12 anomalies. Unknown amount of false negatives. (I didn't count the images of my mum as false positives because they are indeed anomalous!)

# %% [markdown]
# ## 8. Anomaly Summary
#
# Complete list of anomalous files for Lightroom editing.

# %%
if len(anomalies) > 0:
    print(f"\n📋 COMPLETE ANOMALY LIST (for Lightroom editing):")
    print("=" * 60)
    anomalies.sort(key=lambda x: x[0], reverse=False)  # Sort by MSE score (highest first)
    
    for i, (idx, score, filename) in enumerate(anomalies):
        print(f"{i+1:2d}. {filename} (MSE: {score:.4f})")
    
    print("=" * 60)
    print(f"Total anomalies found: {len(anomalies)}")
    print("Check these files in Lightroom for potential issues!")
else:
    print("🎉 No anomalies detected! All images appear normal.")

# %% [markdown]
# ## 9. Save Results
#
# Save the complete anomaly list to a file for easy reference.

# %%
# # Save anomaly results to file
# if len(anomalies) > 0:
#     print("Saving anomaly results...")
#     with open('anomaly_results.txt', 'w') as f:
#         f.write(f"Bird Photo Anomaly Detection Results\n")
#         f.write(f"====================================\n")
#         f.write(f"Dataset: {len(full_dataset)} total images\n")
#         f.write(f"Training: {len(train_indices)} images (20%)\n")
#         f.write(f"Validation: {len(val_indices)} images (20%)\n")
#         f.write(f"Anomaly threshold (95th percentile): {anomaly_threshold:.6f}\n")
#         f.write(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/len(full_mse_scores)*100:.1f}%)\n\n")
        
#         f.write(f"ANOMALIES TO CHECK IN LIGHTROOM:\n")
#         f.write(f"==============================\n")
#         for i, (idx, score, filename) in enumerate(anomalies):
#             f.write(f"{i+1:2d}. {filename} (MSE: {score:.6f})\n")
    
#     print(f"Results saved to 'anomaly_results.txt'")

# %% [markdown]
# # 10. Interactive 

# %%
from tqdm.notebook import tqdm
# Convert img to latent space
latent_space_vals = []
with torch.no_grad():
  for images, filenames in tqdm(full_dataloader):
    images = images.to(device)
    encoded_batch = model.encoder(images)
    latent_space_vals.extend(encoded_batch.cpu().numpy())
  latent_space_vals = np.array(latent_space_vals)

# Plot latent space distribution
fig, ax = plt.subplots()
if model_config['latent_dim'] == 1:
  ax.hist(latent_space_vals.flatten(), 20)
  ax.set_xlabel('Latent Value')
  ax.set_ylabel('Count')
  ax.set_title('1D Latent Space Distribution')
  plt.show()
elif model_config['latent_dim'] == 2:
  ax.hist2d(latent_space_vals[:, 0], latent_space_vals[:, 1], bins=20)
  ax.set_xlabel('Latent dim 1')
  ax.set_ylabel('Latent dim 2')
  ax.set_title('2D Latent Space Distribution')
  plt.colorbar(label='Count')
  plt.show()

# %%
# %matplotlib widget
from bird_watching.autoencoder.interactive_utils import interactive_1d_latent_space, interactive_2d_latent_space

# Interactive exploration
if model_config['latent_dim'] == 1:
  z_min = float(np.min(latent_space_vals))
  z_max = float(np.max(latent_space_vals))
  z_initial = 0.5 * (z_min + z_max)
  interactive_1d_latent_space(model, model_config, z_min=z_min, z_max=z_max, z_initial=z_initial)
elif model_config['latent_dim'] == 2:
  x_min, x_max = float(np.min(latent_space_vals[:, 0])), float(np.max(latent_space_vals[:, 0]))
  y_min, y_max = float(np.min(latent_space_vals[:, 1])), float(np.max(latent_space_vals[:, 1]))
  boundaries = [x_min, x_max, y_min, y_max]
  interactive_2d_latent_space(model, model_config, boundaries)

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

if model_config['latent_dim'] == 2:

    # Collect all latent vectors for the dataset
    all_latents = []

    model.eval()
    with torch.no_grad():
        for batch in full_dataloader:
            # If your dataloader returns (images, filenames), unpack accordingly
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, _ = batch
            else:
                images = batch
            images = images.to(device)
            latents = model.encoder(images)  # shape: (batch_size, 2)
            all_latents.append(latents.cpu().numpy())

    all_latents = np.concatenate(all_latents, axis=0)  # shape: (N, 2)

    # Plot 2D density (hexbin)
    plt.figure(figsize=(6, 6))
    plt.hexbin(all_latents[:, 0], all_latents[:, 1], gridsize=50, cmap='viridis', bins='log')
    plt.colorbar(label='log10(N)')
    plt.xlabel('Latent dim 1')
    plt.ylabel('Latent dim 2')
    plt.title('2D Density of Latent Space Encodings')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# # LOG WITH MLFLOW (do this at the very end)

# %%
# -- IMPORTANT --
# write a short note that summarizes what I learned from this experiment
RUN_NAME = 'fc-autoencoder' + '-'.join([str(x) for x in model_config['layer_sizes']])

NOTE = r"""
This is the second run I'm doing for set_3. I drastically increased  the size of the neural network. The loss function is much smaller than before. I observed that there were far fewer false positives. At least a few false negatives -- which I could tell from the first few images. But overall much better performance. I could also clearly see that the 1D latent space maps to different lighting configurations, which is awesome. You can even see the shadows changing direction, it's really cool. """

# %%
import mlflow
from bird_watching.mlflow_utils import get_input_example, get_output_example
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("fc-autoencoder")



with mlflow.start_run(run_name=RUN_NAME) as run:
  mlflow.log_dict(config, 'config.json')
  mlflow.log_metric(training_config['loss_fn'], results['final_val_loss'])
  mlflow.log_metric('mse', validation_mse_vals[-1])

  # Create input and output examples for model signature
  # Use a clean approach to avoid context variable issues
  device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')

  # Get a sample from the dataloader and immediately convert to numpy to avoid context issues
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


  # Store the run_id for later in case I want to update
  run_id = run.info.run_id

  signature = mlflow.models.infer_signature(input_example, output_example)
  # mlflow.pytorch.log_model(model, artifact_path='autoencoder_model', signature=signature, input_example=input_example)

  # Log the model params
  mlflow.log_params(model_config)
  
  # Log 'path' separate from the rest of the data params
  mlflow.log_params({'data_path': data_config['path']})
  _data_config = {k: v for k,v in data_config.items() if k != 'path'}
  mlflow.log_params(_data_config)

  mlflow.set_tags({
    "architecture": "fully_connected",
    "encoder_layers": "-".join([str(x) for x in model_config['layer_sizes']]),
    "latent_dim": model_config['latent_dim'],
    "latent_norm": model_config.get('softmax', 'none'),
    "activation": "relu",
    "experiment_group": "architecture_comparison"
  })

  if NOTE:
    mlflow.set_tags({"mlflow.note.content" : NOTE})

# %%
# from mlflow import MlflowClient
# client = MlflowClient()
# mlflow.pytorch.load_model(f'models:/{mlflow.last_logged_model().model_id}')
# mlflow.pytorch.load_model(f'models:/m-21c6bf9cd36f4fb09af16f0c8e80fd7a')
