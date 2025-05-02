# ===== IMPORT LIBRARIES =====

import os
import torch
import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

from monai.data import DataLoader, CacheDataset
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityd, SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandAffined, RandBiasFieldd,
    RandGaussianNoised, RandAdjustContrastd,
    ToTensord, EnsureTyped
)
from utils import get_data_list


# ========== SETTINGS ==========

# Defining the seed for reproducibility
set_determinism(seed=42)
# Set the random seed for NumPy and PyTorch
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available and set the device accordingly
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    # error 
    device = torch.device("cpu")


# ========== VARIABLES ==========

# ---------- PATHS ----------
# Directories for training data
train_data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/train"

# Ensure the directory exists before saving the model
save_dir = os.path.expanduser("~/HNTS-MRG/results_UNet_v1")
os.makedirs(save_dir, exist_ok=True)

# Save the model
save_dir_models = os.path.join(save_dir, "models")
os.makedirs(save_dir_models, exist_ok=True)

# Save losses for plotting
losses_file = os.path.join(save_dir, "losses.txt")

# ---------- HYPERPARAMETERS ----------
# Define the batch size
batch_size = 2

# Define the training and testing sizes
# These are the proportions of the data to be used for training and testing
train_size = 0.9
test_size = 0.1

# Define the region of interest size for sliding window inference
# This is the size of the patches that will be extracted from the input images during inference
roi_size = (192, 192, 96)

# ---------- OPTIMIZER ----------

# Define the maximum number of epochs for training
max_epochs = 300

# Define the learning rate for the optimizer
learning_rate = 1e-3

# Define the step size and gamma for the learning rate scheduler
step_size = 100
gamma = 0.1


# ========== MODEL ==========

# Define the model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    kernel_size=3,
    dropout=0.4,
    act="LeakyReLU",
).to(device)


# ========== LOSS ==========

# classic one
# loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
# loss_function = DiceLoss()

# Dice + CrossEntropy Loss
# This is a good choice if you have a multi-class segmentation problem and want to balance the contribution of each class.
loss_function = DiceCELoss(
    to_onehot_y=True,  # Transform the labels to one-hot encoding
    softmax=True,      # Apply softmax to the output
    include_background=False,  # Exclude background class from the loss calculation
    weight=torch.tensor([0.2, 1.0, 1.4]).to(device)  # Background, GTVp and GTVn
)

metric_function = DiceLoss(
    to_onehot_y=True,
    softmax=True,
    include_background=False
)

# Adjust alpha (false negative penalty) and beta (false positive penalty) based on your task. This is particularly good if your tumor is very small in volume.
# loss_function = TverskyLoss(to_onehot_y=True, softmax=True, alpha=0.7, beta=0.3)


# ========== TRANSFORMATIONS ==========

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode="minimum"),
    ScaleIntensityd(keys=["image"]),

    # Crop positive and negative patches
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=roi_size, pos=2, neg=1, num_samples=4, 
        # pos=2, neg=1 makes the model focus more on the positive class
        image_key="image", image_threshold=0
    ),

    # Spatial augmentations
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
    
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandAffined(
        keys=["image", "label"],
        prob=0.5,
        rotate_range=(0.1, 0.1, 0.1),
        shear_range=None,
        translate_range=(10, 10, 5),
        scale_range=(0.1, 0.1, 0.1),
        mode=("bilinear", "nearest")
    ),

    # Intensity augmentations
    RandBiasFieldd(keys=["image"], prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
    RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 1.5)),

    ToTensord(keys=["image", "label"]),
])

# Validation transforms
val_transforms = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode="minimum"),
    EnsureTyped(keys=["image", "label"]),
]


# ========== ALL SETTINGS DONE ==========


# ========== CREATING DICTS ==========

print("Creating dicts")
# Obtain the complete training data
train_data_full = get_data_list(train_data_dir)

# Divide the training data into train and validation sets
train_data, val_data = train_test_split(train_data_full, train_size=train_size, test_size=test_size)

# ========== TRANSFORMING TRAINING DATAS ==========

print("Transforming training datas")

# Datasets and Loaders
train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1)


# ========== DEFINE THE OPTIMIZER AND SCHEDULER ==========

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# ========== TRAINING ==========

print("Training")
# Training loop

best_loss = float("inf")
for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0

    tqdm_bar = tqdm.tqdm(train_loader, desc="Training", unit="batch")
    for batch_data in tqdm_bar:
        tqdm_bar.set_description(f"Epoch {epoch+1}/{max_epochs} - Training")
        tqdm_bar.set_postfix(loss=epoch_loss)
        tqdm_bar.refresh()
        
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        tqdm_bar.set_postfix(loss=epoch_loss / len(train_loader))
        tqdm_bar.refresh()

    print(f"Train loss: {epoch_loss / len(train_loader):.4f}")

    # ====== VALIDATION ======
    model.eval()

    with torch.no_grad():
        val_loss = 0
        for val_batch in val_loader:
            val_inputs, val_labels = val_batch["image"].to(device), val_batch["label"].to(device)

            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )

            val_loss += metric_function(val_outputs, val_labels).item()
            
        val_loss /= len(val_loader)

    print(f"Validation loss: {val_loss:.4f}")

    # ----------- SAVING ----------
    if val_loss < best_loss:
        best_loss = val_loss
        # Save the model if validation loss improves
        torch.save(model, os.path.join(save_dir_models, f"best_model{epoch+1}.pth"))
    
    with open(losses_file, "a") as f:
        f.write(f"Epoch {epoch+1}, Train Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}\n")

# Save the final model
torch.save(model, os.path.join(save_dir_models,"final_model.pth"))
print("Training completed.")

