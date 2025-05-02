# ===== IMPORT LIBRARIES =====

import imageio
import re
from collections import defaultdict
import os
import torch
import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from monai.data import DataLoader, CacheDataset
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityd, SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandAffined, RandBiasFieldd,
    RandGaussianNoised, RandAdjustContrastd,
    ToTensord, EnsureTyped, AsDiscreted
)
from monai.networks.utils import one_hot
from utils import get_data_list, plot_prediction


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
# Directories for testing data
test_data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/test"

# Ensure the directory exists before saving the model
save_dir = os.path.expanduser("~/HNTS-MRG/results_UNet_v1")
os.makedirs(save_dir, exist_ok=True)

# Save the model
save_dir_models = os.path.join(save_dir, "models")
os.makedirs(save_dir_models, exist_ok=True)

# Directory for saving test images and GIFs
save_images_dir = os.path.join(save_dir, "test_images")
os.makedirs(save_images_dir, exist_ok=True)
output_dir = os.path.join(save_dir, "gifs")
os.makedirs(output_dir, exist_ok=True)

# ----------- HYPERPARAMETERS ----------
# Define the region of interest size for sliding window inference
# This is the size of the patches that will be extracted from the input images during inference
roi_size = (192, 192, 96)


# ========== MODEL ==========

model_name = "final_model.pth"
model_path = os.path.join(save_dir, model_name)


# ========== METRICS ==========

test_dice_metric = DiceMetric(include_background=True, reduction="mean")
metrics_list = [ "sensitivity", "specificity", "accuracy", "precision", "f1"]
confusion_metric = ConfusionMatrixMetric(include_background=True, reduction="mean", metric_name=metrics_list)


# ========== ALL SETTINGS DONE ==========


# ========== CREATING DICTS ==========

print("Creating dicts")
# Obtain the test data
test_data = get_data_list(test_data_dir)


# ========== LOADING TEST DATAS ==========

test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"])
])

test_ds = CacheDataset(data=test_data, transform=test_transforms, cache_rate=1.0)
test_loader = DataLoader(test_ds, batch_size=1)


# =========== LOADING MODEL ==========

model = torch.load(model_path, weights_only=False)
model.to(device)
model.eval()

# ========== TESTING ==========

print("Testing the model")
with torch.no_grad():
    test_dice_metric.reset()
    confusion_metric.reset()

    tqdm_bar = tqdm.tqdm(test_loader, desc="Testing", unit="batch")
    for i, test_data in enumerate(tqdm_bar):
        tqdm_bar.set_description("Testing")
        tqdm_bar.refresh()

        test_inputs = test_data["image"].to(device)
        test_labels = test_data["label"].to(device)
        
        # Perform sliding window inference
        test_outputs = sliding_window_inference(test_inputs, roi_size=roi_size, sw_batch_size=4, predictor=model)

        test_labels = one_hot(test_labels, num_classes=test_outputs.shape[1])

        test_dice_metric(y_pred=test_outputs, y=test_labels)
        confusion_metric(y_pred=test_outputs, y=test_labels) 

        # -------- CREATING IMAGES --------
        # For each image in the batch
        for j in range(test_inputs.shape[4]):
            # For each slice in the image
            image_np = test_inputs[0, 0].cpu().numpy()
            label_np = test_labels[0].cpu().numpy()
            label_np = np.argmax(label_np, axis=0) if label_np.ndim == 4 else label_np

            output_np = test_outputs[0].cpu().detach().numpy()
            pred_np = np.argmax(output_np, axis=0)

            plot_prediction(
                image_np,
                label_np,
                pred_np,
                slice_index=j,
                title=f"Test Case #{i}_{j}",
                save_path=os.path.join(save_images_dir, f"test_vis_{i:03d}_{j}.png")
            )

        test_metric = test_dice_metric.aggregate().item()
        test_dice_metric.reset()

    cm_stats = confusion_metric.aggregate()

    for name, value in zip(metrics_list, cm_stats):
        print(f"{name}: {value.item():.4f}")

    confusion_metric.reset()

    print(f"Test Dice: {test_metric:.4f}")


# =========== CREATING GIFS ===========

# Group images by patient ID
patient_images = defaultdict(list)
pattern = r"test_vis_(\d+)_(\d+)\.png"

# List and sort images
for filename in os.listdir(image_dir):
    match = re.match(pattern, filename)
    if match:
        patient_id = int(match.group(1))
        slice_idx = int(match.group(2))
        patient_images[patient_id].append((slice_idx, filename))

# Create a GIF for each patient
for patient_id, slices in patient_images.items():
    # Sort slices by index
    sorted_slices = sorted(slices, key=lambda x: x[0])
    images = [imageio.imread(os.path.join(image_dir, f)) for _, f in sorted_slices]
    
    # Save path for the GIF
    gif_path = os.path.join(output_dir, f"patient_{patient_id:03d}.gif")
    
    # Create the GIF
    imageio.mimsave(gif_path, images, duration=1)
    print(f"GIF saved: {gif_path}")
