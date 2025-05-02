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

# ======= CREATING DICTS =======
def get_data_list(root_dir):
    data_dicts = []
    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id, "preRT")
        if not os.path.isdir(patient_path):
            print(f"⚠️ Le répertoire {patient_path} n'existe pas ou n'est pas un répertoire.")
            continue
        # Trouver fichiers T2 et masque
        t2_files = os.path.join(patient_path, f"{patient_id}_preRT_T2.nii.gz")
        mask_files = os.path.join(patient_path, f"{patient_id}_preRT_mask.nii.gz")

        if t2_files and mask_files:
            data_dicts.append({
                "image": t2_files,
                "label": mask_files
            })
        else:
            print(f"⚠️ Fichiers manquants pour le patient {patient_id}")
    return data_dicts

# Construction des jeux d'entraînement et de validation
train_data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/train"
test_data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/test"

print("Creating dicts")
# Obtenir les données d'entraînement complètes
train_data_full = get_data_list(train_data_dir)

# Diviser les données d'entraînement en train et validation
train_data, val_data = train_test_split(train_data_full, train_size=0.1, test_size=0.05)

# Obtenir les données de test
test_data = get_data_list(test_data_dir)

# Ensure the directory exists before saving the model
save_dir = os.path.expanduser("~/HNTS-MRG/results")
os.makedirs(save_dir, exist_ok=True)

# Check if CUDA is available and set the device accordingly
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

# ======= LOADING TEST DATAS =======

test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"]),
    # AsDiscreted(keys="label")
])

test_ds = CacheDataset(data=test_data, transform=test_transforms, cache_rate=1.0)
test_loader = DataLoader(test_ds, batch_size=1)

# ====== LOADING MODEL ======
model_name = "best_model_old.pth"
model_path = os.path.join(save_dir, model_name)
model = torch.load(model_path, weights_only=False)
model.eval()

# ====== METRICS ======
test_dice_metric = DiceMetric(include_background=True, reduction="mean")
metrics_list = [ "sensitivity", "specificity", "accuracy", "precision", "f1"]
confusion_metric = ConfusionMatrixMetric(include_background=True, reduction="mean", metric_name=metrics_list)

# ===== TESTING =====
print("Testing")

with torch.no_grad():
    test_dice_metric.reset()
    confusion_metric.reset()

    tqdm_bar = tqdm.tqdm(test_loader, desc="Testing", unit="batch")
    for test_data in tqdm_bar:
        tqdm_bar.set_description("Testing")
        tqdm_bar.refresh()

        test_inputs = test_data["image"].to(device)
        test_labels = test_data["label"].to(device)
        
        test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), sw_batch_size=4, predictor=model)

        test_labels = one_hot(test_labels, num_classes=test_outputs.shape[1])

        test_dice_metric(y_pred=test_outputs, y=test_labels)
        confusion_metric(y_pred=test_outputs, y=test_labels)
        tqdm_bar.set_postfix(dice=test_dice_metric.aggregate().item())
        tqdm_bar.refresh()

    test_metric = test_dice_metric.aggregate().item()
    test_dice_metric.reset()

    cm_stats = confusion_metric.aggregate()

    # print(cm_stats)
    for name, value in zip(metrics_list, cm_stats):
        print(f"{name}: {value.item():.4f}")

    confusion_metric.reset()

    print(f"Test Dice: {test_metric:.4f}")

# ====== CREATING IMAGES ======
def plot_prediction(image, label, prediction, slice_index=None, title="", save_path=None):
    """
    Affiche ou sauvegarde une image d'une coupe axiale du volume.
    - image, label, prediction : np.array (3D)
    - slice_index : int ou None (si None => coupe centrale)
    - save_path : chemin de sauvegarde de l'image (si non None, ne montre pas)
    """
    if slice_index is None:
        slice_index = image.shape[2] // 2

    image_slice = image[:, :, slice_index]
    label_slice = label[:, :, slice_index]
    pred_slice = prediction[:, :, slice_index]

    plt.figure(figsize=(15, 5))
    plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.imshow(image_slice, cmap="gray")
    plt.title("Image MRI")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(label_slice, cmap="viridis", interpolation="none")
    plt.title("Vérité Terrain")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_slice, cmap="viridis", interpolation="none")
    plt.title("Prédiction")
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


save_images_dir = os.path.join(save_dir, "test_images")
os.makedirs(save_images_dir, exist_ok=True)

model.eval()

for i, test_data in enumerate(test_loader):
   
    test_inputs = test_data["image"].to(device)
    test_labels = test_data["label"].to(device)
    test_labels = one_hot(test_labels, num_classes=test_outputs.shape[1])

    test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), sw_batch_size=4, predictor=model)

    # Pour chaque image du batch
    tqdm_bar = tqdm.tqdm(range(test_inputs.shape[4]), desc="Creating images", unit="image")

    for j in range(test_inputs.shape[4]):
        tqdm_bar.set_description(f"Test {i+1}/{len(test_loader)} - Image {j+1}/{test_inputs.shape[4]}")
        tqdm_bar.refresh()
        # Pour chaque slice de l'image

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
        tqdm_bar.set_postfix(image=j)
        tqdm_bar.refresh()
    tqdm_bar.close()

print("Test images created.")

# ====== CREATING GIFS ======

# Répertoire contenant les images
image_dir = os.path.expanduser("~/HNTS-MRG/results/test_images")
output_dir = os.path.join(save_dir, "gifs")
os.makedirs(output_dir, exist_ok=True)

# Regrouper les images par patient ID
patient_images = defaultdict(list)
pattern = r"test_vis_(\d+)_(\d+)\.png"

# Lister et classer les images
for filename in os.listdir(image_dir):
    match = re.match(pattern, filename)
    if match:
        patient_id = int(match.group(1))
        slice_idx = int(match.group(2))
        patient_images[patient_id].append((slice_idx, filename))

# Créer un GIF par patient
for patient_id, slices in patient_images.items():
    # Trier les slices par index
    sorted_slices = sorted(slices, key=lambda x: x[0])
    images = [imageio.imread(os.path.join(image_dir, f)) for _, f in sorted_slices]
    
    # Chemin de sauvegarde du GIF
    gif_path = os.path.join(output_dir, f"patient_{patient_id:03d}.gif")
    
    # Créer le GIF
    imageio.mimsave(gif_path, images, duration=1)
    print(f"✅ GIF sauvegardé : {gif_path}")
