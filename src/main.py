import os
import torch
import tqdm
from sklearn.model_selection import train_test_split
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityd, RandCropByPosNegLabeld, RandFlipd,
    RandRotate90d, ToTensord
)
from monai.data import Dataset, DataLoader, CacheDataset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.transforms import Compose
from monai.transforms import SpatialPadd
from monai.transforms import AsDiscreted
from monai.losses import DiceLoss



# Set seed for reproducibility
set_determinism(seed=42)

# ======= Creating dicts =======
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

print("Searching datas...")
# Obtenir les données d'entraînement complètes
train_data_full = get_data_list(train_data_dir)

# Diviser les données d'entraînement en train et validation
train_data, val_data = train_test_split(train_data_full, test_size=0.8, random_state=42)

# Obtenir les données de test
test_data = get_data_list(test_data_dir)

# Exemple d'aperçu
print(f"Train data: {len(train_data)} samples")
print(f"Validation data: {len(val_data)} samples")
print(f"Test data: {len(test_data)} samples")

# Exemple d'aperçu
print(train_data[0])
# {'image': './train/10/preRT/10_preRT_T2.nii.gz', 'label': './train/10/preRT/10_preRT_mask.nii.gz'}

train_files = train_data
val_files = val_data
test_files = test_data

# ======= Training =======

print("Transforming datas...")
# Transforms
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4
    ),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    ToTensord(keys=["image", "label"]),
])

# val_transforms = [
#     LoadImaged(keys=["image", "label"]),
#     EnsureChannelFirstd(keys=["image", "label"]),
#     Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
#     Orientationd(keys=["image", "label"], axcodes="RAS"),
#     ScaleIntensityd(keys=["image"]),
#     ToTensord(keys=["image", "label"]),
# ]

# import nibabel as nib
# for entry in train_files:
#     if not os.path.exists(entry["image"]):
#         print(f"❌ Image not found: {entry['image']}")
#     if not os.path.exists(entry["label"]):
#         print(f"❌ Label not found: {entry['label']}")
#     img = nib.load(entry["image"])
#     print(img.shape)
#     label = nib.load(entry["label"])
#     print(label.shape)

# Datasets and Loaders
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
# val_loader = DataLoader(val_ds, batch_size=1)

# Model
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

print("Training...")
# Training loop
max_epochs = 1
for epoch in range(max_epochs):
    print(f"Epoch {epoch+1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    for batch_data in train_loader:
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Train loss: {epoch_loss / len(train_loader):.4f}")

    # # Validation
    # model.eval()
    # with torch.no_grad():
    #     dice_metric.reset()
    #     for val_data in val_loader:
    #         val_inputs = val_data["image"].to(device)
    #         val_labels = val_data["label"].to(device)
    #         val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
    #         dice_metric(y_pred=val_outputs, y=val_labels)
    #     metric = dice_metric.aggregate().item()
    #     dice_metric.reset()
    #     print(f"Validation Dice: {metric:.4f}")

# ======= Testing =======

test_transforms = Compose([
    AsDiscreted(keys="pred", argmax=True),
    AsDiscreted(keys="label", to_onehot=4),
    AsDiscreted(keys="pred", to_onehot=4),
])

test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
test_loader = DataLoader(test_ds, batch_size=1)

print("Testing...")
# Testing loop
model.eval()
test_dice_metric = DiceMetric(include_background=False, reduction="mean")
with torch.no_grad():
    test_dice_metric.reset()
    for test_data in test_loader:
        test_inputs = test_data["image"].to(device)
        test_labels = test_data["label"].to(device)# val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
# val_loader = DataLoader(val_ds, batch_size=1)
        test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), 4, model)
        test_dice_metric(y_pred=test_outputs, y=test_labels)
    test_metric = test_dice_metric.aggregate().item()
    test_dice_metric.reset()
    print(f"Test Dice: {test_metric:.4f}")