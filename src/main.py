# ===== IMPORT LIBRARIES =====
import os
import torch
import tqdm
from sklearn.model_selection import train_test_split

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
train_data, val_data = train_test_split(train_data_full, train_size=0.9, test_size=0.1)

# Obtenir les données de test
test_data = get_data_list(test_data_dir)

# ======= TRANSFORMING TRAINING DATAS =======

print("Transforming training datas")
# Transforms
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),

    # Crop positive and negative patches
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4,
        image_key="image", image_threshold=0
    ),

    # Spatial augmentations
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
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
    RandBiasFieldd(keys=["image"], prob=0.3),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),

    ToTensord(keys=["image", "label"]),
])
val_transforms = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"]),
]

# Datasets and Loaders
train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=3, shuffle=True)

val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1)

# ====== MODEL ======

# Check if CUDA is available and set the device accordingly
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

# Define the model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=6,
    dropout=0.3,
).to(device)

# ====== LOSS ======
# classic one
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
# loss_function = DiceLoss()

# Dice + CrossEntropy Loss
# This is a good choice if you have a multi-class segmentation problem and want to balance the contribution of each class.
# loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
# loss_function = DiceCELoss(include_background=False)

# Adjust alpha (false negative penalty) and beta (false positive penalty) based on your task. This is particularly good if your tumor is very small in volume.
# loss_function = TverskyLoss(to_onehot_y=True, softmax=True, alpha=0.7, beta=0.3)

# ====== METRICS ======
dice_metric = DiceMetric(include_background=False, reduction="mean")

# ====== OPTIMIZER ======
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
max_epochs = 200

# ====== TRAINING ======
print("Training")
# Training loop
# Ensure the directory exists before saving the model
save_dir = os.path.expanduser("~/HNTS-MRG/results")
os.makedirs(save_dir, exist_ok=True)
best_dice = 0.0
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

    with torch.no_grad():
        dice_metric.reset()

        tqdm_bar = tqdm.tqdm(val_loader, desc="Validation", unit="batch")
        for val_data in tqdm_bar:
            tqdm_bar.set_description(f"Epoch {epoch+1}/{max_epochs} - Validation")
            tqdm_bar.refresh()

            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)

            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), sw_batch_size=4, predictor=model)

            dice_metric(y_pred=val_outputs, y=val_labels)
            # confusion_metric(y_pred=val_outputs, y=val_labels)
            tqdm_bar.set_postfix(dice=dice_metric.aggregate().item())
            tqdm_bar.refresh()

        dice_score = dice_metric.aggregate().item()
        print(f"Validation Dice: {dice_score:.4f}")

        dice_metric.reset()

        # Save model
        torch.save(model.state_dict(), os.path.join(save_dir, "model_epoch_{}.pth".format(epoch)))
        # save best model
        if epoch == 0 or dice_score > best_dice:
            best_dice = dice_score
            torch.save(model.state_dict(), os.path.join(save_dir,"best_model.pth"))
            print(f"Best model saved with Dice: {best_dice:.4f}")
        else:
            print(f"Model not improved. Current best Dice: {best_dice:.4f}")

# Save the final model
torch.save(model.state_dict(), os.path.join(save_dir,"final_model.pth"))
print("Training completed.")

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
model_name = "best_model.pth"
model_path = os.path.join(save_dir, model_name)
model.load_state_dict(torch.load(model_path))

# ====== METRICS ======
test_dice_metric = DiceMetric(include_background=False, reduction="mean")
metrics_list = [ "sensitivity", "specificity", "accuracy", "precision", "f1"]
confusion_metric = ConfusionMatrixMetric(include_background=False, reduction="mean", metric_name=metrics_list)

# ===== TESTING =====
print("Testing")
model.eval()

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