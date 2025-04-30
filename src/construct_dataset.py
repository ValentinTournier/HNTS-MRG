import os
from sklearn.model_selection import train_test_split

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
    return sorted(data_dicts, key=lambda x: x["image"])

# Construction des jeux d'entraînement et de validation
train_data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/train"
test_data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG/test"

# Obtenir les données d'entraînement complètes
train_data_full = get_data_list(train_data_dir)

# Diviser les données d'entraînement en train et validation
train_data, val_data = train_test_split(train_data_full, test_size=0.2, random_state=42)

# Obtenir les données de test
test_data = get_data_list(test_data_dir)

# Exemple d'aperçu
print(f"Train data: {len(train_data)} samples")
print(f"Validation data: {len(val_data)} samples")
print(f"Test data: {len(test_data)} samples")

# Exemple d'aperçu
print(train_data[0])
# {'image': './train/10/preRT/10_preRT_T2.nii.gz', 'label': './train/10/preRT/10_preRT_mask.nii.gz'}
