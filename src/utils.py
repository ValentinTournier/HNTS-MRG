import os

def get_data_list(root_dir):
    """
    This function creates a list of dictionaries containing the paths to T2 and mask files for each patient.
    Each dictionary contains the keys "image" and "label".
    :param root_dir: The root directory containing patient folders.
    :return: A list of dictionaries with keys "image" and "label".
    """
    data_dicts = []
    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id, "preRT")
        if not os.path.isdir(patient_path):
            print(f"⚠️ Le répertoire {patient_path} n'existe pas ou n'est pas un répertoire.")
            continue
        # Find T2 and mask files
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