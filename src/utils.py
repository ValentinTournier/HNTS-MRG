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
            print(f"Directory {patient_path} does not exist or isn't a directory.")
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
            print(f"Files missing for patient {patient_id}")
    return data_dicts

def plot_prediction(image, label, prediction, slice_index=None, title="", save_path=None):
    """
    Displays or saves an image of an axial slice of the volume.
    - image, label, prediction: np.array (3D)
    - slice_index: int or None (if None => central slice)
    - save_path: path to save the image (if not None, does not display)
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