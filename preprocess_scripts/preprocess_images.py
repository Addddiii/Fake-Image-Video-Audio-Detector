import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import io
import os
import glob
import torch
from pathlib import Path

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)

patch_pipline = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

def image_processor(image_file):
    """in back end when an image gets uploaded we might primarely only use this function
       """
    # STEP 1: first we use PILL as the front door to accept different file formats and turn them all into RGB
    try:
        pil_image = Image.open(image_file).convert('RGB')
    except Exception as e:
        print(f"Failed to load image: {e}")
        return None, None

    # STEP 2: resize image to 224x224 and convert to tensor
    spatial_tensor_patch = patch_pipline(pil_image)

    # STEP 3: now we use openCV to extract frequency data from the image
    open_cv_array = np.array(pil_image.resize(IMG_SIZE))
    gray_image = cv2.cvtColor(open_cv_array, cv2.COLOR_RGB2GRAY)
    
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)

    frequency_spectrum = 20*np.log(np.abs(f_shift)+ 1e-8)

    # normalize frequency spectrum
    frequency_spectrum = frequency_spectrum.astype(np.float32)
    min_val = frequency_spectrum.min()
    max_val = frequency_spectrum.max()
    if max_val > min_val:
        frequency_spectrum = (frequency_spectrum - min_val) / (max_val - min_val)
    else:
        frequency_spectrum = np.zeros_like(frequency_spectrum, dtype=np.float32)

    return spatial_tensor_patch, frequency_spectrum


def folder_processor(folder_path, output_folder):
    print(f"\nScanning folder: {folder_path}")

    search_patterns = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []

    for pattern in search_patterns:
        image_files.extend(glob.glob(os.path.join(folder_path, "**", pattern), recursive=True))

    total = len(image_files)

    if total == 0:
        print("No valid images found")
        return

    print(f"Total images found: {total}")
    print("Starting processing...\n")

    for i, img in enumerate(image_files):
        with open(img, 'rb') as file_stream:
            spatial_patch, frequency_spectrum = image_processor(file_stream)

            if spatial_patch is not None and frequency_spectrum is not None:
                frequency_tensor = torch.tensor(frequency_spectrum, dtype = torch.float32).unsqueeze(0)

                file_name = Path(img).stem + ".pt"
                save_path = os.path.join(output_folder, file_name)

                os.makedirs(output_folder, exist_ok=True)

                torch.save({
                    "spatial_tensor": spatial_patch,
                    "frequency_tensor": frequency_tensor,
                    "original_file": img
                }, save_path)

                # Progress info
                percent = ((i+1) / total) * 100
                print(f"[{i+1}/{total}] {percent:.1f}% - processed: {file_name}")

            else:
                print(f"Failed: {os.path.basename(img)}")

    print(f"\nFinished folder: {folder_path}")


# =========================
# RUN FOR FULL DATASET
# =========================
RAW_ROOT = r"D:\FakeDetection\raw_datasets\image"
OUT_ROOT = r"D:\FakeDetection\processed_datasets\image_tensors"

splits = ["train", "eval", "test"]
classes = ["real", "fake"]

for split in splits:
    for cls in classes:
        in_folder = os.path.join(RAW_ROOT, split, cls)
        out_folder = os.path.join(OUT_ROOT, split, cls)

        if os.path.exists(in_folder):
            print(f"\n==============================")
            print(f"Processing {split}/{cls}")
            print(f"==============================")
            folder_processor(in_folder, out_folder)
        else:
            print(f"Missing folder: {in_folder}")

print("\nAll image preprocessing done.")