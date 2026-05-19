# ============================================================
# Image Preprocessing Script
# Converts raw images into clean 256x256 RGB JPG images.
# Output is used by train_image.py.
# ============================================================

from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm


# =========================
# PATHS
# =========================

SRC_ROOT = Path(r"D:\Images\raw")
DST_ROOT = Path(r"D:\Images\processed")


# =========================
# SETTINGS
# =========================

SPLITS = ["train", "eval", "test"]
CLASSES = ["real", "fake"]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

IMAGE_SIZE = 256
JPEG_QUALITY = 95

CLEAN_OUTPUT = False


# =========================
# HELPERS
# =========================

def create_folder(path):
    path.mkdir(parents=True, exist_ok=True)


def get_image_files(folder):
    if not folder.exists():
        print(f"Missing folder: {folder}")
        return []

    files = []

    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            files.append(file_path)

    return sorted(files)


def center_crop_square(image):
    """
    Crop image to a centered square before resizing.

    This avoids stretching faces/objects.
    """

    width, height = image.size

    if width == height:
        return image

    side = min(width, height)

    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side

    return image.crop((left, top, right, bottom))


def process_image(src_path, dst_path):
    """
    Process one image:
    1. Open image
    2. Fix orientation using EXIF
    3. Convert to RGB
    4. Center crop to square
    5. Resize to 256x256
    6. Save as high-quality JPEG
    """

    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        img = center_crop_square(img)

        img = img.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )

        img.save(
            dst_path,
            "JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
        )


def process_split_class(split, class_name):
    src_folder = SRC_ROOT / split / class_name
    dst_folder = DST_ROOT / split / class_name

    create_folder(dst_folder)

    images = get_image_files(src_folder)

    print(f"\n{split}/{class_name}: found {len(images)} images")

    saved_count = 0
    skipped_count = 0

    for src_path in tqdm(
        images,
        desc=f"Processing {split}/{class_name}",
        unit="image",
    ):
        try:
            saved_count += 1

            filename = f"{class_name}_{saved_count:06d}.jpg"
            dst_path = dst_folder / filename

            process_image(src_path, dst_path)

        except Exception as e:
            saved_count -= 1
            skipped_count += 1
            print(f"Skipped {src_path}: {e}")

    print(f"{split}/{class_name} saved: {saved_count}")
    print(f"{split}/{class_name} skipped: {skipped_count}")


def print_final_counts():
    print("\nFinal processed image counts:")

    for split in SPLITS:
        for class_name in CLASSES:
            folder = DST_ROOT / split / class_name

            if folder.exists():
                count = len([
                    p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() == ".jpg"
                ])
            else:
                count = 0

            print(f"{split}/{class_name}: {count}")


def clean_output_folder():
    """
    Deletes old processed images only if CLEAN_OUTPUT = True.
    """

    if not DST_ROOT.exists():
        return

    for split in SPLITS:
        for class_name in CLASSES:
            folder = DST_ROOT / split / class_name

            if not folder.exists():
                continue

            for file_path in folder.glob("*.jpg"):
                file_path.unlink()


# =========================
# MAIN
# =========================

def main():
    print("Image preprocessing started")
    print(f"Source: {SRC_ROOT}")
    print(f"Output: {DST_ROOT}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    if CLEAN_OUTPUT:
        print("\nCleaning old processed image files...")
        clean_output_folder()

    for split in SPLITS:
        for class_name in CLASSES:
            process_split_class(split, class_name)

    print_final_counts()

    print("\nDONE")
    print(f"Processed images saved to: {DST_ROOT}")


if __name__ == "__main__":
    main()