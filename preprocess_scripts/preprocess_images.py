# ============================================================
# Image Preprocessing Script
# ============================================================

import os
from PIL import Image

SRC_ROOT = r"D:\FakeDetection\raw_datasets\image"
DST_ROOT = r"D:\FakeDetection\processed_datasets\image"

SPLITS = ["train", "eval", "test"]
CLASSES = ["real", "fake"]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = (224, 224)


def create_folder(path):
    os.makedirs(path, exist_ok=True)  # create folder if needed


def get_image_files(folder):
    if not os.path.exists(folder):
        return []

    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        ext = os.path.splitext(name)[1].lower()

        if os.path.isfile(path) and ext in VALID_EXTENSIONS:
            files.append(path)  # valid image

    return sorted(files)


def process_image(src, dst):
    with Image.open(src) as img:
        img = img.convert("RGB")  # ensure RGB
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)  # resize
        img.save(dst, "JPEG", quality=95)  # save


def process_dataset():
    for split in SPLITS:
        for cls in CLASSES:
            src_folder = os.path.join(SRC_ROOT, split, cls)
            dst_folder = os.path.join(DST_ROOT, split, cls)

            create_folder(dst_folder)
            images = get_image_files(src_folder)

            print(f"\n{split}/{cls}: {len(images)} images")

            count = 0
            for i, src_path in enumerate(images, 1):
                try:
                    count += 1
                    filename = f"{cls}_{count:06d}.jpg"  # rename
                    dst_path = os.path.join(dst_folder, filename)

                    process_image(src_path, dst_path)

                    if i % 500 == 0:
                        print(f"Processed {i}/{len(images)}")  # progress

                except Exception as e:
                    count -= 1
                    print(f"Skipped: {e}")  # skip bad file

            print(f"Saved {count} images")


if __name__ == "__main__":
    process_dataset()
    print("\nDone.")