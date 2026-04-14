import os
from PIL import Image

# =========================
# PATHS
# =========================
SRC_ROOT = r"D:\FakeDetection\raw_datasets\image"
DST_ROOT = r"D:\FakeDetection\processed_datasets\image"

SPLITS = ["train", "eval", "test"]
CLASSES = ["real", "fake"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMG_SIZE = (224, 224)


def ensure_folder(path):
    os.makedirs(path, exist_ok=True)


def get_image_files(folder):
    files = []
    if not os.path.exists(folder):
        return files

    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        ext = os.path.splitext(f)[1].lower()
        if os.path.isfile(full_path) and ext in VALID_EXTS:
            files.append(full_path)

    return sorted(files)


def process_and_save(src_path, dst_path):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img.save(dst_path, "JPEG", quality=95)


for split in SPLITS:
    for class_name in CLASSES:
        src_folder = os.path.join(SRC_ROOT, split, class_name)
        dst_folder = os.path.join(DST_ROOT, split, class_name)

        ensure_folder(dst_folder)

        image_files = get_image_files(src_folder)
        print(f"\nProcessing {split}/{class_name}...")
        print(f"Found {len(image_files)} images")

        count = 1
        for i, src_path in enumerate(image_files, start=1):
            try:
                new_name = f"{class_name}_{count:06d}.jpg"
                dst_path = os.path.join(dst_folder, new_name)

                process_and_save(src_path, dst_path)
                count += 1

                if i % 500 == 0:
                    print(f"Processed {i}/{len(image_files)}")

            except Exception as e:
                print(f"Skipped {src_path} بسبب error: {e}")

        print(f"Done {split}/{class_name}")
        print(f"Saved {count - 1} images to {dst_folder}")

print("\nDone. All images resized and saved.")