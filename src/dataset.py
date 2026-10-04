from pathlib import Path
from collections import defaultdict
import hashlib
import cv2


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRASHNET_DIR = PROJECT_ROOT / "data" / "raw" / "dataset-resized"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external" / "internet"

EXPECTED_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".jfif",
}


# =========================================================
# UTILITIES
# =========================================================

def calculate_file_hash(file_path):
    """Calculate SHA256 hash for exact duplicate detection."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def get_image_files(folder):
    return sorted(
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    )


# =========================================================
# DATASET INSPECTION
# =========================================================

def inspect_dataset(dataset_name, dataset_dir):
    print("\n" + "=" * 60)
    print(dataset_name.upper())
    print("=" * 60)

    if not dataset_dir.exists():
        print(f"[ERROR] Folder tidak ditemukan:")
        print(dataset_dir)
        return None

    existing_classes = sorted(
        folder.name
        for folder in dataset_dir.iterdir()
        if folder.is_dir()
    )

    missing_classes = sorted(
        set(EXPECTED_CLASSES) - set(existing_classes)
    )

    unexpected_classes = sorted(
        set(existing_classes) - set(EXPECTED_CLASSES)
    )

    if missing_classes:
        print("\n[WARNING] Missing classes:")
        for class_name in missing_classes:
            print(f"  - {class_name}")

    if unexpected_classes:
        print("\n[WARNING] Unexpected classes:")
        for class_name in unexpected_classes:
            print(f"  - {class_name}")

    total_images = 0
    corrupted_images = []
    class_counts = {}

    hashes = defaultdict(list)

    print("\nClass distribution:")
    print("-" * 35)

    for class_name in EXPECTED_CLASSES:

        class_dir = dataset_dir / class_name

        if not class_dir.exists():
            class_counts[class_name] = 0
            continue

        image_files = get_image_files(class_dir)

        class_counts[class_name] = len(image_files)
        total_images += len(image_files)

        print(f"{class_name:<12}: {len(image_files)}")

        for image_path in image_files:

            # ---------------------------------------------
            # OpenCV image validation
            # ---------------------------------------------

            image = cv2.imread(str(image_path))

            if image is None:
                corrupted_images.append(image_path)
                continue

            # ---------------------------------------------
            # Exact duplicate detection
            # ---------------------------------------------

            file_hash = calculate_file_hash(image_path)

            hashes[file_hash].append(image_path)

    print("-" * 35)
    print(f"{'TOTAL':<12}: {total_images}")

    print("\nImage validation:")

    if corrupted_images:
        print(f"[WARNING] {len(corrupted_images)} unreadable images")

        for image in corrupted_images:
            print(f"  - {image}")

    else:
        print("[OK] Semua gambar dapat dibaca OpenCV.")

    # ---------------------------------------------
    # Duplicate files inside dataset
    # ---------------------------------------------

    duplicates = {
        hash_value: paths
        for hash_value, paths in hashes.items()
        if len(paths) > 1
    }

    print("\nExact duplicates:")

    if duplicates:
        print(f"[WARNING] Ditemukan {len(duplicates)} grup duplicate.")

        for paths in duplicates.values():

            print("\nDuplicate group:")

            for path in paths:
                print(f"  - {path}")

    else:
        print("[OK] Tidak ditemukan exact duplicate.")

    return {
        "counts": class_counts,
        "total": total_images,
        "corrupted": corrupted_images,
        "hashes": hashes,
    }


# =========================================================
# CROSS DATASET LEAKAGE CHECK
# =========================================================

def check_cross_dataset_duplicates(trashnet_result, external_result):

    print("\n" + "=" * 60)
    print("CROSS-DATASET DUPLICATE CHECK")
    print("=" * 60)

    trashnet_hashes = trashnet_result["hashes"]
    external_hashes = external_result["hashes"]

    common_hashes = (
        set(trashnet_hashes.keys())
        & set(external_hashes.keys())
    )

    if not common_hashes:

        print(
            "[OK] Tidak ditemukan exact duplicate antara "
            "TrashNet dan external dataset."
        )

        return

    print(
        f"[WARNING] Ditemukan {len(common_hashes)} "
        "gambar external yang identik dengan TrashNet."
    )

    for hash_value in common_hashes:

        print("\nTrashNet:")

        for path in trashnet_hashes[hash_value]:
            print(f"  - {path}")

        print("External:")

        for path in external_hashes[hash_value]:
            print(f"  - {path}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    trashnet_result = inspect_dataset(
        "TrashNet Training Dataset",
        TRASHNET_DIR,
    )

    external_result = inspect_dataset(
        "External Internet Dataset",
        EXTERNAL_DIR,
    )

    if trashnet_result and external_result:

        check_cross_dataset_duplicates(
            trashnet_result,
            external_result,
        )