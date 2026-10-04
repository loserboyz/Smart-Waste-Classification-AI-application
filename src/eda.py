from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRASHNET_DIR = PROJECT_ROOT / "data" / "raw" / "dataset-resized"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external" / "internet"

QUALITY_CSV = PROJECT_ROOT / "results" / "image_quality.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
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
# CLASS DISTRIBUTION
# =========================================================

def count_images(dataset_dir):
    counts = {}

    for class_name in CLASSES:

        class_dir = dataset_dir / class_name

        images = [
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        ]

        counts[class_name] = len(images)

    return counts


def plot_class_distribution():

    trashnet_counts = count_images(TRASHNET_DIR)
    external_counts = count_images(EXTERNAL_DIR)

    df = pd.DataFrame({
        "class": CLASSES,
        "trashnet": [
            trashnet_counts[c]
            for c in CLASSES
        ],
        "external": [
            external_counts[c]
            for c in CLASSES
        ],
    })

    # TrashNet
    plt.figure(figsize=(9, 5))

    plt.bar(
        df["class"],
        df["trashnet"]
    )

    plt.title("TrashNet Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "class_distribution_trashnet.png",
        dpi=150
    )

    plt.close()

    # External
    plt.figure(figsize=(9, 5))

    plt.bar(
        df["class"],
        df["external"]
    )

    plt.title("External Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "class_distribution_external.png",
        dpi=150
    )

    plt.close()


# =========================================================
# BRIGHTNESS DISTRIBUTION
# =========================================================

def plot_brightness(df):

    plt.figure(figsize=(10, 5))

    for source in df["source"].unique():

        subset = df[
            df["source"] == source
        ]

        plt.hist(
            subset["brightness"],
            bins=30,
            alpha=0.5,
            label=source
        )

    plt.title("Brightness Distribution")
    plt.xlabel("Mean Brightness")
    plt.ylabel("Frequency")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "brightness_distribution.png",
        dpi=150
    )

    plt.close()


# =========================================================
# SHARPNESS DISTRIBUTION
# =========================================================

def plot_sharpness(df):

    plt.figure(figsize=(10, 5))

    for source in df["source"].unique():

        subset = df[
            df["source"] == source
        ]

        # clip extreme outliers hanya untuk visualisasi
        max_value = subset["sharpness"].quantile(0.98)

        values = subset[
            subset["sharpness"] <= max_value
        ]["sharpness"]

        plt.hist(
            values,
            bins=30,
            alpha=0.5,
            label=source
        )

    plt.title("Sharpness Distribution")
    plt.xlabel("Variance of Laplacian")
    plt.ylabel("Frequency")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "sharpness_distribution.png",
        dpi=150
    )

    plt.close()


# =========================================================
# RESOLUTION DISTRIBUTION
# =========================================================

def plot_resolution(df):

    plt.figure(figsize=(8, 6))

    for source in df["source"].unique():

        subset = df[
            df["source"] == source
        ]

        plt.scatter(
            subset["width"],
            subset["height"],
            alpha=0.4,
            label=source
        )

    plt.title("Image Resolution Distribution")
    plt.xlabel("Width")
    plt.ylabel("Height")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "resolution_distribution.png",
        dpi=150
    )

    plt.close()


# =========================================================
# SAMPLE IMAGES
# =========================================================

def create_sample_grid():

    rows = 2
    cols = 6

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(18, 7)
    )

    for col, class_name in enumerate(CLASSES):

        # TrashNet sample
        trashnet_files = sorted(
            path
            for path in (TRASHNET_DIR / class_name).iterdir()
            if path.suffix.lower() in VALID_EXTENSIONS
        )

        external_files = sorted(
            path
            for path in (EXTERNAL_DIR / class_name).iterdir()
            if path.suffix.lower() in VALID_EXTENSIONS
        )

        trashnet_image = cv2.imread(
            str(trashnet_files[0])
        )

        external_image = cv2.imread(
            str(external_files[0])
        )

        trashnet_image = cv2.cvtColor(
            trashnet_image,
            cv2.COLOR_BGR2RGB
        )

        external_image = cv2.cvtColor(
            external_image,
            cv2.COLOR_BGR2RGB
        )

        axes[0, col].imshow(
            trashnet_image
        )

        axes[0, col].set_title(
            f"TrashNet\n{class_name}"
        )

        axes[0, col].axis("off")

        axes[1, col].imshow(
            external_image
        )

        axes[1, col].set_title(
            f"External\n{class_name}"
        )

        axes[1, col].axis("off")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "sample_images.png",
        dpi=150
    )

    plt.close()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(
        QUALITY_CSV
    )

    plot_class_distribution()
    print("[OK] Class distribution")

    plot_brightness(df)
    print("[OK] Brightness distribution")

    plot_sharpness(df)
    print("[OK] Sharpness distribution")

    plot_resolution(df)
    print("[OK] Resolution distribution")

    create_sample_grid()
    print("[OK] Sample images")

    print("\nEDA completed.")
    print(
        f"Results saved to:\n{RESULTS_DIR}"
    )