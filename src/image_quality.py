from pathlib import Path

import cv2
import pandas as pd


# =========================================================
# PATH CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRASHNET_DIR = PROJECT_ROOT / "data" / "raw" / "dataset-resized"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external" / "internet"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "image_quality.csv"


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
# IMAGE QUALITY FUNCTIONS
# =========================================================

def calculate_brightness(image):
    """
    Menghitung rata-rata brightness berdasarkan grayscale.
    Range secara umum: 0-255.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(gray.mean())


def calculate_sharpness(image):
    """
    Menggunakan variance of Laplacian.

    Nilai yang lebih tinggi biasanya menunjukkan
    gambar yang lebih tajam.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return float(laplacian.var())


def inspect_image(image_path, source, class_name):
    """
    Membaca sebuah gambar dan mengambil metadata
    serta indikator kualitas.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    height, width = image.shape[:2]

    channels = (
        image.shape[2]
        if len(image.shape) == 3
        else 1
    )

    brightness = calculate_brightness(image)
    sharpness = calculate_sharpness(image)

    return {
        "source": source,
        "class": class_name,
        "filename": image_path.name,
        "filepath": str(image_path),
        "width": width,
        "height": height,
        "channels": channels,
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
        "file_size_kb": round(
            image_path.stat().st_size / 1024,
            2
        ),
    }


# =========================================================
# DATASET SCANNER
# =========================================================

def scan_dataset(dataset_dir, source_name):
    records = []

    print("\n" + "=" * 60)
    print(f"SCANNING: {source_name}")
    print("=" * 60)

    for class_name in EXPECTED_CLASSES:

        class_dir = dataset_dir / class_name

        if not class_dir.exists():
            print(
                f"[WARNING] Folder tidak ditemukan: "
                f"{class_dir}"
            )
            continue

        image_files = sorted(
            file
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        )

        print(
            f"{class_name:<12}: "
            f"{len(image_files)} images"
        )

        for image_path in image_files:

            record = inspect_image(
                image_path=image_path,
                source=source_name,
                class_name=class_name,
            )

            if record is not None:
                records.append(record)

    return records


# =========================================================
# SUMMARY
# =========================================================

def print_summary(df):

    print("\n" + "=" * 60)
    print("IMAGE QUALITY SUMMARY")
    print("=" * 60)

    print("\nJumlah gambar berdasarkan source:")
    print(df["source"].value_counts())

    print("\nBrightness summary:")
    print(
        df.groupby("source")["brightness"]
        .describe()
        .round(2)
    )

    print("\nSharpness summary:")
    print(
        df.groupby("source")["sharpness"]
        .describe()
        .round(2)
    )

    print("\nImage width summary:")
    print(
        df.groupby("source")["width"]
        .describe()
        .round(2)
    )

    print("\nImage height summary:")
    print(
        df.groupby("source")["height"]
        .describe()
        .round(2)
    )


def show_low_quality_candidates(df):

    print("\n" + "=" * 60)
    print("LOWEST SHARPNESS CANDIDATES")
    print("=" * 60)

    columns = [
        "source",
        "class",
        "filename",
        "sharpness",
    ]

    print(
        df.sort_values("sharpness")
        [columns]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("DARKEST IMAGE CANDIDATES")
    print("=" * 60)

    columns = [
        "source",
        "class",
        "filename",
        "brightness",
    ]

    print(
        df.sort_values("brightness")
        [columns]
        .head(10)
        .to_string(index=False)
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    all_records = []

    trashnet_records = scan_dataset(
        TRASHNET_DIR,
        "trashnet",
    )

    external_records = scan_dataset(
        EXTERNAL_DIR,
        "external",
    )

    all_records.extend(trashnet_records)
    all_records.extend(external_records)

    df = pd.DataFrame(all_records)

    if df.empty:
        raise RuntimeError(
            "Tidak ada gambar yang berhasil dianalisis."
        )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print_summary(df)

    show_low_quality_candidates(df)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)

    print(
        f"\nFile hasil disimpan di:\n{OUTPUT_CSV}"
    )