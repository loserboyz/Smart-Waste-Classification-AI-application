from pathlib import Path
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset-resized"
)

SPLITS_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

SPLITS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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

RANDOM_SEED = 42


# =========================================================
# HASHING
# =========================================================

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# =========================================================
# BUILD DATASET MANIFEST
# =========================================================

def build_manifest():

    records = []

    for class_name in CLASSES:

        class_dir = (
            DATASET_DIR
            / class_name
        )

        if not class_dir.exists():

            raise FileNotFoundError(
                f"Class folder tidak ditemukan: "
                f"{class_dir}"
            )

        image_files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in VALID_EXTENSIONS
        )

        for image_path in image_files:

            relative_path = (
                image_path
                .relative_to(PROJECT_ROOT)
                .as_posix()
            )

            records.append({
                "filepath": relative_path,
                "label": class_name,
                "sha256": calculate_sha256(
                    image_path
                ),
            })

    return pd.DataFrame(records)


# =========================================================
# HANDLE EXACT DUPLICATES
# =========================================================

def remove_exact_duplicates(df):

    clean_records = []
    excluded_records = []

    grouped = df.groupby(
        "sha256",
        sort=False
    )

    for hash_value, group in grouped:

        group = group.sort_values(
            "filepath"
        )

        labels = group[
            "label"
        ].unique()

        # -----------------------------
        # Unique image
        # -----------------------------

        if len(group) == 1:

            clean_records.append(
                group.iloc[0].to_dict()
            )

            continue

        # -----------------------------
        # Duplicate with conflicting labels
        # -----------------------------

        if len(labels) > 1:

            for _, row in group.iterrows():

                record = row.to_dict()

                record["exclusion_reason"] = (
                    "cross_class_duplicate"
                )

                excluded_records.append(
                    record
                )

            continue

        # -----------------------------
        # Same-class duplicate
        # Keep only one
        # -----------------------------

        clean_records.append(
            group.iloc[0].to_dict()
        )

        for _, row in group.iloc[1:].iterrows():

            record = row.to_dict()

            record["exclusion_reason"] = (
                "same_class_exact_duplicate"
            )

            excluded_records.append(
                record
            )

    clean_df = pd.DataFrame(
        clean_records
    )

    excluded_df = pd.DataFrame(
        excluded_records
    )

    return clean_df, excluded_df


# =========================================================
# STRATIFIED SPLIT
# =========================================================

def create_splits(df):

    # 70% train
    # 30% temporary pool

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    # temporary pool dibagi dua:
    # 15% validation
    # 15% test

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# =========================================================
# INTEGRITY CHECK
# =========================================================

def verify_splits(
    train_df,
    val_df,
    test_df,
):

    print("\n" + "=" * 60)
    print("SPLIT INTEGRITY CHECK")
    print("=" * 60)

    train_paths = set(
        train_df["filepath"]
    )

    val_paths = set(
        val_df["filepath"]
    )

    test_paths = set(
        test_df["filepath"]
    )

    assert train_paths.isdisjoint(
        val_paths
    )

    assert train_paths.isdisjoint(
        test_paths
    )

    assert val_paths.isdisjoint(
        test_paths
    )

    train_hashes = set(
        train_df["sha256"]
    )

    val_hashes = set(
        val_df["sha256"]
    )

    test_hashes = set(
        test_df["sha256"]
    )

    assert train_hashes.isdisjoint(
        val_hashes
    )

    assert train_hashes.isdisjoint(
        test_hashes
    )

    assert val_hashes.isdisjoint(
        test_hashes
    )

    print(
        "[OK] Tidak ada filepath overlap."
    )

    print(
        "[OK] Tidak ada exact duplicate "
        "antar split."
    )


# =========================================================
# SUMMARY
# =========================================================

def print_split_summary(
    train_df,
    val_df,
    test_df,
):

    total = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)

    for name, split_df in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:

        percentage = (
            len(split_df)
            / total
            * 100
        )

        print(
            f"\n{name}: "
            f"{len(split_df)} images "
            f"({percentage:.2f}%)"
        )

        print(
            split_df[
                "label"
            ]
            .value_counts()
            .reindex(CLASSES)
        )


def save_summary(
    train_df,
    val_df,
    test_df,
):

    rows = []

    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:

        counts = (
            split_df["label"]
            .value_counts()
        )

        for class_name in CLASSES:

            rows.append({
                "split": split_name,
                "class": class_name,
                "count": int(
                    counts.get(
                        class_name,
                        0
                    )
                ),
            })

    summary_df = pd.DataFrame(
        rows
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "split_summary.csv",
        index=False,
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TRASHNET STRATIFIED SPLIT")
    print("=" * 60)

    # ---------------------------------
    # Build manifest
    # ---------------------------------

    manifest_df = build_manifest()

    print(
        f"\nRaw images found: "
        f"{len(manifest_df)}"
    )

    # ---------------------------------
    # Duplicate handling
    # ---------------------------------

    clean_df, excluded_df = (
        remove_exact_duplicates(
            manifest_df
        )
    )

    print(
        f"Images eligible for modeling: "
        f"{len(clean_df)}"
    )

    print(
        f"Excluded duplicate/conflict: "
        f"{len(excluded_df)}"
    )

    # Save excluded records

    if not excluded_df.empty:

        excluded_df.to_csv(
            RESULTS_DIR
            / "excluded_duplicates.csv",
            index=False,
        )

    # ---------------------------------
    # Split
    # ---------------------------------

    train_df, val_df, test_df = (
        create_splits(
            clean_df
        )
    )

    # ---------------------------------
    # Verification
    # ---------------------------------

    verify_splits(
        train_df,
        val_df,
        test_df,
    )

    # ---------------------------------
    # Save CSV
    # ---------------------------------

    train_df.to_csv(
        SPLITS_DIR
        / "train.csv",
        index=False,
    )

    val_df.to_csv(
        SPLITS_DIR
        / "val.csv",
        index=False,
    )

    test_df.to_csv(
        SPLITS_DIR
        / "test.csv",
        index=False,
    )

    save_summary(
        train_df,
        val_df,
        test_df,
    )

    print_split_summary(
        train_df,
        val_df,
        test_df,
    )

    print("\n" + "=" * 60)
    print("SPLIT COMPLETED")
    print("=" * 60)

    print(
        f"\nSaved to:\n{SPLITS_DIR}"
    )