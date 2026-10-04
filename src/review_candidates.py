from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "results" / "image_quality.csv"
OUTPUT_CSV = PROJECT_ROOT / "results" / "review_candidates.csv"


df = pd.read_csv(INPUT_CSV)

# ============================================
# BUAT KOLOM FLAG
# ============================================

df["needs_review"] = False
df["review_reason"] = ""


def add_flag(mask, reason):
    df.loc[mask, "needs_review"] = True

    current = df.loc[mask, "review_reason"]

    df.loc[mask, "review_reason"] = current.apply(
        lambda x: f"{x}; {reason}" if x else reason
    )


# ============================================
# ANALISIS PER SOURCE
# supaya TrashNet dan external tidak
# dibandingkan dengan distribusi yang sama
# ============================================

for source in df["source"].unique():

    source_mask = df["source"] == source
    source_df = df[source_mask]

    # Sharpness terendah 2%
    sharpness_threshold = source_df["sharpness"].quantile(0.02)

    add_flag(
        source_mask
        & (df["sharpness"] <= sharpness_threshold),
        "very_low_sharpness"
    )

    # Brightness paling gelap 1%
    dark_threshold = source_df["brightness"].quantile(0.01)

    add_flag(
        source_mask
        & (df["brightness"] <= dark_threshold),
        "very_dark"
    )

    # Brightness paling terang 1%
    bright_threshold = source_df["brightness"].quantile(0.99)

    add_flag(
        source_mask
        & (df["brightness"] >= bright_threshold),
        "very_bright"
    )


# ============================================
# CHANNEL ANOMALY
# ============================================

add_flag(
    df["channels"] != 3,
    "unexpected_channels"
)


# ============================================
# EXTREME ASPECT RATIO
# ============================================

df["aspect_ratio"] = df["width"] / df["height"]

add_flag(
    (df["aspect_ratio"] < 0.4)
    | (df["aspect_ratio"] > 2.5),
    "extreme_aspect_ratio"
)


# ============================================
# VERY SMALL IMAGE
# ============================================

add_flag(
    (df["width"] < 150)
    | (df["height"] < 150),
    "very_small_resolution"
)


# ============================================
# SAVE
# ============================================

review_df = (
    df[df["needs_review"]]
    .sort_values(
        ["source", "class", "sharpness"]
    )
)

review_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("=" * 60)
print("IMAGE REVIEW CANDIDATES")
print("=" * 60)

print(f"\nTotal dataset : {len(df)}")
print(f"Need review   : {len(review_df)}")

print("\nReason counts:")

print(
    review_df["review_reason"]
    .value_counts()
)

print(
    f"\nSaved to:\n{OUTPUT_CSV}"
)