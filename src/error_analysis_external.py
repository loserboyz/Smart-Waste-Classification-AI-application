from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "results"
    / "mobilenet"
    / "external"
    / "predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "mobilenet"
    / "external"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD PREDICTIONS
# =========================================================

df = pd.read_csv(
    PREDICTIONS_CSV
)

wrong_df = df[
    df["correct"] == False
].copy()


print("=" * 60)
print("EXTERNAL ERROR ANALYSIS")
print("=" * 60)

print(
    f"\nTotal images : {len(df)}"
)

print(
    f"Correct      : {df['correct'].sum()}"
)

print(
    f"Wrong        : {len(wrong_df)}"
)


# =========================================================
# CONFUSION PAIRS
# =========================================================

confusions = (
    wrong_df
    .groupby(
        [
            "true_label",
            "predicted_label"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        "count",
        ascending=False
    )
)


print("\nMost common confusion pairs:")

print(
    confusions
    .head(20)
    .to_string(index=False)
)


confusions.to_csv(
    OUTPUT_DIR
    / "confusion_pairs.csv",
    index=False
)


# =========================================================
# HIGH-CONFIDENCE ERRORS
# =========================================================

high_conf_errors = (
    wrong_df
    .sort_values(
        "confidence",
        ascending=False
    )
    .head(20)
)


rows = 4
cols = 5

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(16, 13)
)

axes = axes.flatten()


for ax in axes:
    ax.axis("off")


for index, (_, row) in enumerate(
    high_conf_errors.iterrows()
):

    image_path = Path(
        row["filepath"]
    )

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        axes[index].imshow(
            image
        )

        axes[index].set_title(
            f"True: {row['true_label']}\n"
            f"Pred: {row['predicted_label']}\n"
            f"Conf: {row['confidence']:.2f}",
            fontsize=9
        )

        axes[index].axis("off")

    except Exception as error:

        axes[index].set_title(
            f"ERROR\n{image_path.name}"
        )

        print(
            f"Cannot open {image_path}: "
            f"{error}"
        )


fig.suptitle(
    "MobileNetV3 — External High-Confidence Errors",
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "high_confidence_errors.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# LOW-CONFIDENCE PREDICTIONS
# =========================================================

low_conf = (
    df
    .sort_values(
        "confidence"
    )
    .head(20)
)


fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(16, 13)
)

axes = axes.flatten()


for ax in axes:
    ax.axis("off")


for index, (_, row) in enumerate(
    low_conf.iterrows()
):

    image_path = Path(
        row["filepath"]
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    axes[index].imshow(
        image
    )

    axes[index].set_title(
        f"True: {row['true_label']}\n"
        f"Pred: {row['predicted_label']}\n"
        f"Conf: {row['confidence']:.2f}",
        fontsize=9
    )

    axes[index].axis("off")


fig.suptitle(
    "MobileNetV3 — Lowest Confidence External Predictions",
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "low_confidence_predictions.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# CONFIDENCE STATISTICS
# =========================================================

print("\nConfidence statistics:")

print(
    df.groupby("correct")["confidence"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max"
        ]
    )
    .round(4)
)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)