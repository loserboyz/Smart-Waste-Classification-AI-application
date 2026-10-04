from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD RESULTS
# =========================================================

df = pd.read_csv(
    PREDICTIONS_CSV
)

wrong_df = df[
    df["correct"] == False
].copy()


print("=" * 60)
print("BASELINE ERROR ANALYSIS")
print("=" * 60)

print(
    f"\nTotal predictions : {len(df)}"
)

print(
    f"Correct           : {df['correct'].sum()}"
)

print(
    f"Wrong             : {len(wrong_df)}"
)


# =========================================================
# MOST COMMON CONFUSIONS
# =========================================================

confusions = (
    wrong_df
    .groupby(
        [
            "label",
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


print("\nMost common confusions:")

print(
    confusions
    .head(15)
    .to_string(index=False)
)


confusions.to_csv(
    OUTPUT_DIR
    / "confusion_pairs.csv",
    index=False
)


# =========================================================
# HIGHEST-CONFIDENCE WRONG PREDICTIONS
# =========================================================

wrong_df = wrong_df.sort_values(
    "confidence",
    ascending=False
)

top_wrong = wrong_df.head(20)


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
    top_wrong.iterrows()
):

    if index >= len(axes):
        break

    image_path = (
        PROJECT_ROOT
        / row["filepath"]
    )

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        axes[index].imshow(
            image
        )

        axes[index].set_title(
            f"True: {row['label']}\n"
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
    "Baseline CNN — High-Confidence Errors",
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
# ERRORS PREDICTED AS TRASH
# =========================================================

trash_errors = wrong_df[
    wrong_df["predicted_label"]
    == "trash"
].head(20)


fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(16, 13)
)

axes = axes.flatten()


for ax in axes:
    ax.axis("off")


for index, (_, row) in enumerate(
    trash_errors.iterrows()
):

    if index >= len(axes):
        break

    image_path = (
        PROJECT_ROOT
        / row["filepath"]
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    axes[index].imshow(
        image
    )

    axes[index].set_title(
        f"True: {row['label']}\n"
        f"Pred: trash\n"
        f"Conf: {row['confidence']:.2f}",
        fontsize=9
    )

    axes[index].axis("off")


fig.suptitle(
    "Baseline CNN — False Trash Predictions",
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "false_trash_predictions.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "\nError analysis saved to:"
)

print(
    OUTPUT_DIR
)