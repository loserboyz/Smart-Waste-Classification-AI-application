from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "internet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "external_review"
)

OUTPUT_DIR.mkdir(
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


def create_contact_sheet(class_name):

    class_dir = EXTERNAL_DIR / class_name

    image_files = sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    )

    print(
        f"{class_name}: "
        f"{len(image_files)} images"
    )

    # 20 foto → layout 4 x 5
    rows = 4
    cols = 5

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, 12)
    )

    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for index, image_path in enumerate(image_files):

        if index >= len(axes):
            break

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            axes[index].imshow(image)

            axes[index].set_title(
                image_path.name,
                fontsize=8
            )

            axes[index].axis("off")

        except Exception as error:

            axes[index].set_title(
                f"ERROR\n{image_path.name}",
                fontsize=8
            )

            print(
                f"Cannot open "
                f"{image_path}: {error}"
            )

    fig.suptitle(
        f"External Review — {class_name.upper()}",
        fontsize=18
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"external_{class_name}_review.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":

    print("=" * 60)
    print("EXTERNAL DATASET VISUAL REVIEW")
    print("=" * 60)

    for class_name in CLASSES:
        create_contact_sheet(
            class_name
        )

    print("\nReview sheets completed.")