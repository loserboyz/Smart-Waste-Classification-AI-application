from pathlib import Path
import json
import time

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "internet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_mobilenet.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "mobilenet"
    / "external"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224
BATCH_SIZE = 32


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("MOBILENETV3 — EXTERNAL TEST")
print("=" * 60)

print(f"\nDevice: {DEVICE}")


# =========================================================
# TRANSFORM
# =========================================================

external_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


# =========================================================
# DATASET
# =========================================================

external_dataset = ImageFolder(
    EXTERNAL_DIR,
    transform=external_transform
)

external_loader = DataLoader(
    external_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    f"\nExternal images: "
    f"{len(external_dataset)}"
)

print("\nExternal classes:")
print(external_dataset.classes)


# =========================================================
# LOAD CHECKPOINT
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]


if external_dataset.classes != list(classes):

    raise ValueError(
        "Urutan/nama class external berbeda "
        "dengan checkpoint model.\n"
        f"External : {external_dataset.classes}\n"
        f"Model    : {classes}"
    )


# =========================================================
# BUILD MODEL
# =========================================================

model = mobilenet_v3_small(
    weights=None
)

input_features = (
    model.classifier[3]
    .in_features
)

model.classifier[3] = nn.Linear(
    input_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# =========================================================
# INFERENCE
# =========================================================

all_true = []
all_pred = []
all_confidence = []

total_time = 0.0
total_images = 0


with torch.no_grad():

    for images, labels in external_loader:

        images = images.to(DEVICE)

        start = time.perf_counter()

        outputs = model(images)

        end = time.perf_counter()

        total_time += (
            end - start
        )

        total_images += (
            images.size(0)
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predictions = (
            probabilities.max(
                dim=1
            )
        )

        all_true.extend(
            labels.numpy().tolist()
        )

        all_pred.extend(
            predictions
            .cpu()
            .numpy()
            .tolist()
        )

        all_confidence.extend(
            confidence
            .cpu()
            .numpy()
            .tolist()
        )


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    all_true,
    all_pred
)

macro_precision = precision_score(
    all_true,
    all_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    all_true,
    all_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_true,
    all_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    all_true,
    all_pred,
    average="weighted",
    zero_division=0
)


average_latency_ms = (
    total_time
    / total_images
    * 1000
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 60)
print("EXTERNAL TEST METRICS")
print("=" * 60)

print(
    f"Accuracy        : "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Macro Precision : "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall    : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1        : "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1     : "
    f"{weighted_f1:.4f}"
)

print(
    f"\nAvg inference   : "
    f"{average_latency_ms:.2f} ms/image"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    all_true,
    all_pred,
    target_names=classes,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

report_df.to_csv(
    RESULTS_DIR
    / "classification_report.csv"
)

print("\nPer-class report:")

print(
    report_df[
        [
            "precision",
            "recall",
            "f1-score",
            "support"
        ]
    ].round(4)
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_true,
    all_pred
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

plt.title(
    "MobileNetV3 — External Internet Test"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "confusion_matrix.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# SAVE INDIVIDUAL PREDICTIONS
# =========================================================

records = []

for index, (filepath, true_index) in enumerate(
    external_dataset.samples
):

    predicted_index = all_pred[
        index
    ]

    records.append({

        "filepath":
            filepath,

        "true_label":
            classes[
                true_index
            ],

        "predicted_label":
            classes[
                predicted_index
            ],

        "confidence":
            all_confidence[
                index
            ],

        "correct":
            true_index
            == predicted_index
    })


predictions_df = pd.DataFrame(
    records
)

predictions_df.to_csv(
    RESULTS_DIR
    / "predictions.csv",
    index=False
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics = {

    "model":
        "MobileNetV3-Small",

    "dataset":
        "external_internet",

    "images":
        len(external_dataset),

    "accuracy":
        float(accuracy),

    "macro_precision":
        float(macro_precision),

    "macro_recall":
        float(macro_recall),

    "macro_f1":
        float(macro_f1),

    "weighted_f1":
        float(weighted_f1),

    "avg_inference_ms_per_image":
        float(
            average_latency_ms
        ),
}


with open(
    RESULTS_DIR
    / "metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print("\n" + "=" * 60)
print("EXTERNAL EVALUATION COMPLETED")
print("=" * 60)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)