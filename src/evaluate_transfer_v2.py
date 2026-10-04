from pathlib import Path
import json
import time

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
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

from torch_dataset import WasteDataset


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_mobilenet_v2.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "mobilenet_v2"
    / "test"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224
BATCH_SIZE = 4


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("MOBILENETV3 V2 — INTERNAL TEST")
print("=" * 60)

print(f"\nDevice: {DEVICE}")


# =========================================================
# TEST TRANSFORM
# =========================================================

test_transform = transforms.Compose([

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

test_dataset = WasteDataset(
    TEST_CSV,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print(
    f"\nTest images: "
    f"{len(test_dataset)}"
)


# =========================================================
# LOAD CHECKPOINT
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]

print("\nClasses:")
print(classes)

print(
    f"\nValidation Accuracy : "
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)

print(
    f"Validation Macro F1 : "
    f"{checkpoint['val_macro_f1']:.4f}"
)

print(
    f"Best Stage          : "
    f"{checkpoint['stage']}"
)

print(
    f"Best Epoch          : "
    f"{checkpoint['epoch']}"
)


# =========================================================
# MODEL
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

model = model.to(
    DEVICE
)

model.eval()


# =========================================================
# CLASS ORDER CHECK
# =========================================================

if list(test_dataset.classes) != list(classes):

    raise ValueError(
        "Class order pada test dataset "
        "berbeda dengan checkpoint."
    )


# =========================================================
# INFERENCE
# =========================================================

all_true = []
all_pred = []
all_confidence = []

total_inference_time = 0.0
total_images = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE
        )

        start_time = time.perf_counter()

        outputs = model(
            images
        )

        end_time = time.perf_counter()

        total_inference_time += (
            end_time
            - start_time
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
            predictions.cpu().numpy().tolist()
        )

        all_confidence.extend(
            confidence.cpu().numpy().tolist()
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


# =========================================================
# SPEED & SIZE
# =========================================================

average_latency_ms = (
    total_inference_time
    / total_images
    * 1000
)

model_size_mb = (
    MODEL_PATH.stat().st_size
    / (1024 ** 2)
)


# =========================================================
# PRINT METRICS
# =========================================================

print("\n" + "=" * 60)
print("V2 TEST METRICS")
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
    f"\nModel size      : "
    f"{model_size_mb:.2f} MB"
)

print(
    f"Avg inference   : "
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
    values_format="d",
    colorbar=False
)

plt.title(
    "MobileNetV3 V2 — Internal Test Confusion Matrix"
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
# SAVE PREDICTIONS
# =========================================================

test_df = pd.read_csv(
    TEST_CSV
)

test_df["true_index"] = all_true
test_df["predicted_index"] = all_pred

test_df["predicted_label"] = [
    classes[index]
    for index in all_pred
]

test_df["confidence"] = (
    all_confidence
)

test_df["correct"] = (
    test_df["label"]
    == test_df["predicted_label"]
)

test_df.to_csv(
    RESULTS_DIR
    / "predictions.csv",
    index=False
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics = {

    "model":
        "MobileNetV3-Small-V2",

    "test_images":
        int(len(test_dataset)),

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

    "model_size_mb":
        float(model_size_mb),

    "avg_inference_ms_per_image":
        float(average_latency_ms),
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
print("V2 INTERNAL EVALUATION COMPLETED")
print("=" * 60)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)