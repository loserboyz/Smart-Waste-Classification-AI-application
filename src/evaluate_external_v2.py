from pathlib import Path
import json

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
    / "best_mobilenet_v2.pth"
)

TEMPERATURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "mobilenet_v2"
    / "calibration"
    / "temperature.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "mobilenet_v2"
    / "external"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224
BATCH_SIZE = 4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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
    num_workers=0,
    pin_memory=False
)


# =========================================================
# LOAD MODEL
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]


if external_dataset.classes != list(classes):

    raise ValueError(
        "Nama/urutan class external berbeda "
        "dengan checkpoint.\n"
        f"External: {external_dataset.classes}\n"
        f"Model   : {classes}"
    )


model = mobilenet_v3_small(
    weights=None
)

input_features = (
    model.classifier[3].in_features
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
# LOAD TEMPERATURE
# =========================================================

with open(
    TEMPERATURE_PATH,
    "r"
) as file:

    calibration_data = json.load(file)


temperature = calibration_data[
    "temperature"
]


print("=" * 60)
print("MOBILENETV3 V2 — EXTERNAL DIAGNOSTIC")
print("=" * 60)

print(f"\nDevice      : {DEVICE}")
print(f"Images      : {len(external_dataset)}")
print(f"Temperature : {temperature:.4f}")

print("\nClasses:")
print(classes)


# =========================================================
# ECE FUNCTION
# =========================================================

def calculate_ece(
    probabilities,
    labels,
    n_bins=15
):

    confidences, predictions = (
        probabilities.max(dim=1)
    )

    correct = (
        predictions == labels
    ).float()

    boundaries = torch.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = torch.zeros(1)


    for index in range(n_bins):

        lower = boundaries[index]
        upper = boundaries[index + 1]

        mask = (
            confidences.gt(lower)
            & confidences.le(upper)
        )

        proportion = (
            mask.float().mean()
        )


        if proportion.item() > 0:

            bin_accuracy = (
                correct[mask].mean()
            )

            bin_confidence = (
                confidences[mask].mean()
            )

            ece += (
                torch.abs(
                    bin_confidence
                    - bin_accuracy
                )
                * proportion
            )


    return ece.item()


# =========================================================
# INFERENCE
# =========================================================

all_logits = []
all_labels = []


with torch.no_grad():

    for images, labels in external_loader:

        images = images.to(DEVICE)

        logits = model(images)

        all_logits.append(
            logits.cpu()
        )

        all_labels.append(
            labels.cpu()
        )


logits = torch.cat(
    all_logits,
    dim=0
)

labels = torch.cat(
    all_labels,
    dim=0
)


# =========================================================
# RAW / CALIBRATED PROBABILITIES
# =========================================================

raw_probabilities = torch.softmax(
    logits,
    dim=1
)

calibrated_probabilities = torch.softmax(
    logits / temperature,
    dim=1
)


raw_confidence, predictions = (
    raw_probabilities.max(
        dim=1
    )
)

calibrated_confidence, calibrated_predictions = (
    calibrated_probabilities.max(
        dim=1
    )
)


# Temperature scaling seharusnya tidak mengubah argmax
assert torch.equal(
    predictions,
    calibrated_predictions
)


all_true = labels.numpy()
all_pred = predictions.numpy()


# =========================================================
# CLASSIFICATION METRICS
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
# CALIBRATION METRICS
# =========================================================

ece_before = calculate_ece(
    raw_probabilities,
    labels
)

ece_after = calculate_ece(
    calibrated_probabilities,
    labels
)


correct_mask = (
    predictions == labels
)

wrong_mask = ~correct_mask


raw_correct_mean = (
    raw_confidence[
        correct_mask
    ].mean().item()
)

raw_wrong_mean = (
    raw_confidence[
        wrong_mask
    ].mean().item()
)


cal_correct_mean = (
    calibrated_confidence[
        correct_mask
    ].mean().item()
)

cal_wrong_mean = (
    calibrated_confidence[
        wrong_mask
    ].mean().item()
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 60)
print("EXTERNAL CLASSIFICATION METRICS")
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


print("\n" + "=" * 60)
print("CONFIDENCE CALIBRATION")
print("=" * 60)

print(
    f"ECE before calibration : "
    f"{ece_before:.4f}"
)

print(
    f"ECE after calibration  : "
    f"{ece_after:.4f}"
)

print("\nMean confidence:")

print(
    f"Correct — raw        : "
    f"{raw_correct_mean:.4f}"
)

print(
    f"Wrong   — raw        : "
    f"{raw_wrong_mean:.4f}"
)

print(
    f"Correct — calibrated : "
    f"{cal_correct_mean:.4f}"
)

print(
    f"Wrong   — calibrated : "
    f"{cal_wrong_mean:.4f}"
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
    "MobileNetV3 V2 — External Diagnostic"
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

records = []


for index, (
    filepath,
    true_index
) in enumerate(
    external_dataset.samples
):

    predicted_index = int(
        predictions[index]
    )

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

        "raw_confidence":
            float(
                raw_confidence[
                    index
                ]
            ),

        "calibrated_confidence":
            float(
                calibrated_confidence[
                    index
                ]
            ),

        "correct":
            bool(
                predicted_index
                == true_index
            ),
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
# PREDICTED CLASS DISTRIBUTION
# =========================================================

prediction_distribution = (
    predictions_df[
        "predicted_label"
    ]
    .value_counts()
    .reindex(
        classes,
        fill_value=0
    )
    .reset_index()
)

prediction_distribution.columns = [
    "class",
    "predicted_count"
]


prediction_distribution.to_csv(
    RESULTS_DIR
    / "prediction_distribution.csv",
    index=False
)


print("\nPredicted class distribution:")

print(
    prediction_distribution
    .to_string(
        index=False
    )
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics = {

    "model":
        "MobileNetV3-Small-V2",

    "dataset":
        "external_diagnostic",

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

    "temperature":
        float(temperature),

    "ece_before":
        float(ece_before),

    "ece_after":
        float(ece_after),

    "raw_correct_mean_confidence":
        float(raw_correct_mean),

    "raw_wrong_mean_confidence":
        float(raw_wrong_mean),

    "calibrated_correct_mean_confidence":
        float(cal_correct_mean),

    "calibrated_wrong_mean_confidence":
        float(cal_wrong_mean),
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
print("V2 EXTERNAL EVALUATION COMPLETED")
print("=" * 60)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)