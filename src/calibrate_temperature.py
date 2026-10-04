from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from torch_dataset import WasteDataset


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "val.csv"
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
    / "calibration"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CALIBRATION_PATH = (
    RESULTS_DIR
    / "temperature.json"
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

val_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================================================
# DATASET
# =========================================================

val_dataset = WasteDataset(
    VAL_CSV,
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
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
# COLLECT VALIDATION LOGITS
# =========================================================

all_logits = []
all_labels = []


print("=" * 60)
print("TEMPERATURE CALIBRATION")
print("=" * 60)

print(f"\nDevice: {DEVICE}")
print(f"Validation images: {len(val_dataset)}")


with torch.no_grad():

    for images, labels in val_loader:

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


print(
    f"Logits shape: "
    f"{tuple(logits.shape)}"
)


# =========================================================
# EXPECTED CALIBRATION ERROR
# =========================================================

def calculate_ece(
    logits,
    labels,
    temperature=1.0,
    n_bins=15
):

    scaled_logits = (
        logits
        / temperature
    )

    probabilities = torch.softmax(
        scaled_logits,
        dim=1
    )

    confidences, predictions = (
        probabilities.max(
            dim=1
        )
    )

    accuracies = (
        predictions
        == labels
    ).float()


    bin_boundaries = torch.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = torch.zeros(1)


    for i in range(n_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        in_bin = (
            confidences.gt(lower)
            & confidences.le(upper)
        )

        proportion = (
            in_bin.float().mean()
        )


        if proportion.item() > 0:

            accuracy_in_bin = (
                accuracies[in_bin]
                .mean()
            )

            confidence_in_bin = (
                confidences[in_bin]
                .mean()
            )

            ece += (
                torch.abs(
                    confidence_in_bin
                    - accuracy_in_bin
                )
                * proportion
            )


    return ece.item()


# =========================================================
# RELIABILITY DATA
# =========================================================

def reliability_data(
    logits,
    labels,
    temperature,
    n_bins=10
):

    probabilities = torch.softmax(
        logits / temperature,
        dim=1
    )

    confidence, prediction = (
        probabilities.max(
            dim=1
        )
    )

    correct = (
        prediction == labels
    ).float()


    bins = np.linspace(
        0,
        1,
        n_bins + 1
    )


    bin_confidence = []
    bin_accuracy = []


    confidence_np = confidence.numpy()
    correct_np = correct.numpy()


    for i in range(n_bins):

        lower = bins[i]
        upper = bins[i + 1]

        mask = (
            (confidence_np > lower)
            & (confidence_np <= upper)
        )


        if mask.sum() == 0:

            bin_confidence.append(
                np.nan
            )

            bin_accuracy.append(
                np.nan
            )

            continue


        bin_confidence.append(
            confidence_np[mask].mean()
        )

        bin_accuracy.append(
            correct_np[mask].mean()
        )


    return (
        bin_confidence,
        bin_accuracy
    )


# =========================================================
# METRICS BEFORE CALIBRATION
# =========================================================

criterion = nn.CrossEntropyLoss()


nll_before = criterion(
    logits,
    labels
).item()


ece_before = calculate_ece(
    logits,
    labels,
    temperature=1.0
)


print("\nBefore calibration:")

print(
    f"NLL : {nll_before:.4f}"
)

print(
    f"ECE : {ece_before:.4f}"
)


# =========================================================
# LEARN TEMPERATURE
# =========================================================

# Kita optimise log(T), sehingga T selalu positif.
log_temperature = torch.nn.Parameter(
    torch.zeros(1)
)


optimizer = torch.optim.LBFGS(
    [log_temperature],
    lr=0.1,
    max_iter=100
)


def closure():

    optimizer.zero_grad()

    temperature = torch.exp(
        log_temperature
    )

    loss = criterion(
        logits / temperature,
        labels
    )

    loss.backward()

    return loss


optimizer.step(
    closure
)


temperature = (
    torch.exp(
        log_temperature
    )
    .detach()
    .item()
)


# =========================================================
# METRICS AFTER CALIBRATION
# =========================================================

nll_after = criterion(
    logits / temperature,
    labels
).item()


ece_after = calculate_ece(
    logits,
    labels,
    temperature=temperature
)


print("\nOptimized temperature:")

print(
    f"T = {temperature:.4f}"
)


print("\nAfter calibration:")

print(
    f"NLL : {nll_after:.4f}"
)

print(
    f"ECE : {ece_after:.4f}"
)


# =========================================================
# RELIABILITY DIAGRAM
# =========================================================

before_conf, before_acc = (
    reliability_data(
        logits,
        labels,
        temperature=1.0
    )
)

after_conf, after_acc = (
    reliability_data(
        logits,
        labels,
        temperature=temperature
    )
)


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.plot(
    before_conf,
    before_acc,
    marker="o",
    label="Before calibration"
)

plt.plot(
    after_conf,
    after_acc,
    marker="o",
    label="After calibration"
)

plt.xlabel(
    "Mean confidence"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "MobileNetV3 V2 — Reliability Diagram"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "reliability_diagram.png",
    dpi=150
)

plt.close()


# =========================================================
# SAVE CALIBRATION
# =========================================================

calibration_results = {

    "temperature":
        float(temperature),

    "nll_before":
        float(nll_before),

    "nll_after":
        float(nll_after),

    "ece_before":
        float(ece_before),

    "ece_after":
        float(ece_after),

    "validation_images":
        len(val_dataset),
}


with open(
    CALIBRATION_PATH,
    "w"
) as file:

    json.dump(
        calibration_results,
        file,
        indent=4
    )


print("\n" + "=" * 60)
print("CALIBRATION COMPLETED")
print("=" * 60)

print(
    f"\nSaved to:\n"
    f"{CALIBRATION_PATH}"
)

print(
    f"\nReliability diagram:\n"
    f"{RESULTS_DIR / 'reliability_diagram.png'}"
)