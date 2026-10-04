from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

from sklearn.metrics import f1_score
from tqdm import tqdm

from torch_dataset import WasteDataset


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "val.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "mobilenet"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "best_mobilenet.pth"
)

BATCH_SIZE = 16
IMAGE_SIZE = 224

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 5

STAGE1_LR = 0.001
STAGE2_LR = 0.0001


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("MOBILENETV3 TRANSFER LEARNING")
print("=" * 60)

print(f"\nDevice: {DEVICE}")


# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
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


val_transform = transforms.Compose([

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

train_dataset = WasteDataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = WasteDataset(
    VAL_CSV,
    transform=val_transform
)

classes = train_dataset.classes

num_classes = len(classes)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("\nClasses:")
print(classes)

print(
    f"\nTraining images   : "
    f"{len(train_dataset)}"
)

print(
    f"Validation images : "
    f"{len(val_dataset)}"
)


# =========================================================
# CLASS WEIGHTS
# =========================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

class_counts = (
    train_df["label"]
    .value_counts()
    .reindex(classes)
)

total_samples = len(
    train_df
)


weights = []

for class_name in classes:

    count = class_counts[
        class_name
    ]

    weight = (
        total_samples
        / (
            num_classes
            * count
        )
    )

    weights.append(
        weight
    )


class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(DEVICE)


print("\nClass weights:")

for class_name, weight in zip(
    classes,
    weights
):

    print(
        f"{class_name:<12}: "
        f"{weight:.4f}"
    )


# =========================================================
# LOAD PRETRAINED MOBILENET
# =========================================================

print(
    "\nLoading pretrained "
    "MobileNetV3 Small..."
)

weights_pretrained = (
    MobileNet_V3_Small_Weights.DEFAULT
)

model = mobilenet_v3_small(
    weights=weights_pretrained
)


# =========================================================
# REPLACE CLASSIFIER
# =========================================================

input_features = (
    model.classifier[3]
    .in_features
)

model.classifier[3] = nn.Linear(
    input_features,
    num_classes
)

model = model.to(
    DEVICE
)


# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# =========================================================
# TRAIN / VALIDATE FUNCTIONS
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )


    for images, labels in progress:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions
            == labels
        ).sum().item()


        total += labels.size(0)


        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    epoch_loss = (
        running_loss
        / total
    )

    epoch_accuracy = (
        correct
        / total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


def validate(
    model,
    loader,
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_true = []
    all_pred = []


    with torch.no_grad():

        progress = tqdm(
            loader,
            desc="Validation",
            leave=False
        )


        for images, labels in progress:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions
                == labels
            ).sum().item()


            total += labels.size(0)


            all_true.extend(
                labels.cpu().numpy()
            )

            all_pred.extend(
                predictions.cpu().numpy()
            )


    epoch_loss = (
        running_loss
        / total
    )

    epoch_accuracy = (
        correct
        / total
    )


    macro_f1 = f1_score(
        all_true,
        all_pred,
        average="macro",
        zero_division=0
    )


    return (
        epoch_loss,
        epoch_accuracy,
        macro_f1
    )


# =========================================================
# TRAINING HISTORY
# =========================================================

history = []

best_val_f1 = 0.0


# =========================================================
# STAGE 1
# FREEZE BACKBONE
# =========================================================

print("\n" + "=" * 60)
print("STAGE 1 — TRAIN CLASSIFIER")
print("=" * 60)


for param in model.features.parameters():

    param.requires_grad = False


optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=STAGE1_LR
)


for epoch in range(
    STAGE1_EPOCHS
):

    print(
        f"\nStage 1 — "
        f"Epoch {epoch + 1}/"
        f"{STAGE1_EPOCHS}"
    )


    train_loss, train_acc = (
        train_one_epoch(
            model,
            train_loader,
            optimizer
        )
    )


    val_loss, val_acc, val_f1 = (
        validate(
            model,
            val_loader
        )
    )


    print(
        f"Train Loss : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_acc * 100:.2f}%"
    )

    print(
        f"Val Loss   : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_acc * 100:.2f}%"
    )

    print(
        f"Val Macro F1: "
        f"{val_f1:.4f}"
    )


    history.append({

        "stage": 1,

        "epoch":
            epoch + 1,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_acc,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_acc,

        "val_macro_f1":
            val_f1,
    })


    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save({

            "model_state_dict":
                model.state_dict(),

            "classes":
                classes,

            "architecture":
                "mobilenet_v3_small",

            "image_size":
                IMAGE_SIZE,

            "val_accuracy":
                val_acc,

            "val_macro_f1":
                val_f1,

            "stage":
                1,

        }, MODEL_PATH)


        print(
            "[SAVED] New best model"
        )


# =========================================================
# STAGE 2
# FINE-TUNE LAST FEATURE BLOCKS
# =========================================================

print("\n" + "=" * 60)
print("STAGE 2 — FINE-TUNING")
print("=" * 60)


# backbone tetap frozen
for param in model.features.parameters():

    param.requires_grad = False


# buka 3 block terakhir
for param in model.features[-3:].parameters():

    param.requires_grad = True


# classifier tetap trainable
for param in model.classifier.parameters():

    param.requires_grad = True


optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=STAGE2_LR
)


for epoch in range(
    STAGE2_EPOCHS
):

    print(
        f"\nStage 2 — "
        f"Epoch {epoch + 1}/"
        f"{STAGE2_EPOCHS}"
    )


    train_loss, train_acc = (
        train_one_epoch(
            model,
            train_loader,
            optimizer
        )
    )


    val_loss, val_acc, val_f1 = (
        validate(
            model,
            val_loader
        )
    )


    print(
        f"Train Loss : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_acc * 100:.2f}%"
    )

    print(
        f"Val Loss   : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_acc * 100:.2f}%"
    )

    print(
        f"Val Macro F1: "
        f"{val_f1:.4f}"
    )


    history.append({

        "stage": 2,

        "epoch":
            STAGE1_EPOCHS
            + epoch
            + 1,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_acc,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_acc,

        "val_macro_f1":
            val_f1,
    })


    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save({

            "model_state_dict":
                model.state_dict(),

            "classes":
                classes,

            "architecture":
                "mobilenet_v3_small",

            "image_size":
                IMAGE_SIZE,

            "val_accuracy":
                val_acc,

            "val_macro_f1":
                val_f1,

            "stage":
                2,

        }, MODEL_PATH)


        print(
            "[SAVED] New best model"
        )


# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

history_df = pd.DataFrame(
    history
)

history_path = (
    RESULTS_DIR
    / "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# =========================================================
# TRAINING CURVES
# =========================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "MobileNetV3 Training Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "training_accuracy.png",
    dpi=150
)

plt.close()


plt.figure(
    figsize=(9, 5)
)

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    marker="o",
    label="Train Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "MobileNetV3 Training Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "training_loss.png",
    dpi=150
)

plt.close()


# =========================================================
# FINISH
# =========================================================

print("\n" + "=" * 60)
print("TRANSFER LEARNING COMPLETED")
print("=" * 60)

print(
    f"\nBest validation Macro F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"\nBest model saved to:\n"
    f"{MODEL_PATH}"
)

print(
    f"\nTraining results saved to:\n"
    f"{RESULTS_DIR}"
)