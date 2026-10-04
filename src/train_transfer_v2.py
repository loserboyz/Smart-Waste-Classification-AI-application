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

TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "splits" / "val.csv"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "mobilenet_v2"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_mobilenet_v2.pth"


# =========================================================
# HYPERPARAMETERS
# =========================================================

BATCH_SIZE = 4
IMAGE_SIZE = 224

STAGE1_EPOCHS = 4
STAGE2_EPOCHS = 4
STAGE3_EPOCHS = 3

STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
STAGE3_LR = 1e-5

WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

RANDOM_SEED = 42


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("MOBILENETV3 V2 — ROBUST TRANSFER LEARNING")
print("=" * 60)

print(f"\nDevice: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")


# =========================================================
# RANDOM SEED
# =========================================================

torch.manual_seed(RANDOM_SEED)


# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.65, 1.0),
        ratio=(0.75, 1.33),
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.RandomPerspective(
        distortion_scale=0.15,
        p=0.30
    ),

    transforms.ColorJitter(
        brightness=0.30,
        contrast=0.30,
        saturation=0.30,
        hue=0.05
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
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
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
        /
        (
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


print("\nClass distribution:")
print(class_counts)

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
# PRETRAINED MOBILENET
# =========================================================

print(
    "\nLoading pretrained "
    "MobileNetV3 Small..."
)

pretrained_weights = (
    MobileNet_V3_Small_Weights.DEFAULT
)

model = mobilenet_v3_small(
    weights=pretrained_weights
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
    weight=class_weights,
    label_smoothing=LABEL_SMOOTHING
)


# =========================================================
# TRAINING MODE HELPER
# =========================================================

def set_training_mode(model):

    model.train()

    # Frozen feature blocks dibuat eval agar
    # BatchNorm statistics tidak ikut berubah.
    for block in model.features:

        trainable = any(
            parameter.requires_grad
            for parameter in block.parameters()
        )

        if not trainable:
            block.eval()

    model.classifier.train()


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
):

    set_training_mode(
        model
    )

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


        optimizer.zero_grad(
            set_to_none=True
        )


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


# =========================================================
# VALIDATION
# =========================================================

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
                labels
                .cpu()
                .numpy()
                .tolist()
            )

            all_pred.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
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
# HISTORY & CHECKPOINT
# =========================================================

history = []

best_val_f1 = 0.0


def save_if_best(
    stage,
    epoch,
    val_acc,
    val_f1,
):

    global best_val_f1


    if val_f1 <= best_val_f1:
        return


    best_val_f1 = val_f1


    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "classes":
                classes,

            "architecture":
                "mobilenet_v3_small",

            "version":
                "v2",

            "image_size":
                IMAGE_SIZE,

            "batch_size":
                BATCH_SIZE,

            "val_accuracy":
                val_acc,

            "val_macro_f1":
                val_f1,

            "stage":
                stage,

            "epoch":
                epoch,

            "label_smoothing":
                LABEL_SMOOTHING,

            "weight_decay":
                WEIGHT_DECAY,
        },

        MODEL_PATH
    )


    print(
        "[SAVED] New best V2 model"
    )


# =========================================================
# STAGE RUNNER
# =========================================================

def run_stage(
    stage_number,
    epochs,
    learning_rate,
    global_epoch_start,
):

    optimizer = torch.optim.AdamW(

        filter(
            lambda parameter:
                parameter.requires_grad,

            model.parameters()
        ),

        lr=learning_rate,

        weight_decay=WEIGHT_DECAY
    )


    for local_epoch in range(
        epochs
    ):

        global_epoch = (
            global_epoch_start
            + local_epoch
            + 1
        )


        print("\n" + "-" * 60)

        print(
            f"Stage {stage_number} — "
            f"Epoch {local_epoch + 1}/{epochs}"
        )

        print("-" * 60)


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
            f"Train Loss   : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Acc    : "
            f"{train_acc * 100:.2f}%"
        )

        print(
            f"Val Loss     : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Acc      : "
            f"{val_acc * 100:.2f}%"
        )

        print(
            f"Val Macro F1 : "
            f"{val_f1:.4f}"
        )


        history.append(
            {
                "stage":
                    stage_number,

                "epoch":
                    global_epoch,

                "learning_rate":
                    learning_rate,

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
            }
        )


        save_if_best(
            stage=stage_number,
            epoch=global_epoch,
            val_acc=val_acc,
            val_f1=val_f1,
        )


# =========================================================
# STAGE 1
# CLASSIFIER ONLY
# =========================================================

print("\n" + "=" * 60)
print("STAGE 1 — CLASSIFIER TRAINING")
print("=" * 60)


for parameter in model.features.parameters():

    parameter.requires_grad = False


for parameter in model.classifier.parameters():

    parameter.requires_grad = True


run_stage(
    stage_number=1,
    epochs=STAGE1_EPOCHS,
    learning_rate=STAGE1_LR,
    global_epoch_start=0,
)


# =========================================================
# STAGE 2
# LAST 3 FEATURE BLOCKS
# =========================================================

print("\n" + "=" * 60)
print("STAGE 2 — PARTIAL FINE-TUNING")
print("=" * 60)


for parameter in model.features.parameters():

    parameter.requires_grad = False


for parameter in model.features[-3:].parameters():

    parameter.requires_grad = True


for parameter in model.classifier.parameters():

    parameter.requires_grad = True


run_stage(
    stage_number=2,
    epochs=STAGE2_EPOCHS,
    learning_rate=STAGE2_LR,
    global_epoch_start=STAGE1_EPOCHS,
)


# =========================================================
# STAGE 3
# DEEPER FINE-TUNING
# =========================================================

print("\n" + "=" * 60)
print("STAGE 3 — DEEPER FINE-TUNING")
print("=" * 60)


for parameter in model.features.parameters():

    parameter.requires_grad = False


for parameter in model.features[-6:].parameters():

    parameter.requires_grad = True


for parameter in model.classifier.parameters():

    parameter.requires_grad = True


run_stage(
    stage_number=3,
    epochs=STAGE3_EPOCHS,
    learning_rate=STAGE3_LR,
    global_epoch_start=(
        STAGE1_EPOCHS
        + STAGE2_EPOCHS
    ),
)


# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    RESULTS_DIR
    / "training_history.csv",
    index=False
)


# =========================================================
# ACCURACY CURVE
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
    "MobileNetV3 V2 — Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "training_accuracy.png",
    dpi=150
)

plt.close()


# =========================================================
# LOSS CURVE
# =========================================================

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
    "MobileNetV3 V2 — Loss"
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
# MACRO F1 CURVE
# =========================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    history_df["epoch"],
    history_df["val_macro_f1"],
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Validation Macro F1"
)

plt.title(
    "MobileNetV3 V2 — Validation Macro F1"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "validation_macro_f1.png",
    dpi=150
)

plt.close()


# =========================================================
# FINISH
# =========================================================

print("\n" + "=" * 60)
print("MOBILENETV3 V2 TRAINING COMPLETED")
print("=" * 60)

print(
    f"\nBest validation Macro F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"\nBest V2 model saved to:\n"
    f"{MODEL_PATH}"
)

print(
    f"\nTraining results saved to:\n"
    f"{RESULTS_DIR}"
)