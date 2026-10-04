from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "baseline_cnn.pth"
)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 224


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("BASELINE CNN TRAINING")
print("=" * 60)

print(f"\nDevice: {DEVICE}")


# =========================================================
# IMAGE TRANSFORMS
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
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

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
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


print("\nClasses:")
print(train_dataset.classes)

print(
    f"\nTraining images: "
    f"{len(train_dataset)}"
)

print(
    f"Validation images: "
    f"{len(val_dataset)}"
)


# =========================================================
# DATALOADER
# =========================================================

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


# =========================================================
# CLASS WEIGHTS
# =========================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

classes = train_dataset.classes

class_counts = (
    train_df["label"]
    .value_counts()
    .reindex(classes)
)


total_samples = len(
    train_df
)

num_classes = len(
    classes
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
# BASELINE CNN MODEL
# =========================================================

class BaselineCNN(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            # Reduce feature maps
            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                p=0.3
            ),

            nn.Linear(
                in_features=128,
                out_features=num_classes
            ),
        )


    def forward(self, x):

        x = self.features(
            x
        )

        x = self.classifier(
            x
        )

        return x


model = BaselineCNN(
    num_classes=num_classes
).to(DEVICE)


print("\nModel:")
print(model)


# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        # reset gradient
        optimizer.zero_grad()


        # forward pass
        outputs = model(
            images
        )


        # calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # backward pass
        loss.backward()


        # update weights
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
# VALIDATION FUNCTION
# =========================================================

def validate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

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
# TRAINING LOOP
# =========================================================

best_val_accuracy = 0.0


for epoch in range(
    EPOCHS
):

    print("\n" + "-" * 60)

    print(
        f"Epoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    print("-" * 60)


    train_loss, train_acc = (
        train_one_epoch()
    )


    val_loss, val_acc = (
        validate()
    )


    print(
        f"Train Loss : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_acc:.4f} "
        f"({train_acc * 100:.2f}%)"
    )

    print(
        f"Val Loss   : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_acc:.4f} "
        f"({val_acc * 100:.2f}%)"
    )


    # Save best model
    if val_acc > best_val_accuracy:

        best_val_accuracy = (
            val_acc
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    classes,

                "image_size":
                    IMAGE_SIZE,

                "val_accuracy":
                    best_val_accuracy,
            },

            MODEL_PATH
        )


        print(
            "[SAVED] New best model"
        )


# =========================================================
# FINISH
# =========================================================

print("\n" + "=" * 60)
print("BASELINE TRAINING COMPLETED")
print("=" * 60)

print(
    f"\nBest validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"\nModel saved to:\n"
    f"{MODEL_PATH}"
)