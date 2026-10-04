from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

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

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)

EXTERNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "internet"
)

OOD_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "ood"
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
    / "ood"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224
BATCH_SIZE = 4

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".jfif",
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# TRANSFORM
# =========================================================

eval_transform = transforms.Compose([

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
# OOD DATASET
# =========================================================

class OODDataset(Dataset):

    def __init__(
        self,
        folder,
        transform=None
    ):

        self.folder = Path(folder)
        self.transform = transform

        if not self.folder.exists():
            raise FileNotFoundError(
                f"OOD folder tidak ditemukan: "
                f"{self.folder}"
            )

        self.image_paths = sorted(
            path
            for path in self.folder.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in VALID_EXTENSIONS
        )

        if not self.image_paths:
            raise RuntimeError(
                "Tidak ada image OOD yang ditemukan."
            )


    def __len__(self):
        return len(
            self.image_paths
        )


    def __getitem__(
        self,
        index
    ):

        image_path = (
            self.image_paths[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(
                image
            )

        return (
            image,
            str(image_path)
        )


# =========================================================
# DATASETS
# =========================================================

val_dataset = WasteDataset(
    VAL_CSV,
    transform=eval_transform
)

test_dataset = WasteDataset(
    TEST_CSV,
    transform=eval_transform
)

external_dataset = ImageFolder(
    EXTERNAL_DIR,
    transform=eval_transform
)

ood_dataset = OODDataset(
    OOD_DIR,
    transform=eval_transform
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

external_loader = DataLoader(
    external_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

ood_loader = DataLoader(
    ood_dataset,
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

classes = checkpoint[
    "classes"
]


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
    checkpoint[
        "model_state_dict"
    ]
)

model = model.to(
    DEVICE
)

model.eval()


# =========================================================
# CLASS CHECK
# =========================================================

if list(
    external_dataset.classes
) != list(classes):

    raise ValueError(
        "Class external berbeda "
        "dengan checkpoint model."
    )


# =========================================================
# UNCERTAINTY SCORES
# =========================================================

def compute_scores(logits):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    # ---------------------------------
    # Maximum Softmax Probability
    # semakin tinggi = semakin yakin
    # ---------------------------------

    msp, predictions = (
        probabilities.max(
            dim=1
        )
    )

    # ---------------------------------
    # Normalized Predictive Entropy
    # semakin tinggi = semakin uncertain
    # range kira-kira 0 - 1
    # ---------------------------------

    entropy = -torch.sum(
        probabilities
        * torch.log(
            probabilities + 1e-12
        ),
        dim=1
    )

    entropy = (
        entropy
        / np.log(
            probabilities.shape[1]
        )
    )

    # ---------------------------------
    # Energy
    # lebih tinggi / kurang negatif
    # biasanya lebih OOD
    # ---------------------------------

    energy = -torch.logsumexp(
        logits,
        dim=1
    )


    return (
        predictions,
        msp,
        entropy,
        energy
    )


# =========================================================
# COLLECT KNOWN DATA SCORES
# =========================================================

def collect_known_scores(
    loader,
    source_name
):

    records = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            logits = model(
                images
            )

            (
                predictions,
                msp,
                entropy,
                energy
            ) = compute_scores(
                logits
            )


            for i in range(
                images.size(0)
            ):

                records.append({

                    "source":
                        source_name,

                    "is_ood":
                        0,

                    "true_label":
                        classes[
                            int(
                                labels[i]
                            )
                        ],

                    "predicted_label":
                        classes[
                            int(
                                predictions[i]
                            )
                        ],

                    "msp":
                        float(
                            msp[i]
                        ),

                    "entropy":
                        float(
                            entropy[i]
                        ),

                    "energy":
                        float(
                            energy[i]
                        ),
                })


    return pd.DataFrame(
        records
    )


# =========================================================
# COLLECT OOD SCORES
# =========================================================

def collect_ood_scores():

    records = []


    with torch.no_grad():

        for images, paths in ood_loader:

            images = images.to(
                DEVICE
            )

            logits = model(
                images
            )

            (
                predictions,
                msp,
                entropy,
                energy
            ) = compute_scores(
                logits
            )


            for i in range(
                images.size(0)
            ):

                records.append({

                    "source":
                        "ood",

                    "is_ood":
                        1,

                    "filepath":
                        paths[i],

                    "true_label":
                        "OOD",

                    "predicted_label":
                        classes[
                            int(
                                predictions[i]
                            )
                        ],

                    "msp":
                        float(
                            msp[i]
                        ),

                    "entropy":
                        float(
                            entropy[i]
                        ),

                    "energy":
                        float(
                            energy[i]
                        ),
                })


    return pd.DataFrame(
        records
    )


# =========================================================
# RUN INFERENCE
# =========================================================

print("=" * 60)
print("MOBILENETV3 V2 — OOD EVALUATION")
print("=" * 60)

print(
    f"\nDevice: {DEVICE}"
)

print(
    f"Validation ID : "
    f"{len(val_dataset)}"
)

print(
    f"Internal Test : "
    f"{len(test_dataset)}"
)

print(
    f"External ID   : "
    f"{len(external_dataset)}"
)

print(
    f"OOD images    : "
    f"{len(ood_dataset)}"
)


val_scores = collect_known_scores(
    val_loader,
    "validation_id"
)

test_scores = collect_known_scores(
    test_loader,
    "internal_test"
)

external_scores = collect_known_scores(
    external_loader,
    "external_known"
)

ood_scores = collect_ood_scores()


# =========================================================
# THRESHOLDS FROM VALIDATION ONLY
# =========================================================

# Target:
# retain sekitar 95% validation ID.

MSP_THRESHOLD = (
    val_scores[
        "msp"
    ]
    .quantile(
        0.05
    )
)

ENTROPY_THRESHOLD = (
    val_scores[
        "entropy"
    ]
    .quantile(
        0.95
    )
)

ENERGY_THRESHOLD = (
    val_scores[
        "energy"
    ]
    .quantile(
        0.95
    )
)


thresholds = {

    "msp_accept_threshold":
        float(
            MSP_THRESHOLD
        ),

    "entropy_reject_threshold":
        float(
            ENTROPY_THRESHOLD
        ),

    "energy_reject_threshold":
        float(
            ENERGY_THRESHOLD
        ),
}


with open(
    RESULTS_DIR
    / "thresholds.json",
    "w"
) as file:

    json.dump(
        thresholds,
        file,
        indent=4
    )


print("\n" + "=" * 60)
print("VALIDATION-DERIVED THRESHOLDS")
print("=" * 60)

print(
    f"MSP threshold     : "
    f"{MSP_THRESHOLD:.4f}"
)

print(
    f"Entropy threshold : "
    f"{ENTROPY_THRESHOLD:.4f}"
)

print(
    f"Energy threshold  : "
    f"{ENERGY_THRESHOLD:.4f}"
)


# =========================================================
# REJECTION RATES
# =========================================================

def rejection_summary(
    df,
    name
):

    msp_rejected = (
        df["msp"]
        < MSP_THRESHOLD
    ).mean()

    entropy_rejected = (
        df["entropy"]
        > ENTROPY_THRESHOLD
    ).mean()

    energy_rejected = (
        df["energy"]
        > ENERGY_THRESHOLD
    ).mean()


    return {

        "dataset":
            name,

        "images":
            len(df),

        "msp_rejection_rate":
            float(
                msp_rejected
            ),

        "entropy_rejection_rate":
            float(
                entropy_rejected
            ),

        "energy_rejection_rate":
            float(
                energy_rejected
            ),
    }


rejection_results = pd.DataFrame([

    rejection_summary(
        val_scores,
        "validation_id"
    ),

    rejection_summary(
        test_scores,
        "internal_test"
    ),

    rejection_summary(
        external_scores,
        "external_known"
    ),

    rejection_summary(
        ood_scores,
        "ood"
    ),
])


rejection_results.to_csv(
    RESULTS_DIR
    / "rejection_rates.csv",
    index=False
)


print("\n" + "=" * 60)
print("REJECTION RATES")
print("=" * 60)

print(
    rejection_results
    .round(4)
    .to_string(
        index=False
    )
)


# =========================================================
# AUROC / AUPR
# =========================================================

def detection_metrics(
    known_df,
    ood_df,
    known_name
):

    combined = pd.concat(
        [
            known_df,
            ood_df
        ],
        ignore_index=True
    )


    y_true = (
        combined[
            "is_ood"
        ]
        .values
    )


    score_map = {

        # High = more OOD
        "MSP":
            1
            - combined[
                "msp"
            ].values,

        "Entropy":
            combined[
                "entropy"
            ].values,

        "Energy":
            combined[
                "energy"
            ].values,
    }


    rows = []


    for score_name, scores in (
        score_map.items()
    ):

        auroc = roc_auc_score(
            y_true,
            scores
        )

        aupr = average_precision_score(
            y_true,
            scores
        )


        rows.append({

            "known_dataset":
                known_name,

            "method":
                score_name,

            "auroc":
                auroc,

            "aupr":
                aupr,
        })


    return rows


detection_rows = []

detection_rows.extend(
    detection_metrics(
        test_scores,
        ood_scores,
        "internal_test"
    )
)

detection_rows.extend(
    detection_metrics(
        external_scores,
        ood_scores,
        "external_known"
    )
)


detection_df = pd.DataFrame(
    detection_rows
)

detection_df.to_csv(
    RESULTS_DIR
    / "ood_detection_metrics.csv",
    index=False
)


print("\n" + "=" * 60)
print("OOD DETECTION METRICS")
print("=" * 60)

print(
    detection_df
    .round(4)
    .to_string(
        index=False
    )
)


# =========================================================
# OOD PREDICTED CLASS DISTRIBUTION
# =========================================================

ood_distribution = (
    ood_scores[
        "predicted_label"
    ]
    .value_counts()
    .reindex(
        classes,
        fill_value=0
    )
    .reset_index()
)

ood_distribution.columns = [
    "predicted_class",
    "count"
]


ood_distribution.to_csv(
    RESULTS_DIR
    / "ood_prediction_distribution.csv",
    index=False
)


print("\nOOD predicted classes:")

print(
    ood_distribution
    .to_string(
        index=False
    )
)


# =========================================================
# SAVE ALL SCORES
# =========================================================

all_scores = pd.concat(
    [
        val_scores,
        test_scores,
        external_scores,
        ood_scores,
    ],
    ignore_index=True
)


all_scores.to_csv(
    RESULTS_DIR
    / "all_uncertainty_scores.csv",
    index=False
)


# =========================================================
# DISTRIBUTION PLOTS
# =========================================================

plot_sets = [
    (
        "internal_test",
        test_scores
    ),
    (
        "external_known",
        external_scores
    ),
    (
        "ood",
        ood_scores
    ),
]


for metric in [
    "msp",
    "entropy",
    "energy"
]:

    plt.figure(
        figsize=(9, 5)
    )

    for name, df in plot_sets:

        plt.hist(
            df[metric],
            bins=20,
            alpha=0.5,
            label=name
        )


    plt.xlabel(
        metric.upper()
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"{metric.upper()} Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / f"{metric}_distribution.png",
        dpi=150
    )

    plt.close()


# =========================================================
# FINISH
# =========================================================

print("\n" + "=" * 60)
print("OOD EVALUATION COMPLETED")
print("=" * 60)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)
