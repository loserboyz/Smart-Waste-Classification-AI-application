from pathlib import Path
import json

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_mobilenet_v2.pth"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "results"
    / "mobilenet_v2"
    / "ood"
    / "thresholds.json"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# PREPROCESSING
# =========================================================

transform = transforms.Compose([

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
# LOAD CHECKPOINT
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

CLASSES = checkpoint[
    "classes"
]


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
    len(CLASSES)
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
# LOAD UNCERTAINTY THRESHOLDS
# =========================================================

with open(
    THRESHOLD_PATH,
    "r"
) as file:

    thresholds = json.load(
        file
    )


MSP_THRESHOLD = thresholds[
    "msp_accept_threshold"
]

ENTROPY_THRESHOLD = thresholds[
    "entropy_reject_threshold"
]

ENERGY_THRESHOLD = thresholds[
    "energy_reject_threshold"
]


# =========================================================
# OPENCV IMAGE VALIDATION
# =========================================================

def validate_cv_image(
    image
):

    if image is None:

        raise ValueError(
            "Image tidak dapat dibaca."
        )


    if len(image.shape) != 3:

        raise ValueError(
            "Image harus memiliki 3 channel warna."
        )


    if image.shape[2] != 3:

        raise ValueError(
            "Image harus berupa RGB/BGR 3-channel."
        )


    height, width = (
        image.shape[:2]
    )


    if width < 50 or height < 50:

        raise ValueError(
            "Resolusi image terlalu kecil."
        )


    return True


# =========================================================
# OPENCV → PIL
# =========================================================

def cv2_to_pil(
    image_bgr
):

    validate_cv_image(
        image_bgr
    )


    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


    image_pil = Image.fromarray(
        image_rgb
    )


    return image_pil


# =========================================================
# UNCERTAINTY SCORES
# =========================================================

def compute_uncertainty(
    logits
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )


    # -----------------------------------------
    # Maximum Softmax Probability
    # semakin tinggi → semakin confident
    # -----------------------------------------

    msp, prediction_index = (
        probabilities.max(
            dim=1
        )
    )


    # -----------------------------------------
    # Normalized entropy
    # semakin tinggi → semakin uncertain
    # -----------------------------------------

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


    # -----------------------------------------
    # Energy
    # semakin tinggi / kurang negatif
    # → semakin OOD-like
    # -----------------------------------------

    energy = -torch.logsumexp(
        logits,
        dim=1
    )


    return (
        probabilities,
        prediction_index,
        msp,
        entropy,
        energy,
    )


# =========================================================
# UNCERTAINTY DECISION
# =========================================================

def determine_status(
    msp,
    entropy,
    energy
):

    flags = {
        "low_msp":
            msp < MSP_THRESHOLD,

        "high_entropy":
            entropy > ENTROPY_THRESHOLD,

        "high_energy":
            energy > ENERGY_THRESHOLD,
    }


    number_of_flags = sum(
        flags.values()
    )


    # Conservative heuristic:
    # minimal 2 dari 3 uncertainty indicators
    # harus melewati threshold.
    if number_of_flags >= 2:

        status = "UNCERTAIN"

    else:

        status = "KNOWN_LIKE"


    return (
        status,
        flags,
        number_of_flags
    )


# =========================================================
# CORE PREDICTION
# =========================================================

def predict_cv_image(
    image_bgr
):

    image_pil = cv2_to_pil(
        image_bgr
    )


    tensor = transform(
        image_pil
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )


    with torch.no_grad():

        logits = model(
            tensor
        )


        (
            probabilities,
            prediction_index,
            msp,
            entropy,
            energy,
        ) = compute_uncertainty(
            logits
        )


    prediction_index = int(
        prediction_index.item()
    )

    predicted_class = CLASSES[
        prediction_index
    ]


    msp_value = float(
        msp.item()
    )

    entropy_value = float(
        entropy.item()
    )

    energy_value = float(
        energy.item()
    )


    (
        status,
        flags,
        uncertainty_votes
    ) = determine_status(

        msp=msp_value,

        entropy=entropy_value,

        energy=energy_value,
    )


    class_probabilities = {

        class_name:
            float(
                probabilities[
                    0,
                    index
                ].item()
            )

        for index, class_name
        in enumerate(CLASSES)
    }


    sorted_probabilities = dict(

        sorted(

            class_probabilities.items(),

            key=lambda item:
                item[1],

            reverse=True
        )
    )


    result = {

        "predicted_class":
            predicted_class,

        "confidence":
            msp_value,

        "status":
            status,

        "uncertainty_votes":
            uncertainty_votes,

        "uncertainty": {

            "msp":
                msp_value,

            "entropy":
                entropy_value,

            "energy":
                energy_value,
        },

        "thresholds": {

            "msp":
                MSP_THRESHOLD,

            "entropy":
                ENTROPY_THRESHOLD,

            "energy":
                ENERGY_THRESHOLD,
        },

        "flags":
            flags,

        "probabilities":
            sorted_probabilities,
    }


    return result


# =========================================================
# PREDICT FROM FILE
# =========================================================

def predict_file(
    image_path
):

    image_path = Path(
        image_path
    )


    if not image_path.exists():

        raise FileNotFoundError(
            f"File tidak ditemukan: "
            f"{image_path}"
        )


    image = cv2.imread(
        str(
            image_path
        )
    )


    return predict_cv_image(
        image
    )


# =========================================================
# PREDICT FROM BYTES
# dipakai nanti oleh FastAPI / Streamlit
# =========================================================

def predict_bytes(
    image_bytes
):

    numpy_buffer = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )


    image = cv2.imdecode(
        numpy_buffer,
        cv2.IMREAD_COLOR
    )


    return predict_cv_image(
        image
    )


# =========================================================
# COMMAND-LINE TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMART WASTE INFERENCE")
    print("=" * 60)

    print(
        f"\nDevice  : {DEVICE}"
    )

    print(
        f"Classes : {CLASSES}"
    )

    print("\nThresholds:")

    print(
        f"MSP     : "
        f"{MSP_THRESHOLD:.4f}"
    )

    print(
        f"Entropy : "
        f"{ENTROPY_THRESHOLD:.4f}"
    )

    print(
        f"Energy  : "
        f"{ENERGY_THRESHOLD:.4f}"
    )

    print(
        "\nInference module loaded successfully."
    )