from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)

from src.inference import (
    predict_bytes,
    CLASSES,
    DEVICE,
)


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Smart Waste Classification API",
    description=(
        "Waste classification API using "
        "MobileNetV3 V2 with uncertainty detection."
    ),
    version="1.0.0",
)


# =========================================================
# CONFIGURATION
# =========================================================

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024
# 10 MB


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message":
            "Smart Waste Classification API",

        "status":
            "running",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "device":
            str(DEVICE),

        "model":
            "MobileNetV3-Small-V2",

        "num_classes":
            len(CLASSES),
    }


# =========================================================
# CLASSES
# =========================================================

@app.get("/classes")
def get_classes():

    return {
        "classes":
            CLASSES
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------
    # File type validation
    # -----------------------------------------

    if file.content_type not in ALLOWED_CONTENT_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Use JPG, PNG, WEBP, or BMP."
            ),
        )


    # -----------------------------------------
    # Read image
    # -----------------------------------------

    image_bytes = await file.read()


    if len(image_bytes) == 0:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )


    if len(image_bytes) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Image exceeds 10 MB limit.",
        )


    # -----------------------------------------
    # Model prediction
    # -----------------------------------------

    try:

        result = predict_bytes(
            image_bytes
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {error}",
        )


    # -----------------------------------------
    # Response
    # -----------------------------------------

    return {

        "filename":
            file.filename,

        "prediction":
            result["predicted_class"],

        "confidence":
            result["confidence"],

        "status":
            result["status"],

        "uncertainty_votes":
            result["uncertainty_votes"],

        "uncertainty":
            result["uncertainty"],

        "flags":
            result["flags"],

        "probabilities":
            result["probabilities"],
    }