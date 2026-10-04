import requests
import pandas as pd
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000"

PREDICT_URL = f"{API_URL}/predict"

HEALTH_URL = f"{API_URL}/health"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="wide",
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "♻️ Smart Waste Classification"
)

st.write(
    """
    AI-based waste classification using **MobileNetV3**.

    Supported classes:
    **Cardboard, Glass, Metal, Paper, Plastic, and Trash**.
    """
)


# =========================================================
# API STATUS
# =========================================================

try:

    response = requests.get(
        HEALTH_URL,
        timeout=5
    )

    if response.status_code == 200:

        health = response.json()

        st.success(
            f"API Online — "
            f"Model: {health['model']} | "
            f"Device: {health['device']}"
        )

    else:

        st.warning(
            "API responded but is not healthy."
        )


except requests.exceptions.RequestException:

    st.error(
        "FastAPI server is offline. "
        "Run the API first using:\n\n"
        "`python -m uvicorn api.main:app --reload`"
    )

    st.stop()


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
    ]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------
    # Layout
    # -----------------------------------------

    left_column, right_column = st.columns(
        [1, 1]
    )


    # -----------------------------------------
    # Preview
    # -----------------------------------------

    with left_column:

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            uploaded_file,
            use_container_width=True
        )


    # -----------------------------------------
    # Send to FastAPI
    # -----------------------------------------

    image_bytes = (
        uploaded_file.getvalue()
    )


    files = {

        "file": (
            uploaded_file.name,
            image_bytes,
            uploaded_file.type
        )

    }


    with st.spinner(
        "Running AI inference..."
    ):

        try:

            response = requests.post(
                PREDICT_URL,
                files=files,
                timeout=30
            )


        except requests.exceptions.RequestException as error:

            st.error(
                f"API connection failed: {error}"
            )

            st.stop()


    # -----------------------------------------
    # Handle API errors
    # -----------------------------------------

    if response.status_code != 200:

        st.error(
            f"Prediction failed.\n\n"
            f"{response.text}"
        )

        st.stop()


    result = response.json()


    # -----------------------------------------
    # Result panel
    # -----------------------------------------

    with right_column:

        st.subheader(
            "Prediction Result"
        )


        prediction = result[
            "prediction"
        ]

        confidence = result[
            "confidence"
        ]

        status = result[
            "status"
        ]

        uncertainty_votes = result[
            "uncertainty_votes"
        ]


        st.metric(
            label="Predicted Class",
            value=prediction.upper()
        )


        st.metric(
            label="Model Confidence",
            value=f"{confidence * 100:.2f}%"
        )


        # -------------------------------------
        # Status
        # -------------------------------------

        if status == "KNOWN_LIKE":

            st.success(
                "Input Status: KNOWN-LIKE"
            )

        else:

            st.warning(
                "Input Status: UNCERTAIN"
            )


        st.write(
            f"Uncertainty votes: "
            f"**{uncertainty_votes}/3**"
        )


# =========================================================
# UNCERTAINTY DETAILS
# =========================================================

    st.divider()

    st.subheader(
        "Uncertainty Analysis"
    )


    uncertainty = result[
        "uncertainty"
    ]


    col1, col2, col3 = st.columns(
        3
    )


    col1.metric(
        "MSP",
        f"{uncertainty['msp']:.4f}"
    )


    col2.metric(
        "Entropy",
        f"{uncertainty['entropy']:.4f}"
    )


    col3.metric(
        "Energy",
        f"{uncertainty['energy']:.4f}"
    )


    flags = result[
        "flags"
    ]


    with st.expander(
        "View uncertainty flags"
    ):

        st.json(
            flags
        )


# =========================================================
# CLASS PROBABILITIES
# =========================================================

    st.divider()

    st.subheader(
        "Class Probabilities"
    )


    probabilities = result[
        "probabilities"
    ]


    probability_df = pd.DataFrame({

        "Class":
            list(
                probabilities.keys()
            ),

        "Probability":
            list(
                probabilities.values()
            )
    })


    probability_df[
        "Probability (%)"
    ] = (
        probability_df[
            "Probability"
        ]
        * 100
    )


    st.bar_chart(
        probability_df.set_index(
            "Class"
        )[
            "Probability (%)"
        ]
    )


    st.dataframe(
        probability_df[
            [
                "Class",
                "Probability (%)"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# LIMITATION NOTICE
# =========================================================

    st.divider()


    st.info(
        """
        **Model limitation**

        `KNOWN-LIKE` does not guarantee that an uploaded image
        belongs to one of the six supported waste classes.

        The uncertainty mechanism is heuristic and may fail to
        reject some unsupported or out-of-distribution images.
        """
    )