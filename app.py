
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Skin Lesion AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99,102,241,0.15), transparent 25%),
            radial-gradient(circle at 90% 10%, rgba(14,165,233,0.15), transparent 25%),
            linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #f0f9ff 100%);
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero */
    .hero {
        padding: 2rem 2.5rem;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #312e81 0%,
            #4f46e5 45%,
            #0284c7 100%
        );
        color: white;
        box-shadow: 0 15px 40px rgba(30, 41, 59, 0.18);
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.3rem;
        font-weight: 800;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.92);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(148,163,184,0.25);
        box-shadow: 0 8px 25px rgba(15,23,42,0.08);
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.8rem;
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(
            135deg,
            #ecfeff 0%,
            #eef2ff 50%,
            #f5f3ff 100%
        );
        border: 1px solid #c7d2fe;
        border-radius: 22px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(79,70,229,0.12);
    }

    .prediction-label {
        color: #64748b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    .prediction-class {
        color: #312e81;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .confidence {
        color: #0284c7;
        font-size: 1.4rem;
        font-weight: 700;
    }

    /* Info cards */
    .info-card {
        border-radius: 18px;
        padding: 1.2rem;
        color: white;
        min-height: 120px;
        box-shadow: 0 8px 20px rgba(15,23,42,0.10);
    }

    .info-blue {
        background: linear-gradient(135deg, #2563eb, #06b6d4);
    }

    .info-purple {
        background: linear-gradient(135deg, #7c3aed, #c026d3);
    }

    .info-green {
        background: linear-gradient(135deg, #059669, #14b8a6);
    }

    .info-orange {
        background: linear-gradient(135deg, #ea580c, #f59e0b);
    }

    .info-number {
        font-size: 2rem;
        font-weight: 800;
    }

    .info-text {
        opacity: 0.9;
        font-size: 0.9rem;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.75);
        border-radius: 18px;
        padding: 0.8rem;
        border: 2px dashed #818cf8;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 14px;
        border: none;
        background: linear-gradient(135deg, #4f46e5, #0284c7);
        color: white;
        font-weight: 700;
        padding: 0.7rem 1rem;
        box-shadow: 0 8px 20px rgba(79,70,229,0.22);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(79,70,229,0.3);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 1.5rem;
        margin-top: 2rem;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_PATH = "skin_lesion_efficientnetb0.keras"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

CLASS_LABELS = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    model_status = True
except Exception as e:
    model_status = False
    st.error(f"Could not load model: {e}")
    st.stop()


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

    <h1>🩺 Skin Lesion AI</h1>

    <p>
        EfficientNetB0-powered image classification
        for the HAM10000 skin-lesion dataset
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL INFORMATION CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="info-card info-blue">
        <div class="info-number">7</div>
        <div class="info-text">Lesion Classes</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card info-purple">
        <div class="info-number">224²</div>
        <div class="info-text">Input Resolution</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card info-green">
        <div class="info-number">B0</div>
        <div class="info-text">EfficientNet Model</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="info-card info-orange">
        <div class="info-number">AI</div>
        <div class="info-text">Image Classification</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.success("Model loaded successfully")

    st.markdown("""
    **Architecture**

    EfficientNetB0

    **Dataset**

    HAM10000

    **Input**

    224 × 224 RGB

    **Output**

    7 classes
    """)

    st.divider()

    st.subheader("🧬 Lesion Classes")

    for class_name in CLASS_NAMES:
        st.write(
            f"• **{class_name}** — "
            f"{CLASS_LABELS[class_name]}"
        )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown("""
<div class="card">

<div class="card-title">
📤 Upload Skin Lesion Image
</div>

Upload a JPG, JPEG, or PNG image for model prediction.

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# IMAGE + PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_col, result_col = st.columns(
        [1, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # IMAGE CARD
    # --------------------------------------------------------

    with image_col:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        🖼️ Uploaded Image
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            width="stretch"
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with result_col:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        🤖 AI Analysis
        </div>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image with EfficientNetB0..."
            ):

                # Resize image
                image_resized = image.resize(
                    IMAGE_SIZE
                )

                # Convert to NumPy
                image_array = np.asarray(
                    image_resized,
                    dtype=np.float32
                )

                # Add batch dimension
                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Prediction
                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

            predicted_index = int(
                np.argmax(predictions)
            )

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

            confidence = float(
                predictions[predicted_index]
            )

            # ------------------------------------------------
            # PREDICTION CARD
            # ------------------------------------------------

            st.markdown(f"""
            <div class="prediction-card">

                <div class="prediction-label">
                    Predicted Class
                </div>

                <div class="prediction-class">
                    {predicted_class.upper()}
                </div>

                <div>
                    {CLASS_LABELS[predicted_class]}
                </div>

                <br>

                <div class="confidence">
                    Confidence: {confidence:.2%}
                </div>

            </div>
            """, unsafe_allow_html=True)


    # ========================================================
    # PROBABILITY SECTION
    # ========================================================

    st.write("")

    st.markdown("""
    <div class="card">

    <div class="card-title">
    📊 Class Probability Distribution
    </div>

    </div>
    """, unsafe_allow_html=True)

    # Sort probabilities
    probability_data = sorted(
        zip(CLASS_NAMES, predictions),
        key=lambda x: x[1],
        reverse=True
    )

    for class_name, probability in probability_data:

        col1, col2 = st.columns(
            [2, 8]
        )

        with col1:

            st.write(
                f"**{class_name.upper()}**"
            )

        with col2:

            st.progress(
                float(probability)
            )

            st.caption(
                f"{CLASS_LABELS[class_name]} • "
                f"{probability:.2%}"
            )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem;">

        <div style="font-size:4rem;">
            🖼️
        </div>

        <h2>
            Upload an image to begin
        </h2>

        <p style="color:#64748b;">
            The EfficientNetB0 model will analyze the
            uploaded image and display the class probabilities.
        </p>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.warning(
    "⚠️ **Important:** This application is an educational/research "
    "demonstration. Model predictions are not medical diagnoses and "
    "should not be used to make treatment decisions. Consult a qualified "
    "healthcare professional for medical evaluation."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    🩺 <b>Skin Lesion AI</b><br>

    EfficientNetB0 • HAM10000 • Deep Learning

    <br><br>

    Built for educational and research purposes.

</div>
""", unsafe_allow_html=True)