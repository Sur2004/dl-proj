
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
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc, #eef2ff, #f0f9ff);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #312e81, #4f46e5, #0284c7);
    padding: 35px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.9;
}

/* Cards */
.card {
    background: white;
    padding: 22px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 25px rgba(15,23,42,0.08);
    margin-bottom: 20px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 12px;
}

/* Metric cards */
.metric-card {
    padding: 22px;
    border-radius: 18px;
    color: white;
    text-align: center;
    min-height: 115px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.10);
}

.metric-number {
    font-size: 30px;
    font-weight: 800;
}

.metric-label {
    font-size: 14px;
    opacity: 0.9;
}

.blue {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
}

.purple {
    background: linear-gradient(135deg, #7c3aed, #c026d3);
}

.green {
    background: linear-gradient(135deg, #059669, #14b8a6);
}

.orange {
    background: linear-gradient(135deg, #ea580c, #f59e0b);
}

/* Prediction */
.prediction-card {
    background: linear-gradient(135deg, #eef2ff, #ecfeff);
    border: 2px solid #c7d2fe;
    border-radius: 22px;
    padding: 30px;
    text-align: center;
}

.prediction-class {
    font-size: 34px;
    font-weight: 800;
    color: #312e81;
}

.confidence {
    font-size: 20px;
    font-weight: 700;
    color: #0284c7;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
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


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🩺 Skin Lesion AI</div>
    <div class="hero-subtitle">
        EfficientNetB0-powered image classification for the HAM10000 dataset
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card blue">
        <div class="metric-number">7</div>
        <div class="metric-label">Lesion Classes</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card purple">
        <div class="metric-number">224 × 224</div>
        <div class="metric-label">Input Resolution</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card green">
        <div class="metric-number">B0</div>
        <div class="metric-label">EfficientNet Model</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card orange">
        <div class="metric-number">AI</div>
        <div class="metric-label">Image Classification</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.success("Model loaded successfully")

    st.write("**Architecture**")
    st.write("EfficientNetB0")

    st.write("**Dataset**")
    st.write("HAM10000")

    st.write("**Input**")
    st.write("224 × 224 RGB")

    st.write("**Output**")
    st.write("7 classes")

    st.divider()

    st.subheader("🧬 Lesion Classes")

    for name in CLASS_NAMES:
        st.write(
            f"**{name}** — {CLASS_LABELS[name]}"
        )


# ============================================================
# UPLOAD
# ============================================================

st.markdown("""
<div class="card">
    <div class="card-title">📤 Upload Skin Lesion Image</div>
    <div>
        Upload a JPG, JPEG, or PNG image for model prediction.
    </div>
</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_col, result_col = st.columns(
        [1, 1],
        gap="large"
    )

    with image_col:

        st.markdown("""
        <div class="card">
            <div class="card-title">🖼️ Uploaded Image</div>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            width="stretch"
        )


    with result_col:

        st.markdown("""
        <div class="card">
            <div class="card-title">🤖 AI Analysis</div>
        </div>
        """, unsafe_allow_html=True)

        analyze = st.button(
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True
        )

        if analyze:

            with st.spinner(
                "Analyzing image with EfficientNetB0..."
            ):

                resized = image.resize(IMAGE_SIZE)

                image_array = np.asarray(
                    resized,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

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

            st.markdown(
                f"""
                <div class="prediction-card">

                    <div>Predicted Class</div>

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
                """,
                unsafe_allow_html=True
            )

            # Store predictions so they remain available
            st.session_state["predictions"] = predictions


# ============================================================
# PROBABILITIES
# ============================================================

if "predictions" in st.session_state:

    predictions = st.session_state["predictions"]

    st.write("")

    st.markdown("""
    <div class="card">
        <div class="card-title">
            📊 Class Probability Distribution
        </div>
    </div>
    """, unsafe_allow_html=True)

    probability_data = sorted(
        zip(CLASS_NAMES, predictions),
        key=lambda x: x[1],
        reverse=True
    )

    for name, probability in probability_data:

        col1, col2 = st.columns(
            [2, 8]
        )

        with col1:

            st.write(
                f"**{name.upper()}**"
            )

        with col2:

            st.progress(
                float(probability)
            )

            st.caption(
                f"{CLASS_LABELS[name]} • "
                f"{probability:.2%}"
            )


# ============================================================
# EMPTY STATE
# ============================================================
else:
    st.markdown("## 🖼️ Upload an image to begin")

    st.write(
        "Choose a JPG, JPEG, or PNG skin-lesion image "
        "using the uploader above."
    )

    st.info(
        "🧠 EfficientNetB0   •   📐 224 × 224   •   🧬 7 Classes"
    )
# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "⚠️ **Important:** This application is an "
    "educational/research demonstration. Model predictions "
    "are not medical diagnoses and should not be used to make "
    "treatment decisions. Consult a qualified healthcare "
    "professional for medical evaluation."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

🩺 <b>Skin Lesion AI</b>

<br>

EfficientNetB0 • HAM10000 • Deep Learning

<br><br>

Built for educational and research purposes.

</div>
""", unsafe_allow_html=True)
