
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Skin Lesion Classifier")
st.caption("EfficientNetB0 • HAM10000")
st.warning(
    "Educational/research use only. "
    "This application is not a medical diagnosis tool."
)

MODEL_PATH = "skin_lesion_efficientnetb0.keras"
IMAGE_SIZE = (224, 224)

# Same class ordering used by the notebook:
# sorted(metadata["dx"].unique())
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a skin-lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width="stretch"
    )

    if st.button("🔍 Predict", type="primary", use_container_width=True):

        # EfficientNetB0 input
        image_resized = image.resize(IMAGE_SIZE)
        image_array = np.asarray(
            image_resized,
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

        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])

        st.subheader("Prediction")
        st.success(
            f"Predicted class: {predicted_class}"
        )
        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        st.subheader("Class probabilities")

        for name, probability in zip(
            CLASS_NAMES,
            predictions
        ):
            st.write(
                f"**{name}** — {probability:.2%}"
            )
            st.progress(
                float(probability)
            )

st.divider()
st.caption(
    "Model: EfficientNetB0 trained from the supplied notebook. "
    "Predictions should not be interpreted as a clinical diagnosis."
)
