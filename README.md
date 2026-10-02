# Skin Lesion EfficientNetB0 — Streamlit

This project converts the supplied Google Colab notebook into a Streamlit inference app.

## 1. Train the model

Run the original notebook through the training and saving cells.

The notebook saves:

```text
skin_lesion_efficientnetb0.keras
```

Place that file in this project folder:

```text
skin_lesion_streamlit/
├── app.py
├── requirements.txt
└── skin_lesion_efficientnetb0.keras
```

## 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 3. Run Streamlit

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## 4. Deploy

For Streamlit Community Cloud, upload:

- `app.py`
- `requirements.txt`
- `skin_lesion_efficientnetb0.keras`

Then select `app.py` as the main file.

### Important

The trained `.keras` model is **not included in the supplied notebook file**; the notebook only contains the code that creates and saves it. Therefore, the app is prepared to either:

1. load `skin_lesion_efficientnetb0.keras` from the project folder, or
2. accept a `.keras` model upload from the sidebar.

## Model details from the supplied notebook

- Dataset: HAM10000
- Architecture: EfficientNetB0
- Image size: 224 × 224
- ImageNet pretrained weights
- Frozen EfficientNetB0 base during training
- GlobalAveragePooling2D
- Dense(128, ReLU)
- Dropout(0.2)
- Softmax output
- Optimizer: Adam
- Loss: categorical crossentropy
- Training epochs in notebook: 5
- Batch size: 32

## Classes

The notebook creates class names using:

```python
classes = sorted(metadata["dx"].unique())
```

For the standard HAM10000 dataset this corresponds to:

```text
akiec, bcc, bkl, df, mel, nv, vasc
```

The app also lets you override the class names from the sidebar.

## Medical-use warning

This is an educational/research image-classification demonstration. It must not be presented as a clinically validated diagnostic system or used as a substitute for professional medical evaluation.
