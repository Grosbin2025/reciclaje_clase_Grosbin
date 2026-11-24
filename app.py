
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Clasificador de Residuos",
    page_icon="♻️",
    layout="centered"
)

st.title("Modelo predictivo Reciclaje clase de IA-UTH -2026-Grosbin Palma")
st.write("Sube una imagen y el modelo intentará identificar el tipo de residuo.")

st.caption("Clases: cartón, vidrio, metal, papel, plástico y basura general.")

IMG_SIZE = (224, 224)

MODEL_DIR = Path("modelo_reciclaje_mobilenet")
CLASS_PATH = MODEL_DIR / "class_names.json"

MODEL_PATHS = [
    MODEL_DIR / "waste_mobilenet.keras",
    MODEL_DIR / "waste_mobilenet.h5"
]

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura"
}


@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)

    st.error("No se encontró el modelo. Coloque la carpeta modelo_reciclaje_mobilenet junto a app.py.")
    st.stop()


@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    top3 = np.argsort(preds)[-3:][::-1]

    resultados = [
        (LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in top3
    ]

    return preds, resultados


def detalle_por_clase(preds):
    st.subheader("Detalle por clase")

    for i, prob in enumerate(preds):
        nombre = LABELS_ES.get(clases[i], clases[i])
        porcentaje = float(prob)

        st.write(f"{nombre}: {porcentaje * 100:.2f}%")
        st.progress(max(0.0, min(1.0, porcentaje)))


modelo = cargar_modelo()
clases = cargar_clases()

archivo = st.file_uploader(
    "Seleccione una imagen",
    type=["jpg", "jpeg", "png"]
)

if archivo:
    imagen = Image.open(archivo)

    st.image(
        imagen,
        caption="Imagen cargada",
        use_container_width=True
    )

    preds, resultados = predecir(imagen)

    st.subheader("Predicción principal")

    st.success(
        f"{resultados[0][0]}: {resultados[0][1]:.2f}%"
    )

    st.subheader("Top 3")

    for clase, prob in resultados:
        st.write(f"• {clase}: {prob:.2f}%")

    detalle_por_clase(preds)

else:
    st.info("Cargue una imagen para iniciar la clasificación.")
