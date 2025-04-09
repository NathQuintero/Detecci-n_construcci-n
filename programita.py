import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import tempfile
import os

# Configurar la app
st.set_page_config(page_title="Detector de EPP", page_icon="🦺", layout="centered")
st.title("🛠️ Verificador de Equipos de Protección Personal")
st.markdown("""
Sube una imagen o pega una URL, y analizaremos si estás listo para trabajar 🏗️.
Para estar preparado, necesitas llevar:
- 🥾 Botas
- 👷 Casco
- 🦺 Chaleco
- 🙋 Ser humano detectado
""")

# Cargar el modelo .tflite
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="mimodelitolindo.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # (altura, ancho)

required_classes = {'boots', 'helmet', 'vest', 'human'}
model_classes = ['boots', 'gloves', 'helmet', 'human', 'vest']  # ajusta según tu modelo

# Elegir fuente de imagen
option = st.radio("Selecciona cómo subir la imagen:", ["📂 Archivo", "🌐 URL"])
image = None

if option == "📂 Archivo":
    uploaded_file = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "🌐 URL":
    url = st.text_input("Pega la URL de la imagen")
    if url:
        try:
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        except:
            st.error("No se pudo cargar la imagen desde la URL")

if image:
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesamiento
    resized_img = image.resize((input_shape[1], input_shape[0]))
    input_data = np.expand_dims(resized_img, axis=0).astype(np.float32) / 255.0  # Normalización

    # Inferencia
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Simulación de detecciones (porque no conocemos el formato exacto del output)
    # Aquí asumimos que el modelo devuelve clases en forma de probabilidad
    predicted_labels = [model_classes[i] for i, prob in enumerate(output_data[0]) if prob > 0.5]
    detected_set = set(predicted_labels)

    # Evaluar si está listo para trabajar
    faltantes = required_classes - detected_set

    st.subheader("🔍 Resultados")
    if not faltantes:
        st.success("✅ ¡Estás listo para trabajar! Todos los elementos de seguridad están presentes.")
    else:
        st.error(f"⚠️ Lo siento compañero, no estás preparado para trabajar. Te falta: {', '.join(faltantes)}")

    st.markdown("""
    ---
    **Detectado:**
    - """ + "\n    - ".join(predicted_labels if predicted_labels else ["Nada detectado"]))

else:
    st.info("Por favor, sube una imagen o proporciona una URL para comenzar ✨")
