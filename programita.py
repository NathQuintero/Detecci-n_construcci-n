import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import tempfile
import os
import cv2
import requests
from io import BytesIO
from gtts import gTTS
import base64

# 🌟 CONFIGURACIÓN DE LA APP
st.set_page_config(page_title="Detector de EPP con Glamour", page_icon="🦺", layout="centered")
st.title("🛠️✨ Detector de Equipos de Protección Personal")
st.markdown("""
### 📸 Sube una imagen, toma una foto o pega una URL

Vamos a verificar si estás **listo para entrar a la obra**.  
Necesitas tener:
- 🥾 **Botas**
- 👷 **Casco**
- 🦺 **Chaleco**
- 🙋 **Presencia humana**
""")

# 🎧 Funciones de voz con estilo

def generar_audio(texto):
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def reproducir_audio(mp3_fp):
    audio_bytes = mp3_fp.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# 🚀 Cargar modelo TFLite
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="mimodelitolindo.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # (alto, ancho)

# 🎯 Clases esperadas
required_classes = {'boots', 'helmet', 'vest', 'human'}
model_classes = ['boots', 'gloves', 'helmet', 'human', 'vest']  # Ajusta según el modelo real

# 💬 Nivel de confianza
st.markdown("**Selecciona el nivel mínimo de confianza para aceptar una clase detectada:**")
confianza = st.slider("Confianza (%)", min_value=0, max_value=100, value=50, step=1) / 100.0

# 📤 Carga de imagen
option = st.radio("Selecciona cómo subir la imagen:", ["📂 Archivo", "🌐 URL", "📸 Cámara"])
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

elif option == "📸 Cámara":
    camera_input = st.camera_input("Toma una foto")
    if camera_input:
        image = Image.open(camera_input).convert("RGB")

# 🔍 Procesamiento y predicción
if image:
    st.image(image, caption="📷 Imagen cargada", use_column_width=True)

    # Preprocesamiento
    resized_img = image.resize((input_shape[1], input_shape[0]))
    input_data = np.expand_dims(resized_img, axis=0).astype(np.float32) / 255.0

    # Inferencia
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Interpretación multiclase con umbral de confianza
    predicted_labels = []
    for i, prob in enumerate(output_data[0][:len(model_classes)]):
        flat_prob = np.ravel(prob)
        if flat_prob.size == 1:
            value = float(flat_prob[0])
            if value > confianza:
                predicted_labels.append(model_classes[i])

    detected_set = set(predicted_labels)
    faltantes = required_classes - detected_set

    st.subheader("📊 Resultado de la Predicción")
    if not faltantes:
        st.success("✅ ¡Estás listo para trabajar! Todos los elementos de seguridad están presentes.")
        audio_text = "¡Felicidades compañero! Estás listo para trabajar."
    else:
        st.error(f"⚠️ Lo siento compañero, no estás preparado para trabajar. Te falta: {', '.join(faltantes)}")
        audio_text = f"Lo siento compañero. No estás listo para trabajar. Te falta: {', '.join(faltantes)}."

    # Mostrar clases detectadas
    st.markdown("""
    ---
    **Detectado:**
    - """ + "\n    - ".join(predicted_labels if predicted_labels else ["Nada detectado"]))

    # Reproducir audio del resultado
    mp3 = generar_audio(audio_text)
    reproducir_audio(mp3)

else:
    st.info("✨ Sube una imagen, pega una URL o toma una foto para comenzar.")
