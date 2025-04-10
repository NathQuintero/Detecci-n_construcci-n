import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from ultralytics import YOLO
import tempfile
from gtts import gTTS
import base64
from io import BytesIO

# 🎧 Funciones de voz

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

# 🧠 Cargar modelos
modelo_personas = YOLO("yolov8n.pt")     # Detección de personas
modelo_ppe = YOLO("best.pt")             # Detección de PPE

# 🌟 Configuración de la página
st.set_page_config(page_title="Detector PPE - Angelly & Nathalia 💖", layout="wide")

# Encabezado
st.title("💼 Sistema Inteligente de uso de Equipos de Protección Personal 🦺")

st.markdown("""
Bienvenido al **Sistema Inteligente de uso de Equipos de Protección Personal (PPE)** 💡 desarrollado con cariño por **Angelly y Nathalia** 💖.  
Esta herramienta usa visión por computadora para verificar si estás listo para trabajar de forma segura.

---
""")

# Instrucciones
st.subheader("📌 ¿Cómo usar la app?")
st.markdown("""
1. Carga una imagen o toma una foto 📸.  
2. Haz clic en **Enviar Foto**.  
3. La IA detectará personas y evaluará el uso correcto de **casco**, **chaleco** y **botas**.
""")

# Tabs para imagen o cámara
tab1, tab2 = st.tabs(["📁 Subir Imagen", "📷 Tomar Foto"])
imagen_original = None
procesar = False

with tab1:
    foto = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
    if st.button("📤 Enviar Foto", key="upload"):
        if foto:
            imagen_original = Image.open(foto)
            procesar = True
        else:
            st.warning("Por favor, sube una imagen primero.")

with tab2:
    captura = st.camera_input("Toma una foto")
    if st.button("📤 Enviar Foto", key="camera"):
        if captura:
            imagen_original = Image.open(captura)
            procesar = True
        else:
            st.warning("Por favor, toma una foto antes de continuar.")

# Procesamiento
if procesar and imagen_original:
    st.subheader("🔍 Imagen cargada")
    st.image(imagen_original, use_container_width=True)

    # Convertir imagen a formato OpenCV
    img_cv = np.array(imagen_original)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Detección de personas
    resultados_personas = modelo_personas(img_cv)[0]
    personas_detectadas = [r for r in resultados_personas.boxes.data.cpu().numpy() if int(r[5]) == 0]

    st.subheader(f"👥 Personas detectadas: {len(personas_detectadas)}")

    # Evaluar cada persona
    for i, persona in enumerate(personas_detectadas, start=1):
        x1, y1, x2, y2, conf, clase = map(int, persona[:6])
        persona_img = img_cv[y1:y2, x1:x2]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            cv2.imwrite(temp_file.name, persona_img)
            resultados_ppe = modelo_ppe(temp_file.name)[0]
            etiquetas_detectadas = [modelo_ppe.names[int(d.cls)] for d in resultados_ppe.boxes]

            for box in resultados_ppe.boxes:
                x1o, y1o, x2o, y2o = map(int, box.xyxy[0])
                label = modelo_ppe.names[int(box.cls[0])]
                conf = float(box.conf[0])
                cv2.rectangle(persona_img, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
                cv2.putText(persona_img, f"{label} {conf:.2f}", (x1o, y1o - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            st.markdown(f"### 👤 Persona {i}")
            st.image(persona_img, caption="Objetos detectados", channels="BGR", width=300)
            st.markdown("**🧾 Objetos detectados:** " + ", ".join(etiquetas_detectadas))

            requeridos = {"casco", "chaleco", "botas"}
            presentes = set(etiquetas_detectadas)

            if requeridos.issubset(presentes):
                mensaje = "✅ ¡Estás listo para trabajar compañero!"
                st.success("✅ ¡Estás listo para trabajar compañero!")
            else:
                faltantes = requeridos - presentes
                mensaje = f"❌ Lo siento compañero, no estás listo para trabajar. Te falta: {', '.join(faltantes)}."
                st.error(mensaje)

            # 🎧 Reproducir audio
            audio_fp = generar_audio(mensaje)
            reproducir_audio(audio_fp)

    st.markdown("---")
    st.markdown("**Hecho con 💖 por Angelly y Nathalia - UNAB 2025**")
else:
    st.info("✨ Sube una imagen o toma una foto para comenzar.")
