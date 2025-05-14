import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from ultralytics import YOLO
import tempfile
import requests
from io import BytesIO
from gtts import gTTS
import base64

# 🎧 Funciones de voz
def generar_audio(texto):
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def mostrar_audio(mp3_fp, label="Reproducir audio"):
    audio_bytes = mp3_fp.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    return audio_html

# 🧠 Cargar modelos
modelo_personas = YOLO("yolov8n.pt")
modelo_ppe = YOLO("best.pt")

# 🌟 Configuración de la página
st.set_page_config(page_title="Evaluador PPE Inteligente", layout="wide")

# ✨ Firma superior
st.markdown("""
<center>
    <p style='font-size:18px;'><strong>Hecho con dedicación por Angelly y Nathalia</strong><br>Todos los derechos reservados ©️</p>
</center>
""", unsafe_allow_html=True)

# 🖼️ Banner
st.image("banner.png", use_container_width=True)
st.markdown("""
<center>
    <h2>🦺 Bienvenido compañero a tu trabajo</h2>
    <p style='font-size: 16px;'>Es hora de evaluar tu equipo de protección personal...</p>
</center>
---
""", unsafe_allow_html=True)

# 🔍 Instrucciones con ejemplo visual
with st.expander("📖 ¿Cómo se usa esta herramienta?"):
    st.markdown("""
    - 📁 Puedes **subir una imagen desde tu dispositivo**
    - 🌐 O puedes **pegar la URL de una imagen** desde internet
    - 📷 También puedes **tomarte una foto con la cámara**
    ```
    Luego, selecciona los elementos de PPE a evaluar y haz clic en **'Analizar Imagen'** para verificar si cumples con los requerimientos de seguridad 🏗️🛡️
    ```
    Los resultados incluirán un audio para cada persona que podrás reproducir manualmente.
    """)
    st.image("ejemplo.png", caption="Ejemplo de foto válida", use_container_width=True)
    st.info("Recuerda mostrar todo tu cuerpo y tus elementos de protección en la imagen")

# 🔄 Selección de clases PPE
st.subheader("🛡️ Selecciona los elementos de PPE a evaluar")
clases_disponibles = ["casco", "chaleco", "botas", "guantes"]  # Ajusta según las clases de tu modelo
clases_seleccionadas = st.multiselect(
    "Elige uno o más elementos requeridos",
    options=clases_disponibles,
    default=["casco", "chaleco", "botas"]
)
if not clases_seleccionadas:
    st.warning("⚠️ Por favor, selecciona al menos un elemento de PPE para evaluar.")

# 🔄 Entrada de imagen con selector
st.subheader("📸 Selecciona cómo quieres subir la imagen")
opcion = st.selectbox("¿Cómo deseas ingresar la imagen?", ("Subir desde archivo", "Desde la cámara", "Desde una URL"))

imagen_original = None
procesar = False

if opcion == "Subir desde archivo":
    foto = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if st.button("📤 Analizar Imagen"):
        if foto:
            imagen_original = Image.open(foto)
            procesar = True
        else:
            st.warning("⚠️ Por favor, sube una imagen.")

elif opcion == "Desde la cámara":
    captura = st.camera_input("Toma una foto")
    if st.button("📤 Analizar Imagen"):
        if captura:
            imagen_original = Image.open(captura)
            procesar = True
        else:
            st.warning("⚠️ Toma una foto para continuar.")

elif opcion == "Desde una URL":
    url = st.text_input("🔗 Pega la URL de la imagen aquí")
    if st.button("📤 Analizar Imagen"):
        if url:
            try:
                response = requests.get(url)
                imagen_original = Image.open(BytesIO(response.content))
                procesar = True
            except:
                st.error("🚫 No se pudo cargar la imagen desde la URL proporcionada.")
        else:
            st.warning("⚠️ Ingresa una URL válida.")

# 🔎 Análisis de la imagen
if procesar and imagen_original and clases_seleccionadas:
    st.markdown("---")
    st.markdown("<center><h3>🔍 Imagen cargada</h3></center>", unsafe_allow_html=True)
    st.image(imagen_original, use_container_width=True)

    # Convertir imagen a OpenCV
    img_cv = np.array(imagen_original)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Detección de personas
    resultados_personas = modelo_personas(img_cv)[0]
    personas_detectadas = [r for r in resultados_personas.boxes.data.cpu().numpy() if int(r[5]) == 0]

    st.markdown(f"<center><h4>👥 Personas detectadas: {len(personas_detectadas)}</h4></center>", unsafe_allow_html=True)

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

            st.markdown(f"<center><h4>👤 Persona {i}</h4></center>", unsafe_allow_html=True)
            persona_img_encoded = base64.b64encode(cv2.imencode('.png', persona_img)[1]).decode()
            st.markdown(
                f'<center><img src="data:image/png;base64,{persona_img_encoded}" style="width: 300px; border-radius: 12px; box-shadow: 0px 4px 12px rgba(0,0,0,0.2);"/></center>',
                unsafe_allow_html=True
            )
            st.markdown("**🎒 Elementos detectados:** " + ", ".join(etiquetas_detectadas) if etiquetas_detectadas else "Ninguno")

            requeridos = set(clases_seleccionadas)
            presentes = set(etiquetas_detectadas)

            if requeridos.issubset(presentes):
                mensaje = f"Persona {i}: ¡Estás listo para trabajar compañero!"
                st.success(mensaje)
                st.image("ok.png", use_container_width=True)
                st.balloons()
            else:
                faltantes = requeridos - presentes
                mensaje = f"Persona {i}: Lo siento compañero, no estás listo para trabajar. Te falta: {', '. huis(faltantes)}."
                st.error(mensaje)
                st.image("No.png", use_container_width=True)
                st.snow()

            # Generar audio individual para esta persona
            audio_fp = generar_audio(mensaje)
            st.markdown(f"**🎧 Audio para Persona {i}:**")
            st.markdown(mostrar_audio(audio_fp, f"Reproducir audio Persona {i}"), unsafe_allow_html=True)
            audio_fp.seek(0)
            st.download_button(
                label=f"📥 Descargar audio Persona {i}",
                data=audio_fp,
                file_name=f"resultado_persona_{i}.mp3",
                mime="audio/mp3"
            )

    st.markdown("---")
else:
    st.info("🧠 Esperando imagen o selección de elementos... ¡Selecciona los elementos de PPE, una forma de carga y haz clic en 'Analizar Imagen'!")
