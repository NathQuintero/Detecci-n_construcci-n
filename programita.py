import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO
import tempfile

# Carga el modelo YOLOv8 entrenado
model = YOLO("mimodelitolindo.tflite")  # Si es un modelo YOLOv8 normal usa .pt en vez de .tflite

# Clases requeridas
required_classes = {'helmet', 'boots', 'vest', 'human'}

# Página bonita
st.set_page_config(page_title="Detector de EPP - ¡Listo para trabajar!", page_icon="🦺")
st.title("🛠️ Detector de Equipos de Protección Personal (EPP)")
st.subheader("¿Estás listo para trabajar?")
st.write("Sube una imagen o pega una URL. Detectaremos si estás usando casco, chaleco, botas y... bueno, si eres humano 😅")

# Entrada de imagen
option = st.radio("Selecciona una opción:", ["Subir archivo", "Pegar URL"])

image = None
if option == "Subir archivo":
    uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif option == "Pegar URL":
    url = st.text_input("Pega la URL de la imagen aquí")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except:
            st.error("No se pudo cargar la imagen desde la URL")

if image:
    st.image(image, caption="Imagen original", use_column_width=True)

    # Guardar temporalmente para predecir
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)
        results = model(temp.name)

    # Mostrar resultados y procesar clases
    detected_classes = set()
    annotated_img = results[0].plot()
    for box in results[0].boxes.data:
        cls_id = int(box[-1])
        class_name = model.names[cls_id]
        detected_classes.add(class_name)

    # Mostrar la imagen anotada
    st.image(annotated_img, caption="Predicción del modelo", use_column_width=True)

    # Evaluar si está listo para trabajar
    faltantes = required_classes - detected_classes
    if not faltantes:
        st.success("✅ Esta persona está lista para trabajar 💪")
    else:
        st.error(f"⚠️ Lo siento compañero, no estás preparado para trabajar. Te falta: {', '.join(faltantes)}")

