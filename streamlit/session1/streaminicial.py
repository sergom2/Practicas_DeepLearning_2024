import streamlit as st
from PIL import Image, ImageOps
import torch
from torchvision import models, transforms
import torch.nn as nn
from torchvision.models import ResNet50_Weights
import pandas as pd
import torch.nn.functional as F

def load_model():
    weights = ResNet50_Weights.DEFAULT if False else None
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15)
    model.load_state_dict(torch.load('models/model_1_grande.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

clases = ["Bedroom", "Coast", "Forest", "Highway", "Industrial", 
          "Inside city", "Kitchen", "Living room", "Mountain", "Office", 
          "Open country", "Store", "Street", "Suburb", "Tall building"]

def transformar_imagen(imagen):
    imagen_bn = ImageOps.grayscale(imagen)
    transformaciones = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transformaciones(imagen_bn).unsqueeze(0)

st.title("Clasificador de Imágenes con PyTorch")

archivo_subido = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if archivo_subido is not None:
    imagen = Image.open(archivo_subido).convert('RGB')
    st.image(imagen, caption='Imagen Original', use_column_width=True)
    imagen_transformada = transformar_imagen(imagen)
    with torch.no_grad():
        prediccion = model(imagen_transformada)
        _, predicted_class = torch.max(prediccion, 1)
        predicted_class_name = clases[predicted_class.item()]
    
    st.write(f"Predicción: {predicted_class_name}")

    
    # Aplicar Softmax para obtener las probabilidades
    probabilidades = F.softmax(prediccion, dim=1).squeeze().tolist()
    
    # Crear un DataFrame para mostrar las probabilidades
    probabilidades_df = pd.DataFrame({
        'Clase': clases,
        'Probabilidad': probabilidades
    })

    st.table(probabilidades_df.sort_values('Probabilidad', ascending=False))