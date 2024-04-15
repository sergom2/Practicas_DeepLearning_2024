import streamlit as st
from PIL import Image, ImageOps
import torch
from torchvision import models, transforms
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import joblib
import numpy as np

def load_pytorch_model(model_name):
    if model_name == 'ResNet50':
        model_path = 'models/model_1_grande.pth'
        weights = ResNet50_Weights.DEFAULT if False else None
        model = models.resnet50(weights=weights)
    elif model_name == 'ResNet152':
        model_path = 'models/resnet152_model.pth'
        weights = ResNet152_Weights.DEFAULT if False else None
        model = models.resnet152(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_stacking_model():
    return joblib.load('models/stacked_model.joblib')

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

def get_predictions_stacking(image, model_resnet50, model_resnet152):
    image_tensor = transformar_imagen(image)
    with torch.no_grad():
        features_resnet50 = torch.nn.functional.softmax(model_resnet50(image_tensor), dim=1)
        features_resnet152 = torch.nn.functional.softmax(model_resnet152(image_tensor), dim=1)
    features = np.hstack([features_resnet50.numpy(), features_resnet152.numpy()])
    stacked_model = load_stacking_model()
    predictions = stacked_model.predict_proba(features)[0]
    return predictions

st.title("Clasificador de Imágenes con PyTorch")

model_option = st.selectbox("Selecciona un modelo para la predicción", ["ResNet50", "ResNet152", "Stacking"])
model_resnet50 = load_pytorch_model('ResNet50')
model_resnet152 = load_pytorch_model('ResNet152')

clases = ["Bedroom", "Coast", "Forest", "Highway", "Industrial", 
          "Inside city", "Kitchen", "Living room", "Mountain", "Office", 
          "Open country", "Store", "Street", "Suburb", "Tall building"]

archivo_subido = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if archivo_subido is not None:
    imagen = Image.open(archivo_subido).convert('RGB')
    st.image(imagen, caption='Imagen Original', use_column_width=True)
    
    if model_option == "Stacking":
        probabilities = get_predictions_stacking(imagen, model_resnet50, model_resnet152)
        predicted_class_index = np.argmax(probabilities)
        predicted_class_name = clases[predicted_class_index]
        # Convertir las probabilidades a un formato adecuado para mostrar
        probabilidades_df = pd.DataFrame({
            'Clase': clases,
            'Probabilidad': probabilities
        })
    else:
        model = model_resnet50 if model_option == "ResNet50" else model_resnet152
        imagen_transformada = transformar_imagen(imagen)
        with torch.no_grad():
            prediccion = model(imagen_transformada)
            _, predicted_class = torch.max(prediccion, 1)
            predicted_class_name = clases[predicted_class.item()]
            probabilidades = F.softmax(prediccion, dim=1).squeeze().tolist()
            # Crear un DataFrame para mostrar las probabilidades
            probabilidades_df = pd.DataFrame({
                'Clase': clases,
                'Probabilidad': probabilidades
            })

    st.write(f"Predicción: {predicted_class_name}")
    st.table(probabilidades_df.sort_values('Probabilidad', ascending=False))
