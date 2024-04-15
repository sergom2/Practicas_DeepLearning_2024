import streamlit as st
from PIL import Image, ImageOps
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Cargar modelo ResNet50
def load_resnet50_model():
    weights = ResNet50_Weights.DEFAULT if False else None
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15)
    model.load_state_dict(torch.load('models/model_1_grande.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Cargar modelo ResNet152
def load_resnet152_model():
    weights = ResNet152_Weights.DEFAULT if False else None
    model = models.resnet152(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15)
    model.load_state_dict(torch.load('models/resnet152_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model_resnet50 = load_resnet50_model()
model_resnet152 = load_resnet152_model()

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

clases = ["Bedroom", "Coast", "Forest", "Highway", "Industrial", 
          "Inside city", "Kitchen", "Living room", "Mountain", "Office", 
          "Open country", "Store", "Street", "Suburb", "Tall building"]

archivo_subido = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if archivo_subido is not None:
    imagen = Image.open(archivo_subido).convert('RGB')
    st.image(imagen, caption='Imagen Original', use_column_width=True)
    imagen_transformada = transformar_imagen(imagen)

    # Predicción con ResNet50
    with torch.no_grad():
        prediccion_resnet50 = model_resnet50(imagen_transformada)
        _, predicted_class = torch.max(prediccion_resnet50, 1)
        predicted_class_name = clases[predicted_class.item()]
        probabilidades_resnet50 = F.softmax(prediccion_resnet50, dim=1).squeeze()
        prob_max_resnet50 = probabilidades_resnet50[predicted_class].item()

    # Revisar si la probabilidad máxima es inferior a 0.9
    if prob_max_resnet50 < 0.9:
        # Predicción con ResNet152
        with torch.no_grad():
            prediccion_resnet152 = model_resnet152(imagen_transformada)
            _, predicted_class_152 = torch.max(prediccion_resnet152, 1)
            predicted_class_name_152 = clases[predicted_class_152.item()]
            probabilidades_resnet152 = F.softmax(prediccion_resnet152, dim=1).squeeze()
            prob_max_resnet50 = probabilidades_resnet152[predicted_class].item()

        # Comprobar si las dos principales clases coinciden
        top2_resnet50 = torch.topk(probabilidades_resnet50, 2).indices.tolist()
        top2_resnet152 = torch.topk(probabilidades_resnet152, 2).indices.tolist()
        if top2_resnet50 == top2_resnet152:
            model_used = "ResNet50"
            final_pred = predicted_class_name
            final_probabilities = probabilidades_resnet50.numpy()
        else:
            model_used = "ResNet152"
            final_pred = predicted_class_name_152
            final_probabilities = probabilidades_resnet152.numpy()
    else:
        model_used = "ResNet50"
        final_pred = predicted_class_name
        final_probabilities = probabilidades_resnet50.numpy()

    st.write(f"Predicción hecha por {model_used}: {final_pred}")
    probabilidades_df = pd.DataFrame({
        'Clase': clases,
        'Probabilidad': final_probabilities
    })
    st.table(probabilidades_df.sort_values('Probabilidad', ascending=False))
