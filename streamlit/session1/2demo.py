import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

def load_model(model_choice):
    model = None
    if model_choice == 'model_1_grande':
        model = models.resnet50(weights=None)
    elif model_choice == 'efficientnet_b0_model':
        model = models.efficientnet_b0(weights=None)
    elif model_choice == 'densenet121_model':
        model = models.densenet121(weights=None)
    elif model_choice == 'mobilenet_v2_model':
        model = models.mobilenet_v2(weights=None)
    elif model_choice == 'shufflenet_v2_x1_0_model':
        model = models.shufflenet_v2_x1_0(weights=None)
    
    if model is not None:
        num_ftrs = model.fc.in_features if hasattr(model, 'fc') else model.classifier[1].in_features
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(num_ftrs, 15)
        else:
            model.classifier[1] = nn.Linear(num_ftrs, 15)

        model.load_state_dict(torch.load(f'models/{model_choice}.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        raise ValueError(f"No model found for the choice: {model_choice}")

st.title("Clasificador de Imágenes con PyTorch")

model_options = ['model_1_grande', 'efficientnet_b0_model', 'densenet121_model', 'mobilenet_v2_model', 'shufflenet_v2_x1_0_model']
selected_model = st.selectbox("Selecciona un modelo:", model_options)

model = load_model(selected_model)

clases = ["Bedroom", "Coast", "Forest", "Highway", "Industrial",
          "Inside city", "Kitchen", "Living room", "Mountain", "Office",
          "Open country", "Store", "Street", "Suburb", "Tall building"]

def transformar_imagen(imagen):
    transformaciones = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transformaciones(imagen).unsqueeze(0)

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
        probabilidades = F.softmax(prediccion, dim=1).squeeze().tolist()
        probabilidades_df = pd.DataFrame({
            'Clase': clases,
            'Probabilidad': probabilidades
        })
        st.table(probabilidades_df.sort_values('Probabilidad', ascending=False))
