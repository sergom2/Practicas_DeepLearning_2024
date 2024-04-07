import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from cnn import CNN
from cnn import load_data
import torchvision
from cnn import load_model_weights

# Actualiza esta ruta para que coincida con la ubicación de tu modelo
#modelo_path = 'C:/Users/sergi/OneDrive/Escritorio/MBD/Machine Learning 2/Practicas_DeepLearning_2024/03TransferLearning/models/resnet50-1epoch.pt'
#modelo = torch.load(modelo_path, map_location=torch.device('cpu'))
num_classes = 15
model_weights = load_model_weights('resnet50-1epoch')
modelo = CNN(torchvision.models.resnet50(weights='DEFAULT'), num_classes)
modelo.load_state_dict(model_weights, strict=False)

def transformar_imagen(imagen):
    """Transforma la imagen para que sea adecuada para el modelo."""
    transformaciones = transforms.Compose([
        transforms.Resize((224, 224)), # Tamaño de imagen esperado por el modelo
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalización ImageNet
    ])
    return transformaciones(imagen).unsqueeze(0)

st.title("Clasificador de Imágenes con PyTorch")

archivo_subido = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if archivo_subido is not None:
    imagen = Image.open(archivo_subido).convert('RGB')
    st.image(imagen, caption='Imagen Subida', use_column_width=True)
    imagen_transformada = transformar_imagen(imagen)
    
    st.write("Clasificando...")
    print(imagen_transformada)
    with torch.no_grad():  # Indica a PyTorch que no necesitamos gradiente para esta inferencia
        prediccion = modelo(imagen_transformada)
        # Asume que tu modelo devuelve un tensor logits y que estás interesado en la clase más probable.
        _, predicted_class = torch.max(prediccion, 1)
    
    st.write(f"Predicción: {predicted_class.item()}")  # Ajusta la interpretación según tus clases
