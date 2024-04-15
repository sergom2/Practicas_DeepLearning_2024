import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import torchvision
# Asegúrate de que cnn.py contenga CNN, load_data, y load_model_weights adecuadamente definidos
from cnn import CNN, load_model_weights

# Cargar modelo
num_classes = 15  # Asegúrate de que este número coincida con el número de clases en tu dataset
model_weights = load_model_weights('resnet50-1epoch')
modelo = CNN(torchvision.models.resnet50(pretrained=True), num_classes)
modelo.load_state_dict(model_weights, strict=True)  # Cambiado a strict=True
modelo.eval()  # Asegúrate de poner el modelo en modo evaluación

def transformar_imagen(imagen):
    """Transforma la imagen para que sea adecuada para el modelo."""
    transformaciones = transforms.Compose([
        transforms.Resize((224, 224)),  # Tamaño de imagen esperado por el modelo
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
    with torch.no_grad():
        prediccion = modelo(imagen_transformada)
        probabilidades = F.softmax(prediccion, dim=1)
        _, predicted_class = torch.max(probabilidades, 1)
        
    # Asignar a cada índice su clase correspondiente (ajusta según tus clases)
    clases = ["Bedroom", "Coast", "Forest", "Highway", "Industrial", "Inside city", 
              "Kitchen", "Living room", "Mountain", "Office", "Open country", "Store",
               "Street", "Suburb", "Tall building"]  # Asegúrate de rellenar con tus nombres de clase reales
    # Crear un DataFrame para mostrar probabilidades por clase
    import pandas as pd
    prob_df = pd.DataFrame({
        'Clase': clases,
        'Probabilidad': probabilidades.numpy().flatten()
    }).sort_values('Probabilidad', ascending=False)
    
    st.write(f"Predicción más probable: {clases[predicted_class.item()]} con una probabilidad de {probabilidades[0, predicted_class.item()].item():.4f}")
    st.write("Probabilidades por clase:")
    st.dataframe(prob_df)
