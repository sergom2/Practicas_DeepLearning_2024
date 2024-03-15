import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import skimage
from PIL import Image

# ========================================================================
# Functions
# ========================================================================
def load_default_image():
    img=skimage.data.chelsea()
    # Convert the NumPy array to a PIL Image
    img = Image.fromarray(img)
    # Convert the image to grayscale
    img_gray = transforms.Grayscale(num_output_channels=1)(img)
    # Convert the grayscale image to a tensor
    img_tensor = transforms.ToTensor()(img_gray)
    # Normalize the tensor for plotting
    img_tensor = (img_tensor - torch.min(img_tensor)) / (torch.max(img_tensor) - torch.min(img_tensor))
    return {"img": img, "im_gray": img_gray, "img_tensor": img_tensor}

def get_filter(filter_name):
    if filter_name == "No Filter":
        filter = torch.tensor([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]], dtype=torch.float)

    elif filter_name == "Gaussian":
        filter = torch.tensor([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=torch.float)
    elif filter_name == "Sharpen":
        filter = torch.tensor([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]], dtype=torch.float)
    elif filter_name == "Edge Detection":
        filter = torch.tensor([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]], dtype=torch.float)
    else:
        raise ValueError("Invalid filter name")
    return filter

def convolve_and_plot(image, filter, stride=1, padding='same'):
    # Normalize the filter
    filter /= torch.sum(torch.abs(filter))
    # Reshape the filter to match the expected input shape
    filter = filter.unsqueeze(0).unsqueeze(0)
    # Apply the filter to the image tensor
    output_img = torch.nn.functional.conv2d(image.unsqueeze(0), 
                                            filter, 
                                            bias = None,
                                            stride=stride, 
                                            padding=padding)
    # Normalize the output for plotting
    output_img = (output_img - torch.min(output_img)) / (torch.max(output_img) - torch.min(output_img))
    output_img = output_img.squeeze()
    return output_img

# ========================================================================
# Application Config
# ========================================================================
st.set_page_config(layout="wide", page_title="DL Lab - Session 1")
col01, col02, col03 = st.columns(3)
with col01:
    st.write('# Deep Learning Lab - Session 1')
with col03:
    st.image("/Users/robertokramer/git_repos/personal/DeepLearning_Lab/streamlit/img/icai.png")

st.write("## Convolution Filters")

# Let's plot the default image
col11, col12, col13 = st.columns(3)
with col11:
    st.write("#### Original Image")
with col12:
    selected_filter_name = st.selectbox(
    'Select a Filter',
    ('No Filter','Gaussian', 'Sharpen', 'Edge Detection'))

col21, col22, col23 = st.columns(3)
with col21:
    default_image = load_default_image()
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.imshow(default_image["img_tensor"].squeeze(), cmap='gray')
    plt.colorbar()
    st.pyplot(fig)
with col22:
    filter = get_filter(selected_filter_name)
    output_image = convolve_and_plot(default_image["img_tensor"], filter)
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.imshow(output_image.squeeze(), cmap='gray')
    plt.colorbar()
    st.pyplot(fig)
