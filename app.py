import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Title of the app
st.title("Image Transformation App")

# Sidebar for image upload
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Slider for brightness adjustment
brightness_factor = st.sidebar.slider("Adjust Brightness", -100, 100, 0)

# Slider for saturation adjustment
saturation_factor = st.sidebar.slider("Adjust Saturation", 0, 200, 100)

# Slider for blur intensity
blur_intensity = st.sidebar.slider("Adjust Blur Intensity", 1, 20, 1)

# Slider for resizing
resize_factor = st.sidebar.slider("Resize Factor (%)", 50, 200, 100)

# Checkboxes for other transformations
grayscale = st.sidebar.checkbox("Convert to Grayscale")
rgb_to_hsv = st.sidebar.checkbox("Convert RGB to HSV")

if uploaded_file:
    # Load image
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Original Image", use_container_width=True)

    # Apply brightness adjustment
    if brightness_factor != 0:
        image = cv2.convertScaleAbs(image, alpha=1, beta=brightness_factor)

    # Apply saturation adjustment
    if saturation_factor != 100:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[..., 1] *= (saturation_factor / 100)
        hsv[..., 1][hsv[..., 1] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Apply blur
    if blur_intensity > 1:
        image = cv2.GaussianBlur(image, (blur_intensity * 2 + 1, blur_intensity * 2 + 1), 0)

    # Convert to grayscale
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert RGB to HSV
    if rgb_to_hsv:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Resize the image
    if resize_factor != 100:
        width = int(image.shape[1] * (resize_factor / 100))
        height = int(image.shape[0] * (resize_factor / 100))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # Display the transformed image
    st.image(image, caption="Transformed Image", use_container_width=True)
