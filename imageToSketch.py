import cv2
import streamlit as st
from PIL import Image
import numpy as np

st.title("Image to Sketch Converter")

@st.cache_data
def convert_to_sketch(image, ksize, sigma):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the image
    inverted_image = 255 - gray_image

    # Apply a Gaussian blur to the inverted image
    ksize = (ksize, ksize)
    if ksize[0] % 2 == 0:
        ksize = (ksize[0] + 1, ksize[1])
    if ksize[1] % 2 == 0:
        ksize = (ksize[0], ksize[1] + 1)
    blurred_image = cv2.GaussianBlur(inverted_image, ksize, sigma)

    # Blend the grayscale image with the blurred inverted image using the "color dodge" blend mode
    sketch_image = cv2.divide(gray_image, 255 - blurred_image, scale=256)

    return sketch_image

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Convert the PIL image to a NumPy array for OpenCV processing
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Get kernel size and sigma values from user
    ksize = st.slider("Kernel size", 1, 250, 21, step=20)
    sigma = st.slider("Sigma", 0, 1000, 0, step = 50)

    # Call the image conversion function
    sketch_image = convert_to_sketch(img_np, ksize, sigma)

    # Convert the sketch image back to a PIL image for display
    sketch_image_pil = Image.fromarray(sketch_image)

    st.image(sketch_image_pil, caption="Sketch Image", use_column_width=True)

    # Add a download button for the sketch image
    sketch_image_data = cv2.imencode(".jpg", sketch_image)[1].tobytes()
    st.download_button("Download Sketch Image", sketch_image_data, "sketch.jpg", "image/jpeg")
