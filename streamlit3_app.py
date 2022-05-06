import streamlit as st
from PIL import Image

image = Image.open('Shared bicycle rental process.png')
st.title("Visualizing concepts and qualitative data")  # add a title
st.image(image, caption='Shared bicycle rental process')
