import streamlit as st
from PIL import Image

image = Image.open('Shared bicycle rental process.png')

st.image(image, caption='Shared bicycle rental process')
