import streamlit as st
from PIL import Image
from streamlit_player import st_player

image = Image.open('Shared bicycle rental process.png')
st.title("Visualizing concepts and qualitative data")  # add a title
st.image(image, caption='Shared bicycle rental process')

# Embed a youtube video
st_player("https://www.youtube.com/watch?v=xjmUhpO8vU0")
