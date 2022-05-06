import streamlit as st
import pandas as pd
import numpy as np

st.title("Visualizing geospatial data")  # add a title
df = pd.read_csv('dataset_200k.csv')
st.map(df)
