import streamlit as st
import pandas as pd
import numpy as np

st.title("Visualizing geospatial data")  # add a title
df = pd.read_csv('london_merged.csv', parse_dates=True, index_col='timestamp')
st.map(df)
