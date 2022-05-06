import streamlit as st
import pandas as pd
import numpy as np


df = pd.read_csv('london_merged.csv', parse_dates=True, index_col='timestamp')
st.map(df)
