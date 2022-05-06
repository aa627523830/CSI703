import pandas as pd
import streamlit as st

df = pd.read_csv('london_merged.csv', parse_dates=True, index_col='timestamp')

st.title("Hello world!")  # add a title
st.write(df)  # visualize my dataframe in the Streamlit app
