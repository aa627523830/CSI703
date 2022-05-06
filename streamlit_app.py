import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('london_merged.csv', parse_dates=True, index_col='timestamp')

st.title("Hello world!")  # add a title
st.write(df)  # visualize my dataframe in the Streamlit app


#plt.figure(figsize=(10, 4))
#fig, ax = plt.subplots()
#sns.pointplot(ax = ax, data=df, x=df.index.hour, y='cnt', hue='is_holiday')#.set(title='Bicycle share usage volumn in holiday and non-holiday')
#st.pyplot(fig)

with st.echo(code_location='below'):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(
        df.index.hour,
        df['cnt'],
    )

    ax.set_xlabel("Acceleration")
    ax.set_ylabel("Miles per gallon")
    st.write(fig)
