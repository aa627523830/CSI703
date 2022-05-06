import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('london_merged.csv', parse_dates=True, index_col='timestamp')

st.title("Visualizing correlation, comparisons, and trends")  # add a title
st.write(df)  # visualize my dataframe in the Streamlit app

with st.echo(code_location='below'):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    sns.pointplot(ax =ax,data=df, x=df.index.hour, y='cnt', hue='is_holiday')
    ax.set_title("Bicycle share usage volumn in holiday and non-holiday")
    ax.set_xlabel("Day Hour")
    ax.set_ylabel("Bicycle share usage volumn")
    st.write(fig)

    
with st.echo(code_location='below'):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    sns.pointplot(data=df, x=df.index.hour, y='cnt', hue='is_weekend')
    ax.set_title("Bicycle share usage volumn in weekend and non-weekend")
    ax.set_xlabel("Day Hour")
    ax.set_ylabel("Bicycle share usage volumn")
    st.write(fig)

    
with st.echo(code_location='below'):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    sns.pointplot(data=df, x=df.index.hour, y='cnt', hue='season')
    ax.set_title("Bicycle share usage volumn in different season")
    ax.set_xlabel("Day Hour")
    ax.set_ylabel("Bicycle share usage volumn")
    st.write(fig)
