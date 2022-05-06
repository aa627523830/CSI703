import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

df = pd.read_csv('london_merged.csv', parse_dates=True, index_col='timestamp')
df_1D=df.resample('1D').sum()

st.title("Visualizing distributions and part-to-whole")  # add a title
st.write(df)  # visualize my dataframe in the Streamlit app

#with st.echo(code_location='below'):

  #  fig = plt.figure()
   # ax = fig.add_subplot(1,1,1)

  #  sns.lineplot(x=df_1D.index, y='cnt', data=df_1D)
  #  ax.set_title("The distribution of bicycle usage over time")
   # ax.set_xlabel("timestamp")
  #  ax.set_ylabel("Bicycle share usage volumn")
  #  st.write(fig)
    
#st.line_chart(df_1D.cnt)
#df_melted = pd.melt(df,id_vars=['date'],var_name='parameter', value_name='value')
c = alt.Chart(df_1D, title='measure of different elements over time').mark_line().encode(
     x='cnt', color='parameter')

st.altair_chart(c)
