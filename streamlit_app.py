import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

df = pd.read_csv('london_merged.csv', parse_dates=True, index_col='timestamp')

st.title("Visualizing correlation, comparisons, and trends")  # add a title
st.write(df)  # visualize my dataframe in the Streamlit app
df['is_holiday'].map(dict(yes=1, no=0))
df['is_weekend'].map(dict(yes=1, no=0))
df['season'].map(dict(Spring=0, Summer=1, Autumn=2, Winter=3))

#with st.echo(code_location='below'):

   # fig = plt.figure()
   # ax = fig.add_subplot(1,1,1)

   # sns.pointplot(ax =ax,data=df, x=df.index.hour, y='cnt', hue='is_holiday')
   # ax.set_title("Bicycle share usage volumn in holiday and non-holiday")
   # ax.set_xlabel("Day Hour")
  #  ax.set_ylabel("Bicycle share usage volumn")
   # st.write(fig)


  
st.title("Stroke Prediction Dashboard")
st.markdown("The dashboard will help a researcher to get to know \
more about the given datasets and it's output")
st.sidebar.title("Select Visual Charts")
st.sidebar.markdown("Select the Charts/Plots accordingly:")

  
chart_visual = st.sidebar.selectbox('Select Charts/Plot type', 
                                    ('Line Chart', 'Bar Chart'))
  
st.sidebar.checkbox("Show Analysis by Smoking Status", True, key = 1)
selected_status = st.sidebar.selectbox('Select Smoking Status',
                                       options = ['is_holiday', 
                                                  'is_weekend', 'season'])
  
fig = go.Figure()
  
if chart_visual == 'Line Chart':
    if selected_status == 'is_holiday':
        fig.add_trace(go.Scatter(x = df.cnt, y = df['is_holiday'],
                                 mode = 'lines',
                                 name = 'is_holiday'))
    if selected_status == 'is_weekend':
        fig.add_trace(go.Scatter(x = df.cnt, y = df['is_weekend'],
                                 mode = 'lines', name = 'is_weekend'))
    if selected_status == 'season':
        fig.add_trace(go.Scatter(x = df.cnt, y = df['season'],
                                 mode = 'lines',
                                 name = 'season'))

elif chart_visual == 'Bar Chart':
    if selected_status == 'is_holiday':
        fig.add_trace(go.Scatter(x = df.cnt, y = df[['is_holiday']],
                                 mode = 'lines',
                                 name = 'is_holiday'))
    if selected_status == 'is_weekend':
        fig.add_trace(go.Scatter(x = df.cnt, y = df[['is_weekend']],
                                 mode = 'lines', name = 'is_weekend'))
    if selected_status == 'season':
        fig.add_trace(go.Scatter(x = df.cnt, y = df[['season']],
                                 mode = 'lines',
                                 name = 'season'))
  
st.plotly_chart(fig, use_container_width=True)
