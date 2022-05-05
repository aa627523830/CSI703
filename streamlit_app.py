import streamlit as st
import pandas as pd
from vega_datasets import data


"""
# Using different charting libraries
"""

@st.cache
def get_data():
    return pd.read_json(data.cars.url)

df = get_data()


"""
Streamlit supports different charting libraries, like:

* Matplotlib
* Seaborn
* Altair
* Plotly
* Bokeh
"""


"""
## Altair example
"""

with st.echo(code_location='below'):
    import altair as alt

    st.write(alt.Chart(df).mark_point().encode(
        # The notation below is shorthand for:
        # x = alt.X("Acceleration", type="quantitative", title="Acceleration"),
        x="Acceleration:Q",

        y=alt.Y("Miles_per_Gallon", type="quantitative", title="Miles per gallon"),
    ))


"""
## Plotly example
"""

with st.echo(code_location='below'):
    import plotly.express as px

    fig = px.scatter(
        x=df["Acceleration"],
        y=df["Miles_per_Gallon"],
    )
    fig.update_layout(
        xaxis_title="Acceleration",
        yaxis_title="Miles per gallon",
    )

    st.write(fig)


"""
## Matplotlib example
"""

with st.echo(code_location='below'):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(
        df["Acceleration"],
        df["Miles_per_Gallon"],
    )

    ax.set_xlabel("Acceleration")
    ax.set_ylabel("Miles per gallon")

    st.write(fig)
