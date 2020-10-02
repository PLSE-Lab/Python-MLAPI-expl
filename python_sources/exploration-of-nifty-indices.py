#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# Objective of the notebook is to explore the different NIFTY indices and see how they have changed over time.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
pd.set_option('max_columns', 100)

from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
output_notebook()


# ### NIFTY 50 Index

# In[ ]:


index_df = pd.read_csv("/kaggle/input/nifty-indices-dataset/NIFTY 50.csv", parse_dates=["Date"])

def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

layout = go.Layout(
    title=go.layout.Title(
        text="NIFTY 50 Closing values over time",
        x=0.5
    ),
    xaxis_title="Date",
    yaxis_title="Index close value",
    font=dict(size=14),
    width=800,
    height=500,
)

cnt_srs = index_df["Close"]
cnt_srs.index = index_df["Date"]
trace = scatter_plot(cnt_srs, "blue")
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="fig")


# In[ ]:


index_df["Month"] = index_df["Date"].dt.month
index_df["Year"] = index_df["Date"].dt.year
index_monthly_df = index_df.groupby(["Year", "Month"]).first().reset_index()

trace = go.Heatmap(
        x=(index_monthly_df["Month"].values)[::-1],
        y=(index_monthly_df["Year"].values)[::-1],
        z=(index_monthly_df["P/E"].values)[::-1],
        colorscale="rdylgn_r"
    )

layout = go.Layout(
    title=go.layout.Title(
        text="Historical P/E values of NIFTY 50 (first trading day of each month)",
        x=0.5
    ),
    xaxis_title="Month",
    yaxis_title="Year",
    font=dict(size=14),
    width=800,
    height=800,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.update_xaxes(nticks=24)
fig.update_yaxes(nticks=40)
py.iplot(fig, filename="fig")


# In[ ]:


trace = go.Heatmap(
        x=(index_monthly_df["Month"].values)[::-1],
        y=(index_monthly_df["Year"].values)[::-1],
        z=(index_monthly_df["P/B"].values)[::-1],
        colorscale="rdylgn_r"
    )

layout = go.Layout(
    title=go.layout.Title(
        text="Historical P/B values of NIFTY 50 (first trading day of each month)",
        x=0.5
    ),
    xaxis_title="Month",
    yaxis_title="Year",
    font=dict(size=14),
    width=800,
    height=800,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.update_xaxes(nticks=24)
fig.update_yaxes(nticks=40)
py.iplot(fig, filename="fig")


# ### NIFTY Next 50 Index

# In[ ]:


index_df = pd.read_csv("/kaggle/input/nifty-indices-dataset/NIFTY NEXT 50.csv", parse_dates=["Date"])
index_name = "NIFTY Next 50"

############# Time series graph ################
def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

layout = go.Layout(
    title=go.layout.Title(
        text=f"{index_name} Closing values over time",
        x=0.5
    ),
    xaxis_title="Date",
    yaxis_title="Index close value",
    font=dict(size=14),
    width=800,
    height=500,
)

cnt_srs = index_df["Close"]
cnt_srs.index = index_df["Date"]
trace = scatter_plot(cnt_srs, "blue")
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="fig")


########## P/E Graph ##################
index_df["Month"] = index_df["Date"].dt.month
index_df["Year"] = index_df["Date"].dt.year
index_monthly_df = index_df.groupby(["Year", "Month"]).first().reset_index()

trace = go.Heatmap(
        x=(index_monthly_df["Month"].values)[::-1],
        y=(index_monthly_df["Year"].values)[::-1],
        z=(index_monthly_df["P/E"].values)[::-1],
        colorscale="rdylgn_r"
    )

layout = go.Layout(
    title=go.layout.Title(
        text=f"Historical P/E values of {index_name} (first trading day of each month)",
        x=0.5
    ),
    xaxis_title="Month",
    yaxis_title="Year",
    font=dict(size=14),
    width=800,
    height=800,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.update_xaxes(nticks=24)
fig.update_yaxes(nticks=40)
py.iplot(fig, filename="fig")

########## P/B Plot ##################
trace = go.Heatmap(
        x=(index_monthly_df["Month"].values)[::-1],
        y=(index_monthly_df["Year"].values)[::-1],
        z=(index_monthly_df["P/B"].values)[::-1],
        colorscale="rdylgn_r"
    )

layout = go.Layout(
    title=go.layout.Title(
        text=f"Historical P/B values of {index_name} (first trading day of each month)",
        x=0.5
    ),
    xaxis_title="Month",
    yaxis_title="Year",
    font=dict(size=14),
    width=800,
    height=800,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.update_xaxes(nticks=24)
fig.update_yaxes(nticks=40)
py.iplot(fig, filename="fig")


# ### NIFTY 500 Index

# In[ ]:


index_df = pd.read_csv("/kaggle/input/nifty-indices-dataset/NIFTY 500.csv", parse_dates=["Date"])
index_name = "NIFTY 500"

############# Time series graph ################
def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

layout = go.Layout(
    title=go.layout.Title(
        text=f"{index_name} Closing values over time",
        x=0.5
    ),
    xaxis_title="Date",
    yaxis_title="Index close value",
    font=dict(size=14),
    width=800,
    height=500,
)

cnt_srs = index_df["Close"]
cnt_srs.index = index_df["Date"]
trace = scatter_plot(cnt_srs, "blue")
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="fig")


########## P/E Graph ##################
index_df["Month"] = index_df["Date"].dt.month
index_df["Year"] = index_df["Date"].dt.year
index_monthly_df = index_df.groupby(["Year", "Month"]).first().reset_index()

trace = go.Heatmap(
        x=(index_monthly_df["Month"].values)[::-1],
        y=(index_monthly_df["Year"].values)[::-1],
        z=(index_monthly_df["P/E"].values)[::-1],
        colorscale="rdylgn_r"
    )

layout = go.Layout(
    title=go.layout.Title(
        text=f"Historical P/E values of {index_name} (first trading day of each month)",
        x=0.5
    ),
    xaxis_title="Month",
    yaxis_title="Year",
    font=dict(size=14),
    width=800,
    height=800,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.update_xaxes(nticks=24)
fig.update_yaxes(nticks=40)
py.iplot(fig, filename="fig")

########## P/B Plot ##################
trace = go.Heatmap(
        x=(index_monthly_df["Month"].values)[::-1],
        y=(index_monthly_df["Year"].values)[::-1],
        z=(index_monthly_df["P/B"].values)[::-1],
        colorscale="rdylgn_r"
    )

layout = go.Layout(
    title=go.layout.Title(
        text=f"Historical P/B values of {index_name} (first trading day of each month)",
        x=0.5
    ),
    xaxis_title="Month",
    yaxis_title="Year",
    font=dict(size=14),
    width=800,
    height=800,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.update_xaxes(nticks=24)
fig.update_yaxes(nticks=40)
py.iplot(fig, filename="fig")


# ### NIFTY BANK

# In[ ]:


index_df = pd.read_csv("/kaggle/input/nifty-indices-dataset/NIFTY BANK.csv", parse_dates=["Date"])
index_name = "NIFTY BANK"

############# Time series graph ################
def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

layout = go.Layout(
    title=go.layout.Title(
        text=f"{index_name} Closing values over time",
        x=0.5
    ),
    xaxis_title="Date",
    yaxis_title="Index close value",
    font=dict(size=14),
    width=800,
    height=500,
)

cnt_srs = index_df["Close"]
cnt_srs.index = index_df["Date"]
trace = scatter_plot(cnt_srs, "blue")
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="fig")


########## P/E Graph ##################
index_df["Month"] = index_df["Date"].dt.month
index_df["Year"] = index_df["Date"].dt.year
index_monthly_df = index_df.groupby(["Year", "Month"]).first().reset_index()

trace = go.Heatmap(
        x=(index_monthly_df["Month"].values)[::-1],
        y=(index_monthly_df["Year"].values)[::-1],
        z=(index_monthly_df["P/E"].values)[::-1],
        colorscale="rdylgn_r"
    )

layout = go.Layout(
    title=go.layout.Title(
        text=f"Historical P/E values of {index_name} (first trading day of each month)",
        x=0.5
    ),
    xaxis_title="Month",
    yaxis_title="Year",
    font=dict(size=14),
    width=800,
    height=800,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.update_xaxes(nticks=24)
fig.update_yaxes(nticks=40)
py.iplot(fig, filename="fig")

########## P/B Plot ##################
trace = go.Heatmap(
        x=(index_monthly_df["Month"].values)[::-1],
        y=(index_monthly_df["Year"].values)[::-1],
        z=(index_monthly_df["P/B"].values)[::-1],
        colorscale="rdylgn_r"
    )

layout = go.Layout(
    title=go.layout.Title(
        text=f"Historical P/B values of {index_name} (first trading day of each month)",
        x=0.5
    ),
    xaxis_title="Month",
    yaxis_title="Year",
    font=dict(size=14),
    width=800,
    height=800,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.update_xaxes(nticks=24)
fig.update_yaxes(nticks=40)
py.iplot(fig, filename="fig")


# In[ ]:




