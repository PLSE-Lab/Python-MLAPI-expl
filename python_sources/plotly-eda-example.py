#!/usr/bin/env python
# coding: utf-8

# (Updated on March 27)

# In[ ]:


import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# Plotly installation: https://plot.ly/python/getting-started/#jupyterlab-support-python-35


# In[ ]:


df = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')
df=df[df["Date"]>"2020-03-09"] # Only keep dates with confirmed cases
df.head()


# ## Confirmed Cases

# In[ ]:


# Reference: https://plot.ly/python/time-series/
fig = go.Figure(
    [go.Scatter(x=df['Date'], y=df['ConfirmedCases'])],
    layout_title_text="Confirmed Cases in California"
)
fig.update_layout(
    yaxis_type="log",
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_white")
fig.show()


# ### 4-Day Doubling Baseline
# Source: [Sample Submission: 4-Day Doubling Baseline](https://www.kaggle.com/benhamner/sample-submission-4-day-doubling-baseline)

# In[ ]:


df_test = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')
print(df_test.shape)
df_test.head()


# In[ ]:


public_leaderboard_start_date = "2020-03-12"
last_public_leaderboard_train_date = "2020-03-11"
public_leaderboard_end_date  = "2020-03-26"

cases  = df[df["Date"]==last_public_leaderboard_train_date]["ConfirmedCases"].values[0] * (2**(1/4))
df_test.insert(1, "ConfirmedCases", 0)


# In[ ]:


for i in range(15):
    df_test.loc[i, "ConfirmedCases"] = cases
    cases = cases * (2**(1/4))    
df_test.head()


# In[ ]:


# Reference: https://plot.ly/python/time-series/
fig = go.Figure(
    [
        go.Scatter(x=df['Date'], y=df['ConfirmedCases'], name="actual"),
        go.Scatter(x=df_test['Date'].iloc[:15], y=df_test['ConfirmedCases'], name="predicted"),
    ],
    layout_title_text="Confirmed Cases in California"
)
fig.update_layout(
    yaxis_type="log",
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_white")
fig.show()


# ### Empirical Growth Rate
# `2 ** (1 / 4) = 1.1892` growth rate seems to be too low. Let's get a new estimate from the empirical data.

# In[ ]:


df_growth = pd.DataFrame({
    "Date": df["Date"].iloc[1:].values,
    "Rate": df["ConfirmedCases"].iloc[1:].values / df["ConfirmedCases"].iloc[:-1].values * 100
})


# In[ ]:


# Reference: https://plot.ly/python/bar-charts/
fig = px.bar(df_growth, x='Date', y='Rate', width=600, height=400)
fig.update_layout(
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_white",
    title="Empirical Growth Rate",
    yaxis_title="Rate (%)"
)
fig.update_yaxes(range=[100, 135])
fig.show()


# Using the median rate to make predictions:
# 
# (Note: taking the median actually uses future information, but the alternative is using only one data point. So the better way is probably to use the estimate from first 7 data points and see if it'll hold for the next week.)

# In[ ]:


# rate = df_growth["Rate"].mean() / 100
rate = df_growth["Rate"].iloc[:7].median() / 100
print(f"Rate used: {rate:.4f}")
df_test = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')
public_leaderboard_start_date = "2020-03-12"
last_public_leaderboard_train_date = "2020-03-11"
public_leaderboard_end_date  = "2020-03-26"

cases  = df[df["Date"]==last_public_leaderboard_train_date]["ConfirmedCases"].values[0] * (rate)
df_test.insert(1, "ConfirmedCases", 0)
for i in range(15):
    df_test.loc[i, "ConfirmedCases"] = cases
    cases = cases * rate  
df_test.head()


# In[ ]:


# Reference: https://plot.ly/python/time-series/
fig = go.Figure(
    [
        go.Scatter(x=df['Date'], y=df['ConfirmedCases'], name="actual"),
        go.Scatter(x=df_test['Date'].iloc[:15], y=df_test['ConfirmedCases'], name="predicted"),
    ],
    layout_title_text="Confirmed Cases in California"
)
fig.update_layout(
    yaxis_type="log",
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_white")
fig.show()


# ## Fatalities

# In[ ]:


# Reference: https://plot.ly/python/time-series/
fig = go.Figure(
    [go.Scatter(x=df['Date'], y=df['Fatalities'])],
    layout_title_text="Fatalities in California"
)
fig.update_layout(
    yaxis_type="log",
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_white")
fig.show()


# In[ ]:


df_growth = pd.DataFrame({
    "Date": df["Date"].iloc[1:].values,
    "Rate": df["Fatalities"].iloc[1:].values / df["Fatalities"].iloc[:-1].values * 100
})
# Reference: https://plot.ly/python/bar-charts/
fig = px.bar(df_growth, x='Date', y='Rate', width=600, height=400)
fig.update_layout(
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_white",
    title="Empirical Growth Rate",
    yaxis_title="Rate (%)"
)
fig.update_yaxes(range=[100, 180])
fig.show()


# In[ ]:


rate = df_growth["Rate"].iloc[:7].median() / 100
print(f"Rate used: {rate:.4f}")
df_test = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')
public_leaderboard_start_date = "2020-03-12"
last_public_leaderboard_train_date = "2020-03-11"
public_leaderboard_end_date  = "2020-03-26"

cases  = df[df["Date"]==last_public_leaderboard_train_date]["Fatalities"].values[0] * (rate)
df_test.insert(1, "Fatalities", 0)
for i in range(15):
    df_test.loc[i, "Fatalities"] = cases
    cases = cases * rate  
df_test.head()


# In[ ]:


# Reference: https://plot.ly/python/time-series/
fig = go.Figure(
    [
        go.Scatter(x=df['Date'], y=df['Fatalities'], name="actual"),
        go.Scatter(x=df_test['Date'].iloc[:15], y=df_test['Fatalities'], name="predicted"),
    ],
    layout_title_text="Fatalities in California"
)
fig.update_layout(
    yaxis_type="log",
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_white")
fig.show()

