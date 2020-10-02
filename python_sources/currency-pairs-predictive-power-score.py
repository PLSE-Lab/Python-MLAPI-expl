#!/usr/bin/env python
# coding: utf-8

# # What is Predictive Power Score (PPS)
# 
# The PPS is an asymmetric, data-type-agnostic score that can detect linear or non-linear relationships between two columns. The score ranges from $0$ (no predictive power) to $1$ (perfect predictive power). It can be used as an alternative to the correlation (matrix).

# In[ ]:


get_ipython().system('pip install ppscore')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ppscore as pps

sns.set(style="white")

df = pd.read_csv("../input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv")

del df['Unnamed: 0']
df.rename(columns={'Time Serie': 'DATE',}, inplace=True)
df.sort_values(by=["DATE"], inplace=True)

for col in set(df.columns):
    if col != "DATE":
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if col == "DATE":
        df[col] = pd.to_datetime(df[col], errors="coerce")

df.dropna(inplace=True)
df.set_index("DATE", inplace=True)

df = df.pct_change().dropna()


# # Plot daily percentage change in foreign exchange rates 

# In[ ]:


df.plot(figsize=(15,8))
plt.ylabel('Percent');


# In[ ]:


df.cumsum().plot(figsize=(15,8)).axhline(0, lw=1, color='black')
plt.ylabel('Returns From Start');


# In[ ]:


def plot_matrix(df, kind="corr", vmax=1.0, center=0.0):
    if kind == "corr":
        # Compute the correlation matrix
        corr = df.corr()
    if kind == "pps":
        corr = pps.matrix(df)
    
    corr = np.round(corr, 2)
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.light_palette((210, 90, 60), input="husl")

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=vmax, center=center,
                square=True, linewidths=.5, cbar_kws={"shrink": .5});


# # Can $x$ predict $y$?
# ## Plot Predictive Power Score

# In[ ]:


plot_matrix(df, kind="pps", vmax=1.0, center=0.5)


# In[ ]:


def currency_pairs(c1, c2):
    _ = np.round(pps.score(df, c1, c2, task='regression')["ppscore"], 2)
    print(f"The PPS of {c1} predicting {c2}: {_}")
    g = sns.jointplot(c1, c2, data=df.sample(500), kind="kde", space=0, color="g")


# In[ ]:


c1 = 'EURO AREA - EURO/US$'
c2 = 'DENMARK - DANISH KRONE/US$'

currency_pairs(c1, c2)


# In[ ]:


c1 = 'JAPAN - YEN/US$'
c2 = 'INDIA - INDIAN RUPEE/US$'

currency_pairs(c1, c2)


# In[ ]:




