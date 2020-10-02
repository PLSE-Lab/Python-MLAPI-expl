#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().system('pip install adjustText')

import datetime as dt
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
from adjustText import adjust_text
import matplotlib.patheffects as path_effects
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import curve_fit
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
cmap = mpl.cm.get_cmap('rainbow_r')
line_colors = list(map(mpl.colors.to_hex, [cmap(i) for i in np.linspace(0.0, 1.0, num=28)]))

def datetime(x):
    return np.array(x, dtype=np.datetime64)

def exponenial_func(x, a, b, c):
    return a * np.exp(-b * x) + c


# In[ ]:


df_demo = pd.read_csv("/kaggle/input/covid19-cases-switzerland/demographics.csv", sep=",", index_col=0)
df = pd.read_csv("/kaggle/input/covid19-cases-switzerland/covid19_cases_switzerland.csv", sep=",")

df = df.interpolate(method="polynomial", order=3)
df_pred = df.copy()

l = df.shape[0]

# Fit an exponential
n = 2
future_values = [[] for i in range(n)]
for column in df:
    if column == "Date": continue
    try:
        values = [v for v in df[column] if v == v]
        nan_indices = np.where(np.isnan(df[column]))[0]
        popt, pcov = curve_fit(exponenial_func, list(range(0, len(values))), values, p0=(1, 1e-6, 1))
        xx = range(0, len(df[column]) + n)
        yy = exponenial_func(xx, *popt)
        
        for i in nan_indices:
            df[column].iloc[i] = yy[i]
        
        
        for i in range(n):
            future_values[i].append(yy[l + i])
    except:
        for i in range(n):
            future_values[i].append(df[column].iloc[-1])
          
         
            
for i in range(n):
    future_values[i] = [round(num, 0) for num in future_values[i]]
    
future_values[0] = [dt.datetime.strftime(dt.datetime.strptime(df["Date"].iloc[-1], "%Y-%m-%d") + dt.timedelta(days=1), "%Y-%m-%d")] + future_values[0]
for i in range(1, n):
    future_values[i] = [dt.datetime.strftime(dt.datetime.strptime(future_values[i-1][0], "%Y-%m-%d") + dt.timedelta(days=1), "%Y-%m-%d")] + future_values[i]


for row in future_values:
    df.loc[len(df)] = row

df.fillna(method="pad", inplace=True)
    
for col in df:
    if col == "Date": continue
    df[col] = df[col].astype(int)
    
df.to_csv("combined.csv", index=False)
   
# Done fitting exponential
        
dates = df.pop("Date")
df_diff = df.diff()
df_diff_pct = df.pct_change()
df = df.reindex(df.sum().sort_values(ascending=False).index, axis=1)
df_diff = df_diff.reindex(df.sum().sort_values(ascending=False).index, axis=1)
df_diff_pct = df_diff_pct.reindex(df.sum().sort_values(ascending=False).index, axis=1)
df["Date"] = dates.values
df_date_indexed = df.set_index("Date")

df_pb = df.copy()
for col in df_pb:
    if col == "Date" or col == "CH": continue
    df_pb[col] = df_pb[col] / df_demo.loc[col,:]["Beds"]


df_pt = df.copy()
for col in df_pt:
    if col == "Date" or col == "CH": continue
    df_pt[col] = df_pt[col] / df_demo.loc[col,:]["Population"] * 10000


df_per_capita = df_date_indexed.copy()
for col in df_per_capita:
    if col == "Date" or col == "CH": continue
    df_per_capita[col] = df_per_capita[col] / df_demo.loc[col,:]["Population"]

df_diff["Date"] = dates.values
df_diff_pct["Date"] = dates.values
df_diff_pct_melted = pd.melt(df_diff_pct, id_vars=['Date'])

df_diff_pct = df_diff_pct.fillna(0).replace(np.inf, 0)
df_diff_pct_melted = df_diff_pct_melted.fillna(0).replace(np.inf, 0)
df_diff_pct = df_diff_pct.set_index('Date')

df_melted = pd.melt(df, id_vars=['Date'])


# In[ ]:


df_t = df.T
df_t.columns = [x for x in df_t.iloc[-1]]
df_t = df_t.drop(df_t.index[-1])

world = gpd.read_file("/kaggle/input/covid19-cases-switzerland/map/ne_10m_admin_1_states_provinces.shp")
ch = world[world["iso_a2"] == "CH"].to_crs(epsg=21781)
ch["iso_3166_2"].replace("CH-", "", inplace=True, regex=True)

ch = ch.join(other=df_demo, on="iso_3166_2")
ch = ch.join(other=df_t, on="iso_3166_2")

dates = [col for col in ch if col.startswith('2020')]
fig, axs = plt.subplots(math.ceil(len(dates) / 2), 2, sharex=True, sharey=True, figsize=(25, 50))

for ax in axs.reshape(-1):
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

for i, date in enumerate(dates):
    ax = axs.reshape(-1)[i]
    ax.set_title("Cases on " + date, fontsize=18)
    ch.plot(column=date, edgecolor="black", figsize=(20, 10), cmap="RdPu", ax=ax)
    for index, row in ch.iterrows():
        annotation = ax.annotate(s=row[date], fontsize=18, xy=row.geometry.centroid.coords[0], ha="center")
        annotation.set_path_effects([path_effects.Stroke(linewidth=10, foreground="white"), path_effects.Normal()])
        
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()


# In[ ]:


df_t = df_pt.T
df_t.columns = [x for x in df_t.iloc[-1]]
df_t = df_t.drop(df_t.index[-1])

world = gpd.read_file("/kaggle/input/covid19-cases-switzerland/map/ne_10m_admin_1_states_provinces.shp")
ch = world[world["iso_a2"] == "CH"].to_crs(epsg=21781)
ch["iso_3166_2"].replace("CH-", "", inplace=True, regex=True)

ch = ch.join(other=df_demo, on="iso_3166_2")
ch = ch.join(other=df_t, on="iso_3166_2")

dates = [col for col in ch if col.startswith('2020')]
fig, axs = plt.subplots(math.ceil(len(dates) / 2), 2, sharex=True, sharey=True, figsize=(25, 50))

for ax in axs.reshape(-1):
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

for i, date in enumerate(dates):
    ax = axs.reshape(-1)[i]
    ax.set_title("Cases per 10,000 Inhabitants on " + date, fontsize=18)
    ch.plot(column=date, edgecolor="black", figsize=(20, 10), cmap="RdPu", ax=ax)
    for index, row in ch.iterrows():
        annotation = ax.annotate(s=round(row[date], 2), fontsize=18, xy=row.geometry.centroid.coords[0], ha="center")
        annotation.set_path_effects([path_effects.Stroke(linewidth=8, foreground="white"), path_effects.Normal()])
        
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()


# In[ ]:


df_t = df_pb.T
df_t.columns = [x for x in df_t.iloc[-1]]
df_t = df_t.drop(df_t.index[-1])

world = gpd.read_file("/kaggle/input/covid19-cases-switzerland/map/ne_10m_admin_1_states_provinces.shp")
ch = world[world["iso_a2"] == "CH"].to_crs(epsg=21781)
ch["iso_3166_2"].replace("CH-", "", inplace=True, regex=True)

ch = ch.join(other=df_demo, on="iso_3166_2")
ch = ch.join(other=df_t, on="iso_3166_2")

dates = [col for col in ch if col.startswith('2020')]
fig, axs = plt.subplots(math.ceil(len(dates) / 2), 2, sharex=True, sharey=True, figsize=(25, 50))

for ax in axs.reshape(-1):
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

for i, date in enumerate(dates):
    ax = axs.reshape(-1)[i]
    ax.set_title("Cases per Hospital Bed on " + date, fontsize=18)
    ch.plot(column=date, edgecolor="black", figsize=(20, 10), cmap="RdPu", ax=ax)
    for index, row in ch.iterrows():
        annotation = ax.annotate(s=round(row[date], 2), fontsize=18, xy=row.geometry.centroid.coords[0], ha="center")
        annotation.set_path_effects([path_effects.Stroke(linewidth=8, foreground="white"), path_effects.Normal()])
        
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()


# In[ ]:


f, axs = plt.subplots(3, 2, sharex=True, figsize=(25, 18))
sns.set(style="whitegrid")


ax = sns.lineplot(data=df_date_indexed[["VD", "TI", "ZH", "BS", "GE", "BE"]], palette="tab10", linewidth=2.5, ax=axs[0][0])
ax.set(xlabel="Date", ylabel="Cases", title="Cases for Cantons 1-6")

ax = sns.lineplot(data=df_date_indexed[["BL", "GR", "NE", "FR", "VS", "AG"]], palette="tab10", linewidth=2.5, ax=axs[1][0])
ax.set(xlabel="Date", ylabel="Cases", title="Cases for Cantons 7-12")

ax = sns.lineplot(data=df_date_indexed[["SG", "LU", "SZ", "SO", "JU", "OW"]], palette="tab10", linewidth=2.5, ax=axs[2][0])
ax.set(xlabel="Date", ylabel="Cases", title="Cases for Cantons 13-18")
plt.xticks(rotation=45)

# Per capita
ax = sns.lineplot(data=df_per_capita[["VD", "TI", "ZH", "BS", "GE", "BE"]], palette="tab10", linewidth=2.5, ax=axs[0][1])
ax.set(xlabel="Date", ylabel="Cases", title="Cases for Cantons 1-6 per Capita")

ax = sns.lineplot(data=df_per_capita[["BL", "GR", "NE", "FR", "VS", "AG"]], palette="tab10", linewidth=2.5, ax=axs[1][1])
ax.set(xlabel="Date", ylabel="Cases", title="Cases for Cantons 7-12 per Capita")

ax = sns.lineplot(data=df_per_capita[["SG", "LU", "SZ", "SO", "JU", "OW"]], palette="tab10", linewidth=2.5, ax=axs[2][1])
ax.set(xlabel="Date", ylabel="Cases", title="Cases for Cantons 13-18 per Capita")

f.autofmt_xdate()


# In[ ]:


plt.figure(figsize=(6, 18))
sns.set(style="whitegrid")
ax = sns.heatmap((df_diff_pct.T * 100).astype("int32"), cmap="PuRd", annot=True, fmt="d", cbar=False, square=True).set_title("Daily Increase of Cases in %")

