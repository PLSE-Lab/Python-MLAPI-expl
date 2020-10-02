#!/usr/bin/env python
# coding: utf-8

# ## TL; DR
# * Since the target variable is time series data, it's important to visualize as chart plot.
# * It's can be assumed that all the target variables has periodicity (daily, weekly, seasonally), but there must be more or less differences because each building has each operation's rule. The purpose of this notebook is to reveal the difference by plotting each building's target variable.
# * According to this [EDA](https://www.kaggle.com/jaseziv83/a-deep-dive-eda-into-all-variables), the difference mentioned above can be considered that it mainly comes from primary usage, so the charts are categorized by it.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")
weather_train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")
meta = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")


# In[ ]:


train = train.merge(meta, left_on="building_id", right_on="building_id", how="left")


# In[ ]:


train = train.merge(weather_train, left_on=["site_id", "timestamp"], right_on=["site_id", "timestamp"])


# In[ ]:


train["timestamp"] = pd.to_datetime(train["timestamp"])
train["meter"] = train["meter"].map({0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"})
train["meter"] = train["meter"].astype("category")


# In[ ]:


train.head()


# ## Primary use

# In[ ]:


meta["primary_use"].value_counts().sort_values().plot.barh()


# In[ ]:


import matplotlib.pyplot as plt
from IPython.display import Markdown, display


# In[ ]:


# define plot func
def plot_train_primary_use(train_primary_use):
    n_building = len(train_primary_use["building_id"].unique())
    n_col = 4
    n_row = int(np.ceil(n_building / n_col))
    f, ax = plt.subplots(n_row, n_col, squeeze=False, figsize=(20, 5 * n_row))

    color_dict = {"electricity": "y", "chilledwater": "b", "steam": "#800000", "hotwater": "#ff5500"}

    for i, building_id in enumerate(train_primary_use["building_id"].unique()):
        df_build = train_primary_use.query("building_id == @building_id").reset_index()
        df_build_piv = df_build.pivot(index="timestamp", columns="meter", values="meter_reading")
        df_build_piv.plot(ax=ax[i // n_col, i % n_col], color=[color_dict.get(x, '#000000') for x in df_build_piv.columns], alpha=0.7)
        ax[i // n_col, i % n_col].set_title("building id: {}".format(building_id))
    plt.tight_layout()
    plt.show()


# In[ ]:


for primary_use in train["primary_use"].unique():
    display(Markdown("## {}".format(primary_use)))
    train_primary_use = train.query("primary_use == @primary_use")
    plot_train_primary_use(train_primary_use)


# In[ ]:




