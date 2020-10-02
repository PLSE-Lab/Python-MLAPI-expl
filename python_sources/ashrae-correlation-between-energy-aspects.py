#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim on this notebook is to find out if there is some correlation between energy aspects, as stated in [this post](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/116748).
# 
# What can be suposed is that different energy aspects (electricity, hot water, chilled water and steam) measure different energy sources.
# In instance, for hot water is used natural gas.
# 
# Could be that the building use electricity for heat the water, and the same energy is measured twice? One as electricity and other as hotwater?
# 
# It is supposed to be some correlation, because when there are people in the building, they consume electricity and also hotwater, chilled water and steam. When the building has no activity, the consumption decreases for all energy aspects.

# In[ ]:


# !pip install nb_black
# %load_ext nb_black
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from IPython.core.display import display, HTML
import pickle
import itertools
from tqdm.auto import trange
pio.templates.default = "plotly_white"


# # Import data

# In[ ]:


with open("../input/ashrae-data-wrangling-train/train.pickle", "rb") as f:
    train = pickle.load(f)


# # Get all the correlations between energy aspects

# In[ ]:


correlation = pd.DataFrame(columns={"building_id", "energy_aspects", "correlation"})

for building_id in trange(0, 1449):
    energy = train[train["building_id"] == str(building_id)]
    energy_aspects = energy["meter"].unique().to_list()
    if len(energy_aspects) > 1:
        for combination in itertools.combinations(energy_aspects, 2):
            ea0 = energy[energy["meter"] == combination[0]]["meter_reading"]
            ea1 = energy[energy["meter"] == combination[1]]["meter_reading"]
            corr = np.ma.corrcoef(np.ma.masked_invalid(ea0), np.ma.masked_invalid(ea1))[
                0, 1
            ]
            correlation = correlation.append(
                {
                    "building_id": building_id,
                    "energy_aspects": "-".join(combination),
                    "correlation": corr,
                },
                ignore_index=True,
            )

correlation.to_csv('correlation.csv')
display(correlation)
display(HTML(f'There are {correlation.shape[0]} combinations between energy aspects for the buildings with more than one energy aspect.'))


# In[ ]:


fig = px.histogram(
    correlation,
    x="correlation",
    color="energy_aspects",
    facet_row="energy_aspects",
    nbins=50,
    height=1200,
)
fig.update_layout(showlegend=False, xaxis=dict(range=[-1, 1], dtick = 0.2))
for i in range(0, 6):
    fig.layout.annotations[i].text = fig.layout.annotations[i].text.replace(
        "energy_aspects=", ""
    )
fig.show()


# Some positive correlation is seen in the combination between electricity and chilled water. Maybe the electricity and the chilled water are consumed at the same time, or the chilled water are produced with electricity and its counted double?.
# 
# Electricity and hotwater doesn't seem correlated.
# 
# Chilled water with hot water and with steam have some light negative correlation. When one is consumed, the other not. 

# In[ ]:


correlation[abs(correlation["correlation"]) > 0.80].sort_values(
    by="correlation", ascending=False
)


# There are some buildings with high correlated energy aspects.
# 
# In building 1156, the electricity consumption is very constant where there isn't chilled water. And when there is, the electricity consumption is proportional to chilled water. Can be deduced, that in this building, the chilled water is produced with electricity, and both are measured.
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F645169%2F8108b9b321ec5be9357090438f95bfaf%2Fbuilding_id-1156-site_id-13-Lodging-residential-electricity-chilledwater-steam.png?generation=1573593723628143&alt=media)
# 
# The same happens with building 1031, 1164, 1201, 1151. All of them are lodging residential type.
