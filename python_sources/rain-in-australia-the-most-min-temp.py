#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")
data.drop( labels=["Date", "Evaporation", "Sunshine"], axis=1 )


# In[ ]:


minTemp = data[
    ["Location", "MinTemp"]
].groupby(by="Location").min()
minTemp = minTemp["MinTemp"].sort_values()[::-1]


# In[ ]:


print(minTemp)


# In[ ]:


sns.barplot( minTemp, minTemp.index )

