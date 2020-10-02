#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.display import IFrame


# In[ ]:


df=pd.read_csv("../input/Melbourne_housing_FULL.csv")


# In[ ]:


df.head()


# ### Lets Check Null Values ###

# In[ ]:


df.isna().sum()


# ### Removed null rows ###

# In[ ]:


df=df.dropna()


# ### Northern Metropolitan have more property ###

# In[ ]:


IFrame("https://public.tableau.com/views/RegionwisePropertyCount/RegionwisePropertyCount?:embed=y&:showVizHome=no", width=600, height=500)


# ### Costly Suburb ###
# 
# * Kooyong have more property price 

# In[ ]:


IFrame("https://public.tableau.com/views/CostlySuburb/CostlySuburb?:embed=y&:showVizHome=no", width=1100, height=600)


# ### Which Type of Property in Melbourne ###
# 
# * H- type property have wide spread among all suburb except carnegie,South yarra, St kilda and Hawthron

# In[ ]:


IFrame("https://public.tableau.com/views/MostPopularTypeamongallSuburb/MostPopularTypeamongallSuburb?:embed=y&:showVizHome=no", width=1200, height=800)


# ### More Room More Price? ###
# 
# * obisiously more number of rooms have high property price depcited price is average and there are few owners of more rooms so average price is low for large room
# * surely room is most import feature for prediction

# In[ ]:


IFrame("https://public.tableau.com/views/Moreroommoreprice/Moreroommoreprice?:embed=y&:showVizHome=no", width=700, height=400)


# ### Large Landsize ###
# 
# * Prices are higher for large landsize property

# In[ ]:


IFrame("https://public.tableau.com/views/LargeLandsizemoreprice/LargeLandsizemoreprice?:embed=y&:showVizHome=no", width=1200, height=500)


# ### Stay Connected For More EDA ... ###

# In[ ]:




