#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# For example, here's several helpful packages to load in 
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt 
import seaborn as sns


sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))


# In[ ]:


import pandas as pd 
energy_dataset = pd.read_csv("../input/international-energy-statistics/all_energy_statistics.csv")


# * Please look ove this piece of [code](https://www.kaggle.com/sametgirgin/oil-gas) to understand the columns that the missing values are dominated.  Drop the column named quntity_footnotes which contain many null values

# In[ ]:


energy_dataset.drop("quantity_footnotes", axis=1, inplace=True) 


# In[ ]:


energy_dataset.head(4)


# In[ ]:


energy_dataset["category"].value_counts().count() # How many types of categories are there in dataset? 
energy_dataset["unit"].value_counts().count #Which types of units used in the dataset...


# Energy-Questionnaire-Guidelines.pdf (299.37 KB) ([Kaggle Dataset PDF]: https://www.kaggle.com/unitednations/international-energy-statistics#Energy-Questionnaire-Guidelines.pdf) 
# 
# The guideline above explains the details of the energy resources one by one. In this piece, my aim is to observe the energy balances by resources of the emerging states like **Turkey, South Korea, Australia, Mexico, Indonesia, Brasil and South Africa **. Those states are the emerging states that retain the economic power in their geographic regions. 
# 
# **Products Categories: **
# 
# Coal, Peat and Oil Shale
# 
# Oil
# 
# Natural Gas, Manufactured Gas and Recovered Gas
# 
# Electricity and Heat
# 
# Biofuels and Waste

# In[ ]:


TUR = energy_dataset[energy_dataset.country_or_area.isin(["Turkey"])].sort_values('year')
SKOR = energy_dataset[energy_dataset.country_or_area.isin(["Korea, Republic of"])].sort_values('year')
AUST= energy_dataset[energy_dataset.country_or_area.isin(["Australia"])].sort_values('year') 
INDO= energy_dataset[energy_dataset.country_or_area.isin(["Indonesia"])].sort_values('year') 
MEX= energy_dataset[energy_dataset.country_or_area.isin(["Mexico"])].sort_values('year') 
BR= energy_dataset[energy_dataset.country_or_area.isin(['Brazil'])].sort_values('year')
SAFR= energy_dataset[energy_dataset.country_or_area.isin(['SouthAfrica'])].sort_values('year')


# In[ ]:


TUR


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




