#!/usr/bin/env python
# coding: utf-8

# # Supermarket Expenses In Western Europe

# ### Source: Numbeo.com
# ### This analysis was executed to show supermarket expenses in western european countries.
# ### Currency: Euro.

# ## Relevant Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Data

# In[ ]:


import pandas as pd
data = pd.read_csv("../input/data.csv")
data


# ## Drop Useless Columns

# In[ ]:


data.drop("Rank",axis=1, inplace=True)


# In[ ]:


data.columns[0]


# In[ ]:


data.drop(data.columns[0], axis=1,inplace=True)


# In[ ]:


data.columns.values


# ## Add Column for Total Expenses

# In[ ]:


data["Total"] = data.sum(axis=1)
data


# ## Split the Data by Country

# In[ ]:


data.City.unique()


# ### Germany

# In[ ]:


def Germany(country):
    if "germany" in country.lower():
        return True
    return False

dataGermany = data[data["City"].apply(Germany)]


# In[ ]:


dataGermany


# In[ ]:


GerMean = round(dataGermany["Total"].mean())
GerMean


# ### Austria

# In[ ]:


def Austria(country):
    if "austria" in country.lower():
        return True
    return False
dataAustria = data[data["City"].apply(Austria)]


# In[ ]:


dataAustria


# In[ ]:


AustMean = round(dataAustria["Total"].mean())
AustMean


# ### Switzerland

# In[ ]:


def Switzerland(country):
    if "switzerland" in country.lower():
        return True
    return False

dataSwitzerland = data[data["City"].apply(Switzerland)]


# In[ ]:


dataSwitzerland


# In[ ]:


SwitzMean = round(dataSwitzerland["Total"].mean())
SwitzMean


# ### France

# In[ ]:


def France(country):
    if "france" in country.lower():
        return True
    return False

dataFrance = data[data["City"].apply(France)]


# In[ ]:


dataFrance


# In[ ]:


FranceMean = round(dataFrance["Total"].mean())
FranceMean


# ### Belgium

# In[ ]:


def Belgium(country):
    if "belgium" in country.lower():
        return True
    return False

dataBelgium = data[data["City"].apply(Belgium)]


# In[ ]:


dataBelgium


# In[ ]:


BelgMean = round(dataBelgium["Total"].mean())
BelgMean


# ### Netherlands

# In[ ]:


def Netherlands(country):
    if "netherlands" in country.lower():
        return True
    return False

dataNeth = data[data["City"].apply(Netherlands)]


# In[ ]:


dataNeth


# In[ ]:


NethMean = round(dataNeth["Total"].mean())
NethMean


# ## Bar Plot with Matplotlib

# In[ ]:


x = ["Germany", "Austria", "Switzerland", "France", "Belgium", "Netherlands"]
y = [GerMean, AustMean, SwitzMean, FranceMean, BelgMean, NethMean]


# In[ ]:


plt.figure(figsize=(18,6))
plt.bar(x,y)
plt.ylim(50,160)
plt.xlabel("Countries",size=18)
plt.ylabel("Average Expenses (Euro)", size=18)
plt.title("Supermarket Expenses in Western Europe",size=25)
plt.show()


# In[ ]:




