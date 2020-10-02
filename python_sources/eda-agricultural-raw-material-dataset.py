#!/usr/bin/env python
# coding: utf-8

# ![](https://1.bp.blogspot.com/-umI8KF0zzWY/XZM4o8tkr8I/AAAAAAAAKWs/sQKCWeCDoZIrXx8M05NrKS_GEnNjyvIUgCLcBGAsYHQ/s400/tumblr_ple4kdmuUC1y0yrefo5_400.gif)
# 
# # Introduction
# The purpose of this kernel is to have a basic understanding of the data and also identify any reason for the price change. A better understanding of these prices will allow us to predict price hikes

# In[ ]:


# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
sns.set(rc={'figure.figsize':(11, 4)})


# ## Overview of data

# In[ ]:


def overview():
    data = pd.read_csv("../input/agricultural-raw-material-prices-19902020/agricultural_raw_material.csv")
    print("The first 5 rows of data are:\n")
    print(data.head)
    print("\n\n\nDataset has {} rows and {} columns".format(data.shape[0], data.shape[1]))
    print("\n\n\nDatatype: \n")
    print(data.dtypes)
    print("\n\n\nThe number of null values for each column are: \n")
    print(data.isnull().sum())
    print("\n\n\nData summary: \n")
    print(data.describe())
    return data

# Lastly, assigning a variable to overview()
data = overview()


# ### What do we see here?
# - We see that price % change is deemed as an object. We will need to fix that by removing the "%" sign.
# - We also notice that the percentages of NaN values are significant (more than 5 %). We will drop the NaN values for those columns with 1 NaN value and as for the rest, we will replace them with median value.

# In[ ]:


# Replacing %, "," and "-"
data = data.replace('%', '', regex=True)
data = data.replace(',', '', regex=True)
data = data.replace('-', '', regex=True)
data = data.replace('', np.nan)
data = data.replace('MAY90', np.nan)

# Dropping rows with NaN values
data = data.dropna()

# Check to see if all NaN values are resolved
data.isnull().sum()

# Converting data type to float
lst = ["Coarse wool Price", "Coarse wool price % Change", "Copra Price", "Copra price % Change", "Cotton price % Change","Fine wool Price", "Fine wool price % Change", "Hard log price % Change", "Hard sawnwood price % Change", "Hide price % change", "Plywood price % Change", "Rubber price % Change", "Softlog price % Change", "Soft sawnwood price % Change", "Wood pulp price % Change"]
data[lst] = data[lst].astype("float")

data.dtypes


# In[ ]:


data.Month  = pd.to_datetime(data.Month.str.upper(), format='%b%y', yearfirst=False)

# Indexing month
data = data.set_index('Month')


# ## Coarse wool prices

# In[ ]:


axes = data[["Coarse wool Price", "Coarse wool price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Copra prices 

# In[ ]:


axes = data[["Copra Price", "Copra price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Cotton prices

# In[ ]:


axes = data[["Cotton Price", "Cotton price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Fine wool prices

# In[ ]:


axes = data[["Fine wool Price", "Fine wool price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Hard log price

# In[ ]:


axes = data[["Hard log Price", "Hard log price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Hard sawnwood price

# In[ ]:


axes = data[["Hard sawnwood Price", "Hard sawnwood price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Hide price

# In[ ]:


axes = data[["Hide Price", "Hide price % change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Plywood

# In[ ]:


axes = data[["Plywood Price", "Plywood price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Rubber prices

# In[ ]:


axes = data[["Rubber Price", "Rubber price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Softlog prices

# In[ ]:


axes = data[["Softlog Price", "Softlog price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Soft sawnwood prices

# In[ ]:


axes = data[["Soft sawnwood Price", "Soft sawnwood price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Wood pulp

# In[ ]:


axes = data[["Wood pulp Price", "Wood pulp price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)


# ## Correlation matrix

# In[ ]:


plt.figure(figsize=(30,15))
 
corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

