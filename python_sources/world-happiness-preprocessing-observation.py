#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
data2019["Year"] = 2019
data2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")
data2018["Year"] = 2018
data = pd.concat((data2019,data2018))
data.reset_index(inplace = True,drop=True)
data


# ### Null value control

# In[ ]:


data[data.isna().any(axis=1)]


# In[ ]:


data[data["Country or region"]=="United Arab Emirates"]


# In[ ]:


#Perceptions of corruption depends with all the others.

#And there isn't a certain relationship between Score and Generosity, we will see this

sns.heatmap(data.corr())


# > ### Fill "Perceptions of corruption" value with KNN

# In[ ]:


get_ipython().system('pip install ycimpute')
from ycimpute.imputer import knnimput


# ### we need np.array for ycimpute prediction.

# In[ ]:


#we need numerical values for prediction
num_data = data.select_dtypes(include=["float64","int64"])

#we kept column names for create new similar dataframe 
var_names = list(num_data)
var_names


# In[ ]:


#to take values as np.array
var_values = num_data.values
var_values


# In[ ]:


#after the prediction
completed_values = knnimput.KNN(k=4).complete(var_values)
completed_values


# In[ ]:


#new not-null dataframe
new_num_data = pd.DataFrame(completed_values,columns = var_names)
# and the value we want to find
new_num_data.iloc[175]


# In[ ]:


#check value of data (consistency)
data.iloc[170:180]


# In[ ]:


#change values
data["Perceptions of corruption"] = new_num_data["Perceptions of corruption"]


# In[ ]:


#control
data.iloc[170:180]


# ### Outliers

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
data.boxplot("GDP per capita")
plt.subplot(2,3,2)
data.boxplot("Social support")
plt.subplot(2,3,3)
data.boxplot("Healthy life expectancy")
plt.subplot(2,3,4)
data.boxplot("Freedom to make life choices")
plt.subplot(2,3,5)
data.boxplot("Generosity")
plt.subplot(2,3,6)
data.boxplot("Perceptions of corruption")


# In[ ]:


#Rwanda has srange "Perceptions of corruption" value!
data[data["Perceptions of corruption"]>.35]


# In[ ]:


#we looked to heatmap and we can see here,too
sns.jointplot(x="Score",y="Generosity",data=data,kind="reg")


# In[ ]:


### multiple outliers
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)


# In[ ]:


clf.fit_predict(new_num_data)


# In[ ]:


n_scores = clf.negative_outlier_factor_


# In[ ]:


plt.plot(np.sort(n_scores))


# In[ ]:


#Outlier values
#These cities are best and worst cities and they are out of standart.
new_num_data[n_scores<-1.01]


# In[ ]:


diff = data.groupby("Country or region")["Overall rank"].diff()[156:]


# In[ ]:


country_diff = pd.DataFrame({"country":data["Country or region"][:156].values,"diff":diff.values})
country_diff


# In[ ]:


# Countries which have big score change.
treshold = 10

country_diff = country_diff[(country_diff["diff"] > treshold) | (country_diff["diff"] < -treshold)]


# In[ ]:


plt.figure(figsize=(18,10))
sns.barplot(x="diff",y="country",data=country_diff)


# In[ ]:


# Lastly Our Overall Rank
data[data["Country or region"]=="Turkey"]

