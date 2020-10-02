#!/usr/bin/env python
# coding: utf-8

# 

# **STATISCAL ANALYSIS**

# In[ ]:


import numpy
import pandas
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor


import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[ ]:


# load dataset
dataframe = pandas.read_csv(r"../input/concrete_data.csv")


# In[ ]:


dataframe.head()


# In[ ]:




print("Statistical Description:") 
dataframe.describe()


# In[ ]:


print("Shape:", dataframe.shape)


# In[ ]:


print("Data Types:", dataframe.dtypes)


# In[ ]:


print("Correlation:") 
dataframe.corr(method='pearson')


# 'cement' has the highest correlation with the area of 'concrete_compressive_strength'(which is a positive correlation), followed by 'superplasticizer', which is also a positive correlation, 'fly_ash' has the least correlation

# In[ ]:


dataset = dataframe.values


X = dataset[:,0:8]
Y = dataset[:,8] 


# In[ ]:


#Feature Selection
model = ExtraTreesRegressor()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_) 


# 'cement', 'superplasticizer' and 'age' were top 3 selected features/feature combination for predicting 'Area'
# using Recursive Feature Elimination, the 1st and 2nd selected features were atually among the attributes with the highest correlation with the  'concrete_compressive_strength'

# **VISUALIZATION**

# In[ ]:


plt.hist((dataframe.concrete_compressive_strength))


# Most of the dataset's samples fall between 34 and 42 of 'concrete_compressive_strength' continous output class, with a positive skew

# In[ ]:


dataframe.hist()


# In[ ]:


dataframe.plot(kind='density', subplots=True, layout=(3,4), sharex=False, sharey=False)


# None of the features have a Guassian Distribution. 
# All of them have a positive skew except 'water' and 'fine_aggregate', which have negative skews

# In[ ]:


dataframe.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)


# In[ ]:


scatter_matrix(dataframe)


# In[ ]:





# In[ ]:




