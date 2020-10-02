#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


#Basic imports
import pandas as pd
import numpy as np

#Plotting importrs
import matplotlib.pyplot as plt
import seaborn as sns


# # Basic Data Exploration

# Here, we want to have a initial understanding of the dataset we are going to be working on. Data exploration only takes place in the following section.

# In[ ]:


#Load dataset
data = pd.read_csv('../input/heart-disease-uci/heart.csv')


# In[ ]:


#Print dataset head
data.head()


# In[ ]:


#Print dataset summary
data.describe()


# # Data Exploration

# Now, we proceed to the actual data exploration. Our first goal is to understand how each attribute behaves individually. Second, we investigate how the predictors relate to each other and to the target class.

# ## Features distributions

# In[ ]:


#Define categorical and numerical attributes
cat_feat = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
num_feat = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


# In[ ]:


#Plot categorical features
fig, axes = plt.subplots(3,3, figsize = (10,10))
for i,col in enumerate(cat_feat):
    nrow = i // 3 
    ncol = i - nrow*3
    sns.countplot(data[col], ax = axes[nrow, ncol]).set(title = col, xlabel = '', ylabel = '')
    
plt.subplots_adjust(hspace = 0.3, wspace = 0.3)


# From these plots, we can note most features are fairly well distributed. The exception to this observation is 'fbs'. 
# 

# In[ ]:


#Plot numerical features
fig, axes = plt.subplots(3,2, figsize = (10,10))
for i,col in enumerate(num_feat):
    nrow = i // 2 
    ncol = i - nrow*2
    sns.distplot(data[col], ax = axes[nrow, ncol]).set(title = col, xlabel = '', ylabel = '')

plt.subplots_adjust(hspace = 0.3, wspace = 0.3)


# Out of the five numerical attributes, 'age', 'trestbps', 'chol' and 'thalach' present distributions somewhat close to the normal curve. The 'oldpeak' feature, however, seens to present significative skewness. We are going to investigate this further in more detail.

# ## Features Correlations

# We now look at the correlations among features and target. It is important to note the computed correlation measures the strength of the linear relationship between two variables. It is not unusual these relationship are more complex. This means, these values must be taken merely as an indication of how two features are related.
# 
# Another important matter regards the multiclass categorical features. In order to better describe the importance of these features and avoid any distortion of the predictive model, we first encode them into a binary format.

# In[ ]:


#Encode multiclass categorical features
to_encode = ['cp', 'restecg', 'slope', 'ca', 'thal']
for col in to_encode:
    new_feat = pd.get_dummies(data[col])
    new_feat.columns = [col+str(x) for x in range(new_feat.shape[1])]
    new_feat = new_feat.drop(columns = new_feat.columns.values[0])
    
    data[new_feat.columns.values] = new_feat
    data = data.drop(columns = col)


# In[ ]:


#Plot correlation map
fig = plt.figure(figsize = (16,12))
sns.heatmap(data.corr(), annot = True)


# The heatmap, as it is, it is not very helpful on finding out which relationships are stronger. We then select the stronger ones.

# In[ ]:


#Define function to find correlations whithin a given parameter
def best_corr(data, par):
    corr = data.corr()
    
    best_corr = pd.DataFrame(columns = ['Atr1', 'Atr2', 'Value'])
    atr1, atr2, val = [], [], []
    for c1 in data.columns:
        for c2 in data.drop(columns = c1).columns:
            if abs(corr.loc[c1,c2]) >= par:
                atr1.append(c1)
                atr2.append(c2)
                val.append(corr.loc[c1,c2])
    best_corr['Atr1'], best_corr['Atr2'], best_corr['Value'] = atr1, atr2, val

    #Remove duplicates
    to_remove = []
    for i in best_corr.index:
        for j in range(i,best_corr.shape[0]):
            if best_corr['Atr1'][i] == best_corr['Atr2'][j] and             best_corr['Atr2'][i] == best_corr['Atr1'][j]:
                to_remove.append(i)

    return best_corr.drop(index = to_remove).reset_index(drop = True)


# In[ ]:


#Print correlations equal or better than 0.4
best_corr(data,0.4)


# We can now have a better idea of the best correlations for this dataset. Again, this alone should not be the basis of any feature selection. 
