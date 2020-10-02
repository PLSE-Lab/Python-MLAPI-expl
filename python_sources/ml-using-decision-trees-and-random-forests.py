#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Using Decision Trees and Random Forests

# ### Data: Breast Cancer Wisconsin Diagnostic

# We are going to be using Machine Learning (Decision Trees and Random Forests) to diagnose patients based on the data

# In[102]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime


# In[103]:


# Import the Data
data = pd.read_csv('../input/data.csv')


# In[104]:


# Explore the Data
data.head(8)


# In[105]:


data.describe()


# In[106]:


# Check for missing values
sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[107]:


# Delete the empty column
data.drop('Unnamed: 32', axis = 1, inplace = True)


# There are now no missing values in the data

# ### Data Visualization

# In[108]:


# Visualize the data
sns.set(style = 'darkgrid')
g = sns.countplot(x = "diagnosis", data = data, palette = "Set3")
plt.ylabel("Number of Occurences")
plt.xlabel("Diagnosis")
plt.title("Diagnosis Distribution")


# In[109]:


# drop ID column from our dataset
data.drop('id', axis = 1, inplace = True)


# We have 30 variables
# 
# It is going to be easier to visualize them in smaller groups

# In[110]:


data.columns


# In[111]:


# Create groups of some variables we want to visualize
means = data[['diagnosis', 'radius_mean', 'texture_mean', 'radius_worst', 'texture_worst']]

means2 = data[['diagnosis', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean']]

means3 = data[['diagnosis', 'concave points_mean', 'fractal_dimension_mean']]


# In[112]:


# Use pd.melt to be able to visualize multiple variables at once
melt_means = pd.melt(means, id_vars = 'diagnosis', var_name = "Variables", value_name = "Value")
melt_means2 = pd.melt(means2, id_vars = 'diagnosis', var_name = "Variables", value_name = "Value")
melt_means3 = pd.melt(means3, id_vars = 'diagnosis', var_name = "Variables", value_name = "Value")


# In[113]:


# Boxplots
sns.boxplot(x = "Variables", y = "Value", data = melt_means, hue = 'diagnosis', palette = 'pastel')


# In[114]:


sns.boxplot(x = "Variables", y = "Value", 
            data = melt_means2, hue = 'diagnosis', 
            palette = 'pastel')

plt.xticks(rotation=25)


# In[115]:


sns.boxplot(x = "Variables", y = "Value", data = melt_means3, hue = 'diagnosis', palette = 'pastel')


# There are definitely some significant differences in values between M and B groups
# 
# Let's try different kinds of plots to explore the data a little more

# In[116]:


# We can also see the relationship between multiple variables at once
f = sns.PairGrid(means)
f = f.map_upper(plt.scatter)
f = f.map_lower(sns.kdeplot, cmap = "Purples_d")
f = f.map_diag(sns.kdeplot, lw = 3, legend = False)


# In[117]:


c = sns.swarmplot(x = "Variables", y = "Value", data = melt_means2, hue = 'diagnosis', palette = 'pastel')
plt.xticks(rotation=25)


# You can see the B values being smaller than the M values very clearly here

# In[118]:


# Violin plots
cv = sns.violinplot(x = "Variables", y = "Value", data = melt_means2, hue = 'diagnosis', palette = 'spring')
plt.xticks(rotation=25)


# In[119]:


cv = sns.violinplot(x = "Variables", y = "Value", data = melt_means2, hue = 'diagnosis', palette = 'seismic', split = True)
plt.xticks(rotation=25)


# In[120]:


# If we want to see any specific relationships, we can use this:
sns.jointplot(x = 'texture_mean', y = 'radius_mean', data = means, kind = 'hex', color = "#4CB391")


# Lastly, let's see how all variable scorrelate to one another

# In[121]:


corrmat = data.corr()
fig, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(corrmat, square = True, cmap = "YlGnBu", annot = True, fmt = '.1f', linewidths = .25, linecolor = 'r')


# That's enough visualization
# 
# ## Model Building

# ### Feature Selection
# 
# First, let's see which variables are going to be included in our model

# In[122]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[123]:


# Prepare the data
X = data.drop('diagnosis', axis = 1)
y = data['diagnosis']


# In[124]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[125]:


# Feature Selection
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)


# In[126]:


# Check which features were selected to be the best to use
sel.get_support()


# In[127]:


selected_feat = X_train.columns[(sel.get_support())]
print(selected_feat)


# ### Refit the model with new features

# In[128]:


X = data[['radius_mean', 'perimeter_mean', 'area_mean', 'concave points_mean',
       'radius_worst', 'perimeter_worst', 'area_worst',
       'concave points_worst']]

y = data['diagnosis']


# In[129]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[130]:


dtree = DecisionTreeClassifier()


# In[131]:


dtree.fit(X_train, y_train)


# In[132]:


# Predict values based on selected model
predictions = dtree.predict(X_test)


# In[133]:


# Check how the well the model did
from sklearn.metrics import classification_report, confusion_matrix


# In[134]:


print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))


# 92% acuuracy - Not too great but not too bad

# ## Random Forests

# The dataset is fairly big, we can assume that Random Forests is going to do a better job here

# Let's try to fit a model using random forests

# In[135]:


rfc = RandomForestClassifier(n_estimators = 100)


# In[136]:


# Fit the model
rfc.fit(X_train, y_train)


# In[137]:


# Predict values
rfc_pred = rfc.predict(X_test)


# In[138]:


# Check accuracy
print(confusion_matrix(y_test, rfc_pred))
print('\n')
print(classification_report(y_test, rfc_pred))


# Sure enough!
# 
# We get 96% accuracy.
# 
# As expected, random forests did a better job.

# ## Thanks for checking this out
