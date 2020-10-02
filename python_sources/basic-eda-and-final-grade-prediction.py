#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
sns.set_color_codes("pastel")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# I'm use only student-por.csv
data = pd.read_csv('../input/student-por.csv')


# # Data Overview

# In[ ]:


data.head(5)


# In[ ]:


data.info()


# # Some Visualizations

# Let's do some basic data visualizations.

# ## Sex

# In[ ]:


f, ax = plt.subplots(figsize=(4, 4))
plt.pie(data['sex'].value_counts().tolist(), 
        labels=['Female', 'Male'], colors=['#ffd1df', '#a2cffe'], 
        autopct='%1.1f%%', startangle=90)
axis = plt.axis('equal')


# ## Age

# In[ ]:


fig, ax = plt.subplots(figsize=(5, 4))
sns.distplot(data['age'],  
             hist_kws={"alpha": 1, "color": "#a2cffe"}, 
             kde=False, bins=8)
ax = ax.set(ylabel="Count", xlabel="Age")


# ## Weekly study time

# In[ ]:


f, ax = plt.subplots(figsize=(4, 4))
plt.pie(data['studytime'].value_counts().tolist(), 
        labels=['2 to 5 hours', '<2 hours', '5 to 10 hours', '>10 hours'], 
        autopct='%1.1f%%', startangle=0)
axis = plt.axis('equal')


# ## Romantic Relationship

# In[ ]:


f, ax = plt.subplots(figsize=(4, 4))
plt.pie(data['romantic'].value_counts().tolist(), 
        labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90)
axis = plt.axis('equal')


# ## Alcohol Consumption and other features

# Workday alcohol consumption: number from 1 (very low) to 5 (very high)<br/>
# Weekend alcohol consumption: number from 1 (very low) to 5 (very high)<br/>
# Health - current health status: number from 1 (very bad) to 5 (very good) 

# ### Weekend Alcohol Consumption Distribution

# In[ ]:


fig, ax = plt.subplots(figsize=(5, 4))
sns.distplot(data['Walc'],  
             hist_kws={"alpha": 1, "color": "#a2cffe"}, 
             kde=False, bins=4)
ax = ax.set(ylabel="Students", xlabel="Weekend Alcohol Consumption")


# ### Alcohol Consumption and Health

# In[ ]:


plot1 = sns.factorplot(x="Walc", y="health", hue="sex", data=data)
plot1.set(ylabel="Health", xlabel="Weekend Alcohol Consumption")

plot2 = sns.factorplot(x="Dalc", y="health", hue="sex", data=data)
plot2.set(ylabel="Health", xlabel="Workday Alcohol Consumption")


# ### Alcohol Consumption and Final Grade

# Final grade: number from 0 to 20.

# In[ ]:


plot1 = sns.factorplot(x="G3", y="Walc", data=data)
plot1.set(ylabel="Final Grade", xlabel="Weekend Alcohol Consumption")

plot2 = sns.factorplot(x="G3", y="Dalc", data=data)
plot2.set(ylabel="Final Grade", xlabel="Workday Alcohol Consumption")


# # Final Grade Prediction

# Output target of this dataset is **Final Grade**. Let's use some regression model to predict it. I'll limit myself to 4 simple regression models (without searching the best parameters): decision tree regression, linear regression, lasso and ridge regression.

# ## With G1 and G2 features

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score


# In[ ]:


y = data['G3']
X = data.drop(['G3'], axis=1)


# In[ ]:


X = pd.get_dummies(X)


# In[ ]:


names = ['DecisionTreeRegressor', 'LinearRegression', 'Ridge', 'Lasso']

clf_list = [DecisionTreeRegressor(),
            LinearRegression(),
            Ridge(),
            Lasso()]


# In[ ]:


for name, clf in zip(names, clf_list):
    print(name, end=': ')
    print(cross_val_score(clf, X, y, cv=5).mean())


# It's a really good score. However, let's look at feature importances.

# ## Feature Importances

# In[ ]:


tree = DecisionTreeRegressor()
tree.fit(X, y)


# In[ ]:


importances = tree.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print("%d. Feature %s (%f)" % (f + 1, X.columns.values[indices[f]], importances[indices[f]]))


# Here's big problem. More than 80 percent of the predictive ability of the algorithm achieves with the help only of **G2** feature (second period grade). The remaining features are almost not use in this model. This means that almost all the data in no way help us to predict target feature.

# Let's look at scores of models, trained without G1 and G2 features.

# ## Without G1 and G2 features

# In[ ]:


X = data.drop(['G3', 'G2', 'G1'], axis=1)


# In[ ]:


X = pd.get_dummies(X)


# In[ ]:


for name, clf in zip(names, clf_list):
    print(name, end=': ')
    print(cross_val_score(clf, X, y, cv=5).mean())


# A terrible result. Unfortunately, using only survey questions (without grades for intermediate tests) we will not be able to predict the final grade.
