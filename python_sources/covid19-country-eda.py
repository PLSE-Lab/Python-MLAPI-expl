#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.externals.six import StringIO  
from IPython.display import Image  


# In[ ]:


df = pd.read_csv('../input/covid19-demographic-predictors/covid19.csv')


# In[ ]:


X = df[df.columns[2:]]
y = df['Total Infected']
df[df.columns[1:]].corr()['Total Infected'][:]


# In[ ]:


plt.scatter(df['GDP 2018'], df['Population 2020'])
plt.xlabel('GDP')
plt.ylabel('Population')
plt.title('GDP vs Population')
plt.show()


# The scatter plot shows the following 3 outliers

# In[ ]:


df[(df['GDP 2018'] > 1e13) | (df['Population 2020'] > 400000)]


# China, India and US are outliers in the above dataset

# ### Scale the data and perform a linear regression

# In[ ]:


X_s = StandardScaler().fit_transform(X)


# In[ ]:


est = sm.OLS(y, X_s)
est2 = est.fit()
print(est2.summary(xname=[*X.columns]))
# small pval, reject the null hypothesis that a feature has no effect


# ### Random Forest and single Decision Tree

# In[ ]:


rfr = RandomForestRegressor(max_depth=3)
rfr.fit(X_s, y)
print('score', rfr.score(X_s, y), '\n')
[print(c + '\t', f) for c,f in zip(df[df.columns[2:]], rfr.feature_importances_)]


# In[ ]:


tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_s, y)
print('score', tree.score(X_s, y), '\n')
print('Feature Importances')
[print(c + '\t', f) for c,f in zip(df[df.columns[2:]], tree.feature_importances_)]


# In[ ]:


data = export_graphviz(tree, out_file=None,  
                filled=True, rounded=True,
                special_characters=True, feature_names=df.columns[2:])
graph = graphviz.Source(data)
graph


# In this analysis, population was the single biggest predictor of COVID-19 infections. This makes sense given the huge outliers shown in the scatter plot above.  Next, remove those outliers and rerun the experiment

# ## Remove outliers (China, United States, India) and rerun analysis

# In[ ]:


df_small = df[(df['Country'] != 'United States') & (df['Country'] != 'India') & (df['Country'] != 'China')]
X = df_small[df.columns[2:]]
y = df_small['Total Infected']


# Show countries with highest populations now that outliers have been removed
# 

# In[ ]:


df_small.sort_values(by='Population 2020', ascending=False).head(5)


# GDP and Population are NOT correlated

# In[ ]:


X = df_small[df_small.columns[2:]]
y = df_small['Total Infected']
df_small[df_small.columns[1:]].corr()['Total Infected'][:]


# In[ ]:


plt.scatter(df_small['GDP 2018'], df_small['Population 2020'])
plt.xlabel('GDP')
plt.ylabel('Population')
plt.title('GDP vs Population')
plt.show()


# ### Regression Analysis

# In[ ]:


X_s = StandardScaler().fit_transform(X)


# In[ ]:


est = sm.OLS(y, X_s)
est2 = est.fit()
print(est2.summary(xname=[*X.columns]))
# small pval, reject the null hypothesis that a feature has no effect


# The adjusted R-squared value is quite low; this shows that the model doesn't explain the variation in the dependent variable

# ### Random Forest and single Decision Tree

# In[ ]:


rfr = RandomForestRegressor(max_depth=3)
rfr.fit(X_s, y)
print('score', rfr.score(X_s, y), '\n')
[print(c + '\t', f) for c,f in zip(df_small[df_small.columns[2:]], rfr.feature_importances_)]


# In[ ]:


tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_s, y)
print('score', tree.score(X_s, y), '\n')
print('Feature Importances')
[print(c + '\t', f) for c,f in zip(df_small[df_small.columns[2:]], tree.feature_importances_)]


# In[ ]:


data = export_graphviz(tree, out_file=None,  
                filled=True, rounded=True,
                special_characters=True, feature_names=df.columns[2:])
graph = graphviz.Source(data)
graph


# Final analysis shows that GDP, mean age and smoking are the best predictors of COVID-19 infections. While mean age and smoking make sense, it is not known why there are more infections in countries with higher GDP.

# In[ ]:




