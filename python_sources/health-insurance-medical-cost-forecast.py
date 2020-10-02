#!/usr/bin/env python
# coding: utf-8

# Import the modules

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


# Load the data

# In[3]:


df = pd.read_csv("../input/insurance.csv")


# Look at the data

# In[111]:


df.shape


# In[112]:


df.info()


# <h3>Data Exloration</h3>
# <h4>Label: "charges"</h4>
# Calculate the summary statistics, look at the distribution by histogram

# In[113]:


df["charges"].describe()


# In[114]:


sns.distplot(df["charges"], bins=20, fit=norm)
plt.show()


# <h4>Numeric features: "age", "bmi", "children"</h4>
# Univariate analysis: Calculate the summary statistics, look at the distribution by histogram
# <br>
# Bivariate analysis: Scatter plots
# <br>
# Correlation matrix

# In[115]:


df["age"].describe()


# In[116]:


sns.distplot(df["age"], bins=10, fit=norm)
plt.show()
sns.regplot(x="age", y="charges", data=df);
plt.show()


# In[117]:


df["bmi"].describe()


# In[118]:


sns.distplot(df["bmi"], bins=10, fit=norm)
plt.show()
sns.regplot(x="bmi", y="charges", data=df);
plt.show()


# In[119]:


df["children"].value_counts()


# In[120]:


sns.countplot(x="children", data=df);
plt.show()
sns.boxplot(x="children", y="charges", data=df);
plt.show()


# In[121]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True);
plt.show()


# In[122]:


print(corr.loc["charges"].sort_values(ascending=False).drop("charges"))


# <h4>Categorical features: "sex", "smoker", "region"</h4>
# Univariate analysis: Look at the distribution by count plots
# <br>
# Bivariate analysis: Boxplots

# In[123]:


df["sex"].value_counts()


# In[124]:


sns.countplot(x="sex", data=df);
plt.show()
sns.boxplot(x="sex", y="charges", data=df);
plt.show()


# In[125]:


df["smoker"].value_counts()


# In[126]:


sns.countplot(x="smoker", data=df);
plt.show()
sns.boxplot(x="smoker", y="charges", data=df);
plt.show()


# In[127]:


df["region"].value_counts()


# In[128]:


sns.countplot(x="region", data=df);
plt.show()
sns.boxplot(x="region", y="charges", data=df);
plt.show()


# Make the feature "children" categorical

# In[129]:


df["children"] = df["children"].astype("str")


# Create dummy variables

# In[130]:


df = pd.get_dummies(df, drop_first=True)


# Look at the transformed dataset

# In[131]:


df.shape


# In[132]:


df.info()


# Prepare the data for modeling

# In[133]:


X = df.drop("charges", axis=1)
X = X.values
y = df["charges"]
y = y.values


# Use a linear regression, a decision tree regressor and a random forest regressor

# In[134]:


linreg = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()


# In[135]:


score_lin = round(np.mean(cross_val_score(linreg, X, y, cv=5)),3)
print(score_lin)


# In[136]:


param_grid_dtr = {'max_depth': [2,3,4], 'max_leaf_nodes': [13,14,15]}

grid_dtr = GridSearchCV(dtr, param_grid_dtr, cv=5)

grid_dtr.fit(X,y)

print(grid_dtr.best_params_)


# In[137]:


score_dtr = round(np.mean(cross_val_score(grid_dtr, X, y, cv=5)),3)
print(score_dtr)


# In[138]:


param_grid_rfr = {'n_estimators': [16,17,18,19], 'max_depth': [2,3,4], 'max_leaf_nodes': [19,20,21,22,23]}


grid_rfr = GridSearchCV(rfr, param_grid_rfr, cv=5)

grid_rfr.fit(X,y)

print(grid_rfr.best_params_)


# In[139]:


score_rfr = round(np.mean(cross_val_score(grid_rfr, X, y, cv=5)),3)
print(score_rfr)


# Compare the scores

# In[140]:


print("Linear Regression:" + str(score_lin))
print("Decision Tree Regressor:" + str(score_dtr))
print("Random Forest Regressor:" + str(score_rfr))

