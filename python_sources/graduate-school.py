#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/Admission_Predict.csv')
df.head()


# Now, let us drop the irrelevant column and check if there are any null values in the dataset

# In[2]:


df = df.drop(['Serial No.'], axis=1)
df.isnull().sum()


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
fig = sns.regplot(x="GRE Score", y="TOEFL Score", data=df)
plt.title("GRE Score vs TOEFL Score")
plt.show()


# People with higher GRE Scores also have higher TOEFL Scores which is justified because both TOEFL and GRE have a verbal section which although not similar are relatable

# In[5]:


fig = sns.regplot(x="GRE Score", y="CGPA", data=df)
plt.title("GRE Score vs CGPA")
plt.show()


# Although there are exceptions, people with higher CGPA usually have higher GRE scores maybe because they are smart or hard working

# In[6]:


fig = sns.lmplot(x="CGPA", y="LOR ", data=df, hue="Research")
plt.title("GRE Score vs CGPA")
plt.show()


# In[7]:


fig = sns.lmplot(x="GRE Score", y="LOR ", data=df, hue="Research")
plt.title("GRE Score vs CGPA")
plt.show()


# GRE scores and LORs are also not that related. People with different kinds of LORs have all kinds of GRE scores

# In[8]:


fig = sns.regplot(x="CGPA", y="SOP", data=df)
plt.title("GRE Score vs CGPA")
plt.show()


# In[9]:


fig = sns.regplot(x="GRE Score", y="SOP", data=df ,scatter_kws={"color": "red"}, line_kws={"color": "black"})
plt.title("GRE Score vs CGPA")
plt.show()


# In[10]:


fig = sns.regplot(x="TOEFL Score", y="SOP", data=df)
plt.title("GRE Score vs CGPA")
plt.show()


# In[11]:


fig = sns.regplot(x="University Rating", y="CGPA", data=df, scatter_kws={"color": "blue"}, line_kws={"color": "black"})
plt.title("University Rating vs CGPA")
plt.show()


# **Correlation among variables**

# In[12]:


import numpy as np
corr = df.corr()
fig, ax = plt.subplots(figsize=(8, 8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.show()


# Lets split the dataset with training and testing set and prepare the inputs and outputs

# In[13]:


from sklearn.model_selection import train_test_split

X = df.drop(['Chance of Admit '], axis=1)
y = df['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False)


# Lets use a bunch of different algorithms to see which model performs better

# In[14]:


from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

models = [['DecisionTree :',DecisionTreeRegressor()],
           ['Linear Regression :', LinearRegression()],
           ['RandomForest :',RandomForestRegressor()],
           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)]]

print("Results...")


for name,model in models:
    model = model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))


# In[ ]:





# In[ ]:




