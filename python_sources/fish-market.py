#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import copy


# In[ ]:


data = pd.read_csv('../input/fish-market/Fish.csv')


# In[ ]:


data.rename(columns= {'Length1':'Body_height', 'Length2':'Total_Length', 'Length3':'Diagonal_Length'}, inplace=True)


# ![image.png](attachment:image.png)

# In[ ]:


data.head()


# In[ ]:


data.describe()


# ### 2. Analize data ###

# In[ ]:


sns.countplot(x='Species',data=data);


# In[ ]:


sns.heatmap(data.corr(), annot=True, cmap='BuGn');


# In[ ]:


data = data.drop(columns=['Total_Length', 'Diagonal_Length'])


# From the plot data we can see that parameters Body_height, Total_Length, Diagonal_Length depend each other. 
# Select for data only: Body_height.
# And stay independed parameters:
# * Body_height
# * Height
# * Width
#  
#  Target parameter: 
# * Weight

# On boxplot can watch outliers by weight

# In[ ]:


sns.boxplot(x='Species',y='Weight',data=data,palette='rainbow');


# In[ ]:


names = data['Species'].unique()
columns = data.columns[:]


# ### 3. Remove outliers ###

# In[ ]:


columns = data.columns[1:]

data_outliers = []
for name in names:
    dfw = data[data['Species']==name]
    for col in columns:    
        Q_min = dfw["Weight"].quantile(0.01)
        Q_max = dfw["Weight"].quantile(0.99)
        idx = (data['Species']==name) & ((data["Weight"] < Q_min) | (data["Weight"] > Q_max))
        data_outliers.append(data[idx])

data_outliers = pd.concat(data_outliers)
data_cleared = data.drop(data_outliers.index.unique())
data_cleared


# In[ ]:


data_cleared.describe()


# In[ ]:


def display(y, ypred):
    pd.options.display.max_rows = (len(y))
    df = pd.DataFrame ({'Actual': y, 'Predicted': ypred})
    return df


# In[ ]:


x = data_cleared
y = data_cleared['Weight']
x = x.drop(columns=["Weight"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ### Common LinearRegression ###

# In[ ]:


model = LinearRegression(fit_intercept=True)
model.fit(x.iloc[:,1:],y)
model.score(x.iloc[:,1:], y)
ypredict = model.predict(x.iloc[:,1:])
plt.scatter(y, ypredict);


# In[ ]:


model.score(x.iloc[:,1:], y)


# ### Test on linear independence ###

# As we can see, some 'Weight' are negative value that can not be.
# Let `s to watch on plots:
# 
# On many plots there are no linear independence. The weight depends on the three components together, i.e. the volume. Let's try to multiply the components.

# In[ ]:


g = sns.FacetGrid(data_cleared, col="Species")
g.map(plt.scatter, "Body_height", "Weight", alpha=0.7);


# In[ ]:


g = sns.FacetGrid(data_cleared, col="Species")
g.map(plt.scatter, "Height", "Weight", alpha=0.7);


# In[ ]:


g = sns.FacetGrid(data_cleared, col="Species")
g.map(plt.scatter, "Width", "Weight", alpha=0.7);


# In[ ]:


data_volume = pd.DataFrame([])
data_volume['Species'] = data_cleared['Species']
data_volume['Weight'] = data_cleared['Weight']
data_volume['Volume'] = data_cleared['Body_height'] * data_cleared['Height'] * data_cleared['Width']

data_volume


# In[ ]:


g = sns.FacetGrid(data_volume, col="Species")
g.map(plt.scatter, "Volume", "Weight", alpha=0.7);


# On this plots the linear independence is more explicit. Use this data for learning linear regression by every species of fish.

# ### LinearRegression by every species ###

# In[ ]:


def Regression():
    total_y_test = []
    total_y_pred = []
    model = {}

    for name in names:

        df = data_volume[data_volume['Species']==name]
        x = df.iloc[:,2:]
        y = df['Weight']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)

        model[name] = LinearRegression()
        model[name].fit(x_train, y_train)
        y_pred = model[name].predict(x_test)
        total_y_test.append(y_test)
        total_y_pred.append(pd.Series(y_pred))
        print(name, ": ", r2_score(y_test, y_pred))


    total_y_test = pd.concat(total_y_test, ignore_index=True)
    total_y_pred = pd.concat(total_y_pred, ignore_index=True)

    print("---= TOTAL =---")
    print("R2: ", r2_score(total_y_test, total_y_pred))
    return total_y_pred, total_y_test, model


# In[ ]:


total_y_pred, total_y_test, model = Regression()
plt.scatter(total_y_pred, total_y_test)
display(total_y_test, total_y_pred)


# So we got 97,37% accuracy.
# It is good result.
# 
# ### Happy Coding ###
