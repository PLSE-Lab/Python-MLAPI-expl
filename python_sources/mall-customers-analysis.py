#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import DecisionTreeRegressor
df = pd.read_csv('../input/Mall_Customers.csv')
print('info about table')
print(df.info())


# In[ ]:


print('Number of rows in the dataset: ',df.shape[0])
print('Number of columns in the dataset: ',df.shape[1])


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# There are no null values in the dataset

# In[ ]:


df.describe()


# In[ ]:


male =len(df[df['Gender'] == 'Male'])
female = len(df[df['Gender']== 'Female'])

plt.figure(figsize=(10,8))

# Data to plot
labels = 'Male','Female'
sizes = [male,female]
colors = ['skyblue', 'yellow']
explode = (0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%',
shadow=True, startangle=270)
 
plt.axis('equal')
plt.show()


# In[ ]:


plot = df.Age.value_counts().sort_index().plot(kind = "line", figsize=(15,5), fontsize=13)
plot.set_title("Mall customers: Age distribution", fontsize = 20)


# In[ ]:


plot = df.Age.value_counts().sort_index().plot(kind = "bar", figsize=(15,5), fontsize = 15)
plot.set_title("Mall Customers: Age distribution", fontsize = 20)


# In[ ]:


plot = df['Spending Score (1-100)'].value_counts().sort_index().plot(kind = "bar", figsize=(20,5), fontsize = 15)
plot.set_title("Mall Customers: Spending score", fontsize = 20)
#Spending Score distribution


# In[ ]:


plot = df['Spending Score (1-100)'].value_counts().sort_index().plot(kind = "line", figsize=(15,5), fontsize=13)
plot.set_title("Mall customers: Spending Score ", fontsize = 20)
#Spending Score distribution


# In[ ]:


sns.distplot(df['Annual Income (k$)'],kde=False,bins=60,color='green')
#Annual income distribution


# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='Annual Income (k$)',data = df, hue = 'Gender',palette='GnBu')
plt.show()
#Annual income distribution


# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='Age',data = df, hue = 'Gender',palette='GnBu')
plt.show()
#Age distribution


# In[ ]:


plt.figure(figsize=(12,9))
sns.scatterplot(x='Spending Score (1-100)',y='Annual Income (k$)',data=df,hue='Gender')
plt.show()


# In[ ]:


plt.figure(figsize=(12,9))
sns.scatterplot(x='Spending Score (1-100)',y='Age',data=df,hue='Gender')
plt.show()


# In[ ]:


plt.figure(figsize=(12,9))
sns.scatterplot(x='Annual Income (k$)',y='Age',data=df,hue='Gender')
plt.show()


# In[ ]:


#Trying to predict spending score
df['Gender'] = df['Gender'].apply({'Male':1, 'Female':2}.get)


# In[ ]:


y = df['Spending Score (1-100)']
df_features = ['Gender','Age' ,'Annual Income (k$)']
x = df[df_features]
#Define model
df_model = DecisionTreeRegressor(random_state = 1)
#Fit model
df_model.fit(x,y)
print('prediction for first five rows')
print(df_model.predict(x.head()))


# In[ ]:


from sklearn.metrics import mean_absolute_error
predicted_score = df_model.predict(x)
MAE = mean_absolute_error(y,predicted_score)
print('mean absolute error is '  + str(MAE))


# In[ ]:


from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(x,y,random_state=0)


# In[ ]:


df_mod = DecisionTreeRegressor()
df_mod.fit(train_x, train_y)
score_predictions = df_mod.predict(val_x)
MAE = mean_absolute_error(val_y,score_predictions)
print('mean absolute error is '  + str(MAE))


# In[ ]:


def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x,train_y)
    pred_val = model.predict(val_x)
    MAE = mean_absolute_error(val_y, pred_val)
    return(MAE)
for max_leaf_nodes in [5,50,75,77,80,100,500,1000,2500,5000,10000,1000000]:
    my_mae=get_mae(max_leaf_nodes,train_x,val_x,train_y,val_y)
    print('Max leaf node = ' + str(max_leaf_nodes) + '; MAE = ' + str(my_mae))


# Random forest model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state = 1)
forest_model.fit(train_x,train_y)
df_pred = forest_model.predict(val_x)
print(mean_absolute_error(val_y,df_pred))


# In[ ]:




