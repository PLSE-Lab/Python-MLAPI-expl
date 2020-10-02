#!/usr/bin/env python
# coding: utf-8

# # HI , this kernel shows machine learning basics for the very first beginners 
#  We'll start with a model called the Decision Tree. There are fancier models that give more accurate predictions. But decision trees are easy to understand, and they are the basic building block for some of the best models in data science.
# 
# For simplicity, we'll start with the simplest possible decision tree. 
# 
# ![First Decision Trees](http://i.imgur.com/7tsb5b1.png)
# 
# It divides houses into only two categories. The predicted price for any house under consideration is the historical average price of houses in the same category.
# 
# We use data to decide how to break the houses into two groups,  and then again to determine the predicted price in each group.  This step of capturing patterns from data is called **fitting** or **training** the model. The data used to **fit** the model is called the **training data**.  
# 
# The details of how the model is fit (e.g. how to split up the data) is complex enough that we will save it for later. After the model has been fit, you can apply it to new data to **predict** prices of additional homes.

# # In our Case we won't talk about houses but about VIDEO GAMES HAHAH
# **Don't expect a lot of EDAs ,as the titles says its an intro to machine learning decisions tree model **

# In[ ]:


import pandas as pd # for data preprocessing in the workspace
import numpy as np #calculus and linear algebra
# plotters
import matplotlib.pyplot as plt 
import seaborn as sns

import plotly.plotly as py  # plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[ ]:


Data = pd.read_csv('../input/vgsales.csv')
Data.head(20)# first 20 records 


# In[ ]:


Data.columns


# ## Let's have a look on the game platform which contains the most games.

# In[ ]:


plt.figure(figsize=(15,8))
sns.set()
plt.grid(True)
sort_plat = Data['Platform'].value_counts().sort_values(ascending=False)
sort_plat.head()
sns.barplot(y=sort_plat.index,x=sort_plat.values,orient='h')
plt.xlabel('Values count')
plt.ylabel('Games Platform')
plt.title('Grouped Platforms count')


# ## World rank of the top 100 video games, north america sales and europe sales.

# In[ ]:



df_gl=Data.loc[:99,:] # data.iloc[:100,:] -- data.head(100)

import plotly.graph_objs as go

trace1=go.Scatter(
                x=df_gl.Rank,
                y=df_gl.NA_Sales,
                mode="lines+markers",
                name="North America Sales",
                text=df_gl.Name)
trace2=go.Scatter(
                x=df_gl.Rank,
                y=df_gl.EU_Sales,
                mode="lines",
                name="Europe Sales",
                text=df_gl.Name)

edit_df=[trace1,trace2]
layout=dict(title="World rank of the top 100 video games, EU Sales and NA Sales .",
            xaxis=dict(title="World Rank",tickwidth=5,ticklen=8,zeroline=False))
fig=dict(data=edit_df,layout=layout)
iplot(fig)


# In[ ]:


Data.describe()


# In[ ]:



sns.heatmap(Data.corr(),cbar=True,annot=True)


# As we can clearly see , NA_Sales is the highest correlation but we cant deny EU_Sales and the Other_Sales that are stongly correlated too.

# In[ ]:


max_Sales = Data[Data['Global_Sales']==max(Data['Global_Sales'])]
max_Sales


# # SIMPLE EDA to show the correlation and explains the model choosen

# In[ ]:


sns.set()
sns.regplot(Data['Global_Sales'],Data['NA_Sales'])
plt.xlabel('Global Sales')
plt.ylabel('North America Sales')
plt.title('Global Sales - NA Sales ')


# In[ ]:


sns.regplot(Data['Global_Sales'],Data['EU_Sales'])
plt.xlabel('Global Sales')
plt.ylabel('Europe Sales')
plt.title('Global Sales - EU Sales ')


# In[ ]:


sns.regplot(Data['Global_Sales'],Data['JP_Sales'])
plt.xlabel('Global Sales')
plt.ylabel('Japan Sales')
plt.title('Global Sales - JP Sales ')


# In[ ]:


sns.regplot(Data['Global_Sales'],Data['Other_Sales'])
plt.xlabel('Other countries Sales')
plt.ylabel('North America Sales')
plt.title('Global Sales - Others Sales ')


# In[ ]:


plt.figure(figsize=(15,8))
cop = Data.copy()
cop.sort_values('Global_Sales',ascending=False)
print(cop.shape)
cop1 = cop.head(1000).copy()
sns.barplot(y=cop1['Publisher'],x=cop1['Global_Sales'],orient='h')


# In[ ]:


#Some label encoding since we have some categorical DATA
obj_cols = [col for col in cop.columns if cop[col].dtype=='object']
print('Columns that will be encoded are ='+str(obj_cols))


# In[ ]:


#Quick peak into NA columns

fig = plt.figure(figsize=(15, 8))
cop.isna().sum().sort_values(ascending=True).plot(kind='barh', fontsize=20)


# In[ ]:


cop.drop('Year',axis=1)


# In[ ]:


print(cop.shape)


# In[ ]:


# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: cop[col].nunique(), obj_cols))
d = dict(zip(obj_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])


# **Since we have 4 categorical data with far more than 10 entries , wont be good to OH( One hot encoding) them , label coding will be fair enough but will overfit the model , we will just skip the label encoding **

# In[ ]:


from sklearn.model_selection import train_test_split #Best approach to test the model
from sklearn.metrics import mean_absolute_error # mean absolute error , error = predictions - validation_y then abs for pos value
from sklearn.tree import DecisionTreeRegressor #model
features = ['NA_Sales','EU_Sales','JP_Sales','Other_Sales']#our features
X = cop[features]
y = cop.Global_Sales #target
train_X , val_X , train_y , val_y = train_test_split(X,y,test_size=0.25,random_state=1)
model = DecisionTreeRegressor(random_state=1)
model.fit(train_X,train_y)


# In[ ]:


predictions =model.predict(val_X)
mae = mean_absolute_error(predictions, val_y)
print('Mean absolute error '+str(mae))


# In[ ]:



df = pd.DataFrame({'Actual': val_y, 'Predicted': predictions})
df


# In[ ]:



df1 = df.head(80)
df1.plot(kind='bar',figsize=(15,8))
plt.show()


# In[ ]:


val_X['Global_Sales']=predictions
print(len(df.index))
val_X['Rank'] = df.index
val_X[['Rank','Global_Sales']].to_csv('sub_for_nothing.csv',index=False)
df.to_csv('predvsval_y.csv',index=False)


# ### We reached together the end of this Kernel , if you found it useful an upvote would be very appreciated , thanks see you in the next kernel !
