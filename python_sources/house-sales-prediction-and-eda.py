#!/usr/bin/env python
# coding: utf-8

# # Analysing House Sales in King County, USA
# ## This kernel we will examine house sales from 1970 to 2010 for King county  and we will try to predict house price using multiple linear regression 

# ### Importing libs
# 

# In[ ]:


import pandas as pd # for data analyse and data manupulation
import matplotlib.pyplot as plt # visualization
import numpy as np  
import folium # visualization
import seaborn as sns # visualization


# ## Understanding data 

# In[ ]:


data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
data.head()


# In[ ]:


data.describe()


# ## Visualize  data to  understand some important variable 

# In[ ]:


groups = data.groupby(['grade'])['price'].mean()
plt.figure(figsize=(10, 5))
plt.xlabel('price')
groups.plot.barh()


# In[ ]:


groups = data.groupby(['bedrooms'])['price'].mean()
plt.figure(figsize=(10, 5))
plt.xlabel('price')
groups.plot.barh()


# In[ ]:


groups = data.groupby(['bathrooms'])['price'].mean()
plt.figure(figsize=(10, 10))
groups.plot.barh()


# In[ ]:


sns.countplot(data.bathrooms, order = data['bathrooms'].value_counts().index)


# In[ ]:


sns.countplot(data.bedrooms, order = data['bedrooms'].value_counts().index)


# In[ ]:


sns.countplot(data.grade, order = data['grade'].value_counts().index)


# In[ ]:


sns.countplot(data.condition, order = data['condition'].value_counts().index)


# ## Now , we will examine  house sales  from 1970 to 2010   with geographical heat map
# ###  I will  separate dataset ;
# ### from 1970 to 1980 ,
# ### 1980 to 1990,
# ### 1990 to 2000,
# ### and 2000 to 2010
# 
# ###  I created a  function to generate map graph 

# In[ ]:


def generateBaseMap(map_location=[47.5,-122.161], zoom=9):
    base_map = folium.Map(location=map_location, control_scale=True, zoom_start=zoom)
    return base_map


# In[ ]:


from folium.plugins import HeatMap
df_copy = data[np.logical_and(data.yr_built<=1980,data.yr_built >= 1970)] 
df_copy['count'] = 1
base_map = generateBaseMap()
HeatMap(data=df_copy[['lat', 'long', 'count']].groupby(['lat', 'long']).sum().reset_index().values.tolist(), radius=8, max_zoom=15).add_to(base_map)
base_map


# In[ ]:


df_copy = data[np.logical_and(data.yr_built<=1990,data.yr_built >= 1980)] 
df_copy['count'] = 1
base_map = generateBaseMap()
HeatMap(data=df_copy[['lat', 'long', 'count']].groupby(['lat', 'long']).sum().reset_index().values.tolist(), radius=8, max_zoom=15).add_to(base_map)
base_map


# In[ ]:


df_copy = data[np.logical_and(data.yr_built<=2000,data.yr_built >= 1990)] 
df_copy['count'] = 1
base_map = generateBaseMap()
HeatMap(data=df_copy[['lat', 'long', 'count']].groupby(['lat', 'long']).sum().reset_index().values.tolist(), radius=8, max_zoom=15).add_to(base_map)
base_map


# In[ ]:


df_copy = data[np.logical_and(data.yr_built<=2010,data.yr_built >= 2000)] 
df_copy['count'] = 1
base_map = generateBaseMap()
HeatMap(data=df_copy[['lat', 'long', 'count']].groupby(['lat', 'long']).sum().reset_index().values.tolist(), radius=8, max_zoom=15).add_to(base_map)
base_map


# ### Prepare regression data for prediction
# #### we will create a corralation table 

# In[ ]:


neededCols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
            'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
            'yr_renovated', 'sqft_living15', 'sqft_lot15']


corr = data[neededCols].corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


dataForRegression = data[neededCols]


# In[ ]:


dataForRegression.head()


# In[ ]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

import statsmodels.api as sm


# ### Stats model  provide us some significant values like r2  
# 

# In[ ]:


X=dataForRegression.drop('price',axis=1)
y=dataForRegression['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=42)
lm = linear_model.LinearRegression() 
model = lm.fit(X_train[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15',
       'sqft_lot15']], y_train)

lm = sm.OLS(y_train, X_train)
model1 = lm.fit()
model1.summary()


# ### r2 is 0.881 
# ### our dataset is good for regression 

# In[ ]:




print('model accuracy is : ',model.score(X_test,y_test))


# ### the model have %65 accuracy but ;we will check cross validation score for better  verification 

# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


np.sqrt(mean_squared_error(y_train, model.predict(X_train)))


# In[ ]:


np.sqrt(mean_squared_error(y_test, model.predict(X_test)))


# In[ ]:


cross_val_score(model, X_train, y_train, cv = 100, scoring = "r2").mean()


# ### Now, we will try to find our model's prediction on the dataset and what is differece between real prices and predicted price 

# In[ ]:


predictedDatas=[]
for  row in range(0,len(dataForRegression)):
    a=(model.predict([[dataForRegression['bedrooms'].values[row],dataForRegression['bathrooms'].values[row],dataForRegression['sqft_living'].values[row],dataForRegression['sqft_lot'].values[row],dataForRegression['floors'].values[row],
        dataForRegression['waterfront'].values[row],dataForRegression['view'].values[row],dataForRegression['condition'].values[row],dataForRegression['grade'].values[row],dataForRegression['sqft_above'].values[row],
        dataForRegression['sqft_basement'].values[row],dataForRegression['yr_built'].values[row],dataForRegression['yr_renovated'].values[row],dataForRegression['sqft_living15'].values[row],dataForRegression['sqft_lot15'].values[row]
        ]]))
    a=round(a[0],0)
    predictedDatas.append(a)


# In[ ]:


final_df = dataForRegression.price.values
final_df = pd.DataFrame(final_df,columns=['Real_price'])
final_df['predicted_prices'] = predictedDatas
final_df['difference'] = abs(final_df['Real_price'] - final_df['predicted_prices'])
final_df.tail(20)


# ### If we want to predict a house price  except from this  dataset , we can predict this way;
# #### we write our 15 criteria  to predict method 

# In[ ]:


prediction= model.predict([[2,0,1180,6000,1,0,0,4,7,1180,0,1995,2010,1340,6000]]) 
prediction=round(prediction [0],0)
prediction


# # As a result , 
# ### Popular areas from 1970 to 2010   are map valley, sammammish
# ###  Most sales house type  has 3 bedroom, 2.5 bathroom , 7 grade level  and 3 Condition level 
# ### Also , we learnt how to make prediction with Multiple linear regression

# In[ ]:




