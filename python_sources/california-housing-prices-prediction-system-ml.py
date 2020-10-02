#!/usr/bin/env python
# coding: utf-8

# # California Housing Prices Prediction 

# This data contains information from the 1990 California census. 

# In[ ]:


# Importing the libraries :

# Explore the Data :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Data preprocessing Libraries :
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Regression Libraries :
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score


# ### Import the DataSet (csv file ):

# In[ ]:


# read the data :
df_house = pd.read_csv('../input/housing.csv')


# In[ ]:


df_house.columns


# In[ ]:


List_of_Labels = list(df_house['median_house_value'].head(10))
List_of_Labels


# Note :                          
# 1.It is a 'Regression' problem because the label 'median_house_value' is continuous .           
# 2.We will apply all types of regression and predict the median_house_value .               
# 3.We will compare which type of regression is best for this prediction .

# ### Sample of our Data Set :

# In[ ]:


df_house.head(5)


# In[ ]:


df_house.tail(5)


# In[ ]:


df_house.describe()


# ### Checking for NaN value :

# In[ ]:


df_house.isnull().sum()


# In[ ]:


df_house.isnull().sum().plot(kind = 'bar')


# You can see there are NaN values in coulmn named 'total_bedrooms' so we have to deal with that 

# In[ ]:


# filling zero on the place of NaN values in the data set 
df_house['total_bedrooms'].fillna(0,inplace = True)


# In[ ]:


df_house.isnull().sum()


# # Exploratory Data Analysis (EDA)

# In[ ]:


plt.figure(figsize=(15,6))
plt.subplots_adjust(hspace = .25)
plt.subplot(1,2,1)
plt.xlabel('ocean_proximity',fontsize=12)
plt.ylabel('median_house_value',fontsize=12)
sns.stripplot(data=df_house,x='ocean_proximity',y='median_house_value',)
plt.subplot(1,2,2)
plt.xlabel('ocean_proximity',fontsize=12)
plt.ylabel('median_house_value',fontsize=12)
sns.boxplot(data=df_house,x='ocean_proximity',y='median_house_value')
plt.plot()


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplots_adjust(hspace = .25)
plt.subplot(1,2,1)
plt.title('Corelation b/w longtitude and median_house_value')
plt.xlabel('longitude',fontsize=12)
plt.ylabel('median_house_value',fontsize=12)
plt.scatter(df_house['longitude'].head(100),df_house['median_house_value'].head(100),color='g')
plt.subplot(1,2,2)
plt.title('Corelation b/w latitude and median_house_value')
plt.xlabel('latitude',fontsize=12)
plt.ylabel('median_house_value',fontsize=12)
plt.scatter(df_house['latitude'].head(100),df_house['median_house_value'].head(100),color='r')


# In[ ]:


df_house.plot(kind='scatter', x='longitude', y='latitude', alpha=0.9, 
    s=df_house['population']/100, label='population', figsize=(14,10), 
    c='median_house_value', cmap=plt.get_cmap('prism'), colorbar=True)


# In[ ]:


df_house.plot(kind='scatter', x='longitude', y='latitude', alpha=0.9, 
    s=df_house['population']/10, label='population', figsize=(14,10), 
    c='median_house_value', cmap=plt.get_cmap('cool'), colorbar=True)


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(df_house['median_house_value'],color='red')
plt.show()


# In[ ]:


df_house[df_house['median_house_value']>450000]['median_house_value'].value_counts().head()


# In[ ]:


df_house=df_house.loc[df_house['median_house_value']<500001,:]
df_house=df_house[df_house['population']<25000]
plt.figure(figsize=(10,6))
sns.distplot(df_house['median_house_value'],color='yellow')
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))
plt.subplots_adjust(hspace = .25)
plt.subplot(1,2,1)
df_house['ocean_proximity'].value_counts().plot(kind = 'pie',colormap = 'jet')
plt.subplot(1,2,2)
df_house['median_income'].hist(color='purple')


# In[ ]:


df_house.hist(bins=100, figsize=(20,20) , color = 'b')


# ### Corelation Matrics 

# In[ ]:


plt.figure(figsize=(10,4))
sns.heatmap(cbar=False,annot=True,data=df_house.corr(),cmap='Blues')
plt.title('% Corelation Matrix')
plt.show()


# # Data Preprocessing :

# In[ ]:


x=df_house.iloc[:,:-1].values
print(x)


# In[ ]:


y=df_house['median_house_value'].values
print(y)


# ### Encoding the Categorical values :

# In[ ]:


df_house['ocean_proximity'].value_counts().plot(kind = 'bar')


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:, 8] = labelencoder.fit_transform(x[:, 8])
onehotencoder = OneHotEncoder(categorical_features = [8])
x = onehotencoder.fit_transform(x).toarray()


# ### Spliting the data set into train and test set :

# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:


print('xtrain :')
print(xtrain)
print('xtest :')
print(xtest)


# In[ ]:


print('ytrain :')
print(ytrain)
print('ytest :')
print(ytest)


# ### Linear Regression :

# In[ ]:


from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(xtrain,ytrain)


# In[ ]:


# predict the value of dependent variable y
ypred = linear_regressor.predict(xtest)
ypred


# In[ ]:


from sklearn.metrics import mean_squared_error
predictions = linear_regressor.predict(xtest)
lin_mse = mean_squared_error(ytest,predictions)
lin_rmse = np.sqrt(lin_mse)
print('rmse value is : ',lin_rmse)


# In[ ]:


lin_reg_score = linear_regressor.score(xtest,ytest)
print('r squared value is : ',lin_reg_score )


# ### Decision Tree Regression :

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(xtrain,ytrain)


# In[ ]:


y_pred = tree_regressor.predict(xtest)
y_pred


# In[ ]:


from sklearn.metrics import mean_squared_error
predictions = tree_regressor.predict(xtest)
lin_mse = mean_squared_error(ytest,predictions)
lin_rmse = np.sqrt(lin_mse)
print('rmse value is : ',lin_rmse)


# In[ ]:


tree_score = tree_regressor.score(xtest,ytest)
print('r squared value is : ',tree_score )


# ### Random forest Regression :

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rn_forest_regressor = RandomForestRegressor(n_estimators=50,random_state=0)
rn_forest_regressor.fit(xtrain,ytrain)


# In[ ]:


rn_forest_regressor.predict(xtest)


# In[ ]:


from sklearn.metrics import mean_squared_error
predictions = rn_forest_regressor.predict(xtest)
lin_mse = mean_squared_error(ytest,predictions)
lin_rmse = np.sqrt(lin_mse)
print('rmse value is : ',lin_rmse)


# In[ ]:


rsq_rn_forest = rn_forest_regressor.score(xtest,ytest)
print('r squared value is : ',rsq_rn_forest )


# We can see that by the use of Random forest regression we are getting r squared value is 0.79 .

# Hence  ' RANDOM FOREST ' could be the best model because of low mean squred error and a high r squared value .

# In[ ]:




