#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df=pd.read_csv('../input/who-suicide-statistics/who_suicide_statistics.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


# checking missing values
df.isnull().sum()


# we can see that total number of recored are 43776 and 'suicide_no' and 'population' has large number of missing values , 2256 & 5460 respectively

# In[ ]:


# sorting in the ascending order by 'year'
df= df.sort_values(['year'],ascending=True)


# In[ ]:


df.head()


# In[ ]:


# getting the total number of different countries in our dataset
print(" number of countries:" , df['country'].nunique())


# In[ ]:


# checking percentage of missing values
missing_percent= df.isnull().sum()/df.shape[0]
print(missing_percent*100)


# we can see that 'suicide_no' holds 5.15 % of missing values & 'population' holds 12.47 % of missing values

# In[ ]:


# checking correlation between population and number of suicide in a country

corr= df['suicides_no'].corr(df['population'])
print(corr) # correlation betwwen suicides and country


# # #  VISUALIZATIONS

# In[ ]:


# top 10 countries with suicide cases
df[['country','suicides_no']].groupby(['country']).agg('sum').sort_values(by='suicides_no',ascending=False).head(10)


# In[ ]:


# looking at it graphically
import seaborn as sns
plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = (15, 9)

color = plt.cm.winter(np.linspace(0, 10, 100))
x = pd.DataFrame(df.groupby(['country'])['suicides_no'].sum().reset_index())
x.sort_values(by = ['suicides_no'], ascending = False, inplace = True)

sns.barplot(x['country'].head(10), y = x['suicides_no'].head(10), data= x, palette = 'winter')
plt.title('Top 10 Countries in Suicides', fontsize = 20)
plt.xlabel('Name of Country')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.show()


# In[ ]:


# top 10 countries with minimum sucide cases
df[['country','suicides_no']].groupby(['country']).agg('sum').sort_values(by='suicides_no',ascending=True).head(10)


# In[ ]:


# graphically
plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = (15, 9)

color = plt.cm.winter(np.linspace(0, 10, 100))
x = pd.DataFrame(df.groupby(['country'])['suicides_no'].sum().reset_index())
x.sort_values(by = ['suicides_no'], ascending = True , inplace = True)

sns.barplot(x['country'].head(10), y = x['suicides_no'].head(10), data= x, palette = 'winter')
plt.title('Top 10 Countries least number Suicides', fontsize = 20)
plt.xlabel('Name of Country')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.show()


# In[ ]:


# year with highest number of suicides
df[['year','suicides_no']].groupby('year').agg('sum').sort_values(by='suicides_no',ascending=False).head(10)


# In[ ]:


# graphically
plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = (15, 9)

color = plt.cm.winter(np.linspace(0, 10, 100))
x = pd.DataFrame(df.groupby(['year'])['suicides_no'].sum().reset_index())
x.sort_values(by = ['suicides_no'], ascending = False , inplace = True)

sns.barplot(x['year'].head(10), y = x['suicides_no'].head(10), data= x, palette = 'winter')
plt.title('suicides in a years', fontsize = 20)
plt.xlabel('year')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.show()


# In[ ]:


# age group with highest no. of suicides
df[['age','suicides_no']].groupby('age').agg('sum').sort_values(by='suicides_no',ascending=False).head(10)


# In[ ]:


# Looking at it graphically
plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = (15, 9)

color = plt.cm.winter(np.linspace(0, 10, 100))
x = pd.DataFrame(df.groupby(['age'])['suicides_no'].sum().reset_index())
x.sort_values(by = ['suicides_no'], ascending = False , inplace = True)

sns.barplot(x['age'].head(10), y = x['suicides_no'].head(10), data= x, palette = 'winter')
plt.title('Top age groups with highest number Suicides', fontsize = 20)
plt.xlabel('age group')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.show()


# In[ ]:


# number of suicides with respect to sex
df[['sex','suicides_no']].groupby('sex').agg('sum').sort_values(by='suicides_no',ascending='False').head()


# we can clearly see that males committed suicide more than females

# # # HANDELING CATEGORICAL DATA 

# In[ ]:


#label encoder for 'sex' column

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
df['sex']= le.fit_transform(df['sex'])
df['age']=le.fit_transform(df['age'])


# In[ ]:


# dropping unnecessary column which is 'country'
df.drop(['country'],axis=1,inplace=True)


# In[ ]:



df.head()


# # #  HANDELING MISSING VALUES

# In[ ]:


# filling missing values
df['suicides_no'].fillna(0,inplace=True)

# we will fill mean of population in missinf values of 'population' feature
df['population'].fillna(1664090,inplace=True)

# converting in inerger type
df['suicides_no']=df['suicides_no'].astype(int)
df['population']=df['population'].astype(int)


# In[ ]:


# correlation betwwen the features

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr() ,annot=True,cmap="RdYlGn")


# extreme green--> high correlation,   extreme red--> least correlation

# In[ ]:


# spliting into dependent and independent variables
X= df.drop(['suicides_no'],axis=1)
y= df['suicides_no']


# In[ ]:


# splitting into train_test_split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# creating a scaler
mm = MinMaxScaler()

# scaling the independent variables
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)


# # # LINEAR REGRESSION

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# creating the model
model = LinearRegression()

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)

# calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)

#calculating the r2 score
r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)


# # RANDOM FOREST REGRESSION

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# creating the model
model = RandomForestRegressor()

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)

# calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)

#calculating the r2 score
r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)


# # # DECISION TREE REGRESSOR

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

# creating the model
model = DecisionTreeRegressor()

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)

# calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)

#calculating the r2 score
r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)


# In[ ]:





# CONCLUSION:
# 
# R2 score is best for Random Forest Regressor &
# 
# RMSE is best for simple linear regression
