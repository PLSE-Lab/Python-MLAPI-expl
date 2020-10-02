#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import data

# In[ ]:


data=pd.read_csv('../input/english-premier-league-players-dataset/epldata_final.csv')
data.head()


# In[ ]:


data.describe()


# In[ ]:


missing_val_count_by_column = (data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


data.info()


# **Lets do some exploraition of the data **
# first lets see the sum of "market value" of big and not big teams

# How many players play are for big clubs and how many are not?

# In[ ]:


data['big_club'].value_counts()
sns.countplot(x='big_club',data=data,palette='Set1')


# Where do they come from? and what is the average market value for each country 

# In[ ]:


import plotly.express as px
import plotly.graph_objects as go


# Unite UK countries

# In[ ]:


byNation=data.groupby('nationality')
Nation=byNation.count()
Nation.reset_index(inplace=True)
Nation1=pd.DataFrame(Nation[['nationality','name']])
Nation1['count']=Nation['name']
Nation1.drop('name',inplace=True,axis=1)

Nation1.set_index('nationality',inplace=True)
UK={'nationality':['UK'],'count':[int(Nation1.loc['England']+Nation1.loc['Scotland']+Nation1.loc['Wales']+Nation1.loc['Northern Ireland'])]}
df=pd.DataFrame(UK)
df.set_index('nationality',inplace=True)
Nation1=pd.concat([Nation1,df])


# In[ ]:


Nation1['averageVal']=round(byNation['market_value'].mean(),2)
UKCountries=['England','Scotland','Wales','Northern Ireland']
top=0
bottom=Nation1.loc['UK']['count']
for country in UKCountries:
    top+=Nation1['averageVal'].loc[country]*Nation1['count'].loc[country]

AvgUK=top/bottom
AvgUK
Nation1['averageVal']['UK']=round(AvgUK,2)


# In[ ]:


Nation1['Text']=None
for nation in Nation1.index:
    Nation1['Text'][nation]='Average Market Value: '+ str(Nation1.loc[nation]['averageVal'])+"M$"+'<br>'+    'Number of Players:' + str(Nation1.loc[nation]['count'])


# In[ ]:


nationData = dict(type = 'choropleth',

            locations = Nation1.index,
            locationmode = 'country names',
            colorscale= 'Picnic',
            text=Nation1['Text'],
            marker_line_color='darkgray',
            marker_line_width=0.9,
            #z=Nation1['count'],
            z=Nation1['averageVal'],
            colorbar_tickprefix = '$',
            colorbar = {'title':'Avg Value<br>M US$'})
layout = dict(title="Number of Players" ,geo = {'showframe':True,'scope':'world','projection':{'type':'natural earth'}})
choromap = go.Figure(data = [nationData],layout = layout)
choromap


# The relationship between searches in the and market value

# In[ ]:


def tmin():
    return 3


# In[ ]:


sns.jointplot(x='market_value',
              y='page_views',
              kind='reg',
              data=data,
              marginal_kws={
                            'color':'red'}) 


# In[ ]:





# In[ ]:


data[(data['page_views']>7000) & (data['market_value']<30)]


# In[ ]:


sns.lmplot(x='market_value', y='page_views', hue='big_club',palette="RdBu",markers='*', data=data[(data['page_views']<3000) & (data['market_value']<30)]) 


# Seems while you are not playing for a big club, if you're market value is higher, you are more popular  

# In[ ]:


sns.clustermap(data.corr(),annot=True)


# In[ ]:


data.corr()['market_value']


# In[ ]:


data['fpl_sel']=data['fpl_sel'].apply(lambda x:float(x.split('%')[0]))
data['fpl_sel']


# In[ ]:


data.corr()['market_value'].sort_values()


# In[ ]:


plt.figure(figsize=(10,7))

sns.boxplot(x='position_cat',y='market_value',data=data)


# In[ ]:


X=data[['position_cat','age','fpl_sel','fpl_points','page_views','fpl_value']]
y=data['market_value']


# In[ ]:


# sns.pairplot(data[['big_club','fpl_points','page_views','position_cat','fpl_sel','market_value']])


# In[ ]:


sns.distplot(y)


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:



coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[ ]:





# In[ ]:


predictions = lm.predict( X_test)


# In[ ]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[ ]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


sns.distplot((y_test-predictions),bins=20);


# Make ANN Model

# In[ ]:


X=data[['position_cat','age','fpl_sel','fpl_points','page_views','fpl_value']].values
y=data['market_value'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler=MinMaxScaler()


# In[ ]:


X_train= scaler.fit_transform(X_train)


# In[ ]:


X_test = scaler.transform(X_test)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


# In[ ]:


X_train.shape


# In[ ]:


model = Sequential()
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# In[ ]:


model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),
         batch_size=128,epochs=300)


# In[ ]:


losses=pd.DataFrame(model.history.history)


# In[ ]:


losses.plot()


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score


# In[ ]:


pred = model.predict(X_test)


# In[ ]:


mean_squared_error(y_test,pred)**0.5


# In[ ]:


mean_absolute_error(y_test,pred)


# In[ ]:


data['market_value'].describe()


# In[ ]:


explained_variance_score(y_test,pred)


# In[ ]:


plt.figure(figsize=(15,9))
plt.scatter(y_test,pred)
plt.plot(y_test,y_test,'r')


# Let's take player as an example

# In[ ]:


player = data[['position_cat','age','fpl_sel','fpl_points','page_views','fpl_value']].iloc[0]


# In[ ]:


player.values


# In[ ]:


player = scaler.transform(player.values.reshape(-1,6))


# In[ ]:


float(model.predict(player))


# In[ ]:


float(data.head(1)['market_value'])


# In[ ]:


f"Relative Error: {round(float(abs((model.predict(player)-float(data.head(1)['market_value']))/float(data.head(1)['market_value']))*100),2)}%"

