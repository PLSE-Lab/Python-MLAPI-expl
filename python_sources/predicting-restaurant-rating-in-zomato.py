# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
train=pd.read_csv("../input/zomato.csv")
train.head()
train.dtypes
train.columns
train['listed_in(type)'].unique()
train=train.drop(['url','address','phone'],axis=1)
train=train.rename(columns={"approx_cost(for two people)":"cost","listed_in(type)":"type","listed_in(city)":"city"})
train.columns
train.isnull().sum()
train['rate'].unique()
train['rate']=train['rate'].astype('str')
def stripping(y):
    b=y.replace('/5','').strip()
    return b
train['rate']=train.rate.apply(stripping)
train['rate']=train.rate.apply(lambda x:x.replace('NEW',str(np.nan)))
train['rate']=train.rate.apply(lambda x:x.replace('-',str(np.nan)))
train['rate']=train.rate.astype('float')

train['rate']
train.dtypes
train.isnull().sum()
train.dropna(subset=['rate','cost'],inplace=True)
rest=train['rest_type'].unique()
resto={}
for i in range(0,len(rest)):
    resto[rest[i]]=i
train['rest_types']=train.rest_type.map(resto).astype('int')    
dishes=train['dish_liked'].unique()
dish={}
for i in range(0,len(dishes)):
    dish[dishes[i]]=i
train['dish_like']=train.dish_liked.map(dish).astype('int')
train['dish_like']
train['cost'].unique()
train['cost']=train.cost.astype('str')
def costcut(x):
    y=x.replace(',','')
    return(y)
train['cost']=train.cost.apply(costcut).astype('float')
train['cost'].isna().sum()
train['city']
cities=train['city'].unique()
city_liked={}
for i in range(0,len(cities)):
    city_liked[cities[i]]=i
train['location']=train.city.map(city_liked).astype('int')  
train['location']
train['online_order'].unique()
def order(x):
    if x=='Yes':
        return 1
    if x=='No':
        return 0
train['online_order']=train.online_order.apply(order).astype('int')   
train['book_table'].unique()
def book(x):
    if x=='Yes':
        return 1
    if x=='No':
        return 0
train['book_table']=train.book_table.apply(book).astype('int') 
train['reviews_list'].unique()
train['menu_item'].shape
import re
all_ratings=[]
for ratings in (train['reviews_list']):
    ratings=eval(ratings)
    for score, doc in ratings:
        if score:
            doc = doc.strip('RATED').strip()
            all_ratings.append(doc)
all_ratings
types=train['type'].unique()
type_resto={}
for i in range(0,len(types)):
    type_resto[types[i]]=i
train['type_restaurant']=train.type.map(type_resto)    
train['type_restaurant']    
x=train[['online_order','book_table','votes','location','cost','rest_types','dish_like','type_restaurant']]
y=train['rate']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
from sklearn.linear_model import LinearRegression
regr=LinearRegression()
regr.fit(x_train,y_train)
pred1=regr.predict(x_test)
pred1[2]
y_test.iloc[2]
from sklearn.metrics import mean_squared_error
mean_squared_error(pred1,y_test)
n_neighbours=5
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=6)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
knn.score(x_test,y_test)
pred[10]
y_test.iloc[10]
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,pred)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.