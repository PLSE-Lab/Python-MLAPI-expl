#!/usr/bin/env python
# coding: utf-8

# **Import package**

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# Load Data

# In[ ]:


data = pd.read_csv('../input/renfe.csv', index_col=0)
data.head()


# In[ ]:


data["price"].describe()


# **Data Analyze**

# What are the data types for various features?

# In[ ]:


data.columns.values


# In[ ]:


data.info()


# Categorical: origin  destination train_type train_class fare
# 
# numerical:price
# 
# ![](http://)should change these columns(insert_data start_date end_data ) into Datetime later.

# **Analyze by visualizing data**
# 
# ****1. Dealwith the Nan value ****

# In[ ]:


data.shape


# Calculate the missed percentage

# In[ ]:


data.isnull().sum()/data.shape[0] * 100


# Fill the missed value in the "price" column with mean of the ticket price.
# 

# In[ ]:


data['price'].fillna(data['price'].mean(),inplace=True)


# Fill the missed value in the "train_class" and "fare" column with the value has the most frequency.

# In[ ]:


data['train_class'].value_counts()


# The "Turista" has every high frequency in the train_class column.

# In[ ]:


data['train_class'].fillna("Turista",inplace=True)


# In[ ]:


data['fare'].value_counts()


# The "Promo" has every high frequency in the fare column.

# In[ ]:


data['fare'].fillna("Promo",inplace=True)


# In[ ]:


data.isnull().sum()


# ****2. Analyze the price ****

# In[ ]:


data['price'].describe()


# Distribution of the ticket prices

# In[ ]:


sns.distplot(data['price'])


# ****3. train_type and price ****

# In[ ]:



# train_type
var = 'train_type'
data_train_type = pd.concat([data['price'], data[var]], axis=1)
plt.subplots(figsize=(15,6))
sns.boxplot(x=var, y="price",data=data_train_type)


# "AVE" and "AVE-TGV" are higher compared with other train types.
# 

# ****4. train_class and price ****

# In[ ]:


# train_class
var = 'train_class'
data_train_class = pd.concat([data['price'], data[var]], axis=1)
plt.subplots(figsize=(15,6))
sns.boxplot(x=var, y="price",data=data_train_class)


# "Cama G. Clase" is the train class with the highest ticket price. "Turista con enlace" is the train class with the lowest price.
# 

# ****5. origin and price ****

# In[ ]:


# origin
var = 'origin'
data_origin = pd.concat([data['price'], data[var]], axis=1)
plt.subplots(figsize=(15,6))
sns.boxplot(x=var, y="price",data=data_origin)


# In[ ]:





# ****6. destination and price ****

# In[ ]:


#  destination
var = 'destination'
data_dest = pd.concat([data['price'], data[var]], axis=1)
plt.subplots(figsize=(15,6))
sns.boxplot(x=var, y="price",data=data_dest)


# ****7. deal with the datetime ****

# The "insert_date" column just store the time when the ticket is sold,it's no releated with the price,so just delet this column. 

# In[ ]:


data.drop('insert_date',axis=1,inplace=True)


# ![](http://)The "start_date" and "end_date" just the departure time in origin and the Arrival  time in destination.So we just use the timedelta.

# In[ ]:


#strp time
#data['start_date'] = pd.to_datetime(data['start_date'])
#data['end_date'] = pd.to_datetime(data['end_date'])


# Calculate the  time delta between the place of origin and destination

# In[ ]:


def dataInterval(start,end):
    start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    delta = end - start
    return delta.seconds/3600


# In[ ]:


data['data_interval'] = data.apply(lambda x:dataInterval(x['start_date'],x['end_date']),axis = 1)


# In[ ]:


data.drop(['start_date','end_date'],axis=1,inplace=True)


# ****7. Completing a categorical feature****

# origin  destination train_type train_class fare

# In[ ]:


data.head()


# In[ ]:


f_names = ["origin","destination","train_type","train_class","fare"]
for x in f_names:
    label = preprocessing.LabelEncoder()
    data[x] = label.fit_transform(data[x])


# In[ ]:


data.head()


# ****Feature Engineering****

# In[ ]:


corrmat = data.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)


# ****Model, predict and solve****

# split the training and testing data

# In[ ]:


# Get the date 
cols = ['origin','destination', 'train_type', 'train_class', 'fare', 'data_interval']
x = data[cols].values
y = data['price'].values
x_scaled = preprocessing.StandardScaler().fit_transform(x)
y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))
X_train,X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.1, random_state=10)


# 
# Use Support Vector Machines , RandomForestRegressor and BayesianRidge mode to run  in the workflow. And compare the result with the each mode

# In[ ]:


clfs ={
    "LGBMRegressor":LGBMRegressor(),
    "RandomForestRegressir":RandomForestRegressor(n_estimators=200),
    'BayesianRidge':linear_model.BayesianRidge()
}
for clf in clfs:
    try:
        clfs[clf].fit(X_train, y_train)
        #y_pred = clfs[clf].predict(X_test)
        #print(clf + " cost:" + str(np.sum(y_pred-y_test)/len(y_pred)) )
        print(clf+" score:"+ str(clfs[clf].score(X_test,y_test)))
    except Exception as e:
        print(clf + " Error:")
        print(str(e))


# compare with the score of each mode,the RandomForestRegressir has high score(0.8293834131266755)
