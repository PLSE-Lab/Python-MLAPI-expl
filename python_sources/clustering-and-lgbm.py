# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train=pd.read_csv('/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv')
train.head()
train.columns
result=[]
for i in train.columns:
    result.append((train[i].nunique(),train[i].isnull().sum(),train[i].dtypes))
results=pd.DataFrame(result,columns=['unique','missing','data-type']) 
results
train=train.dropna()
train.isnull().sum()
train.describe()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,15))
sns.countplot('Weekend',data=train)
plt.title('Purchase of Weekends')
plt.xlabel('Weekend or not',fontsize=15)
plt.ylabel('Count')
train['VisitorType'].value_counts()
train['Browser'].unique()
plt.figure(figsize=(15,15))
fig1,ax1=plt.subplots(1,1)
ax1.pie(train['VisitorType'].value_counts(),autopct='%1.1f%%',labels=['Returning Visitor','New Visitor','Others'])
plt.title('Types of Visitors')
train['TrafficType'].value_counts()
plt.figure(figsize=(15,15))
sns.distplot(train['TrafficType'],color='red')
plt.ylabel('Count')
plt.title('Distribution of Traffic Type')
plt.figure(figsize=(15,15))
sns.distplot(train['Region'],color='g')
plt.ylabel('Count')
plt.title('Distribution of Region Type')
train['Revenue'].value_counts()
#Graph of Informational duration vs Revenue
plt.figure(figsize=(15,15))
sns.boxenplot(x='Informational_Duration',y='Revenue',data=train,palette='rainbow')
plt.xlabel('Information Duration')
plt.ylabel('Revenue')
plt.title('Information duration vs Revenue')
#Graph of Administrative Duration vs Revenue
plt.figure(figsize=(15,15))
sns.boxenplot(train['Revenue'],train['Administrative_Duration'],palette='pastel')
plt.xlabel('Administrative_Duration')
plt.ylabel('Revenue')
plt.title('Administrative duration vs Revenue')
#Graph of Product Duration vs Revenue
plt.figure(figsize=(15,15))
sns.boxenplot(train['Revenue'],train['ProductRelated_Duration'],palette='pastel')
plt.xlabel('Product_Duration')
plt.ylabel('Revenue')
plt.title('Product duration vs Revenue')
#exit rates vs revenue
plt.figure(figsize=(15,15))
sns.boxenplot(train['Revenue'],train['ExitRates'],palette='dark')
plt.xlabel('exit rate')
plt.ylabel('Revenue')
plt.title('exit rate vs Revenue')
#bounce rates vs revenues
plt.figure(figsize=(15,15))
sns.boxenplot(train['Revenue'],train['BounceRates'],palette='dark')
plt.xlabel('Bounce Rates')
plt.ylabel('Revenue')
plt.title('Bounce Rates vs Revenue')
#weekends vs revenues
sns.barplot(train['Weekend'],train['Revenue'])
#Clustering Analysis
#administrative_duration vs bounce_rates
X=train.iloc[:,[1,6]].values
X.shape
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    km = KMeans(n_clusters = i,
              init = 'k-means++',
              max_iter = 300,
              n_init = 10,
              random_state = 0,
              algorithm = 'elkan',
              tol = 0.001)
    km.fit(X)
    labels = km.labels_
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('No of clusters')
plt.ylabel('wss')
plt.show()
km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(X)
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'pink', label = 'Un-interested Customers')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=100,c='green',label='Target Customers')
plt.title('Informative Duration vs Bounce Rates', fontsize = 20)
plt.grid()
plt.xlabel('Informative Duration')
plt.ylabel('Bounce Rates')
plt.legend()
plt.show()
#Information Duration vs Bounce Rates
X=train.iloc[:,[1,7]].values
X.shape
wcss=[]
for i in range(1,11):
    km = KMeans(n_clusters = i,
              init = 'k-means++',
              max_iter = 300,
              n_init = 10,
              random_state = 0,
              algorithm = 'elkan',
              tol = 0.001)
    km.fit(X)
    labels = km.labels_
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('No of clusters')
plt.ylabel('wss')
plt.show()
km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(X)
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'pink', label = 'Un-interested Customers')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=100,c='green',label='Target Customers')
plt.title('Administrative Duration vs Exit Rates', fontsize = 20)
plt.grid()
plt.xlabel('Administrative Duration')
plt.ylabel('Exit Rate')
plt.legend()
plt.show()
#Administrative Duration vs Region 
X=train.iloc[:,[1,13]].values
X.shape
wcss=[]
for i in range(1,11):
    km = KMeans(n_clusters = i,
              init = 'k-means++',
              max_iter = 300,
              n_init = 10,
              random_state = 0,
              algorithm = 'elkan',
              tol = 0.001)
    km.fit(X)
    labels = km.labels_
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('No of clusters')
plt.ylabel('wss')
plt.show()
km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(X)
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'pink', label = 'Un-interested Customers')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=100,c='green',label='Target Customers')
plt.title('Administrative Duration vs Exit Rates', fontsize = 20)
plt.grid()
plt.xlabel('Administrative Duration')
plt.ylabel('Region')
plt.legend()
plt.show()
def revenue(x):
    if x==True:
        return 1
    else:
        return 0
train['Revenue']=train['Revenue'].apply(revenue)    
train=pd.get_dummies(train,columns=['Month','VisitorType','Weekend'])
train.columns
x=train.drop('Revenue',axis=1)
y=train['Revenue']
#train.dtypes
#train.corr()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)
import lightgbm as lgb
d_train = lgb.Dataset(x_train,y_train)
params = {}
params['learning_rate'] = 0.005
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 100
params['min_data'] = 50
params['max_depth'] = 25

model=lgb.train(params,d_train,num_boost_round=2000)
model
pred=model.predict(x_test)
pred[0]
x_test.shape[0]
for i in range(0,3079):
    if pred[i]>=.5:       # setting threshold to .5
       pred[i]=1
    else:  
       pred[i]=0
from sklearn.metrics import accuracy_score
print('Accuracy Score')
accuracy_score(pred,y_test)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
os.listdir('../input')
# Any results you write to the current directory are saved as output.