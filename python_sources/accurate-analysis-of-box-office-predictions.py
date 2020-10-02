#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from collections import OrderedDict

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


#loading the data
tr=pd.read_csv('../input/train.csv')


# In[ ]:


test=pd.read_csv('../input/train.csv')


# here the correlation is high with revenue by the budget that is 0.75

# In[ ]:


# the correlation between numerical variables
corr = tr.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


# In[ ]:


tr.head()


#                       **#visualizations****

# In[ ]:


tr['title'].value_counts()[:50].plot(kind='bar',figsize=(60,20),fontsize=10)


# In[ ]:


tr.plot.scatter('budget','revenue')


# In[ ]:


tr_mean = tr.groupby('title').mean().sort_values(by='runtime', ascending=False)[['runtime']]


# In[ ]:


tr_mean.plot(kind='bar',figsize=(80,20),color='green').set_title('runtime for title')


# In[ ]:


tr['original_language'].value_counts().plot(kind='bar',figsize=(60,10),fontsize=20).set_title('plotting original languages count')


# In[ ]:


tr[['title','revenue']].groupby(['title']).sum().plot()


# In[ ]:


#visualizing original title vs revenue
trg=tr[:20]
trg[['revenue','original_title']].groupby(['original_title']).sum().plot(kind='bar',figsize=(50,10), fontsize=40,color='blue').set_title('original title vs revenue for first 20 data',fontsize=40)


# In[ ]:


#visualizing original title vs revenue least 50 data
trg=tr[-50:]
trg[['revenue','original_title']].groupby(['original_title']).sum().plot(kind='bar',figsize=(50,10), fontsize=40,color='blue').set_title('original title vs revenue for least 50 data',fontsize=40)


# In[ ]:


#visualizing original title vs budget
trg=tr[:20]
trg[['budget','original_title']].groupby(['original_title']).sum().plot(kind='bar',figsize=(50,10), fontsize=40,color='pink').set_title('original title vs budget for first 20 data',fontsize=40)


# In[ ]:


#visualizing original title vs budget least 50
trg=tr[-50:]
trg[['budget','original_title']].groupby(['original_title']).sum().plot(kind='bar',figsize=(50,10), fontsize=40,color='pink').set_title('original title vs budget least 50 data',fontsize=40)


# In[ ]:


#visualizing original title vs budget least 50
trg=tr[-50:]
trg[['popularity','original_title']].groupby(['original_title']).sum().plot(kind='hist',figsize=(50,10), fontsize=40,color='pink').set_title('original title vs budget least 50 data',fontsize=40)


# In[ ]:


tr.head()


# In[ ]:


df=tr[:10]
ta=df.sort_values('revenue',ascending=False)
tit = df['title']
rbr = ta['revenue']
colors  = ("red", "green", "orange", "cyan", "brown", 
"grey","blue","indigo", "beige", "yellow")
plt.pie(rbr, labels=tit, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Top 10 data high revenue titles',fontsize=40)
plt.show()


# In[ ]:


df=tr[:10]
ta=df.sort_values('revenue',ascending=False)
tit = df['title']
rbr = ta['runtime']
colors  = ("red", "green", "orange", "cyan", "brown", 
"grey","blue","indigo", "beige", "yellow")
plt.pie(rbr, labels=tit, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Top 10 data more runtime titles',fontsize=40)
plt.show()


# In[ ]:


df=tr[:10]
ta=df.sort_values('budget',ascending=False)
tit = df['title']
rbr = ta['budget']
colors  = ("red", "green", "orange", "cyan", "brown", 
"grey","blue","indigo", "beige", "yellow")
plt.pie(rbr, labels=tit, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Top 10 high budget titles',fontsize=40)
plt.show()


# **#performing predictions**

# In[ ]:


tr.info()


# In[ ]:


tr.dtypes


# In[ ]:


tr.head()


# In[ ]:


tr.shape


# In[ ]:


# missing values in the datasaet
tr.isnull().sum()


# In[ ]:


tr.fillna(0,inplace=True)


# splitting the data into train and test data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


y=tr['revenue']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(tr, y, test_size=0.2,random_state=0)


# In[ ]:


x_train=tr.drop('revenue',axis=1)


# In[ ]:


x_test=tr.drop('revenue',axis=1)


# In[ ]:


y_train=tr['revenue']


# In[ ]:


y_test=tr['revenue']


# In[ ]:


# checking the shape of X_train, y_train, X_val and y_val
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[ ]:


#performing linear regression as the target variable is a continuous data i.e.,revenue


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


x_train=pd.get_dummies(x_train)


# In[ ]:


x_test=pd.get_dummies(x_test)


# In[ ]:


#filling NaN values


# In[ ]:


x_train.fillna(0,inplace=True)


# In[ ]:


x_test.fillna(0,inplace=True)


# In[ ]:


# applying dummies on the train dataset
tr=pd.get_dummies(tr)


# # performing linear regression model

# In[ ]:


lg=LinearRegression()


# In[ ]:


# fitting the model on X_train and y_train
lg.fit(x_train,y_train)


# In[ ]:


#post processing predicting values


# In[ ]:


# making prediction on validation set
pred=lg.predict(x_test)


# In[ ]:


pred


# In[ ]:


sol=pd.DataFrame()


# In[ ]:


sol['predicted values']=pred


# sol.to_csv('../input/sample_submission.csv', header=True, index=False)

# **saving predicted data in**** sol.csv**

# In[ ]:


pred1=lg.predict(x_train)


# In[ ]:


pred1


# # evaluating performance

# In[ ]:


lg.score(x_train,y_train)


# In[ ]:


lg.score(x_test,y_test)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


mean_squared_error(y_test,pred,multioutput='raw_values')


# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


mean_absolute_error(y_train,pred1,multioutput='raw_values')


# # R2 value

# In[ ]:


# defining a function which will return the rmsle score
def rmsle(y, y_):
    y = np.exp(y),   # taking the exponential as we took the log of target variable
    y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


# In[ ]:


rmsle(y_test,pred)


# In[ ]:


rmsle(y_train,pred1)


# #performing decision tree classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# defining the decision tree model with depth of 4, you can tune it further to improve the accuracy score
clf = DecisionTreeClassifier(max_depth=4, random_state=0)


# In[ ]:


# fitting the decision tree model
clf.fit(x_train,y_train)


# In[ ]:


# making prediction on the validation set
predict = clf.predict(x_test)


# In[ ]:


predict

