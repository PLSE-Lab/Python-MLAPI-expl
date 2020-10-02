#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/Melbourne_housing_FULL.csv')
train.head()


# In[ ]:


train=train.dropna()
train.head()


# In[ ]:


plt.subplots(figsize=(12,9))
sns.distplot(train['Price'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['Price'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(train['Price'], plot=plt)
plt.show()


# In[ ]:


train['Price'] = np.log1p(train['Price'])

#Check again for more normal distribution

plt.subplots(figsize=(12,9))
sns.distplot(train['Price'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['Price'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(train['Price'], plot=plt)
plt.show()


# In[ ]:


train.columns[train.isnull().any()]


# In[ ]:


train.isnull().sum()


# In[ ]:


train.dtypes


# In[ ]:


corr = train.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)


# In[ ]:


top_feature = corr.index[abs(corr['Price']>0.5)]
plt.subplots(figsize=(12, 8))
top_corr = train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


# In[ ]:


train.Bedroom2.unique()


# In[ ]:


print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(['Price'], ascending=False, inplace=True)
corr.Price


# In[ ]:


plt.figure(figsize=(10, 5))
sns.heatmap(train.isnull())


# In[ ]:


train.columns


# In[ ]:


cols=('Suburb', 'Address', 'Type', 'Method', 'SellerG',
       'Date', 'Regionname', 'Propertycount')


# In[ ]:


from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))


# In[ ]:


train.shape


# In[ ]:




#Take targate variable into y
y = train['Price']


# In[ ]:


train.head()


# In[ ]:


del train['CouncilArea']


# In[ ]:


del train['Price']
train.head()


# In[ ]:


X = train.values
y = y.values
print(y)


# In[ ]:


# Split data into train and test formate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# In[ ]:


#linear Regression
#Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()


# In[ ]:


#Fit the model
model.fit(X_train, y_train)


# In[ ]:


#Prediction
print("Predict value " + str(model.predict([X_test[142]])))
print("Real value " + str(y_test[142]))
y_predict = model.predict(X_test)
Actual_Price=y_test


# In[ ]:


out = pd.DataFrame({'Actual_Price': Actual_Price, 'predict_Price': y_predict,'Diff' :(Actual_Price-y_predict)})


# In[ ]:


out[['Actual_Price','predict_Price','Diff']].head(10)


# In[ ]:


print("Accuracy --> ", model.score(X_test, y_test)*100)


# In[ ]:


#Train the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)


# In[ ]:


#Fit
model.fit(X_train, y_train)


# In[ ]:


#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)


# In[ ]:


#Train the model
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)


# In[ ]:


#Fit
GBR.fit(X_train, y_train)


# In[ ]:


print("Accuracy --> ", GBR.score(X_test, y_test)*100)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
dataset = MinMaxScaler().fit_transform(X)
X_trainn, X_testt, y_trainn, y_testt = train_test_split(dataset, y, test_size=0.3, random_state=40)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


reg2 = DecisionTreeRegressor()
reg3 = ExtraTreesRegressor()
reg4 = XGBRegressor()
reg5 = SVR()


reg2.fit( X_trainn,y_trainn )
reg3.fit( X_trainn,y_trainn )
reg4.fit( X_trainn,y_trainn )
reg5.fit( X_trainn,y_trainn )

label2 = reg2.predict( X_testt )
label3 = reg3.predict( X_testt )
label4 = reg4.predict( X_testt )
label5 = reg5.predict( X_testt )


# In[ ]:


# compare the loss of different models

from sklearn.metrics import mean_squared_error
print(label2)
print( 'the loss of DecisionTreeRegressor is ',mean_squared_error(y_testt,label2) )
print( 'the loss of ExtraTreesRegressor is ',mean_squared_error(y_testt,label3) )
print( 'the loss of XGBRegressor is ',mean_squared_error(y_testt,label4) )
print( 'the loss of SVR is ',mean_squared_error(y_testt,label5) )

print( '==='*10 )


# In[ ]:


# to compare the r^2 value of different regression models
# to chech the percentage of explained samples

from sklearn.metrics import r2_score
print( 'the r2 of DecisionTreeRegressor is ',r2_score(y_testt,label2) )
print( 'the r2 of ExtraTreesRegressor is ',r2_score(y_testt,label3) )
print( 'the r2 of XGBRegressor is ',r2_score(y_testt,label4) )
print( 'the r2 of SVR is ',r2_score(y_testt,label5) )

print( '++'*10 )


# In[ ]:


print( 'aparently, ExtraTreeRegressor performs the best in all the linear models with the loss presenting the least' )
print('---'*10)


# In[ ]:


#Score/Accuracy
print("Accuracy --> ", reg2.score(X_testt, y_testt)*100)
print("Accuracy --> ", reg3.score(X_testt, y_testt)*100)
print("Accuracy --> ", reg4.score(X_testt, y_testt)*100)
print("Accuracy --> ", reg5.score(X_testt, y_testt)*100)

