#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:


trainData=pd.read_csv("../input/restaurant-revenue-prediction/train.csv.zip")


# In[ ]:


df=pd.read_csv("../input/restaurant-revenue-prediction/test.csv.zip")


# In[ ]:


column_names = ["id","Prediction"]
sample=pd.DataFrame(columns=column_names)
sample["id"]=df["Id"]


# In[ ]:


trainData


# # first modifying The OpenDate column 

# In[ ]:


trainData['Open Date'] = pd.to_datetime(trainData['Open Date'], format='%m/%d/%Y')
df['Open Date'] = pd.to_datetime(df['Open Date'], format='%m/%d/%Y')


# In[ ]:


trainData['OpenDays']=""
df['OpenDays']=""


# In[ ]:


dateLastTrain = pd.DataFrame({'Date':np.repeat(['01/01/2015'],[len(trainData)]) })
dateLastTest = pd.DataFrame({'Date':np.repeat(['01/01/2015'],[len(df)]) })


# In[ ]:


dateLastTrain['Date'] = pd.to_datetime(dateLastTrain['Date'], format='%m/%d/%Y') 
dateLastTest['Date'] = pd.to_datetime(dateLastTest['Date'], format='%m/%d/%Y') 


# In[ ]:


df.head()


# In[ ]:


trainData['OpenDays'] = dateLastTrain['Date'] - trainData['Open Date']
df['OpenDays'] = dateLastTest['Date'] - df['Open Date']


# In[ ]:


trainData['OpenDays'] = trainData['OpenDays'].astype('timedelta64[D]').astype(int)
df['OpenDays'] = df['OpenDays'].astype('timedelta64[D]').astype(int)


# # now looking at the City group column

# In[ ]:


cityPerc = trainData[["City Group", "revenue"]].groupby(['City Group'],as_index=False).mean()


# In[ ]:


sns.barplot(x='City Group', y='revenue', data=cityPerc)


# In[ ]:


citygroupDummy = pd.get_dummies(trainData['City Group'])
citygroupD = pd.get_dummies(df['City Group'])
citygroupDummy


# In[ ]:


trainData = trainData.join(citygroupDummy)
df = df.join(citygroupD)


# In[ ]:


trainData = trainData.drop('City Group', axis=1)
df = df.drop('City Group', axis=1)


# In[ ]:


trainData = trainData.drop('Open Date', axis=1)
df = df.drop('Open Date', axis=1)


# #we have to label encode the city column but we can't do directly
# #so we create a new column of city mean revenue
# 

# In[ ]:


trainData[["City","revenue"]].groupby(["City"]).mean().plot(kind="bar")


# In[ ]:


mean_revenue_per_city = trainData[['City', 'revenue']].groupby('City', as_index=False).mean()
mean_revenue_per_city['revenue'] = mean_revenue_per_city['revenue'].apply(lambda x: int(x/1e6)) 


# In[ ]:


mean_revenue_per_city


# In[ ]:


mean_dict = dict(zip(mean_revenue_per_city.City, mean_revenue_per_city.revenue))


# In[ ]:


mean_dict


# In[ ]:


trainData.replace({"City":mean_dict}, inplace=True)


# In[ ]:


trainData.City.unique()


# In[ ]:


trainData.City.mean()


# In[ ]:


df.City.unique()


# In[ ]:


df.replace({"City":mean_dict}, inplace=True)


# In[ ]:


#adding 4 as it was the mean in traindata column


# In[ ]:


df['City'] = df['City'].apply(lambda x: 4 if isinstance(x,str) else x)


# In[ ]:





# #now looking the Type Column 

# In[ ]:


trainData.Type.unique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


lr = LabelEncoder()
lr2=LabelEncoder()


# In[ ]:


trainData["Type"]=lr.fit_transform(trainData["Type"])
df["Type"]=lr2.fit_transform(df["Type"])


# In[ ]:


X = trainData.drop(['revenue', 'Id'],axis=1)


# In[ ]:


X


# In[ ]:


df = df.drop(['Id'],axis=1)


# In[ ]:


Y=trainData["revenue"]


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# In[ ]:


from math import sqrt


# In[ ]:


cv = KFold(n_splits=10, shuffle=True, random_state=108)
model = LGBMRegressor(n_estimators=200, learning_rate=0.01, subsample=0.7, colsample_bytree=0.8)

scores = []
for train_idx, test_idx in cv.split(X):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[test_idx]
    y_train = Y.iloc[train_idx]
    y_val = Y.iloc[test_idx]
    
    model.fit(X_train,y_train)
    preds = model.predict(X_val)
    
    rmse = sqrt(mean_squared_error(y_val, preds))
    print(rmse)
    scores.append(rmse)

print("\nMean score %d"%np.mean(scores))


# In[ ]:


predictions = model.predict(df)
predictions


# In[ ]:


sns.distplot(predictions, bins=20)


# In[ ]:





# #checking the weightage of all the column

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance


# In[ ]:


X = trainData.drop(['revenue', 'Id'],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

from xgboost import XGBRegressor
xgb=XGBRegressor()
xgb.fit(X_train,Y_train)


# In[ ]:


perm = PermutationImportance(xgb, random_state=1).fit(X_train,Y_train)
eli5.show_weights(perm, feature_names = X_train.columns.to_list())

#adding new columns accordingly
# In[ ]:


trainData['P29_to_City_mean'] = trainData.groupby('City')['P29'].transform('mean')
trainData['P17_to_City_mean'] = trainData.groupby('City')['P17'].transform('mean')
trainData['P28_to_City_mean'] = trainData.groupby('City')['P28'].transform('mean')
trainData['P1_to_City_mean'] = trainData.groupby('City')['P1'].transform('mean')
trainData['P27_to_City_mean'] = trainData.groupby('City')['P27'].transform('mean')
trainData['P20_to_City_mean'] = trainData.groupby('City')['P20'].transform('mean')


# In[ ]:


df['P29_to_City_mean'] = df.groupby('City')['P29'].transform('mean')
df['P17_to_City_mean'] = df.groupby('City')['P17'].transform('mean')
df['P28_to_City_mean'] = df.groupby('City')['P28'].transform('mean')
df['P1_to_City_mean'] = df.groupby('City')['P1'].transform('mean')
df['P27_to_City_mean'] = df.groupby('City')['P27'].transform('mean')
df['P20_to_City_mean'] = df.groupby('City')['P20'].transform('mean')


# In[ ]:


X = trainData.drop(['revenue', 'Id'],axis=1)


# In[ ]:


X


# #Applying model once again so that we can get our new Model with more accuracy

# In[ ]:


cv = KFold(n_splits=10, shuffle=True, random_state=108)
model = LGBMRegressor(n_estimators=200, learning_rate=0.01, subsample=0.7, colsample_bytree=0.8)

scores = []
for train_idx, test_idx in cv.split(X):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[test_idx]
    y_train = Y.iloc[train_idx]
    y_val = Y.iloc[test_idx]
    
    model.fit(X_train,y_train)
    preds = model.predict(X_val)
    
    rmse = sqrt(mean_squared_error(y_val, preds))
    print(rmse)
    scores.append(rmse)

print("\nMean score %d"%np.mean(scores))


# In[ ]:


predictions = model.predict(df)
sample['Prediction'] = predictions


# In[ ]:


sample


# In[ ]:


sample['Prediction']=sample['Prediction'].apply(lambda x: round((float(x/1e6)*1000000),1))


# In[ ]:


sample['Prediction']


# In[ ]:


sample.to_csv('submission.csv', index=False)

