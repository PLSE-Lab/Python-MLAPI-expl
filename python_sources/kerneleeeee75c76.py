#!/usr/bin/env python
# coding: utf-8

# In[101]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[102]:


get_ipython().system('pip install regressors')


# In[103]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from regressors import stats
import statsmodels.formula.api as sm
from sklearn.preprocessing import PolynomialFeatures,FunctionTransformer 
from sklearn.linear_model import LogisticRegression

import statsmodels.formula.api as sm
import statsmodels.api as sma
from mlxtend.feature_selection import SequentialFeatureSelector as sfs # Forward selection
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score,cross_val_predict,LeaveOneOut,train_test_split


# In[104]:


dfTrain = pd.read_csv("../input/train.csv")
dfTrain.head()


# In[105]:


#Clean the Data

dfTrain['WeatherCat'] = dfTrain['Weather'].map({'Clear': 0, 'Cloudy': 1,'Light Rain': 2, ' Heavy Rain': 3})
dfTrain['SeasonCat'] = dfTrain['Season'].map({'Fall': 0, 'Spring': 1,'Summer': 2, 'Winter': 3})
dfTrain['TimeDiv'] = dfTrain['Time'].map({'00:00:00': 0,'01:00:00': 1, '02:00:00': 2, '03:00:00': 3,'04:00:00': 4,'05:00:00': 5,'06:00:00': 6,'07:00:00': 7,'08:00:00': 8,'09:00:00': 9,'10:00:00': 10,'11:00:00': 11,'12:00:00': 12,'13:00:00': 13,'14:00:00': 14,'15:00:00': 15,'16:00:00': 16,'17:00:00': 17,'18:00:00': 18,'19:00:00': 19,'20:00:00': 20,'21:00:00': 21,'22:00:00': 22,'23:00:00': 23,'24:00:00': 0})
dfTrain = dfTrain.dropna()


# In[106]:


dfTrain.info()


# In[107]:


inputDF = dfTrain[["IsHoliday","IsWorkingDay","WeatherCat","Temperature","WindSpeed","SeasonCat","AdoptedTemperature","Humidity"]]
outputDF = dfTrain[["Demand"]]

model = sfs(LinearRegression(),k_features=5,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')
model.fit(inputDF,outputDF)

model.k_feature_names_


# In[108]:


#Model 1 - Evaluation using train_test_split
#inputDF = dfTrain[['IsHoliday', 'WeatherCat', 'Temperature', 'SeasonCat', 'AdoptedTemperature']]
inputDF = dfTrain[['TimeDiv', 'Humidity', 'Temperature', 'AdoptedTemperature']]
outputDF = dfTrain[["Demand"]]

X_train, X_test, y_train, y_test = train_test_split(inputDF, outputDF, test_size=0, random_state=0) 

model = lm.LinearRegression()
results = model.fit(X_train,y_train)

print("R - Squared value:\n",stats.adj_r2_score(model, X_train, y_train)) 

print(model.intercept_, model.coef_)


# In[129]:


chol1 = sm.ols(formula="Demand ~ 1",data=dfTrain).fit()
chol2 = sm.ols(formula="Demand ~ TimeDiv",data=dfTrain).fit()
chol3 = sm.ols(formula="Demand ~ TimeDiv+IsHoliday",data=dfTrain).fit()
chol4 = sm.ols(formula="Demand ~ TimeDiv*IsHoliday",data=dfTrain).fit()
#inter = sm.ols(formula="Demand ~ IsHoliday*TimeDiv",data=dfTrain).fit()
print(chol4.summary())
#print(sma.stats.anova_lm(chol1,chol2,chol3,chol4))


# In[110]:


dfTest = pd.read_csv("../input/test.csv")
dfTest.head()
#model.predict

#Clean the Data

dfTest['WeatherCat'] = dfTest['Weather'].map({'Clear': 0, 'Cloudy': 1,'Light Rain': 2, ' Heavy Rain': 3})
dfTest['SeasonCat'] = dfTest['Season'].map({'Fall': 0, 'Spring': 1,'Summer': 2, 'Winter': 3})
dfTest['TimeDiv'] = dfTest['Time'].map({'00:00:00': 0,'01:00:00': 1, '02:00:00': 2, '03:00:00': 3,'04:00:00': 4,'05:00:00': 5,'06:00:00': 6,'07:00:00': 7,'08:00:00': 8,'09:00:00': 9,'10:00:00': 10,'11:00:00': 11,'12:00:00': 12,'13:00:00': 13,'14:00:00': 14,'15:00:00': 15,'16:00:00': 16,'17:00:00': 17,'18:00:00': 18,'19:00:00': 19,'20:00:00': 20,'21:00:00': 21,'22:00:00': 22,'23:00:00': 23,'24:00:00': 0})
# dfTrain = dfTrain.dropna()

#xtest = dfTest[['IsHoliday', 'WeatherCat', 'Temperature', 'SeasonCat', 'AdoptedTemperature']]
xtest = dfTest[['TimeDiv', 'Humidity', 'Temperature', 'AdoptedTemperature']]
dfTest["Demand"] = ""

dfTest["Demand"]=model.predict(xtest)

dfTest.head()


# In[111]:


dfTrain["DemandNew"]=""

dfTrain["DemandNew"]=model.predict(inputDF)

print("RMSE:\n", np.sqrt(metrics.mean_squared_error(dfTrain["Demand"], dfTrain["DemandNew"])))


# In[112]:


dfTest.Demand = dfTest.Demand.round()
dfTest.head()


# In[113]:


sample = pd.read_csv("../input/sample_submission.csv")
sample.head()


# In[114]:


sample["Demand"]=dfTest["Demand"]


# In[115]:


sample.head()


# In[116]:


#Submission:
submissionDF = pd.DataFrame({"Id": dfTest["Id"],"Demand":dfTest["Demand"]})
submissionDF.to_csv('sample_submission.csv',index=False)


# In[ ]:





# In[ ]:




