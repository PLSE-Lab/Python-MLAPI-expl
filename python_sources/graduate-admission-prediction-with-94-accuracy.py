#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")


# In[ ]:


df.head()


# In[ ]:


df.corr()


# In[ ]:


df.isnull().sum()


# In[ ]:


# Performing Normalization for GRE Score and TOEFL Score columns
from sklearn import preprocessing
minmax_scaler = preprocessing.MinMaxScaler()
minmax_scaler_fit=minmax_scaler.fit(df[['GRE Score', 'TOEFL Score']])
NormalizedGREScoreAndTOEFLScore = minmax_scaler_fit.transform(df[['GRE Score', 'TOEFL Score']])


# Creating a separate Data Frame just to store new standardized columns
NormalizedGREScoreAndTOEFLScoreData=pd.DataFrame(NormalizedGREScoreAndTOEFLScore,columns=['GRE Score', 'TOEFL Score'])
NormalizedGREScoreAndTOEFLScoreData.head()


# In[ ]:


df['GRE Score']=NormalizedGREScoreAndTOEFLScoreData['GRE Score']
df['TOEFL Score']=NormalizedGREScoreAndTOEFLScoreData['TOEFL Score']
df.head()


# In[ ]:


PredictorColumns=list(df.columns)
PredictorColumns.remove('Serial No.')
PredictorColumns.remove('Chance of Admit ')


# In[ ]:


X=df[PredictorColumns].values
y=df['Chance of Admit '].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)


# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=100,criterion='mse')
#print(clf)
RF=clf.fit(X_train,y_train)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Plotting the importance of variables
feature_importances = pd.Series(RF.feature_importances_, index=PredictorColumns)

# Plotting the feature importance for Top 10 most important columns
feature_importances.nlargest(10).plot(kind='barh')


# In[ ]:


# Predicting the Rating values for testing data
PredAdmit = RF.predict(X_test)

# Creating a DataFrame of Zomato Testing data
AdmitData=pd.DataFrame(X_test, columns=PredictorColumns)
AdmitData['ChancesOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PredAdmit
AdmitData.head()


# In[ ]:


# Calculating the Absolute Percentage Error committed in each prediction
AdmitData['APE']=100 * (abs(AdmitData['ChancesOfAdmit'] - AdmitData['PredictedChancesOfAdmit'])/AdmitData['ChancesOfAdmit'])


# In[ ]:


# Final accuracy of the model
print('Mean Absolute Percent Error(MAPE): ',round(np.mean(AdmitData['APE'])), '%')
print('Average Accuracy of the model: ',100 - round(np.mean(AdmitData['APE'])), '%')

