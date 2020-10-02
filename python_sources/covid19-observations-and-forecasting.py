#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Simple EDA

# In[ ]:


covid19_train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')


# In[ ]:


covid19_train_df.describe()


# In[ ]:


covid19_train_df.head()


# In[ ]:


covid19_train_df.sample()


# In[ ]:


covid19_train_df.sample(n=10)


# ### Simple Data Visualization

# In[ ]:


import seaborn as sns


# In[ ]:


sns.countplot(y="Country_Region", data=covid19_train_df,order=covid19_train_df["Country_Region"].value_counts(ascending=False).iloc[:10].index)


# In[ ]:


sns.countplot(y="Province_State", data=covid19_train_df,order=covid19_train_df["Province_State"].value_counts(ascending=False).iloc[:20].index)


# In[ ]:


sns.regplot(x=covid19_train_df["ConfirmedCases"], y=covid19_train_df["Fatalities"], fit_reg=False)


# In[ ]:


sns.regplot(x=covid19_train_df["ConfirmedCases"], y=covid19_train_df["Fatalities"])


# In[ ]:


sns.jointplot(x=covid19_train_df["ConfirmedCases"], y=covid19_train_df["Fatalities"], kind='scatter')


# In[ ]:


sns.set(style="darkgrid")
sns.lineplot(x="Date",y="ConfirmedCases",hue="Country_Region", 
             data=covid19_train_df)


# In[ ]:


sns.set(style="darkgrid")
sns.lineplot(x="Date",y="Fatalities", hue="Country_Region",
             data=covid19_train_df)


# In[ ]:


sns.residplot(x=covid19_train_df["ConfirmedCases"], y=covid19_train_df["Fatalities"], lowess=True, color="g")


# In[ ]:


#data_4.to_csv("submission.csv", index=False)


# ### XGB Boost 

# In[ ]:


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# In[ ]:


trainData = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
testData = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

print(trainData.shape)
print(testData.shape)


# In[ ]:


convertDict = {'Province_State': str,'Country_Region':str}
trainData = trainData.astype(convertDict)
testData = testData.astype(convertDict)


# In[ ]:




trainData['Date'] = pd.to_datetime(trainData['Date'], infer_datetime_format=True)

testData['Date'] = pd.to_datetime(testData['Date'], infer_datetime_format=True)


# In[ ]:



trainData.loc[:, 'Date'] = trainData.Date.dt.strftime('%m%d')
trainData.loc[:, 'Date'] = trainData['Date'].astype(int)


testData.loc[:, 'Date'] = testData.Date.dt.strftime('%m%d')
testData.loc[:, 'Date'] = testData['Date'].astype(int)


# In[ ]:


trainData['Country_Region'] = np.where(trainData['Province_State'] == 'nan',
                                       trainData['Country_Region'],trainData['Province_State']+
                                       trainData['Country_Region'])
testData['Country_Region'] = np.where(testData['Province_State'] == 'nan',
                                      testData['Country_Region'],testData['Province_State']+
                                      testData['Country_Region'])


# In[ ]:


trainData = trainData.drop(columns=['Province_State'])
testData = testData.drop(columns=['Province_State'])

print(trainData.head(),testData.head())
print(trainData.shape,testData.shape)


# In[ ]:


#list of categorical variables
categoryObject = (trainData.dtypes == 'object')
objectData = list(categoryObject[categoryObject].index)


# In[ ]:


objectData


# In[ ]:


labelEncoder = LabelEncoder()
trainData['Country_Region'] = labelEncoder.fit_transform(trainData['Country_Region'])
testData['Country_Region'] = labelEncoder.transform(testData['Country_Region'])


# In[ ]:


trainData.head()


# In[ ]:


testData.head()


# In[ ]:


testForecastId = testData.ForecastId


# In[ ]:


trainData.drop(['Id'], axis=1, inplace=True)
testData.drop('ForecastId', axis=1, inplace=True)


# In[ ]:


trainData.head(), trainData.shape


# In[ ]:


testData.head(), testData.shape


# In[ ]:


#pip install pandas-profiling


# In[ ]:


import pandas_profiling


# #### train data EDA using pandas profiling

# In[ ]:


trainData.profile_report()


# #### test data profile report

# In[ ]:


testData.profile_report()


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


X_train = trainData[['Country_Region','Date']]
y_train = trainData[['ConfirmedCases', 'Fatalities']]


# In[ ]:


x_train = X_train.iloc[:,:].values
x_test = testData.iloc[:,:].values


# In[ ]:


model = MultiOutputRegressor(XGBRegressor(n_estimators=1500, max_depth=20, random_state=0))
model.fit(x_train, y_train)
predict = MultiOutputRegressor(model.predict(x_test))


# In[ ]:


submissionData = pd.DataFrame()
submissionData['ForecastId'] = testForecastId
submissionData['ConfirmedCases'] = np.round(predict.estimator[:,0],2)
submissionData['Fatalities'] = np.round(predict.estimator[:,1],2)

submissionData.to_csv('submission.csv', index=False)


# In[ ]:


submissionData.sample(5), submissionData.shape


# In[ ]:





# In[ ]:





# In[ ]:




