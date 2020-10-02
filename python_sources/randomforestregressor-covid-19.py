#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # LoadData

# In[ ]:


train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv').fillna('-')
tempTrain = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv').fillna('-')
tempTest = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv').fillna('-')
submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv').fillna('-')


# # import Lib

# In[ ]:


import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px # install plotly
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# change matplotlib.pyplot to plotly.express because text overlap can't read
def drawPie(dataFrame, indexValue, label, title="Default"):
    fig = px.pie(train, values=indexValue, names=label, title=title)
    fig.update_traces(textposition='inside')
    fig.show()


# In[ ]:


# Get Top 15 Country 
getTopList = 15
grouped_multiple = train.groupby(['Country_Region'], as_index=False)['TargetValue'].sum()
countryTop = grouped_multiple.nlargest(getTopList, 'TargetValue')['Country_Region']
newList = train[train['Country_Region'].isin(countryTop.values)]
line = newList.groupby(['Date', 'Country_Region'], as_index=False)['TargetValue'].sum()
line = line[line['TargetValue'] >= 0]


# In[ ]:


line.pivot(index="Date", columns="Country_Region", values="TargetValue").plot(figsize=(10,5))
plt.grid(zorder=0)
plt.title('Top ' + str(getTopList) +' ConfirmedCases & Fatalities', fontsize=18, pad=10)
plt.ylabel('People')
plt.xlabel('Date')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()


# In[ ]:


drawPie(train, 'TargetValue', 'Target', 'Summary ConfirmedCases & Fatalities')


# In[ ]:


drawPie(train, 'TargetValue', 'Country_Region', 'Percent Target ConfirmedCases & Fatalities')


# In[ ]:


# Check Relationships
sns.pairplot(train)


# # Create Model RandomForestRegressor

# In[ ]:


# Convert string to Date
redate = pd.to_datetime(tempTrain['Date'], errors='coerce')
tempTrain['Date']= redate.dt.strftime("%Y%m%d").astype(int)


# In[ ]:


targets = train['Target'].unique()
for index in range(0, len(targets)):
    tempTrain['Target'].replace(targets[index], index, inplace=True)


# In[ ]:


# Get features
feature_cols = ['Population', 'Weight', 'Date', 'Target']
X = tempTrain[feature_cols] # Features
y = tempTrain['TargetValue'] # Target variable


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Change n_estimators = 50 because give score > .9
model = RandomForestRegressor(n_jobs=-1, n_estimators = 50)
# Fit on training data
model.fit(X_train, y_train)


# In[ ]:


# Score
score = model.score(X_test, y_test)
print("Score: "+ str(score))


# # Prediction
# 

# In[ ]:


# Convert string to Date
redate = pd.to_datetime(tempTest['Date'], errors='coerce')
tempTest['Date']= redate.dt.strftime("%Y%m%d").astype(int)


# In[ ]:


for index in range(0, len(targets)):
    tempTest['Target'].replace(targets[index], index, inplace=True)


# In[ ]:


# Get features
featureCols = ['Population', 'Weight', 'Date', 'Target']
testData = tempTest[featureCols]


# In[ ]:


# predictions
predic = model.predict(testData)


# In[ ]:


# Set Format
listPrediction = [int(x) for x in predic]
newDF = pd.DataFrame({'number': testData.index, 'Population': testData['Population'], 'val': listPrediction})


# In[ ]:


Q05 = newDF.groupby('number')['val'].quantile(q=0.05).reset_index()
Q50 = newDF.groupby('number')['val'].quantile(q=0.5).reset_index()
Q95 = newDF.groupby('number')['val'].quantile(q=0.95).reset_index()

Q05.columns=['number','0.05']
Q50.columns=['number','0.5']
Q95.columns=['number','0.95']


# In[ ]:


concatDF = pd.concat([Q05,Q50['0.5'],Q95['0.95']],1)
concatDF['number'] = concatDF['number'] + 1
concatDF.head(10)


# In[ ]:


sub = pd.melt(concatDF, id_vars=['number'], value_vars=['0.05','0.5','0.95'])
sub['ForecastId_Quantile']=sub['number'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head(10)


# In[ ]:




