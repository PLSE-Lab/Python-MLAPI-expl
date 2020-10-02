#!/usr/bin/env python
# coding: utf-8

# ## Description of this kernel

# In this kernel I used a very basic xgboost regressor algorithm to predict Covid-19 cases and then I see which states give rise the the largest error. 
# 
# A big error in predicting 'ConfirmedCases' means that the trend cannot be easily predicted by a simple model, there are several possible explanations : 
# 
# * not enough test to detect all covid-19 cases
# * Local policies that are difficult to predict
# * environmental factor
# * atypical place (e.g. China, origin of the epidemy)
# 
# On the other hand, a big error in predicting'Fatalities' cannot be explained by the first reason.

# ## Data processing

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


# In[ ]:


from tqdm import tqdm
from xgboost import XGBRegressor


# In[ ]:


X_train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv' )
X_test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv' )


# In[ ]:


sample_submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv' )


# First, we process the data so that the time is given by an integer and we simplify the name of the features.

# In[ ]:


X_train['Date'] = pd.to_datetime(X_train['Date'], infer_datetime_format=True)
X_test['Date'] = pd.to_datetime(X_test['Date'], infer_datetime_format=True)

X_train.loc[:, 'Date'] = X_train.Date.dt.strftime("%m%d")
X_train["Date"]  = X_train["Date"].astype(int)

X_test.loc[:, 'Date'] = X_test.Date.dt.strftime("%m%d")
X_test["Date"]  = X_test["Date"].astype(int)

X_train.rename(columns={'Country_Region':'Country'}, inplace=True)
X_test.rename(columns={'Country_Region':'Country'}, inplace=True)

X_train.rename(columns={'Province_State':'State'}, inplace=True)
X_test.rename(columns={'Province_State':'State'}, inplace=True)


# Rename nan states as the string "nan". We will need this later.

# In[ ]:


X_train.loc[X_train['State'].isna(), 'State']="nan"
X_test.loc[X_test['State'].isna(), 'State']="nan"


# ## Prediction with xgboost and comparison of error on several countries.

# In[ ]:


from sklearn.base import BaseEstimator

# Create a regressor that does not give negative results
class booster(BaseEstimator):
    def __init__(self, **params):
        self.reg=XGBRegressor(**params)

    def fit(self, X, y=None):
        self.reg.fit(X,y)
        return self

    def predict(self,X):
        pred=self.reg.predict(X)
        pred[pred<0]=0
        return pred
    
    def set_params(self,**params):
        self.reg.set_params(**params)
    


# We make a model for each country/state. We use TimeSeriesSplit a cross-validation adapted to time-series to get validation scores. Instead of the mean_squared_log_error used in the leaderboard we use r2 score as it seems less influenced by the total population of a country.

# In[ ]:


countries = X_train['Country'].unique()

from sklearn import preprocessing, clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
tscv = TimeSeriesSplit(n_splits=10)
cv_score=[]
cs=[]

le = preprocessing.LabelEncoder()

xout = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
cv_score=[]

for country in countries:
    states = X_train.loc[X_train['Country'] == country, :]['State'].unique()
    #print(country, states)
    # check whether string is nan or not
    for state in states:
        X_train_CS = X_train.loc[(X_train['Country'] == country) & (X_train['State'] == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]

        y1_train_CS = X_train_CS.loc[:, 'ConfirmedCases']
        y2_train_CS = X_train_CS.loc[:, 'Fatalities']


        X_train_CS = X_train_CS.loc[:, ['State', 'Country', 'Date']]

        X_train_CS['Country'] = le.fit_transform(X_train_CS['Country'])
        X_train_CS['State'] = le.fit_transform(X_train_CS['State'])

        X_test_CS = X_test.loc[(X_test['Country'] == country) & (X_test['State'] == state), ['State', 'Country', 'Date', 'ForecastId']]

        X_test_CS_Id = X_test_CS.loc[:, 'ForecastId']
        X_test_CS = X_test_CS.loc[:, ['State', 'Country', 'Date']]

        X_test_CS['Country'] = le.fit_transform(X_test_CS['Country'])
        X_test_CS['State'] = le.fit_transform(X_test_CS['State'])

        regressor = booster(mad_depth=3,n_estimators=10)

        # cross-validation for confirmed cases
        cv=[]
        for train_index, test_index in tscv.split(X_train_CS):
            xtrain, ytrain = X_train_CS.iloc[train_index], y1_train_CS.iloc[train_index]
            xtest, ytest = X_train_CS.iloc[test_index], y1_train_CS.iloc[test_index]
            reg=clone(regressor)
            reg.fit(xtrain,ytrain)
            cv+=[np.mean(r2_score(ytest,reg.predict(xtest)))]
            
        # cross validation for fatalities
        cv2=[]
        for train_index, test_index in tscv.split(X_train_CS):
            xtrain, ytrain = X_train_CS.iloc[train_index], y2_train_CS.iloc[train_index]
            xtest, ytest = X_train_CS.iloc[test_index], y2_train_CS.iloc[test_index]
            reg=clone(regressor)
            reg.fit(xtrain,ytrain)
            cv2+=[np.mean(r2_score(ytest,reg.predict(xtest)))]
            
        cv_score += [[np.mean(cv), np.mean(cv2)]]

        cs += [(country, state)]


# ## Results

# In[ ]:


cv_score=np.array(cv_score)
np.mean(cv_score, axis=0)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(cv_score[:,0], label='Error for Confirmed Cases')
sns.distplot(cv_score[:,1], label='Error for Fatalities')
plt.legend()


# * See which are the 15 most "Confirmed cases" difficult to predict with our model (it can change according to which loss you choose)

# In[ ]:


country_state= [country+'/'+state for country,state in cs]
N=15
ind_sort_cc=np.argsort(cv_score[:,0])

print(pd.DataFrame({'Coutry/State':np.array(country_state)[ind_sort_cc[:N]],'error':np.array(cv_score)[ind_sort_cc[:N],0] }))


# * We do the same with the number of fatalities

# In[ ]:


ind_sort_f=np.argsort(cv_score[:,1])

print(pd.DataFrame({'Coutry/State':np.array(country_state)[ind_sort_f[:N]],'error':np.array(cv_score)[ind_sort_f[:N],1] }))


# Without surprise, China present a lot of difficult to predict regions because of its place as origin of the epidemy.

# ## A small vizualization

# In[ ]:


import plotly.graph_objects as go #Plotlygo for plotting

#Plotting a bar graph for error by couple (country,state).
N=50 # number of country/state to plot
indices=ind_sort_cc[:N]
scores = {'Country/State' : np.array(country_state)[indices], 'error': np.array(cv_score[:,0])[indices]}
scores_df = pd.DataFrame(scores)

#Plotting the Graph.

fig = go.Figure()
fig.add_trace(go.Bar(x=scores_df['Country/State'], y=scores_df['error'], name='States most difficult to predict for ConfirmedCases'))

fig.show()


# In[ ]:



indices=ind_sort_f[:N]
scores = {'Country/State' : np.array(country_state)[indices], 'error': np.array(cv_score[:,1])[indices]}
scores_df = pd.DataFrame(scores)

#Plotting the Graph.

fig = go.Figure()
fig.add_trace(go.Bar(x=scores_df['Country/State'], y=scores_df['error'], name='States most difficult to predict for Fatalities'))

fig.show()


# In[ ]:




