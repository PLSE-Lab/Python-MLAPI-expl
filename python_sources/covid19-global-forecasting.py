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


# In[ ]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
train.head()


# In[ ]:


train.dtypes


# In[ ]:


train.info()


# In[ ]:


train.shape


# In[ ]:


Total_Country = list(train["Country/Region"].unique())
print(Total_Country)


# In[ ]:


print(" No of Country ",len(Total_Country))


# In[ ]:


train["Province/State"].fillna("none", inplace=True)


# In[ ]:


CountryState = train.groupby(['Country/Region', 'Province/State']) 
CountryState.first()


# In[ ]:


train.groupby(['Country/Region','Province/State'])['ConfirmedCases', 'Fatalities'].max().sort_values(by='ConfirmedCases', ascending=False).head(10)


# In[ ]:


train['month'] = train['Date'].str.extract(r'[0-9]+[-]([0-9]+)[-]')
train['day'] = train['Date'].str.extract(r'[0-9]+[-][0-9]+[-]([0-9]+)')
train = train.drop('Date', axis = 1)
train.shape


# In[ ]:


train.head()


# In[ ]:


la = LabelEncoder()
la.fit(train['Province/State'].unique())
train['Province/State'] = la.transform(train['Province/State'])
la.fit(train['Country/Region'])
train['Country/Region'] = la.transform(train['Country/Region'])
new = train


# In[ ]:


new.head()


# In[ ]:


X = train.drop(['Id', 'ConfirmedCases', 'Fatalities'], axis = 1)
y = train.ConfirmedCases


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[ ]:


y.shape


# In[ ]:


X.shape


# In[ ]:


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=2020)


# In[ ]:


X_train.shape,y_train.shape


# In[ ]:


X_test.shape ,y_test.shape


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[ ]:


regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print( np.sqrt( mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


# In[ ]:


new.head()


# In[ ]:


X = train.drop(['Id', 'ConfirmedCases', 'Fatalities'], axis = 1)
y = train.Fatalities


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[ ]:


X.head()


# In[ ]:


scaler = StandardScaler()
scaler.fit(X)


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[ ]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state=2020)


# In[ ]:


X_train.shape ,y_train.shape


# In[ ]:


X_test.shape ,y_test.shape


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()


# In[ ]:


regressor1.fit(X_train, y_train)


# In[ ]:


y_prediction = regressor1.predict(X)


# In[ ]:


import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print( np.sqrt( mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


# In[ ]:


test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
test.head()


# In[ ]:


test.shape


# In[ ]:


test.info()


# In[ ]:


test['month'] = test['Date'].str.extract(r'[0-9]+[-]([0-9]+)[-]')
test['day'] = test['Date'].str.extract(r'[0-9]+[-][0-9]+[-]([0-9]+)')
test = test.drop('Date', axis = 1)


# In[ ]:


test["Province/State"].fillna("none", inplace=True)


# In[ ]:


la = LabelEncoder()
la.fit(test['Province/State'].unique())
test['Province/State'] = la.transform(test['Province/State'])
la.fit(test['Country/Region'])
test['Country/Region'] = la.transform(test['Country/Region'])
X = test.drop(['ForecastId'], axis = 1)
X


# In[ ]:


y_predictions = regressor.predict(X)


# In[ ]:


y_prediction = regressor1.predict(X)


# In[ ]:


sub = pd.DataFrame({'ForecastId': test.ForecastId, 'ConfirmedCases': y_predictions,"Fatalities": y_prediction})
sub = sub[['ForecastId', 'ConfirmedCases',"Fatalities"]]
filename = 'submission.csv'
sub.to_csv(filename, index=False) 
sub.head()

