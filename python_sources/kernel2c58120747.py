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
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train= pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
submission = pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')


# In[ ]:


train['Dates'] = pd.to_datetime(train.Date)
train['Year'] = train.Dates.dt.year
train['Month'] = train.Dates.dt.month
train['Week'] = train.Dates.dt.week
train['Name of month'] = train.Dates.dt.month_name()
train['Week_day_name'] = train.Dates.dt.weekday_name
train['Day_of_year'] = train.Dates.dt.dayofyear


# In[ ]:


test['Dates'] = pd.to_datetime(test.Date)
test['Year'] = test.Dates.dt.year
test['Month'] = test.Dates.dt.month
test['Week'] = test.Dates.dt.week
test['Name of month'] = test.Dates.dt.month_name()
test['Week_day_name'] = test.Dates.dt.weekday_name
test['Day_of_year'] = test.Dates.dt.dayofyear


# In[ ]:


train[['ConfirmedCases','Fatalities']].sum().plot(kind = 'pie', autopct = '%.1f%%', legend = 'best', 
                                                 explode = [0, 0.5], label = '')
plt.title('Percentage of total confirmed and fatalities')
plt.legend()
plt.show()


# In[ ]:


train.groupby('Country_Region')['ConfirmedCases'].sum().nlargest(5)


# In[ ]:


train.groupby('Country_Region')['Fatalities'].sum().nlargest(5)


# In[ ]:


plt.subplot(1,2,1)
train.groupby('Country_Region')['ConfirmedCases'].sum().nlargest(5).plot(kind = 'pie',
                                                                         label ='',autopct ='%0.1f%%')
plt.title('Top 5 highest confirmed cases')
plt.show()
plt.subplot(1,2,2)
train.groupby('Country_Region')['Fatalities'].sum().nlargest(5).plot(kind ='pie',label = '',autopct ='%0.1f%%')
plt.title('Top 5 highest fatalities')
plt.tight_layout()
plt.show()


# In[ ]:


#The month with the highest confirmed cases for five top contries
train.groupby(['Country_Region', 'Name of month'])['ConfirmedCases'].sum().nlargest(5)


# In[ ]:


#The month with the highest fatalities for 5 top countries 
train.groupby(['Country_Region', 'Name of month'])['Fatalities'].sum().nlargest(5)


# In[ ]:


#Countries and week with the highest confirmed cases
train.groupby(['Country_Region', 'Week'])['ConfirmedCases'].sum().nlargest(10)


# In[ ]:


#Countries and week with the highest fatalities
train.groupby(['Country_Region', 'Week'])['Fatalities'].sum().nlargest(10)


# In[ ]:


x = train[['Month','Week','Day_of_year']]
y = train['Fatalities']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 40)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dec = DecisionTreeClassifier(max_features=0.1)
dec.fit(x_train,y_train)
dec.score(x_test,y_test)


# In[ ]:


x1 = test[['Month','Week','Day_of_year']]


# In[ ]:


pred2 = dec.predict(x1)


# In[ ]:


x2 = train[['Month','Week','Day_of_year']]
y2 = train[['ConfirmedCases']]


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x2,y2, test_size= 0.3, random_state = 40)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dec= DecisionTreeClassifier(max_features=0.1)
dec.fit(x_train,y_train)
dec.score(x_test,y_test)


# In[ ]:


x3 = test[['Month','Week','Day_of_year']]


# In[ ]:


pred1 = dec.predict(x3)


# In[ ]:


submission.ConfirmedCases=pred1 
submission.Fatalities= pred2


# In[ ]:


submission.to_csv('submission.csv', index = False)


# In[ ]:




