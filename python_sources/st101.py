#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing


import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


raw_data = pd.read_csv('../input/temps.csv')
df = raw_data.copy()
df.head()


# In[ ]:


df = df.drop(['year', 'forecast_noaa', 'forecast_acc', 'forecast_under'], axis=1)


# In[ ]:


day_of_the_week = pd.get_dummies(df['week'])
day_of_the_week.head()


# In[ ]:


df = pd.concat([df, day_of_the_week], axis=1)
df2 = df.copy()
df2 = df.drop(['week'], axis=1)


# In[ ]:


df2.columns.values


# In[ ]:


## Arrange the table
df2 = df2[['month', 'day', 'Sun','Mon', 'Tues', 'Wed','Thurs',
       'Fri', 'Sat' ,'friend', 'temp_2', 'temp_1', 'average', 'actual', ]]
df2.head()


# In[ ]:


df2.info()


# In[ ]:


features = np.array(df2.iloc[:, :-1])
targets = np.array(df2.iloc[:, -1:])


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, random_state=20)


# In[ ]:


rf = RandomForestRegressor(n_estimators=1500, random_state=50)
rf.fit(x_train, y_train)


# In[ ]:


print('Training Accuracy:', round(rf.score(x_train, y_train)*100, 2), '%')


# In[ ]:


rf.feature_importances_


# In[ ]:


features = ['month', 'day', 'Sun','Mon', 'Tues', 'Wed','Thurs',
       'Fri', 'Sat' ,'friend', 'temp_2', 'temp_1', 'average']


# In[ ]:


summary_table = pd.DataFrame({'Features':features, 'Values': rf.feature_importances_})
summary_table


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.style.use('dark_background')

x = np.arange(len(rf.feature_importances_))
y = list(rf.feature_importances_)
plt.xticks(x, features, rotation='vertical')
plt.bar(x,y,orientation='vertical')
plt.ylabel('Level of Importance')
plt.title('Feature Importances')
plt.show()


# In[ ]:


print('Test Accuracy:', round(rf.score(x_test, y_test)*100, 2), '%')


# In[ ]:





# In[ ]:




