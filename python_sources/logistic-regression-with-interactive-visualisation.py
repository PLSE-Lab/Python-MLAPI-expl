#!/usr/bin/env python
# coding: utf-8
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
# In[ ]:


import pandas as pd
df=pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
df.head()


# In[ ]:


df.describe()


# In[ ]:


#checking NA's in dataframe
df.isna().sum()


# In[ ]:


df=df.drop(['company'],axis=1)


# In[ ]:


for col in ['country','agent','children']:
    df[col].fillna(df[col].mode()[0],inplace=True)


# In[ ]:


df.isna().sum()


# In[ ]:


df.info()


# In[ ]:


from sklearn import preprocessing 
dff=df.apply(preprocessing.LabelEncoder().fit_transform)
dff


# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import AdaBoostRegressor


# In[ ]:


X = dff.drop(['is_canceled'], axis = 1)
y = dff['is_canceled']


# In[ ]:


dff['is_canceled'].unique()


# In[ ]:





# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


logreg = LogisticRegression(solver = 'lbfgs')
# fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

print('Mean Absolute Error_logreg:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_logreg:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_logreg:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_logreg:', r2_score(y_test, y_pred).round(3))


# In[ ]:


import matplotlib.pyplot as plt
con=confusion_matrix(y_test, y_pred)
print(con)
acc=accuracy_score(y_test, y_pred)
print(acc)

import seaborn as sns
ax= plt.subplot()
sns.heatmap(con, annot=True, ax = ax,fmt='g',cmap='gist_rainbow'); 


# In[ ]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
pre = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
pre


# In[ ]:


import chart_studio.plotly as py
#import chart_studio.graph_objs as go
from plotly import graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


# In[ ]:


df['country'].iplot(kind='hist', xTitle='month',
                  yTitle='count', title='Arrivals in a year',colors='Blue')


# In[ ]:


df.groupby(['arrival_date_month','arrival_date_year'])['children', 'babies','adults'].sum().plot.bar(figsize=(15,5))


# In[ ]:


plt.figure(figsize=(15,5))
sns.lineplot(x= 'arrival_date_month', y = 'lead_time', data = df)


# In[ ]:


df_ct=dff['customer_type'].unique()


# In[ ]:


df['customer_type'].unique()


# In[ ]:


explode = (0, 0.1, 0, 0)

labels = ['Transient', 'Contract', 'Transient-Party', 'Group']
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
fig1, ax1 = plt.subplots()
ax1.pie(df_ct, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.tight_layout()
plt.show()

