#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Visualization 2

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from sklearn_pandas import CategoricalImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Set-up

# In[ ]:


df = pd.read_csv('/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
df.head()


# In[ ]:


percent_missing = df.isnull().sum() * 100 / len(df)
missing_values = pd.DataFrame({'column_name': df.columns,
                               'percent_missing': percent_missing})
missing_values


# In[ ]:


# imputer = CategoricalImputer()
# df['Open'] = imputer.fit_transform(df['Open'].values)
# df['High'] = imputer.fit_transform(df['High'].values)
# df['Low'] = imputer.fit_transform(df['Low'].values)
# df['Close'] = imputer.fit_transform(df['Close'].values)
# df['Volume_(BTC)'] = imputer.fit_transform(df['Volume_(BTC)'].values)
df = df.dropna()
percent_missing = df.isnull().sum() * 100 / len(df)
missing_values = pd.DataFrame({'column_name': df.columns,
                               'percent_missing': percent_missing})
missing_values


# In[ ]:


df = df.drop(['Timestamp'], axis=1)
#df = df.apply(preprocessing.LabelEncoder().fit_transform)
#df = pd.get_dummies(df)
df = df[:2000]
df.head()


# In[ ]:


len(df)


# In[ ]:


X = df.drop(['Close'], axis=1)
y = df['Close'].values
scaler = StandardScaler().fit(X)
X2 = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.05, random_state=42)


# In[ ]:


regr = RandomForestRegressor(max_depth=200, random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mean_squared_error(y_test, y_pred)


# In[ ]:


plt.figure(figsize=(12,7))
plt.plot(y_pred, color='green', marker='o', linestyle='dashed', 
         label='Predicted Price')
plt.plot(y_test, color='red', label='Actual Price')
#plt.xticks(np.arange(1486,1856, 60), df['Date'][1486:1856:60])
plt.title('Sales Prediction')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()


# ## Prediction Table

# In[ ]:


get_ipython().system(' pip install chart-studio')


# In[ ]:


import chart_studio
chart_studio.tools.set_credentials_file(username='TODO', api_key='TODO')


# In[ ]:


import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly

def predreport(y_pred, Y_Test):
    diff = y_pred.flatten() - Y_Test.flatten()
    perc = (abs(diff)/y_pred.flatten())*100
    priority = []
    for i in perc:
        if i > 0.4:
            priority.append(3)
        elif i> 0.1:
            priority.append(2)
        else:
            priority.append(1)


    print("Error Importance 1 reported in ", priority.count(1), "cases \n")
    print("Error Importance 2 reported in ", priority.count(2), "cases \n")
    print("Error Importance 3 reported in ", priority.count(3), "cases \n")
    colors = ['rgb(102, 153, 255)','rgb(0, 255, 0)', 'rgb(255, 153, 51)',
              'rgb(255, 51, 0)']

    fig = go.Figure(data=[go.Table(header=dict(values=['Actual Values', 'Predictions', 
                                                       '% Difference', "Error Importance"],
                                                        line_color=[np.array(colors)[0]],
                                                        fill_color=[np.array(colors)[0]],
                                                        align='left'),
                     cells=dict(values=[y_pred.flatten(), Y_Test.flatten(), perc, priority],
                                        line_color=[np.array(colors)[priority]], 
                                        fill_color=[np.array(colors)[priority]],
                                        align='left'))
                         ])

    init_notebook_mode(connected=False)
    #py.plot(fig, filename = 'Predictions_Table', auto_open=True)
    fig.show()


# In[ ]:


predreport(y_pred[:200], y_test[:200])


# ## dtreeviz : Decision Tree Visualization

# In[ ]:


get_ipython().system('pip install dtreeviz')


# In[ ]:


regr = DecisionTreeRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mean_squared_error(y_test, y_pred)


# In[ ]:


from dtreeviz.trees import *

viz = dtreeviz(regr,
               X_train,
               y_train,
               target_name='Close',
               feature_names=list(X.columns))
              
viz


# In[ ]:


#viz.svg()

