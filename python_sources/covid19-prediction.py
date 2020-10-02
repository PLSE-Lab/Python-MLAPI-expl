#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for visualization
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_dark"

import os
# printing the directories
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# read the dataset
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
test= pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
train.head()


# In[ ]:


# describe the dataset
def description(data):
    print('Total no of Countries :', data['Country/Region'].nunique())
    print('Total no of Provinces :', data['Province/State'].nunique())
    print('Total no of records :', data.shape[0])
    print('Range of dates from ' +str(data['Date'].min()) + ' to ' + str(data['Date'].max()))

print('Description of the training set :\n')
description(train)
print('\nDescription of the test set :\n')
description(test)


# ### There are overlapping in the train and test data from 12-03-2020 to 24-03-2020. We need to take care of this during training models.

# In[ ]:


# count of provinces for each of the countries
train.groupby('Country/Region')['Country/Region', 'Province/State'].nunique()                                                                   .drop('Country/Region', axis=1).reset_index()                                                                   .sort_values(by='Province/State', ascending=False)


# In[ ]:


train['Country&Province'] = train['Country/Region'] + ',' + train['Province/State'].astype(str).replace('nan', '')
train['Country&Province'] = train['Country&Province'].str.rstrip(',')


# In[ ]:


NY = train[train['Country&Province'] == 'US,New York'][['Date', 'ConfirmedCases', 'Fatalities']].reset_index(drop=True)
NY.head()


# In[ ]:


# plotting only for NY
# plot
fig = go.Figure(layout=dict(title=dict(text='New York')))

fig.add_trace(go.Scatter(x=NY['Date'], y=NY['ConfirmedCases'], mode='lines+markers', name='ConfirmedCases'))
fig.add_trace(go.Scatter(x=NY['Date'], y=NY['Fatalities'], mode='lines+markers', name='Fatalities'))
        
fig.show()


# In[ ]:




