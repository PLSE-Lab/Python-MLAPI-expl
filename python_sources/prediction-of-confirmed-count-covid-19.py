#!/usr/bin/env python
# coding: utf-8

# The number of confirmed cases of covid-19 in Korea has been soaring since February 18, a week ago. It's nearly a 30-fold increase in just a week, so it's hard to imagine what the next week will be like. Based on the data for about a month, I tried to draw simple predictions through regression.

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta


# In[ ]:


path = '/kaggle/input/coronavirusdataset/patient.csv'
df = pd.read_csv(path)
df.head()


# In[ ]:


# type casting : object -> datetime
df.confirmed_date = pd.to_datetime(df.confirmed_date)

# get daily confirmed count
daily_count = df.groupby(df.confirmed_date).id.count()

# get accumulated confirmed count
accumulated_count = daily_count.cumsum()


# In[ ]:


daily_count.plot()
plt.title('Daily Confirmed Count');


# In[ ]:


accumulated_count.plot()
plt.title('Accumulated Confirmed Count');


# In[ ]:


# fill missing dates with zero
data = daily_count.resample('D').first().fillna(0).cumsum()

# use only recent data
data = data[30:]

# prepare data for regressor
x = np.arange(len(data)).reshape(-1, 1)
y = data.values

# train simple MLPRgressor
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)
_=model.fit(x, y)


# In[ ]:


# predict accumlated cofirmed count over the week
test = np.arange(len(data)+7).reshape(-1, 1)
pred = model.predict(test)

# get time sequence data as pd.Series
prediction = pred.round().astype(int)
week = [data.index[0] + timedelta(days=i) for i in range(len(prediction))]
dt_idx = pd.DatetimeIndex(week)
predicted_count = pd.Series(prediction, dt_idx)


# In[ ]:


# plot predicted data
accumulated_count.plot()
predicted_count.plot()

plt.title('Prediction of Accumulated Confirmed Count')
plt.legend(['current confirmd count', 'predicted confirmed count'])
plt.show()


# In[ ]:


# print predicted data
print(predicted_count[-7:])

