#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet
import matplotlib.pyplot as plt
from fbprophet.plot import add_changepoints_to_plot
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data=pd.read_csv('/kaggle/input/retail-grocery-store-sales-data-from-20162019/fruithut_data_ordered_csv_file_1_1.csv')
print(data)
columns=data.columns
print(columns)



import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from fbprophet.plot import add_changepoints_to_plot
from sklearn.neighbors import LocalOutlierFactor
import numpy as np


def filter(df):
    print(df.head(30))
    new_df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.DATENEW)))
    # new_df=new_df.drop(['DATENEW'], axis=1)
    new_df = new_df['UNITS'].resample('w').sum()

    new_df_ = pd.DataFrame(data=new_df, index=new_df.index, columns=['DATENEW', 'UNITS'])
    new_df_['DATENEW'] = new_df_.index
    a = new_df_.reset_index(drop=True)
    print(a)

    return a


def time_series_fbprophet(df):
   # df = pd.read_csv('fruithut_data_ordered_csv_file_1_1.csv')
    df = df[df['NAME'] == "Orange navel"]
    print(df)
    train_dataset = df[['DATENEW', 'UNITS']]
    result = filter(train_dataset)
    filtered_data = result.rename(columns={"DATENEW": "ds", "UNITS": "y"}, errors="raise")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.8,interval_width=0.95,
                    changepoint_range=0.9)
    model.fit(filtered_data)

    future = model.make_future_dataframe(periods=int(365), freq='D')

    forecast = model.predict(future)
    fig1 = model.plot(forecast)
    a = add_changepoints_to_plot(fig1.gca(), model, forecast)
    plt.show()
    fig2 = model.plot_components(forecast)
    plt.show()
    # fig1 = m.plot(forecast)
    # plot_data = plt.show()


time_series_fbprophet(data)


# Any results you write to the current directory are saved as output.


# In[ ]:




