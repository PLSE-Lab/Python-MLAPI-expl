#!/usr/bin/env python
# coding: utf-8

# ## This notebook's purpose is to provide an EDA on the TransactionDT

# This kernel has been inspired by https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb

# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling as pp
import random

#plotly packages
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tools

#Modelling
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')\n\ntrain_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')\n\nsample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')\n\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ntest = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)\n\nprint(train.shape)\nprint(test.shape)\n\ny_train = train['isFraud'].copy()\ndel train_transaction, train_identity, test_transaction, test_identity\n\n# Drop target, fill in NaNs\nX_train = train.drop('isFraud', axis=1)\nX_test = test.copy()\n\ndel train, test\n\n# Label Encoding\nfor f in X_train.columns:\n    if X_train[f].dtype=='object' or X_test[f].dtype=='object': \n        lbl = preprocessing.LabelEncoder()\n        lbl.fit(list(X_train[f].values) + list(X_test[f].values))\n        X_train[f] = lbl.transform(list(X_train[f].values))\n        X_test[f] = lbl.transform(list(X_test[f].values))   ")


# In[ ]:



# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
# WARNING! THIS CAN DAMAGE THE DATA 
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
X_train = reduce_mem_usage(X_train)
X_test = reduce_mem_usage(X_test)


# In[ ]:


def missing_values(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    return missing_value_df

data = [go.Bar(
            x=missing_values(X_train).column_name,
            y=missing_values(X_train).percent_missing
    )]
layout = go.Layout(
    autosize=False,
    width=1000,
    height=500,
title = 'Missing Values by Column')
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# ### TransactionDT Variable

# In[ ]:


X_train['data'] = 'X_train'
X_test['data'] = 'X_test'

X_train_test = pd.concat([X_train, X_test], axis = 1)
print(X_train_test.shape)


# In[ ]:


data = [
    go.Scatter(
        x=X_train.TransactionDT, # assign x as the dataframe column 'x'
        y=X_train.TransactionAmt,
        name='X_train'
    )
#     ,
#     go.Scatter(
#         x=X_test.TransactionDT, # assign x as the dataframe column 'x'
#         y=X_test.TransactionAmt,
#         name='X_test'
#     )
]
fig = data
py.offline.iplot(fig)


# In[ ]:


def pick_color():
    colors = ["blue","black","brown","red","yellow","green","orange","beige","turquoise","pink"]
    random.shuffle(colors)
    return colors[0]

def Hist_plot(data,i):
    trace0 = go.Histogram(
        x= data.iloc[:,i],
        name = str(data.columns[i]),
        nbinsx = 100,
        marker= dict(
            color=pick_color(),
            line = dict(
                color = 'black',
                width = 0.5
              ),
        ),
        opacity = 0.70,
  )
    fig_list = [trace0]
    title = str(data.columns[i])
    return fig_list, title
    
def Plot_grid(data, ii, ncols=2):
    plot_list = list()
    title_list = list()
    
    #Saving all the plots in a list
    for i in range(ii):
        p = Hist_plot(data,i)
        plot_list.append(p[0])
        title_list.append(p[1])
    
    #Creating the grid
    nrows = max(1,ii//ncols)
    i = 0
    fig = tools.make_subplots(rows=nrows, cols=ncols, subplot_titles = title_list)
    for rows in range(1,nrows+1):
        for cols in range(1,ncols+1):
            fig.append_trace(plot_list[i][0], rows, cols)
            i += 1
    fig['layout'].update(height=400*nrows, width=1000)
    return py.offline.iplot(fig)


# In[ ]:


Plot_grid(X_train, 6,2)


# In[ ]:




