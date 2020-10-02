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


import ml_trading_strategy as mlt


# In[ ]:


raw = mlt.read_data('../input/bitcoin-dataset/binance-BTCUSDT-1d.csv')
data = raw


# In[ ]:


SMA1 = 60
SMA2 = 100
    
mlt.moving_averages_crossover_strategy(data, 60, 100)


# In[ ]:


mlt.plotfig(data)


# In[ ]:


mlt.position(data)


# In[ ]:


mlt.plot_position(data)


# In[ ]:


mlt.vectorized_backtesting(data)


# In[ ]:


sma1 = range(20, 61, 4)
sma2 = range(100, 281, 10)
mlt.optimization(SMA1, SMA2, sma1, sma2, data)


# In[ ]:


mlt.random_walk_hypothesis(data)


# In[ ]:


data = pd.DataFrame(raw['Close'])
mlt.linear_ols_regression(data)


# In[ ]:


#Reference Python for Finance 2nd Edition

