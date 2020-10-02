#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')


# In[ ]:


import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


market_train, news_train = market_train_df.copy(), news_train_df.copy()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff
import plotly.plotly as py
import plotly.graph_objs as go


# In[ ]:


get_ipython().run_cell_magic('time', '', "def data_prep(market_train,news_train):\n    market_train.time = market_train.time.dt.date\n    news_train.time = news_train.time.dt.hour\n    news_train.sourceTimestamp= news_train.sourceTimestamp.dt.hour\n    news_train.firstCreated = news_train.firstCreated.dt.date\n    news_train['assetCodesLen'] = news_train['assetCodes'].map(lambda x: len(eval(x)))\n    news_train['assetCodes'] = news_train['assetCodes'].map(lambda x: list(eval(x))[0])\n    kcol = ['firstCreated', 'assetCodes']\n    news_train = news_train.groupby(kcol, as_index=False).mean()\n    market_train = pd.merge(market_train, news_train, how='left', left_on=['time', 'assetCode'], \n                            right_on=['firstCreated', 'assetCodes'])\n    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}\n    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)\n    \n    \n    market_train = market_train.dropna(axis=0)\n    \n    return market_train\n\nmarket_train = data_prep(market_train_df, news_train_df)\nmarket_train.shape")


# In[ ]:


get_ipython().run_cell_magic('time', '', "from datetime import datetime, date\n# The target is binary\nmarket_train = market_train.loc[market_train['time_x']>=date(2009, 1, 1)]\nup = market_train.returnsOpenNextMktres10 >= 0\nfcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', \n                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', \n                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# We still need the returns for model tuning\nX = market_train[fcol].values\nup = up.values\nr = market_train.returnsOpenNextMktres10.values\n\n# Scaling of X values\n# It is good to keep these scaling values for later\nmins = np.min(X, axis=0)\nmaxs = np.max(X, axis=0)\nrng = maxs - mins\nX = 1 - ((maxs - X) / rng)\n\n# Sanity check\nassert X.shape[0] == up.shape[0] == r.shape[0]')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from xgboost import XGBClassifier\nfrom sklearn import model_selection\nfrom sklearn.metrics import accuracy_score\nimport time\n\nX_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.25, random_state=99)')


# In[ ]:


xgb_up = XGBClassifier(n_jobs=4,n_estimators=250,max_depth=9,eta=0.08)


# In[ ]:


t = time.time()
print('Fitting Up')
xgb_up.fit(X_train,up_train)
print(f'Done, time = {time.time() - t}')


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(xgb_up.predict(X_test),up_test)


# In[ ]:




