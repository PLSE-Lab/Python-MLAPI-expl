#!/usr/bin/env python
# coding: utf-8

# # EDA & Gaussian KDE Baseline Model

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
submission_file = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')


# In[ ]:


days = range(1, 1913 + 1)
time_series_columns = [f'd_{i}' for i in days]

time_series_data = train_sales[time_series_columns]


# # Visualizing the Time Series
# 
# Below I make a simple plot of the first time series in the data.

# In[ ]:


figsize = (25, 5)
time_series_data.iloc[0, :].plot(figsize=figsize)
plt.grid()


# # Gaussian Kernel Density Estimator (KDE) Sampling

# In[ ]:


from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


# In[ ]:


time_series_data.head(5)


# In[ ]:


params = {
    'kernel': ['gaussian'],
    'bandwidth': [0.1]
         }


# In[ ]:


N = time_series_data.shape[0]
random_samples = []
for i in range(N):
    model = GridSearchCV(KernelDensity(), params)
    df = time_series_data.loc[[i]].T
    df = df.values.reshape(-1,1)
    model.fit(df)
    
    # randomly sample 28 times from the KDE 
    kde_sample = time_series_data.loc[[i]].T.sample(28,random_state=42).T
    random_samples += [kde_sample]
    
    # print only selected outputs
    if i < 10:
        print('Iteration ' + str(i) + ' done. Best model params is: ' + str(model.best_params_))
    elif i == N-5:
        print('...')
    elif i > N-5:
        print('Iteration ' + str(i) + ' done. Best model params is: ' + str(model.best_params_))


# In[ ]:


forecasts = pd.concat(random_samples)
forecasts.columns = [f'F{i}' for i in range(1, 28 + 1)]
forecasts.reset_index(inplace=True)
forecasts.drop(['index'], axis=1, inplace=True)

forecasts.head(5)


# # Predictions
# 
# We need to provide predictions for the next 28 days for each of the series. For the validation series that is days 1914 - 1941 and for the evaluation that is days 1942 - 1969.

# In[ ]:


validation_ids = train_sales['id'].values
evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]


# In[ ]:


ids = np.concatenate([validation_ids, evaluation_ids])


# In[ ]:


predictions = pd.DataFrame(ids, columns=['id'])
forecasts = pd.concat([forecasts] * 2).reset_index(drop=True)
predictions = pd.concat([predictions, forecasts], axis=1)


# In[ ]:


predictions.head()


# In[ ]:


predictions.to_csv('submission.csv', index=False)


# Feel free to add in more parameters for grid search! This is only a baseline working version.
# 
# If you find this helpful and/or insightful, an upvote will be appreciated!
# 
# I'll try to answer questions to the best of my ability! :)
