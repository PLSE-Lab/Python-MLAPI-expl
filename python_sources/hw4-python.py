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

# %% [code]
train_data = pd.read_csv("/kaggle/input/sputnik/train.csv")
train_data.head()

# %% [code]
train_data.describe()

# %% [code]
train_data['epoch'] = pd.to_datetime(train_data.epoch,format='%Y-%m-%d %H:%M:%S')

# %% [code]
train_data.index  = train_data.epoch
train_data.drop('epoch', axis = 1, inplace = True)

# %% [code]
train_data.head()

# %% [code]
train_data['error']  = np.linalg.norm(train_data[['x', 'y', 'z']].values - train_data[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

# %% [code]
train_data[train_data.sat_id == 12].error.plot()

# %% [code]
train_data['year'] = train_data.index.year
train_data['month'] = train_data.index.month
train_data['hour'] = train_data.index.hour

# %% [code]
train_data['day'] = train_data.index.day

# %% [code]
def smape(y_true, y_pred): 
    return np.mean(np.abs(2*(y_true - y_pred)) / (np.abs(y_true) + np.abs(y_pred)))  * 100

from statsmodels.tsa.api import ExponentialSmoothing
result = {}
for name, group in train_data.groupby('sat_id'):
#for sid in [0, 1, 2, 3]:
    #sat = train_data[train_data.sat_id == sid]
    train_idx = group[group.type == 'train'].index
    test_idx = group[group.type == 'test'].index
    #train_idx = sat[sat.type == 'train'].index
    #test_idx = sat[sat.type == 'test'].index
    
    model = ExponentialSmoothing(np.asarray(group.error.loc[train_idx]) ,seasonal_periods=24 , 
                                 seasonal='additive').fit()
    forecast = pd.Series(model.forecast(len(test_idx)))

    result[name] = forecast

# %% [code]
df = train_data[train_data.type == 'test'][['id', 'sat_id']]

# %% [code]


# %% [code]
df.index = df.id

# %% [code]
df['error'] = 0

# %% [code]
for sat_id in result:
    indices = df[df.sat_id == sat_id].index
    df.loc[indices, 'error'] = result[sat_id].values

# %% [code]
df

# %% [code]
df[['id','error']].to_csv('submission.csv', index = False)
