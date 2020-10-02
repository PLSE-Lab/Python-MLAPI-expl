#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import lightgbm as lgb
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


INPUT_PATH = '/kaggle/input/liverpool-ion-switching'
train_df = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
test_df = pd.read_csv(os.path.join(INPUT_PATH, 'test.csv'))
sample_sub_df = pd.read_csv(os.path.join(INPUT_PATH, 'sample_submission.csv'))


# In[ ]:


sns.set(rc={'figure.figsize':(11,8)})
sns.set(style="whitegrid")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


sample_sub_df.head()


# In[ ]:


print(f'Shape of training dataset: {train_df.shape}')
print(f'Shape of test dataset: {test_df.shape}')


# In[ ]:


train_df.info()


# In[ ]:


train_df.nunique()


# In[ ]:


def print_description(df, column):
    print(f'Column: {column}: Min: {df[column].min()} Max: {df[column].max()} Mean: {df[column].mean()}')


# In[ ]:


print_description(train_df, 'open_channels')
print_description(train_df, 'signal')


# ### Distribution of Signals

# In[ ]:


sns.distplot(train_df['signal'], kde=False)
plt.show()


# ### Distribution of Target variable

# In[ ]:


sns.distplot(train_df['open_channels'], kde=False)
plt.show()


# ### Batchwise target distribution

# In[ ]:


# Visualize signals of all the different batches
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
count=0
for row in ax:
    for col in row:
        col.title.set_text(f'Batch #{count}')
        col.bar(train_df['open_channels'][count*500000: (count+1)*500000].value_counts().index.values, train_df['open_channels'][count*500000: (count+1)*500000].value_counts().values)
        count += 1
plt.show()


# ### Batchwise signal distributation

# In[ ]:


# Visualize signals of all the different batches
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
count=0
for row in ax:
    for col in row:
        col.title.set_text(f'Batch #{count}')
        col.plot(train_df['time'][count*500000: (count+1)*500000], train_df['signal'][count*500000: (count+1)*500000])
        count += 1
plt.show()


# ### Feature Extraction

# In[ ]:


window_sizes = [50, 100, 1000, 5000, 10000, 25000]
for window in window_sizes:
    train_df[f'rolling_mean{window}'] = train_df['signal'].rolling(window).mean()
    train_df[f'rolling_std{window}'] = train_df['signal'].rolling(window).std()
    train_df[f'rolling_min{window}'] = train_df['signal'].rolling(window).min()
    train_df[f'rolling_max{window}'] = train_df['signal'].rolling(window).max()
    a = (train_df['signal'] - train_df['rolling_min' + str(window)]) / (train_df['rolling_max' + str(window)] - train_df['rolling_min' + str(window)])
    train_df["norm" + str(window)] = a * (np.floor(train_df['rolling_max' + str(window)]) - np.ceil(train_df['rolling_min' + str(window)]))
    
train_df = train_df.fillna(train_df.mean())


# In[ ]:


X_train = train_df.drop(['time', 'open_channels'], axis=1)
Y_train = train_df['open_channels']


# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)


# In[ ]:


for window in window_sizes:
    test_df[f'rolling_mean{window}'] = test_df['signal'].rolling(window).mean()
    test_df[f'rolling_std{window}'] = test_df['signal'].rolling(window).std()
    test_df[f'rolling_min{window}'] = test_df['signal'].rolling(window).min()
    test_df[f'rolling_max{window}'] = test_df['signal'].rolling(window).max()
    a = (test_df['signal'] - test_df['rolling_min' + str(window)]) / (test_df['rolling_max' + str(window)] - test_df['rolling_min' + str(window)])
    test_df["norm" + str(window)] = a * (np.floor(test_df['rolling_max' + str(window)]) - np.ceil(test_df['rolling_min' + str(window)]))
    
test_df = test_df.fillna(test_df.mean())


# In[ ]:


test_df = test_df.drop('time', axis=1)


# In[ ]:


test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)


# In[ ]:


print(f'Shape of training dataset after feature extraction: {X_train.shape}')
print(f'Shape of test dataset after feature extraction: {test_df.shape}')
print(f'Shape of training labels: {Y_train.shape}')


# ### Defining and training the Model using Cross Validation

# In[ ]:


folds = 10
seed = 666

kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

models = []
for fold, (train_index, val_index) in enumerate(kf.split(X_train, Y_train)):
    x_train = X_train.iloc[train_index]
    x_val = X_train.iloc[val_index]
    
    y_train = Y_train.iloc[train_index]
    y_val = Y_train.iloc[val_index]
    
    rgsr = RandomForestRegressor(n_estimators=32, 
                                 oob_score=True, 
                                 n_jobs=-1,
                                 verbose=100,
                                 random_state=seed)
    rgsr.fit(x_train, y_train)
    
    score = rgsr.score(x_val, y_val)
    
    print(f'[{fold}] score: {score}')
    
    models.append(rgsr)


# ### Making Predictions

# In[ ]:


predictions = sum([model.predict(test_df) for model in tqdm(models, total=folds)]) / folds
predictions


# In[ ]:


sample_sub_df['open_channels'] = predictions
sample_sub_df['open_channels'] = sample_sub_df['open_channels'].apply(lambda x: int(x))  # Converting into int 
sample_sub_df.loc[sample_sub_df['open_channels'] < 0, 'open_channels'] = 0  # Clipping the -ve values
sample_sub_df.to_csv('submission.csv', index=False, float_format='%.4f')
sample_sub_df.head()


# In[ ]:


sample_sub_df['open_channels'].unique()

