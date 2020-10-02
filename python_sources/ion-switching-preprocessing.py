#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from pykalman import KalmanFilter
from scipy import signal


# ## **0. Introduction**
# Competition data has only three features `signal`, `time` and `open_channels`. `open_channels` is the target and `signal` is the only predictor. It is a very small dataset in terms of features and samples.
# 
# > **IMPORTANT**: While the time series appears continuous, the data is from discrete batches of 50 seconds long 10 kHz samples (500,000 rows per batch). In other words, the data from 0.0001 - 50.0000 is a different batch than 50.0001 - 100.0000, and thus discontinuous between 50.0000 and 50.0001.
# 
# This is a very important detail to consider. Every **500,000** samples are coming from different batches. Data analysis and feature engineering should be done on batches independently.
# 
# Competition test data is split by 3/7. First **30%** is the public test set and the other **70%** is the private test set. It means the public leaderboard scores are based on entire **Test Batch 0** and first 1/5 of **Test Batch 1**.

# In[ ]:


df_train = pd.read_csv('../input/liverpool-ion-switching/train.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels': np.uint8})
df_test = pd.read_csv('../input/liverpool-ion-switching/test.csv', dtype={'time': np.float32, 'signal': np.float32})

print('Training Set Shape = {}'.format(df_train.shape))
print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
print('Training Set Batches = {}'.format(int(len(df_train) / 500000)))
print('Test Set Shape = {}'.format(df_test.shape))
print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))
print('Test Set Batches = {}'.format(int(len(df_test) / 500000)))


# In[ ]:


BATCH_SIZE = 500000

for i in range(10):
    df_train.loc[i * BATCH_SIZE:((i + 1) * BATCH_SIZE) - 1, 'batch'] = i
    
for i in range(4):
    df_test.loc[i * BATCH_SIZE:((i + 1) * BATCH_SIZE) - 1, 'batch'] = i
    
df_train['batch'] = df_train['batch'].astype(np.uint8)
df_test['batch'] = df_test['batch'].astype(np.uint8)


# ## **1. Target (Ion Channels)**
# > Many diseases, including cancer, are believed to have a contributing factor in common. **Ion channels** are pore-forming proteins present in animals and plants. They encode learning and memory, help fight infections, enable pain signals, and stimulate muscle contraction.
# 
# `open_channels` is the open ion channels at the given time (**0.0001 seconds**). There are **11** classes (from 0 to 10) to predict. Unique `open_channels` values are different in every batch so the rules learned from a batch may not generalize to another batch. This can be both regression or classification problem. However, regression models have worked better so far.

# ### **1.1. Global Target Distribution**
# Classes are not balanced in training set. Most of the time, when the `open_channels` increases, its value count decreases, but not necessarily. Higher `open_channels` values are less common. **0** is the most common class and **10** is the least common class.

# In[ ]:


fig = plt.figure(figsize=(15, 7))
sns.barplot(x=df_train['open_channels'].value_counts().index, y=df_train['open_channels'].value_counts().values)

plt.xlabel('Open Ion Channels', size=15, labelpad=20)
plt.ylabel('Value Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.title('Open Ion Channels Value Counts in Training Set', size=15)

plt.show()


# ### **1.2. Batch Target Distribution**
# Batches have very different target distributions as seen below. Every class from **0** to **10** doesn't exist in every batch. However, every batch is almost identical with another batch in terms of target distribution. Those similar batch pairs might be from the same sample.
# 
# * **Training Batch 0** and **Training Batch 1** have the same class distribution. They both have extremely high number of **0**  `open_channels` and low number of **1** `open_channels`.
# * **Training Batch 2** and **Training Batch 6** have the same class distribution. They both have high number of **1** `open_channels` and low number of **0** `open_channels`.
# * **Training Batch 3** and **Training Batch 7** have the same class distribution. They both almost have the same number of **0**, **1**, **2**, **3** `open_channels`.
# * **Training Batch 5** and **Training Batch 8** have the same class distribution. They both almost have the same number of **0**, **1**, **2**, **3**, **4**, **5** `open_channels`.
# * **Training Batch 4** and **Training Batch 9** have the same class distribution. They both almost have the same number of **1**, **2**, **3**, **4**, **5**, **6**, **7**, **8**, **9**, **10** `open_channels`. The only difference between those two batches is there are 2 **0** `open_channels` in **Training Batch 4** which doesn't exist in **Training Batch 9**.

# In[ ]:


training_batches = df_train.groupby('batch')

fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(20, 20), dpi=100)
for i, batch in training_batches:
    ax = plt.subplot(5, 2, i + 1) 
    
    sns.barplot(x=batch['open_channels'].value_counts().index, y=batch['open_channels'].value_counts().values)
    
    plt.xlabel('')
    plt.ylabel('')
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    ax.set_title(f'Batch {i} ({batch.index.min()}-{batch.index.max()}) Target Distribution', size=15)
    
plt.tight_layout()
plt.show()


# ## **2. Noise Types**

# ### **2.1. Outliers**
# There are very few outliers in the entire dataset. They can be seen on **Training Batch 0**, **Training Batch 7**, **Test Batch 2** and **Test Batch 3**. Those outliers could be dangerous since lots of them are in private test set. It is very hard to predict class thresholds because of the outliers, so they have to be dealt with.

# ### **2.2. Decimal Noise**
# `signal` is written with **4** decimal places in dataset. Random noise might be applied to decimal places of `signal`. Rounding `signal` could be useful in some models.

# ### **2.3. Ramp Transformation**
# Ramp transformation function is defined as $\mathcal{R}(X) = X + ((r \times 3) \div N)$
# 
# * $X$ is the given `signal` vector with size $N$
# * $r$ is a straight line with size $N$
# * $3$ is the slope of straight line, it is a constant because all of the ramp transformations are using $3$ for slope in entire dataset
# 
# Ramp transformation can be seen on several batches. Instead of entire batches, this transformation is applied to **10** second time frames. It can be seen on **Training Batch 1 first 100,000 samples**, **Test Batch 0 first, second and fifth 100,000 samples**, **Test Batch 1 second third and forth 100,000 samples**.
# 
# This transformation looks very unnatural because it always starts from and ends at multiples of 10. That's why it has to be removed in order to make the distribution smoother. Ramp removal function is basically the same function above with subtraction instead of addition.

# In[ ]:


def remove_ramp(X, constant=3):
    r = np.arange(len(X))        
    return X - ((r * constant) / len(X))


# ### **2.4. Sine Transformation**
# Sine transformation function is defined as $\mathcal{S}(X) = X + (5 \times \sin(s \times \pi \div N))$
# 
# * $X$ is the given `signal` vector with size $N$
# * $s$ is a straight line with size $N$
# * $5$ is the constant multiplied with $sin$
# 
# Sine transformation can be seen on several batches. Instead of **10** second time frames, this transformation is applied to entire batches. It can be seen on **Training Batch 6**, **Training Batch 7**, **Training Batch 8**, **Training Batch 9**, **Test Batch 2**.
# 
# This transformation looks very unnatural because it looks like a seasonal trend. It messes up the ranges of `open_channels` and it has to be removed in order to make the distribution smoother. Sine removal function is basically the same function above with subtraction instead of addition.

# In[ ]:


def remove_sine(X, constant=5):
    s = np.arange(len(X))        
    return X - (constant * (np.sin(np.pi * s / len(X))))


# ## **3. Batches**
# Every batch is unique, and have its own trends, characteristics and anomalies even though they have similar target distribution. They have to be analyzed separately. Necessary preprocessing steps can be decided more accurately that way. Filtering, normalizing and scaling should be done on batch level.
# 
# Every batch have one thing in common. If the noises are ignored, `signal` and `open_channels` have same number of levels in every batch of training set. There are concurrent spikes in `signal` and `open_channels`. That's why correlation between those two features is very high.
# 
# After the `signal` is cleaned, it becomes easier to count levels in batches. By counting the levels in a batch, the lower and upper bounds can be predicted easily. Predicted `open_channels` values can be clipped according to lower and upper bounds of the current batch. This postprocessing operation is going to help labeling the classes more accurately.
# 
# `signal_processed` feature is created for keeping both raw and processed `signal`. All changes made during the preprocessing will be written to `signal_processed`.

# In[ ]:


df_train['signal_processed'] = df_train['signal'].copy()
df_test['signal_processed'] = df_test['signal'].copy()


# `report_training_batch` and `report_test_batch` helper functions are made for visualizing the `signal` and `open_channels` on `time`. They also print descriptive statistics such as:
# * unique `open_channels` values
# * correlation between `signal` and `open_channels`
# * `signal` and `open_channels` mean
# * `signal` and `open_channels` std
# * `signal` range of every `open_channels` value

# In[ ]:


def report_training_batch(df, feature, batch):
    
    print(f'Training Batch {batch} - Unique Open Channel Values = {df[df["batch"] == batch]["open_channels"].unique()}')
    signal_openchannel_corr = np.corrcoef(df[df['batch'] == batch][feature], df[df['batch'] == batch]['open_channels'])[0][1]
    print(f'Training Batch {batch} - Correlation between Signal and Open Channels = {signal_openchannel_corr:.4}')
    print(f'Training Batch {batch} - Signal Mean = {df[df["batch"] == batch][feature].mean():.4} and Open Channels Mean = {df[df["batch"] == batch]["open_channels"].mean():.4}')
    print(f'Training Batch {batch} - Signal Std = {df[df["batch"] == batch][feature].std():.4} and Open Channels Std = {df[df["batch"] == batch]["open_channels"].std():.4}')
    print(f'Training Batch {batch} - Open Channels Range:')
    for value in df[df['batch'] == batch]['open_channels'].unique():
        print(f'                   Open Channels {value} - Min = {df.query("batch == @batch and open_channels == @value")[feature].min():.6} and Max = {df.query("batch == @batch and open_channels == @value")[feature].max():.6}')
    
    fig = plt.figure(figsize=(16, 6), dpi=100)
    df[df['batch'] == batch].set_index('time')[feature].plot(label='Signal')
    df[df['batch'] == batch].set_index('time')['open_channels'].plot(label='Open Channels')
        
    plt.xlabel('Time', size=15)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.legend()
    title = f'Training Batch {batch} ({df[df["batch"] == batch].index.min()}-{df[df["batch"] == batch].index.max()})'
    plt.title(title, size=15)
    
    plt.show()
    
def report_test_batch(df, feature, batch):
    
    print(f'Test Batch {batch} - Signal Mean = {df[df["batch"] == batch][feature].mean():.4}')
    print(f'Test Batch {batch} - Signal Std = {df[df["batch"] == batch][feature].std():.4}')
    
    fig = plt.figure(figsize=(16, 6), dpi=100)
    df[df['batch'] == batch].set_index('time')[feature].plot(label='Signal')
        
    plt.xlabel('Time', size=15)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.legend()
    title = f'Test Batch {batch} ({df[df["batch"] == batch].index.min()}-{df[df["batch"] == batch].index.max()})'
    plt.title(title, size=15)
    
    plt.show()
    


# ### **3.1. Training Batch 0**
# 
# **Training Batch 0** is the first batch of training set. It has a very simple distribution. There are lots of concurrent spikes in `signal` and `open_channels` which indicate the increases in target. `open_channels` and `signal` are correlated with each other because of this reason.
# 
# `open_channels` range is not accurate because there are couple outliers that are greater than **0** in `signal`. Either they can be removed or replaced with max `signal` value of its class (**0**).

# In[ ]:


report_training_batch(df_train, 'signal', 0)


# In[ ]:


outlier_idx = df_train.query('signal > 0 and batch == 0').index
batch0_target0_mean = df_train.drop(outlier_idx).query('batch == 0 and open_channels == 0')['signal'].mean()
df_train.loc[outlier_idx, 'signal_processed'] = batch0_target0_mean


# All outliers in **Training Batch 0** are **0** `open_channels`. Mean signal for **0** `open_channels` is calculated for **Training Batch 0** after dropping those samples, and `signal` of those samples are replaced with it. After dealing with the outliers, distribution looks more natural and correlation increased by **0.0006**.
# 
# ### Training Batch 0 Levels
# 
# There are **2** levels in **Training Batch 0**:
# * **0** `open_channels` are between **-3.8506** and **-0.8545**
# * **1** `open_channels` are between **-2.4036** and **-0.4163**

# In[ ]:


report_training_batch(df_train, 'signal_processed', 0)


# ### **3.2. Training Batch 1**
# **Training Batch 1** has a very simple distribution. There are lots of concurrent spikes in `signal` and `open_channels` which indicate the increases in target. 
# 
# `open_channels` and `signal` are not as highly correlated with each other as in **Training Batch 0** because there is a ramp at the beginning. `open_channels` range, `signal` mean and std are not accurate because of that ramp. Besides the ramp, there is no outlier in this batch.

# In[ ]:


report_training_batch(df_train, 'signal', 1)


# In[ ]:


batch1_slice = df_train.loc[df_train.query('time <= 60.0000 and batch == 1').index, 'signal']
df_train.loc[df_train.query('time <= 60.0000 and batch == 1').index, 'signal_processed'] = remove_ramp(batch1_slice)


# Removing the ramp transformation from first **100,000** samples of **Training Batch 1** increased the correlation from **0.3353** to **0.6902**.
# 
# ### Training Batch 1 Levels
# 
# There are **2** levels in **Training Batch 1**:
# * **0** `open_channels` are between **-3.9021** and **-1.53369**
# * **1** `open_channels` are between **-2.45985** and **-0.45438**

# In[ ]:


report_training_batch(df_train, 'signal_processed', 1)


# ### **3.3. Training Batch 2**
# In **Training Batch 2**, `open_channels` is extremely correlated with `signal`. There are very few **0** `open_channels` in **Training Batch 2**, that's why instead of concurrent spikes, there are vertical white lines.
# 
# Descriptive statistics are accurate because there are no outliers, trends or anomalies in **Training Batch 2**.

# In[ ]:


report_training_batch(df_train, 'signal', 2)


# 50X zoom is applied to `signal` between **143-144** to see the levels clearly. The boundaries of levels are from whole **Training Batch 2**, not from the slice.
# 
# ### Training Batch 2 Levels
# 
# There are **2** levels in **Training Batch 2**:
# * **0** `open_channels` are between **-3.9107** and **-1.4957**
# * **1** `open_channels` are between **-2.7586** and **-0.3525**

# In[ ]:


report_training_batch(df_train.loc[df_train.query('batch == 2 and 143 < time <= 144').index], 'signal', 2)


# ### **3.4. Training Batch 3**
# In **Training Batch 3**, `open_channels` is extremely correlated with `signal`. Instead of **2**, there are **4** levels in **Training Batch 3** unlike previous batches. Those levels represent every **4** unique `open_channels` values.
# 
# Descriptive statistics are accurate because there are no outliers, trends or anomalies in **Training Batch 3**.

# In[ ]:


report_training_batch(df_train, 'signal', 3)


# 50X zoom is applied to `signal` between **190-191** to see the levels clearly. The boundaries of levels are from whole **Training Batch 3**, not from the slice.
# 
# ### Training Batch 3 Levels
# 
# There are **4** levels in **Training Batch 3**:
# * **0** `open_channels` are between **-3.7073** and **-1.7493**
# * **1** `open_channels` are between **-2.7763** and **-0.3116**
# * **2** `open_channels` are between **-1.5271** and **0.9555**
# * **3** `open_channels` are between **-0.2341** and **2.2404**

# In[ ]:


report_training_batch(df_train.loc[df_train.query('batch == 3 and 190 < time < 191').index], 'signal', 3)


# ### **3.5. Training Batch 4**
# In **Training Batch 4**, `open_channels` is extremely correlated with `signal` and there are **11** unique values in it. It is the only batch with every unique `open_channels` value. Based on the examples, there has to be **11** `signal` and `open_channels` levels in **Training Batch 4**, but it is very hard to detect them without zooming. Extremely high correlation might be a proof of those **11** levels.
# 
# Descriptive statistics are accurate because there are no outliers, trends or anomalies in **Training Batch 4**

# In[ ]:


report_training_batch(df_train, 'signal', 4)


# 50X zoom is applied to `signal` between **239-240** to see the levels clearly but it still very hard to separate levels from each other. The boundaries of levels are from whole **Training Batch 4**, not from the slice.
# 
# ### Training Batch 4 Levels
# 
# There are **11** levels in **Training Batch 4**:
# * **0** `open_channels` are between **-5.7965** and **-5.7481**
# * **1** `open_channels` are between **-5.3438** and **-3.3162**
# * **2** `open_channels` are between **-4.6254** and **-1.6356**
# * **3** `open_channels` are between **-3.1726** and **-0.3517**
# * **4** `open_channels` are between **-2.1492** and **0.9949**
# * **5** `open_channels` are between **-0.9996** and **2.44**
# * **6** `open_channels` are between **0.1303** and **3.7926**
# * **7** `open_channels` are between **1.348** and **4.9947**
# * **8** `open_channels` are between **2.4193** and **6.0338**
# * **9** `open_channels` are between **3.9629** and **7.1794**
# * **10** `open_channels` are between **5.3285** and **8.6131**

# In[ ]:


report_training_batch(df_train.loc[df_train.query('batch == 4 and 239 < time <= 240').index], 'signal', 4)


# ### **3.6. Training Batch 5**
# In **Training Batch 5**, `open_channels` is extremely correlated with `signal` and there are **6** unique values in it. It is not hard to detect levels compared to previous example since there are few of them.
# 
# Descriptive statistics are accurate because there are no outliers, trends or anomalies in **Training Batch 5**

# In[ ]:


report_training_batch(df_train, 'signal', 5)


# 50X zoom is applied to `signal` between **295-296** to see the levels clearly. The boundaries of levels are from whole **Training Batch 5**, not from the slice.
# 
# ### Training Batch 5 Levels
# 
# There are **6** levels in **Training Batch 5**:
# * **0** `open_channels` are between **-3.8174** and **-1.7285**
# * **1** `open_channels` are between **-2.8902** and **-0.3566**
# * **2** `open_channels` are between **-1.5576** and **0.9042**
# * **3** `open_channels` are between **-0.2754** and **2.2316**
# * **4** `open_channels` are between **0.9769** and **3.4705**
# * **5** `open_channels` are between **2.1698** and **4.7929**

# In[ ]:


report_training_batch(df_train.loc[df_train.query('batch == 5 and 295 < time <= 296').index], 'signal', 5)


# ### **3.7. Training Batch 6**
# In **Training Batch 6**, `open_channels` is not correlated with `signal` because the entire batch is transformed with sine function. If the effects of that transformation are ignored, **Training Batch 6** has the same distribution with **Training Batch 2**. There are very few **0** `open_channels` in **Training Batch 6** that's why instead of concurrent spikes, there are vertical white lines. Those concurrent vertical white lines are not effected by sine transformation so it is safe to remove it. Besides the sine transformation, there is no outlier in this batch.

# In[ ]:


report_training_batch(df_train, 'signal', 6)


# In[ ]:


batch6 = df_train.loc[df_train.query('batch == 6').index]['signal']
df_train.loc[df_train.query('batch == 6').index, 'signal_processed'] = remove_sine(batch6)


# Removing the effects of sine transformation from `signal` cleans the entire **Training Batch 6**. Cleaning **Training Batch 6** increased the correlation from **0.319** to **0.9062**.
# 
# ### Training Batch 6 Levels
# 
# There are **2** levels in **Training Batch 6**:
# * **0** `open_channels` are between **-3.90221** and **-1.69979**
# * **1** `open_channels` are between **-2.60893** and **-0.285418**

# In[ ]:


report_training_batch(df_train, 'signal_processed', 6)


# ### **2.8. Training Batch 7**
# 
# In **Training Batch 7**, `open_channels` is not correlated with `signal` because the entire batch is transformed with sine function. Another problem is, middle part has higher standard deviation than other parts. If the effects of sine transformation is ignored and middle part is scaled, **Training Batch 7** has the same distribution with **Training Batch 3**. There are **4** unique `open_channels` in **Training Batch 7** just like **Training Batch 3**, and their `open_channels` ranges are very similar.

# In[ ]:


report_training_batch(df_train, 'signal', 7)


# In[ ]:


batch7 = df_train.loc[df_train.query('batch == 7').index]['signal']
df_train.loc[df_train.query('batch == 7').index, 'signal_processed'] = remove_sine(batch7)


# Removing the effects of sine transformation from `signal` doesn't clean the entire **Training Batch 7**. Removing sine transformation from **Training Batch 7** increased the correlation from **0.5126** to **0.8247**. **Training Batch 7** levels are not clear yet because of the outliers at the middle section.

# In[ ]:


report_training_batch(df_train, 'signal_processed', 7)


# It is not possible to filter every outlier by using `signal` values in **Training Batch 7**. Even though some of the samples are outliers, they can't be seen at first glance. It is better to filter entire middle part in which the outliers occur. Those samples are not dropped yet because models could perform better if filtered samples are assigned with lower sample weights. 
# 
# `is_filtered` feature is created for marking the filtered samples. The entire middle part between the first and the last outlier in **Training Batch 7** is filtered.

# In[ ]:


df_train['is_filtered'] = 0
df_train['is_filtered'] = df_train['is_filtered'].astype(np.uint8)
batch7_outlier_idx = pd.Int64Index(range(3641000, 3829000))
df_train.loc[batch7_outlier_idx, 'is_filtered'] = 1


# Dropping the outliers from **Training Batch 7** increased the correlation from **0.8247** to **0.9583** and `open_channels` ranges are almost back to normal.
# 
# ### Training Batch 7 Levels
# 
# There are **4** levels in **Training Batch 7**:
# * **0** `open_channels` are between **-3.80866** and **-1.59495**
# * **1** `open_channels` are between **-2.86314** and **0.146036**
# * **2** `open_channels` are between **-1.75331** and **1.01144**
# * **3** `open_channels` are between **-0.343348** and **-2.17848**

# In[ ]:


report_training_batch(df_train.drop(batch7_outlier_idx), 'signal_processed', 7)


# Another approach is replacing the outliers with the mean `signal_processed` calculated on **Training Batch 3** and **Training Batch 7** (except the noisy part) for every `open_channels`. Using raw means might be prone to overfitting so adding a small random noise is a better option.

# In[ ]:


open_channels0_mean = df_train[((df_train['batch'] == 3) | (df_train['batch'] == 7)) & (df_train['is_filtered'] == 0) & (df_train['open_channels'] == 0)]['signal_processed'].mean()
open_channels1_mean = df_train[((df_train['batch'] == 3) | (df_train['batch'] == 7)) & (df_train['is_filtered'] == 0) & (df_train['open_channels'] == 1)]['signal_processed'].mean()
open_channels2_mean = df_train[((df_train['batch'] == 3) | (df_train['batch'] == 7)) & (df_train['is_filtered'] == 0) & (df_train['open_channels'] == 2)]['signal_processed'].mean()
open_channels3_mean = df_train[((df_train['batch'] == 3) | (df_train['batch'] == 7)) & (df_train['is_filtered'] == 0) & (df_train['open_channels'] == 3)]['signal_processed'].mean()

df_train.loc[(df_train['is_filtered'] == 1) & (df_train['open_channels'] == 0), 'signal_processed'] = open_channels0_mean
df_train.loc[(df_train['is_filtered'] == 1) & (df_train['open_channels'] == 1), 'signal_processed'] = open_channels1_mean
df_train.loc[(df_train['is_filtered'] == 1) & (df_train['open_channels'] == 2), 'signal_processed'] = open_channels2_mean
df_train.loc[(df_train['is_filtered'] == 1) & (df_train['open_channels'] == 3), 'signal_processed'] = open_channels3_mean

batch7_filtered_part = df_train.loc[df_train['is_filtered'] == 1, 'signal_processed']
df_train.loc[df_train['is_filtered'] == 1, 'signal_processed'] = batch7_filtered_part + np.random.normal(0, 0.3, size=len(batch7_filtered_part)) 


# In[ ]:


report_training_batch(df_train, 'signal_processed', 7)


# ### **2.9. Training Batch 8**
# In **Training Batch 8**, `open_channels` is not correlated with `signal` because the entire batch is transformed with sine function. If the effects of that transformation are ignored, **Training Batch 8** has the same distribution with **Training Batch 5**. Besides the sine transformation, there is no outlier in this batch.

# In[ ]:


report_training_batch(df_train, 'signal', 8)


# In[ ]:


batch8 = df_train.loc[df_train.query('batch == 8').index]['signal']
df_train.loc[df_train.query('batch == 8').index, 'signal_processed'] = remove_sine(batch8)


# Removing the effects of sine transformation from `signal` cleans the entire **Training Batch 8**. Cleaning **Training Batch 8** increased the correlation from **0.6202** to **0.9742**.

# In[ ]:


report_training_batch(df_train, 'signal_processed', 8)


# 50X zoom is applied to `signal_processed` between **430-431** to see the levels clearly. The boundaries of levels are from whole **Training Batch 8**, not from the slice.
# 
# ### Training Batch 8 Levels
# 
# There are **6** levels in **Training Batch 8**:
# * **0** `open_channels` are between **-3.65724** and **-1.6207**
# * **1** `open_channels` are between **-2.55889** and **-0.412766**
# * **2** `open_channels` are between **-1.45358** and **0.921311**
# * **3** `open_channels` are between **-0.236957** and **2.196**
# * **4** `open_channels` are between **0.875905** and **3.4344**
# * **5** `open_channels` are between **2.22102** and **4.6432**

# In[ ]:


report_training_batch(df_train.loc[df_train.query('batch == 8 and 430 < time <= 431').index], 'signal_processed', 8)


# ### **2.10. Training Batch 9**
# In **Training Batch 9**, `open_channels` is not correlated with `signal` because the entire batch is transformed with sine function. If the effects of that transformation are ignored, **Training Batch 9** has the same distribution with **Training Batch 4**. Besides the sine transformation, there is no outlier in this batch.

# In[ ]:


report_training_batch(df_train, 'signal', 9)


# In[ ]:


batch9 = df_train.loc[df_train.query('batch == 9').index]['signal']
df_train.loc[df_train.query('batch == 9').index, 'signal_processed'] = remove_sine(batch9)


# Removing the effects of sine transformation from `signal` cleans the entire **Training Batch 9**. Cleaning **Training Batch 9** increased the correlation from **0.7469** to **0.9739**.

# In[ ]:


report_training_batch(df_train, 'signal_processed', 9)


# 50X zoom is applied to `signal` between **470-471** to see the levels clearly but it still very hard separate levels from each other. The boundaries of levels are from whole **Training Batch 9**, not from the slice.
# 
# ### Training Batch 9 Levels
# 
# There are **10** levels in **Training Batch 9**:
# * **1** `open_channels` are between **-4.99713** and **-3.1768**
# * **2** `open_channels` are between **-4.2639** and **-1.45995**
# * **3** `open_channels` are between **-3.4975** and **-0.473576**
# * **4** `open_channels` are between **-2.04313** and **1.04354**
# * **5** `open_channels` are between **-0.895961** and **2.26227**
# * **6** `open_channels` are between **0.251721** and **3.80713**
# * **7** `open_channels` are between **1.29101** and **4.95493**
# * **8** `open_channels` are between **2.68158** and **6.20222**
# * **9** `open_channels` are between **4.02257** and **7.29556**
# * **10** `open_channels` are between **5.02223** and **8.51555**

# In[ ]:


report_training_batch(df_train.loc[df_train.query('batch == 9 and 470 < time <= 471').index], 'signal_processed', 9)


# ### **2.10. Test Batch 0**
# **Test Batch 0** is the first batch of test set and 5/6 of public leaderboard. It is very different from training batches. Every **100,000** samples (**10** seconds) have different distributions. Distribution change in every precise **10** seconds doesn't look very natural. This could be the random noise applied by the organizers. Every **100,000** samples have to be analyzed and cleaned separately for **Test Batch 0**.

# In[ ]:


report_test_batch(df_test, 'signal', 0)


# In[ ]:


batch0_1 = df_test.loc[:100000 - 1, 'signal']
df_test.loc[:100000 - 1, 'signal_processed'] = remove_ramp(batch0_1)


# First **100,000** samples of **Test Batch 0** have ramp transformation. After the ramp is removed, levels can be seen clearly. The distribution looks very similar to **Training Batch 0** and **Training Batch 1**.
# 
# ### Test Batch 0 (1st 100,000) Levels
# 
# There are **2** levels in **Test Batch 0** first **100,000** samples:
# * Values between **-3.25** and **-2** are more likely to be **0** `open_channels` 
# * Spikes greater than **-1** are more likely to be **1** `open_channels`

# In[ ]:


report_test_batch(df_test.loc[:100000 - 1], 'signal_processed', 0)


# In[ ]:


batch0_2 = df_test.loc[100000:200000 - 1, 'signal']
df_test.loc[100000:200000 - 1, 'signal_processed'] = remove_ramp(batch0_2)


# Second **100,000** samples of **Test Batch 0** have ramp transformation. After the ramp is removed, levels can be seen clearly. The distribution looks very similar to **Training Batch 3** and **Training Batch 7**.
# 
# ### Test Batch 0 (2nd 100,000) Levels
# 
# There are **4** levels in **Test Batch 0** second **100,000** samples:
# * Values closer to **-3** are more likely to be **0** `open_channels` 
# * Values closer to **-2** are more likely to be **1** `open_channels`
# * Values between to **-1** are more likely to be **2** `open_channels`
# * Values greater than **1** are more likely to be **3** `open_channels`

# In[ ]:


report_test_batch(df_test.loc[100000:200000 - 1], 'signal_processed', 0)


# Third **100,000** samples of **Test Batch 0** doesn't have noise. The distribution looks very similar to **Training Batch 5** and **Training Batch 8**. Those two batches have at least **6** levels in them so it is very hard to detect them in test set without `open_channels`.
# 
# ### Test Batch 0 (3rd 100,000) Levels
# 
# There are **6** levels in **Test Batch 0** third **100,000** samples:
# * Values closer to **-3** are more likely to be **0** `open_channels` 
# * Values closer to **-2** are more likely to be **1** `open_channels`
# * Values closer to **-1** are more likely to be **2** `open_channels`
# * Values closer to **1** are more likely to be **3** `open_channels`
# * Values closer to **2** are more likely to be **4** `open_channels`
# * Values greater than **3** are more likely to be **5** `open_channels`

# In[ ]:


report_test_batch(df_test.loc[200000:300000 - 1], 'signal', 0)


# Fourth **100,000** samples of **Test Batch 0** might have outliers. The distribution is unique and it doesn't look like any batch from training set, but it could have the same levels with **Training Batch 3** and **Training Batch 7** or **Training Batch 0** and **Training Batch 1**.
# 
# ### Test Batch 0 (4th 100,000) Levels
# 
# There are **3** or **4** levels in **Test Batch 0** fourth **100,000** samples:
# * Values between **-3** and **-2** are more likely to be **0** `open_channels` 
# * Values closer to **-2** are more likely to be **1** `open_channels` 
# * Values closer to **-1** are more likely to be **2** `open_channels` or they are outliers
# * Values greater than **-1** are more likely to be **3** `open_channels` or they are outliers
# 

# In[ ]:


report_test_batch(df_test.loc[300000:400000 - 1], 'signal', 0)


# In[ ]:


batch0_5 = df_test.loc[400000:500000 - 1, 'signal']
df_test.loc[400000:500000 - 1, 'signal_processed'] = remove_ramp(batch0_5)


# Fifth **100,000** samples of **Test Batch 0** have ramp transformation. After the ramp is removed, levels can be seen clearly. The distribution looks very similar to **Training Batch 2** and **Training Batch 6** with more frequent vertical white lines.
# 
# ### Test Batch 0 (5th 100,000) Levels
# 
# There are **2** levels in **Test Batch 0** last **100,000** Samples:
# * Values smaller than **-2** are more likely to be **0** `open_channels` 
# * Values greater than **-2** are more likely to be **1** `open_channels`

# In[ ]:


report_test_batch(df_test.loc[400000:500000 - 1], 'signal_processed', 0)


# After all of the ramp transformations are removed from **Test Batch 0**, entire distribution looks cleaner. Every **10** second time frames are similar to different batches from training set. There might be outliers in fourth **100,000** samples of **Test Batch 0**, but it is hard to separate them from another level.

# In[ ]:


report_test_batch(df_test, 'signal_processed', 0)


# ### **2.11. Test Batch 1**
# **Test Batch 1** is the second batch of test set and its first **100,000** samples are the last 1/6 of public leaderboard. It is similar to **Test Batch 0** in terms of different distributions. Every **100,000** samples have to be analyzed and cleaned separately for **Test Batch 1**.

# In[ ]:


report_test_batch(df_test, 'signal', 1)


# First **100,000** samples of **Test Batch 1** doesn't have noise. The distribution looks very similar to **Training Batch 4** and **Training Batch 9**. Those two batches have at least **9** levels so it is very hard to detect them in test set.
# 
# ### Test Batch 1 (1st 100,000) Levels
# 
# There are **10** levels in **Test Batch 0** first **100,000** samples:
# * Values closer to **-5** are more likely to be **0** `open_channels` 
# * Values closer to **-4** are more likely to be **1** `open_channels`
# * Values closer to **-3** are more likely to be **2** `open_channels`
# * Values closer to **-2** are more likely to be **3** `open_channels`
# * Values closer to **-1** are more likely to be **4** `open_channels`
# * Values closer to **0** are more likely to be **5** `open_channels`
# * Values closer to **2** are more likely to be **6** `open_channels`
# * Values closer to **4** are more likely to be **7** `open_channels`
# * Values closer to **5** are more likely to be **8** `open_channels`
# * Values closer to **6** are more likely to be **9** `open_channels`
# * Values greater than **7** are more likely to be **10** `open_channels`

# In[ ]:


report_test_batch(df_test.loc[500000:600000 - 1], 'signal_processed', 1)


# In[ ]:


batch1_2 = df_test.loc[600000:700000 - 1, 'signal']
df_test.loc[600000:700000 - 1, 'signal_processed'] = remove_ramp(batch1_2)


# Second **100,000** samples of **Test Batch 1** have ramp transformation. After the ramp is removed, levels can be seen clearly. The distribution looks very similar to **Training Batch 5** and **Training Batch 8**.
# 
# ### Test Batch 1 (2nd 100,000) Levels
# 
# There are **6** levels in **Test Batch 0** second **100,000** samples:
# * Values smaller than **-3** are more likely to be **0** `open_channels` 
# * Values closer to **-2** are more likely to be **1** `open_channels`
# * Values closer to **-0.5** are more likely to be **2** `open_channels`
# * Values closer to **0.5** are more likely to be **3** `open_channels`
# * Values between **2** and **3** are more likely to be **4** `open_channels`
# * Values greater than **4** are more likely to be **5** `open_channels`

# In[ ]:


report_test_batch(df_test.loc[600000:700000 - 1], 'signal_processed', 1)


# In[ ]:


batch1_3 = df_test.loc[700000:800000 - 1, 'signal']
df_test.loc[700000:800000 - 1, 'signal_processed'] = remove_ramp(batch1_3)


# Third **100,000** samples of **Test Batch 1** have ramp transformation. After the ramp is removed, the distribution looks very similar to **Training Batch 4** and **Training Batch 9**. Those two batches have at least **9** levels so it is very hard to detect them in test set.
# 
# ### Test Batch 1 (3rd 100,000) Levels
# 
# There are **10** levels in **Test Batch 0** third **100,000** samples:
# * Values closer to **-4** are more likely to be **0** `open_channels` 
# * Values closer to **-3** are more likely to be **1** `open_channels`
# * Values closer to **-2** are more likely to be **2** `open_channels`
# * Values closer to **-1** are more likely to be **3** `open_channels`
# * Values closer to **1** are more likely to be **4** `open_channels`
# * Values closer to **2** are more likely to be **5** `open_channels`
# * Values closer to **3** are more likely to be **6** `open_channels`
# * Values closer to **4** are more likely to be **7** `open_channels`
# * Values closer to **5** are more likely to be **8** `open_channels`
# * Values closer to **6** are more likely to be **9** `open_channels`
# * Values greater than **7** are more likely to be **10** `open_channels`

# In[ ]:


report_test_batch(df_test.loc[700000:800000 - 1], 'signal_processed', 1)


# In[ ]:


batch1_4 = df_test.loc[800000:900000 - 1, 'signal']
df_test.loc[800000:900000 - 1, 'signal_processed'] = remove_ramp(batch1_4)


# Fourth **100,000** samples of **Test Batch 1** have ramp transformation. After the ramp is removed, the distribution looks very similar to **Test Batch 0** fourth **100,000** samples. The distribution is unique and it doesn't look like any batch from training set, but it could have the same levels with **Training Batch 3** and **Training Batch 7**.
# 
# ### Test Batch 1 (4th 100,000) Levels
# 
# There are **4** levels in **Test Batch 0** fourth **100,000** samples:
# * Values between **-3** and **-2** are more likely to be **0** `open_channels` 
# * Values closer to **-2** are more likely to be **1** `open_channels` 
# * Values closer to **-1** are more likely to be **2** `open_channels` 
# * Values greater than **-1** are more likely to be **3** `open_channels` 

# In[ ]:


report_test_batch(df_test.loc[800000:900000 - 1], 'signal_processed', 1)


# Fifth **100,000** samples of **Test Batch 1**  doesn't have noise. The distribution looks very similar to **Training Batch 5** and **Training Batch 8**.
# 
# ### Test Batch 1 (5th 100,000) Levels
# 
# There are **6** levels in **Test Batch 1** fifth **100,000** samples:
# * Values smaller than **-3** are more likely to be **0** `open_channels` 
# * Values closer to **-2** are more likely to be **1** `open_channels`
# * Values closer to **-1** are more likely to be **2** `open_channels`
# * Values closer to **0** are more likely to be **3** `open_channels`
# * Values closer to **1** are more likely to be **4** `open_channels`
# * Values greater than **1.5** are more likely to be **5** `open_channels`

# In[ ]:


report_test_batch(df_test.loc[900000:1000000 - 1], 'signal_processed', 1)


# After all of the ramp transformations are removed from **Test Batch 1**, entire distribution looks cleaner. Every **10** second time frames are similar to different batches from training set. There might be outliers in fourth **100,000** samples of **Test Batch 0**, but it is hard to separate them from another level.

# In[ ]:


report_test_batch(df_test, 'signal_processed', 1)


# ### **2.12. Test Batch 2**
# **Test Batch 2** is transformed with sine function. If the effects of that transformation are ignored, **Test Batch 2** has the same distribution **Test Batch 0**'s and **Test Batch 1**'s fourth **100,000** samples.

# In[ ]:


report_test_batch(df_test, 'signal', 2)


# In[ ]:


batch2 = df_test.loc[df_test.query('batch == 2').index]['signal']
df_test.loc[df_test.query('batch == 2').index, 'signal_processed'] = remove_sine(batch2)


# Removing the effects of sine transformation from `signal` cleans the entire **Test Batch 2**. The distribution is unique and it doesn't look like any batch from training set, but it could have the same levels with **Training Batch 3** and **Training Batch 7**. There might be outliers in **Test Batch 2**, but it is hard to separate them from another level.

# In[ ]:


report_test_batch(df_test, 'signal_processed', 2)


# 50X zoom is applied to `signal_processed` between **624-625** to see levels clearly. The assumptions of levels are based on entire **Test Batch 2**, not the slice.
# 
# 
# ### Test Batch 2 Levels
# 
# There are **3** or **4** levels in **Test Batch 2**:
# * Values between **-4** and **-2** are more likely to be **0** `open_channels` 
# * Values closer to **-2** are more likely to be **1** `open_channels` 
# * Values closer to **-1** are more likely to be **2** `open_channels` 
# * Values greater than **-1** are more likely to be **3** `open_channels`

# In[ ]:


report_test_batch(df_test.loc[df_test.query('batch == 2 and 624 < time <= 625').index], 'signal_processed', 2)


# ### **2.13. Test Batch 3**
# **Test Batch 3** is the last batch of test set. It doesn't have noise. **Test Batch 3** has the same distribution with **Test Batch 0**'s and **Test Batch 1**'s fourth **100,000** samples, and entire **Test Batch 2**.

# In[ ]:


report_test_batch(df_test, 'signal', 3)


# 50X zoom is applied to `signal_processed` between **681-682** to see levels clearly. The assumptions of levels are based on entire **Test Batch 2**, not the slice.
# 
# ### Test Batch 3 Levels
# 
# There are **3** or **4** levels in **Test Batch 3**:
# * Values between **-4** and **-2** are more likely to be **0** `open_channels` 
# * Values closer to **-2** are more likely to be **1** `open_channels` 
# * Values closer to **-1** are more likely to be **2** `open_channels` 
# * Values greater than **-1** are more likely to be **3** `open_channels`

# In[ ]:


report_test_batch(df_test.loc[df_test.query('batch == 3 and 681 < time <= 682').index], 'signal_processed', 3)


# ## **4. Models**

# ### **4.1. Model Classification**
# 
# After `signal` is processed, the distribution becomes smooth and easier to predict. Every batch pair in training set have different unique `open_channels` values, `signal_processed` mean, std and ranges. There are **5** different distribution types in training and and **6** in test set. One extra distribution doesn't exist in training set so **5** or **6** different models have to be created for those different distributions.
# 
# One extra distribution in test set is something between 0-1 and 0-1-2-3 distributions. It could be both 0-1, 0-1-2 and 0-1-2-3, but it is hard to tell without `open_channels`. It is classified as **Model 1.5**. It can be predicted with **Model 0**, **Model 2** or resampled **Model 2** without 3 `open_channels`.
# 
# | Model                        | Training      | Test                                                                                |
# |------------------------------|---------------|-------------------------------------------------------------------------------------|
# | 0-1 with 0s              | Batch 0 and 1 | Batch 0 1st 100,000 samples         |
# | 0-1 with 1s               | Batch 2 and 6 | Batch 0 5th 100,000 samples                                                       |
# | 0-1-2-3 or 0-1-2 |               | **Batch 0 4th 100,000 samples, Batch 1 4th 100,000 samples, Batch 2 and Batch 3** |
# | 0-1-2-3           | Batch 3 and 7 | Batch 0 2nd 100,000 samples and Batch 1 5th 100,000 samples                    |
# | 0-1-2-3-4-5                  | Batch 5 and 8 | Batch 0 3rd 100,000 samples and Batch 1 2nd 100,000 samples                    |
# | 0-1-2-3-4-5-6-7-8-9-10       | Batch 4 and 9 | Batch 1 1st and 3rd 100,000 samples                                             |

# In[ ]:


# model 0
model0_trn_idx = df_train.query('batch == 0 or batch == 1').index
model0_tst_idx = df_test.query('batch == 0 and (500 < time <= 510)').index

df_test.loc[model0_tst_idx, 'model'] = 0
df_train.loc[model0_trn_idx, 'model'] = 0

# model 1
model1_trn_idx = df_train.query('batch == 2 or batch == 6').index
model1_tst_idx = df_test.query('batch == 0 and (540 < time <= 550)').index

df_train.loc[model1_trn_idx, 'model'] = 1
df_test.loc[model1_tst_idx, 'model'] = 1

# model 1.5
model15_tst_idx = df_test.query('(batch == 0 and (530 < time <= 540)) or (batch == 1 and (580 < time <= 590)) or batch == 2 or batch == 3').index
df_test.loc[model15_tst_idx, 'model'] = 1.5

# model 2
model2_trn_idx = df_train.query('batch == 3 or batch == 7').index
model2_tst_idx = df_test.query('(batch == 0 and (510 < time <= 520)) or (batch == 1 and (590 < time <= 600))').index

df_train.loc[model2_trn_idx, 'model'] = 2
df_test.loc[model2_tst_idx, 'model'] = 2

# model 3
model3_trn_idx = df_train.query('batch == 5 or batch == 8').index
model3_tst_idx = df_test.query('(batch == 0 and (520 < time <= 530)) or (batch == 1 and (560 < time <= 570))').index

df_train.loc[model3_trn_idx, 'model'] = 3
df_test.loc[model3_tst_idx, 'model'] = 3

# model 4
model4_trn_idx = df_train.query('batch == 4 or batch == 9').index
model4_tst_idx = df_test.query('(batch == 1 and (550 < time <= 560)) or (batch == 1 and (570 < time <= 580))').index

df_train.loc[model4_trn_idx, 'model'] = 4
df_test.loc[model4_tst_idx, 'model'] = 4

for model in [0, 1, 1.5, 2, 3, 4]:
    print(f'\n---------- Model {model} ----------\n')
    for batch in df_train[df_train['model'] == model]['batch'].unique():
        model_signal_mean = df_train[(df_train['model'] == model) & (df_train['batch'] == batch)]['signal_processed'].mean()
        model_signal_std = df_train[(df_train['model'] == model) & (df_train['batch'] == batch)]['signal_processed'].std()
        model_signal_min = df_train[(df_train['model'] == model) & (df_train['batch'] == batch)]['signal_processed'].min()
        model_signal_max = df_train[(df_train['model'] == model) & (df_train['batch'] == batch)]['signal_processed'].max()
        print(f'Training Set Model {model} Batch {batch} signal_processed mean = {model_signal_mean:.4}, std = {model_signal_std:.4}, range = {model_signal_min:.4} - {model_signal_max:.4}')

    for batch in df_test[df_test['model'] == model]['batch'].unique():
        model_signal_mean = df_test[(df_test['model'] == model) & (df_test['batch'] == batch)]['signal_processed'].mean()
        model_signal_std = df_test[(df_test['model'] == model) & (df_test['batch'] == batch)]['signal_processed'].std()
        model_signal_min = df_test[(df_test['model'] == model) & (df_test['batch'] == batch)]['signal_processed'].min()
        model_signal_max = df_test[(df_test['model'] == model) & (df_test['batch'] == batch)]['signal_processed'].max()
        print(f'Test Set Model {model} Batch {batch} signal_processed mean = {model_signal_mean:.4}, std = {model_signal_std:.4}, range = {model_signal_min:.4} - {model_signal_max:.4}')
        
print('\n---------- Training Set Model Value Counts ----------\n')
print(df_train['model'].value_counts())
print('\n---------- Test Set Model Value Counts ----------\n')
print(df_test['model'].value_counts())


# In[ ]:


fig, axes = plt.subplots(nrows=2, figsize=(20, 14), dpi=100)

df_train.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[0])
for batch in np.arange(0, 550, 50):
    axes[0].axvline(batch, color='r', linestyle='--', lw=2)
    
df_test.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[1])

for batch in np.arange(500, 600, 10):
    axes[1].axvline(batch, color='r', linestyle='--', lw=2)
for batch in np.arange(600, 700, 50):
    axes[1].axvline(batch, color='r', linestyle='--', lw=2)
    
axes[1].axvline(560, color='y', linestyle='dotted', lw=8)

for i in range(2):    
    for batch in np.arange(0, 550, 50):
        axes[i].axvline(batch, color='r', linestyle='--', lw=2)
        
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', labelsize=15)
    axes[i].tick_params(axis='y', labelsize=15)
    axes[i].legend()
    
axes[0].set_title('Training Set Batches', size=18, pad=18)
axes[1].set_title('Public/Private Test Set Batches and Sub-batches', size=18, pad=18)

plt.show()


# ### **4.2. Ghost Drift**
# Even though `signal_processed` is smooth for every batch, there is an anomaly on model 4's distribution. Model 4's signal is negatively shifted. That distribution has different means than other groups for every `open_channels` value. It can be shifted back to normal by adding `np.exp(1)` to `signal_processed`.

# In[ ]:


SHIFT_CONSTANT = np.exp(1)

df_train.loc[df_train['model'] == 4, 'signal_processed'] += SHIFT_CONSTANT
df_test.loc[df_test['model'] == 4, 'signal_processed'] += SHIFT_CONSTANT


# In[ ]:


fig, axes = plt.subplots(nrows=2, figsize=(20, 14), dpi=100)

df_train.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[0])
for batch in np.arange(0, 550, 50):
    axes[0].axvline(batch, color='r', linestyle='--', lw=2)
    
df_test.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[1])

for batch in np.arange(500, 600, 10):
    axes[1].axvline(batch, color='r', linestyle='--', lw=2)
for batch in np.arange(600, 700, 50):
    axes[1].axvline(batch, color='r', linestyle='--', lw=2)
    
axes[1].axvline(560, color='y', linestyle='dotted', lw=8)

for i in range(2):    
    for batch in np.arange(0, 550, 50):
        axes[i].axvline(batch, color='r', linestyle='--', lw=2)
        
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', labelsize=15)
    axes[i].tick_params(axis='y', labelsize=15)
    axes[i].legend()
    
axes[0].set_title('Training Set Batches without Ghost Drift', size=18, pad=18)
axes[1].set_title('Public/Private Test Set Batches and Sub-batches without Ghost Drift', size=18, pad=18)

plt.show()


# ### **4.3. Periodic Noise**
# 
# First step of removing the periodic noise is calculating `signal_processed` mean for every `open_channels` value. In order to do that, entire **Batch 7** has to be removed from training set. Even though **Batch 7** looks clean, it can't be used for `signal_processed` mean calculation because the middle part was already replaced with mean values + random noise and it doesn't reflect the real data. 
# 
# Training set `signal_processed` is separated into two parts `signalA` (every batch except **Batch 7**) and `signalB` (**Batch 7**). Clean parts of **Batch 7** can be used but it makes it harder to split the `signal_processed` into equal pieces because there are less than 500000 samples in it. For the sake of simplicity, **Batch 7** is omitted and it will be denoised separately.
# 
# Finally, `signalC` is basically test set `signal_processed` and `channelsC` is test set predictions done by a high scoring model.

# In[ ]:


df_train['signal_processed_denoised'] = df_train['signal_processed'].copy(deep=True)
df_test['signal_processed_denoised'] = df_test['signal_processed'].copy(deep=True)

# Clean parts of training set signal
signalA = df_train[df_train['batch'] != 7]['signal_processed_denoised'].values
channelsA = df_train[df_train['batch'] != 7]['open_channels'].values

# Replacing a hidden outlier
signal1 = signalA[:1000000]
channels1 = channelsA[:1000000]
median = np.median(signal1[channels1 == 0])
condition = (signal1 > -1) & (channels1 == 0)
signal1[condition] = median
signalA[:1000000] = signal1

# Batch 7 first clean part and second clean part separated
signalB_good1 = df_train.loc[3_500_000:3_642_932 - 1]['signal_processed_denoised'].values
signalB_good2 = df_train.loc[3_822_753 + 1:4_000_000 - 1]['signal_processed_denoised'].values
channelsB_good1 = df_train.loc[3_500_000:3_642_932 - 1]['open_channels'].values
channelsB_good2 = df_train.loc[3_822_753 + 1:4_000_000 - 1]['open_channels'].values

# Test set signal and Bidirectional Viterbi predictions
signalC = df_test['signal_processed_denoised'].values
channelsC = pd.read_csv('../input/ion-switching-0945-predictions/gbdt_blend_submission.csv')['open_channels'].astype(np.uint8)


# Means of `signal_processed` for different `open_channels` have to be subtracted from itself in order to make the signal flat. Otherwise, FFT can't isolate periodicity without the normalization. It is easy to find means of different groups when batches are aligned. Means of `signal_processed` for every `open_channels` value for training set is listed below in an order. Those values can also be used for test set `signal_processed` normalization.

# In[ ]:


label = np.arange(len(signalA))
channel_list = np.arange(11)
n_list = np.empty(11)
mean_list = np.empty(11)
std_list = np.empty(11)
stderr_list = np.empty(11)

fig, axes = plt.subplots(ncols=2, figsize=(25, 8), dpi=100)

for i in range(11):
    x = label[channelsA == i]
    y = signalA[channelsA == i]
    n_list[i] = np.size(y)
    mean_list[i] = np.mean(y)
    std_list[i] = np.std(y)
    
    axes[0].plot(x, y, '.', markersize=0.5, alpha=0.02)    
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[0].set_title('Training Set Signal Processed Open Channels', size=18, pad=18)
    
stderr_list = std_list / np.sqrt(n_list)
sample_weight = 1 / stderr_list
channel_list = channel_list.reshape(-1, 1)

lr = LinearRegression()
lr.fit(channel_list, mean_list, sample_weight=sample_weight)
mean_predictA = lr.predict(channel_list)

x = np.linspace(-0.5, 10.5, 5)
y = lr.predict(x.reshape(-1, 1))
axes[1].plot(x, y, label='Predicted Means')
axes[1].plot(channel_list, mean_list, '.', markersize=8, label='Original Means')
axes[1].legend()

axes[1].tick_params(axis='x', labelsize=15)
axes[1].tick_params(axis='y', labelsize=15)
axes[1].set_title('Training Set Signal Processed Means', size=18, pad=18)

print('Predicted Means of signalA (Training Clean Signal):\n', mean_predictA)
plt.show()


# After the means are subtracted from every group, `signal_processed` is normalized and becomes flat. Test set doesn't have the labels so predictions of a strong model used for grouping.
# 
# Batches can still be identified by their lengths but `signal_processed` is on the same level for every batch. Training set **Batch 7** and test set have some spikes because it is harder normalize them.

# In[ ]:


def remove_target_mean(signal, target, means):
    signal_out = signal.copy()
    for i in range(11):
        signal_out[target == i] -= means[i]
    return signal_out

sig_A = remove_target_mean(signalA, channelsA, mean_predictA)
sig_B1 = remove_target_mean(signalB_good1, channelsB_good1, mean_predictA)
sig_B2 = remove_target_mean(signalB_good2, channelsB_good2, mean_predictA)
sig_C = remove_target_mean(signalC, channelsC, mean_predictA)


# In[ ]:


fig, axes = plt.subplots(nrows=3, figsize=(25, 20), dpi=100)
axes[0].plot(sig_A, linewidth=1)
axes[1].plot(np.hstack((sig_B1, sig_B2)), linewidth=1)
axes[2].plot(sig_C, linewidth=1)

for i in range(3):
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=12)
axes[0].set_title('Training Set (Without Batch 7) Signal Processed After Means are Subtracted', size=18, pad=18)
axes[1].set_title('Training Set Batch 7 (Without Noisy Part) Signal Processed After Means are Subtracted', size=18, pad=18)
axes[2].set_title('Test Set Signal Processed After Means are Subtracted', size=18, pad=18)

plt.show()


# In[ ]:


def bandstop(x, samplerate=1000000, fp=np.array([4925, 5075]), fs=np.array([4800, 5200])):    
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn
    gpass = 1
    gstop = 10.0

    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, 'bandstop')
    y = signal.filtfilt(b, a, x)
    return y

def bandpass(x, samplerate=1000000, fp=np.array([4925, 5075]), fs=np.array([4800, 5200])):
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn
    gpass = 1
    gstop = 10.0

    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "bandpass")
    y = signal.filtfilt(b, a, x)
    return y


# In[ ]:


train_normalized_signals = np.split(sig_A, 9)
train_original_signals = np.split(signalA, 9)
train_filtered_signals = []
train_supervised_noise = []

# Denoising training batches except Batch 7
for batch, original_signal in enumerate(train_original_signals):
    
    normalized_signal = train_normalized_signals[batch]    
    filtered_signal = bandstop(normalized_signal)
    noise = bandpass(normalized_signal)
    
    if batch >= 7:
        batch += 1
    
    plt.figure(figsize=(25, 5))
    plt.title(f'Open Channels Denormalized - Training Batch {batch}', size=18, pad=18)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.plot(normalized_signal, label='Original Signal', linewidth=0.5, alpha=0.5)
    plt.plot(filtered_signal, label = 'Filtered Signal', linewidth=0.5, alpha=0.5)
    plt.show()
    
    clean_signal = original_signal - noise
    plt.figure(figsize=(25, 5))
    plt.title(f'Signal Space - Training Batch {batch}', size=18, pad=18)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.plot(original_signal, linewidth=0.5, alpha=0.5)
    plt.plot(clean_signal, linewidth=0.5, alpha=0.5)
    plt.show()

    plt.figure(figsize=(25, 5))
    plt.title(f'Signal Space - Training Batch {batch}', size=18, pad=18)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.plot(original_signal, linewidth=0.5, alpha=0.5)
    plt.twinx()
    plt.plot(filtered_signal, linewidth=0.5, alpha=0.5, c='orange')
    plt.show()
   
    train_filtered_signals.append(clean_signal)
    train_supervised_noise.append(noise)
    
# Denoising Batch 7 Part 1
batch7_filtered_signal1 = bandstop(sig_B1)
batch7_noise1 = bandpass(sig_B1)

plt.figure(figsize=(25, 5))
plt.title(f'Open Channels Denormalized - Training Batch 7 Part 1', size=18, pad=18)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.plot(sig_B1, label='Batch 7 Normalized Signal Part 1', linewidth=0.5, alpha=0.5)
plt.plot(batch7_filtered_signal1, label = 'Batch 7 Filtered Signal Part 1', linewidth=0.5, alpha=0.5)
plt.show()
    
batch7_clean_signal1 = signalB_good1 - batch7_noise1
plt.figure(figsize=(25, 5))
plt.title(f'Batch 7 Signal Space - Part 1', size=18, pad=18)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.plot(signalB_good1, linewidth=0.5, alpha=0.5)
plt.plot(batch7_filtered_signal1, linewidth=0.5, alpha=0.5)
plt.show()

plt.figure(figsize=(25, 5))
plt.title(f'Batch 7 Signal Space - Part 1', size=18, pad=18)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.plot(signalB_good1, linewidth=0.5, alpha=0.5)
plt.twinx()
plt.plot(batch7_filtered_signal1, linewidth=0.5, alpha=0.5, c='orange')
plt.show()

# Denoising Batch 7 Part 2
batch7_filtered_signal2 = bandstop(sig_B2)
batch7_noise2 = bandpass(sig_B2)

plt.figure(figsize=(25, 5))
plt.title(f'Open Channels Denormalized - Training Batch 7 Part 2', size=18, pad=18)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.plot(sig_B2, label='Batch 7 Normalized Signal Part 2', linewidth=0.5, alpha=0.5)
plt.plot(batch7_filtered_signal2, label = 'Batch 7 Filtered Signal Part 2', linewidth=0.5, alpha=0.5)
plt.show()
    
batch7_clean_signal2 = signalB_good2 - batch7_noise2
plt.figure(figsize=(25, 5))
plt.title(f'Batch 7 Signal Space - Part 2', size=18, pad=18)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.plot(signalB_good2, linewidth=0.5, alpha=0.5)
plt.plot(batch7_filtered_signal2, linewidth=0.5, alpha=0.5)
plt.show()

plt.figure(figsize=(25, 5))
plt.title(f'Batch 7 Signal Space - Part 2', size=18, pad=18)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.plot(signalB_good2, linewidth=0.5, alpha=0.5)
plt.twinx()
plt.plot(batch7_filtered_signal2, linewidth=0.5, alpha=0.5, c='orange')
plt.show()
    
df_train.loc[df_train['batch'] == 0, 'signal_processed_denoised'] = train_filtered_signals[0]
df_train.loc[df_train['batch'] == 1, 'signal_processed_denoised'] = train_filtered_signals[1]
df_train.loc[df_train['batch'] == 2, 'signal_processed_denoised'] = train_filtered_signals[2]
df_train.loc[df_train['batch'] == 3, 'signal_processed_denoised'] = train_filtered_signals[3]
df_train.loc[df_train['batch'] == 4, 'signal_processed_denoised'] = train_filtered_signals[4]
df_train.loc[df_train['batch'] == 5, 'signal_processed_denoised'] = train_filtered_signals[5]
df_train.loc[df_train['batch'] == 6, 'signal_processed_denoised'] = train_filtered_signals[6]
df_train.loc[df_train['batch'] == 6, 'signal_processed_denoised'] = train_filtered_signals[6]
df_train.loc[3500000:3642932 - 1, 'signal_processed_denoised'] = batch7_clean_signal1
df_train.loc[3822753 + 1:4000_000 - 1, 'signal_processed_denoised'] = batch7_clean_signal2
df_train.loc[df_train['batch'] == 8, 'signal_processed_denoised'] = train_filtered_signals[7]
df_train.loc[df_train['batch'] == 9, 'signal_processed_denoised'] = train_filtered_signals[8]


# In[ ]:


test_normalized_signals1 = np.split(sig_C[:1000000], 10)
test_original_signals1 = np.split(signalC[:1000000], 10)
test_normalized_signal2 = sig_C[1000000:]
test_original_signal2 = signalC[1000000:]
test_filtered_signals = []
test_supervised_noise = []

# Denoising test set sub batches part by part
for sub_batch, original_signal in enumerate(test_original_signals1):
    
    normalized_signal = test_normalized_signals1[sub_batch]    
    filtered_signal = bandstop(normalized_signal)
    noise = bandpass(normalized_signal)
        
    plt.figure(figsize=(25, 5))
    plt.title(f'Open Channels Denormalized - Test Sub-Batch {sub_batch}', size=18, pad=18)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.plot(normalized_signal, label='Original Signal', linewidth=0.5, alpha=0.5)
    plt.plot(filtered_signal, label = 'Filtered Signal', linewidth=0.5, alpha=0.5)
    plt.show()
    
    clean_signal = original_signal - noise
    plt.figure(figsize=(25, 5))
    plt.title(f'Signal Space - Test Sub-Batch {sub_batch}', size=18, pad=18)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.plot(original_signal, linewidth=0.5, alpha=0.5)
    plt.plot(clean_signal, linewidth=0.5, alpha=0.5)
    plt.show()

    plt.figure(figsize=(25, 5))
    plt.title(f'Signal Space - Test Sub-Batch {sub_batch}', size=18, pad=18)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.plot(original_signal, linewidth=0.5, alpha=0.5)
    plt.twinx()
    plt.plot(filtered_signal, linewidth=0.5, alpha=0.5, c='orange')
    plt.show()
   
    test_filtered_signals.append(clean_signal)
    test_supervised_noise.append(noise)
        
# Denoising test set second half
test_filtered_signal2 = bandstop(test_normalized_signal2)
test_noise2 = bandpass(test_normalized_signal2)

plt.figure(figsize=(25, 5))
plt.title(f'Open Channels Denormalized - Test Batch 2 & 3 (Second Half)', size=18, pad=18)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.plot(test_normalized_signal2, label='Original Signal', linewidth=0.5, alpha=0.5)
plt.plot(test_filtered_signal2, label = 'Filtered Signal', linewidth=0.5, alpha=0.5)
plt.show()

test_clean_signal2 = test_original_signal2 - test_noise2
plt.figure(figsize=(25, 5))
plt.title(f'Signal Space - Test Batch 2 & 3 (Second Half)', size=18, pad=18)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.plot(test_original_signal2, linewidth=0.5, alpha=0.5)
plt.plot(test_clean_signal2, linewidth=0.5, alpha=0.5)
plt.show()

plt.figure(figsize=(25, 5))
plt.title(f'Signal Space - Test Batch 2 & 3 (Second Half)', size=18, pad=18)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.plot(test_original_signal2, linewidth=0.5, alpha=0.5)
plt.twinx()
plt.plot(test_filtered_signal2, linewidth=0.5, alpha=0.5, c='orange')
plt.show()

df_test.loc[0:100000 - 1, 'signal_processed_denoised'] = test_filtered_signals[0]
df_test.loc[100000:200000 - 1, 'signal_processed_denoised'] = test_filtered_signals[1]
df_test.loc[200000:300000 - 1, 'signal_processed_denoised'] = test_filtered_signals[2]
df_test.loc[300000:400000 - 1, 'signal_processed_denoised'] = test_filtered_signals[3]
df_test.loc[400000:500000 - 1, 'signal_processed_denoised'] = test_filtered_signals[4]
df_test.loc[500000:600000 - 1, 'signal_processed_denoised'] = test_filtered_signals[5]
df_test.loc[600000:700000 - 1, 'signal_processed_denoised'] = test_filtered_signals[6]
df_test.loc[700000:800000 - 1, 'signal_processed_denoised'] = test_filtered_signals[7]
df_test.loc[800000:900000 - 1, 'signal_processed_denoised'] = test_filtered_signals[8]
df_test.loc[900000:1000000 - 1, 'signal_processed_denoised'] = test_filtered_signals[9]
df_test.loc[1000000:, 'signal_processed_denoised'] = test_clean_signal2


# In[ ]:


fig, axes = plt.subplots(nrows=2, figsize=(20, 14), dpi=100)

df_train.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[0], alpha=0.5)
df_train.set_index('time')['signal_processed_denoised'].plot(label='Signal Denoised', ax=axes[0], alpha=0.5)
for batch in np.arange(0, 550, 50):
    axes[0].axvline(batch, color='r', linestyle='--', lw=2)
    
df_test.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[1], alpha=0.5)
df_test.set_index('time')['signal_processed_denoised'].plot(label='Signal Denoised', ax=axes[1], alpha=0.5)

for batch in np.arange(500, 600, 10):
    axes[1].axvline(batch, color='r', linestyle='--', lw=2)
for batch in np.arange(600, 700, 50):
    axes[1].axvline(batch, color='r', linestyle='--', lw=2)
    
axes[1].axvline(560, color='y', linestyle='dotted', lw=8)

for i in range(2):    
    for batch in np.arange(0, 550, 50):
        axes[i].axvline(batch, color='r', linestyle='--', lw=2)
        
    axes[i].set_xlabel('Time', size=15)
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=12)
    axes[i].legend()
    
axes[0].set_title('Training Set Batches Processed and Denoised', size=18, pad=18)
axes[1].set_title('Public/Private Test Set Batches and Sub-batches Processed and Denoised', size=18, pad=18)

plt.show()


# ## **5. Scaling and Filtering**

# ### **5.1. Time Scaled**
# In some batches, `signal` is affected by `time`. That effect is clearly visible on **Batch 6**, **Batch 7**, **Batch 8**, **Batch 9** in training set, and **Batch 2** in test set, but their `signal` is cleaned. `time` can't be used as a predictor by itself because of the covariance shift, but it can be scaled on `batch` level. The values of `time` will be between **0** and **1** when it is scaled.

# In[ ]:


scaler = MinMaxScaler()

for i, batch in enumerate(df_train.groupby('batch')):
    time = df_train.loc[(df_train['batch'] == i), 'time'].values.reshape(-1, 1)
    df_train.loc[(df_train['batch'] == i), 'time_scaled'] = scaler.fit_transform(time)

for i, batch in enumerate(df_test.groupby('batch')):
    time = df_test.loc[(df_test['batch'] == i), 'time'].values.reshape(-1, 1)
    df_test.loc[(df_test['batch'] == i), 'time_scaled'] = scaler.fit_transform(time)
    
df_train['time_scaled'] = df_train['time_scaled'].astype(np.float32)
df_test['time_scaled'] = df_test['time_scaled'].astype(np.float32)


# ### **5.2. Kalman Filter**
# Kalman filter is applied to `signal_processed` in order to reduce the noise, but it is applied separately to every different signal. For training set, it is applied to every batch separately, and for test set, it is applied to every sub batch separately.
# 
# Every `signal_covariance` is tuned based on the filtered `signal_processed` and `open_channels` correlation. Mean of covariances returning the highest correlation in training set, are used for the test set signals.

# In[ ]:


def kalman(signal, signal_covariance):
    
    kf = KalmanFilter(initial_state_mean=signal[0], 
                      initial_state_covariance=signal_covariance,
                      observation_covariance=signal_covariance, 
                      transition_covariance=0.1,
                      transition_matrices=1)
    
    pred_state, state_cov = kf.smooth(signal)
    return pred_state


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# filter model 0\nprint(f'\\n---------- Model 0 ----------\\n')\n\nbatch0_corr = np.corrcoef(df_train[df_train['batch'] == 0]['signal_processed'], df_train[df_train['batch'] == 0]['open_channels'])[0][1]\ndf_train.loc[df_train['batch'] == 0, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 0]['signal_processed'].values, 0.6)\nfiltered_batch0_corr = np.corrcoef(df_train[df_train['batch'] == 0]['signal_processed_kalman'], df_train[df_train['batch'] == 0]['open_channels'])[0][1]\nprint(f'Training Batch 0 - Correlation between Signal and Open Channels increased from {batch0_corr:.6} to {filtered_batch0_corr:.6} (Covariance: {0.6})')\n\nbatch1_corr = np.corrcoef(df_train[df_train['batch'] == 1]['signal_processed'], df_train[df_train['batch'] == 1]['open_channels'])[0][1]\ndf_train.loc[df_train['batch'] == 1, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 1]['signal_processed'].values, 0.45)\nfiltered_batch1_corr = np.corrcoef(df_train[df_train['batch'] == 1]['signal_processed_kalman'], df_train[df_train['batch'] == 1]['open_channels'])[0][1]\nprint(f'Training Batch 1 - Correlation between Signal and Open Channels increased from {batch1_corr:.6} to {filtered_batch1_corr:.6} (Covariance: {0.45})')\n\nprint(f'Test Batch 0 Sub-batch 0 (Mean Model 0 Covariance: {0.525})')\ndf_test.loc[df_test.query('batch == 0 and (500 < time <= 510)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 0 and (500 < time <= 510)').index, 'signal_processed'].values, 0.525)\n\n# filter model 1\nprint(f'\\n---------- Model 1 ----------\\n')\n\nbatch2_corr = np.corrcoef(df_train[df_train['batch'] == 2]['signal_processed'], df_train[df_train['batch'] == 2]['open_channels'])[0][1]\ndf_train.loc[df_train['batch'] == 2, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 2]['signal_processed'].values, 0.03)\nfiltered_batch2_corr = np.corrcoef(df_train[df_train['batch'] == 2]['signal_processed_kalman'], df_train[df_train['batch'] == 2]['open_channels'])[0][1]\nprint(f'Training Batch 2 - Correlation between Signal and Open Channels increased from {batch2_corr:.6} to {filtered_batch2_corr:.6} (Covariance: {0.03})')\n\nbatch6_corr = np.corrcoef(df_train[df_train['batch'] == 6]['signal_processed'], df_train[df_train['batch'] == 6]['open_channels'])[0][1]\ndf_train.loc[df_train['batch'] == 6, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 6]['signal_processed'].values, 0.03)\nfiltered_batch6_corr = np.corrcoef(df_train[df_train['batch'] == 6]['signal_processed_kalman'], df_train[df_train['batch'] == 6]['open_channels'])[0][1]\nprint(f'Training Batch 6 - Correlation between Signal and Open Channels increased from {batch6_corr:.6} to {filtered_batch6_corr:.6} (Covariance: {0.03})')\n\nprint(f'Test Batch 0 Sub-batch 4 (Mean Model 1 Covariance: {0.03})')\ndf_test.loc[df_test.query('batch == 0 and (540 < time <= 550)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 0 and (540 < time <= 550)').index, 'signal_processed'].values, 0.03)\n\n# filter model 1.5\nprint(f'\\n---------- Model 1.5 ----------\\n')\n\nprint(f'Test Batch 0 Sub-batch 3 (Covariance: {0.1})')\ndf_test.loc[df_test.query('batch == 0 and (530 < time <= 540)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 0 and (530 < time <= 540)').index, 'signal_processed'].values, 0.1)\n\nprint(f'Test Batch 1 Sub-batch 3 (Covariance: {0.1})')\ndf_test.loc[df_test.query('batch == 1 and (580 < time <= 590)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 1 and (580 < time <= 590)').index, 'signal_processed'].values, 0.1)\n\nprint(f'Test Batch 2 (Covariance: {0.1})')\ndf_test.loc[df_test.query('batch == 2').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 2').index, 'signal_processed'].values, 0.1)\nprint(f'Test Batch 3 (Covariance: {0.1})')\ndf_test.loc[df_test.query('batch == 3').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 3').index, 'signal_processed'].values, 0.1)\n\n# filter model 2\nprint(f'\\n---------- Model 2 ----------\\n')\n\nbatch3_corr = np.corrcoef(df_train[df_train['batch'] == 3]['signal_processed'], df_train[df_train['batch'] == 3]['open_channels'])[0][1]\ndf_train.loc[df_train['batch'] == 3, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 3]['signal_processed'].values, 0.01)\nfiltered_batch3_corr = np.corrcoef(df_train[df_train['batch'] == 3]['signal_processed_kalman'], df_train[df_train['batch'] == 3]['open_channels'])[0][1]\nprint(f'Training Batch 3 - Correlation between Signal and Open Channels increased from {batch3_corr:.6} to {filtered_batch3_corr:.6} (Covariance: {0.01})')\n\nbatch7_corr = np.corrcoef(df_train[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1)]['signal_processed'], df_train[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1)]['open_channels'])[0][1]\ndf_train.loc[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1), 'signal_processed_kalman'] = kalman(df_train[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1)]['signal_processed'].values, 0.01)\nfiltered_batch7_corr = np.corrcoef(df_train[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1)]['signal_processed_kalman'], df_train[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1)]['open_channels'])[0][1]\nprint(f'Training Batch 7 - Correlation between Signal and Open Channels increased from {batch7_corr:.6} to {filtered_batch7_corr:.6} (Covariance: {0.01})')\n\nprint(f'Test Batch 0 Sub-batch 1 (Mean Model 2 Covariance: {0.01})')\ndf_test.loc[df_test.query('batch == 0 and (510 < time <= 520)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 0 and (510 < time <= 520)').index, 'signal_processed'].values, 0.01)\nprint(f'Test Batch 1 Sub-batch 4 (Mean Model 2 Covariance: {0.01})')\ndf_test.loc[df_test.query('batch == 1 and (590 < time <= 600)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 1 and (590 < time <= 600)').index, 'signal_processed'].values, 0.01)\n\n# filter model 3\nprint(f'\\n---------- Model 3 ----------\\n')\n\nbatch5_corr = np.corrcoef(df_train[df_train['batch'] == 5]['signal_processed'], df_train[df_train['batch'] == 5]['open_channels'])[0][1]\ndf_train.loc[df_train['batch'] == 5, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 5]['signal_processed'].values, 0.005)\nfiltered_batch5_corr = np.corrcoef(df_train[df_train['batch'] == 5]['signal_processed_kalman'], df_train[df_train['batch'] == 5]['open_channels'])[0][1]\nprint(f'Training Batch 5 - Correlation between Signal and Open Channels increased from {batch5_corr:.6} to {filtered_batch5_corr:.6} (Covariance: {0.005})')\n\nbatch8_corr = np.corrcoef(df_train[df_train['batch'] == 8]['signal_processed'], df_train[df_train['batch'] == 8]['open_channels'])[0][1]\ndf_train.loc[df_train['batch'] == 8, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 8]['signal_processed'].values, 0.005)\nfiltered_batch8_corr = np.corrcoef(df_train[df_train['batch'] == 8]['signal_processed_kalman'], df_train[df_train['batch'] == 8]['open_channels'])[0][1]\nprint(f'Training Batch 8 - Correlation between Signal and Open Channels increased from {batch8_corr:.6} to {filtered_batch8_corr:.6} (Covariance: {0.005})')\n\nprint(f'Test Batch 0 Sub-batch 2 - (Mean Model 3 Covariance: {0.005})')\ndf_test.loc[df_test.query('batch == 0 and (520 < time <= 530)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 0 and (520 < time <= 530)').index, 'signal_processed'].values, 0.005)\nprint(f'Test Batch 1 Sub-batch 1 - (Mean Model 3 Covariance: {0.005})')\ndf_test.loc[df_test.query('batch == 1 and (560 < time <= 570)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 1 and (560 < time <= 570)').index, 'signal_processed'].values, 0.005)\n\n# filter model 4\nprint(f'\\n---------- Model 4 ----------\\n')\n\nbatch4_corr = np.corrcoef(df_train[df_train['batch'] == 4]['signal_processed'], df_train[df_train['batch'] == 4]['open_channels'])[0][1]\ndf_train.loc[df_train['batch'] == 4, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 4]['signal_processed'].values, 0.005)\nfiltered_batch4_corr = np.corrcoef(df_train[df_train['batch'] == 4]['signal_processed_kalman'], df_train[df_train['batch'] == 4]['open_channels'])[0][1]\nprint(f'Training Batch 4 - Correlation between Signal and Open Channels increased from {batch4_corr:.6} to {filtered_batch4_corr:.6} (Covariance: {0.005})')\n\nbatch9_corr = np.corrcoef(df_train[df_train['batch'] == 9]['signal_processed'], df_train[df_train['batch'] == 9]['open_channels'])[0][1]\ndf_train.loc[df_train['batch'] == 9, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 9]['signal_processed'].values, 0.005)\nfiltered_batch9_corr = np.corrcoef(df_train[df_train['batch'] == 9]['signal_processed_kalman'], df_train[df_train['batch'] == 9]['open_channels'])[0][1]\nprint(f'Training Batch 9 - Correlation between Signal and Open Channels increased from {batch9_corr:.6} to {filtered_batch9_corr:.6} (Covariance: {0.005})')\n\nprint(f'Test Batch 1 Sub-batch 0 - (Mean Model 4 Covariance: {0.005})')\ndf_test.loc[df_test.query('batch == 1 and (550 < time <= 560)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 1 and (550 < time <= 560)').index, 'signal_processed'].values, 0.005)\nprint(f'Test Batch 1 Sub-batch 2 - (Mean Model 4 Covariance: {0.005})')\ndf_test.loc[df_test.query('batch == 1 and (570 < time <= 580)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 1 and (570 < time <= 580)').index, 'signal_processed'].values, 0.005)")


# In[ ]:


fig, axes = plt.subplots(nrows=2, figsize=(20, 14), dpi=100)

df_train.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[0], alpha=0.4)
df_train.set_index('time')['signal_processed_kalman'].plot(label='Signal Kalman Filtered', ax=axes[0], alpha=0.8)
for batch in np.arange(0, 550, 50):
    axes[0].axvline(batch, color='r', linestyle='--', lw=2)
    
df_test.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[1], alpha=0.4)
df_test.set_index('time')['signal_processed_kalman'].plot(label='Signal Kalman Filtered', ax=axes[1], alpha=0.8)

for batch in np.arange(500, 600, 10):
    axes[1].axvline(batch, color='r', linestyle='--', lw=2)
for batch in np.arange(600, 700, 50):
    axes[1].axvline(batch, color='r', linestyle='--', lw=2)
    
axes[1].axvline(560, color='y', linestyle='dotted', lw=8)

for i in range(2):    
    for batch in np.arange(0, 550, 50):
        axes[i].axvline(batch, color='r', linestyle='--', lw=2)
        
    axes[i].set_xlabel('Time', size=15)
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=12)
    axes[i].legend()
    
axes[0].set_title('Training Set Batches Raw/Filtered', size=18, pad=18)
axes[1].set_title('Public/Private Test Set Batches and Sub-batches Raw/Filtered', size=18, pad=18)

plt.show()


# ## **6. Conclusion**

# In[ ]:


df_train.to_pickle('train.pkl')
df_test.to_pickle('test.pkl')

print('Training Set Shape = {}'.format(df_train.shape))
print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
print('Test Set Shape = {}'.format(df_test.shape))
print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))

