#!/usr/bin/env python
# coding: utf-8

# # Code overview 
# This code give various way to do the data visualization (exploratory data analysis), need to implement noise reduce, batching, data handle. 
# Please leave a comment below, and let me know what you think!
# 
# 
# ![](https://www.news-medical.net/image.axd?picture=2018%2f10%2fshutterstock_480412786.jpg&ts=20181025104435&ri=673)this 

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


import math
import pandas as pd
import numpy as np

import lightgbm as lgb
import time
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel

from itertools import cycle

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

pd.set_option("display.precision", 8)

INPUT = '/kaggle/input/liverpool-ion-switching'


# # Data overview and understand

# In[ ]:


train_df = pd.read_csv(INPUT + '/train.csv')
train_df.shape


# In[ ]:


test = pd.read_csv(INPUT + '/test.csv')
test.shape


# In[ ]:


train_df.head(10)


# In[ ]:


test.head(10)


# # Memory usage reduction

# In[ ]:


# thanks to https://www.kaggle.com/vbmokin/ion-switching-lgb-mlp-regr-confmatrices
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if col != 'time':
            col_type = df[col].dtypes
            if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


train_df = reduce_mem_usage(train_df)
test = reduce_mem_usage(test)


# In[ ]:


train_df.describe()


# In[ ]:


train_df.describe()     .T.round(4) 


# In[ ]:


test.describe()     .T.round(4) 


# In[ ]:


train_df["open_channels"].value_counts()


# In[ ]:


train_df["time"].value_counts()


# In[ ]:


test["time"].value_counts()


# # Data visualization

# In[ ]:


import matplotlib as mpl
print(mpl.rcParams['agg.path.chunksize'])
mpl.rcParams['agg.path.chunksize'] = 10000
print(mpl.rcParams['agg.path.chunksize'])

fig = plt.figure(figsize = (20,15))
fig.suptitle("plot var corrlation and visualzation")

ax1 = fig.add_subplot(231)
ax1.set_title("time vs open_channel")
ax1.plot(train_df["time"],
        train_df["open_channels"],
        color = "green")

ax2 = fig.add_subplot(232)
ax2.set_title("time vs signal")
#plot each 100s point
ax2.plot(train_df["signal"][::100],
        train_df["open_channels"][::100],
        color = "green")


ax3 = fig.add_subplot(233)
ax3.set_title("signal vs time")
#plot each 100s point
ax3.plot(train_df["signal"][::10],
        train_df["time"][::10],
        color = "green")



ax4 = fig.add_subplot(234)
ax4.set_title("signal vs time")
#plot each 100s point
ax4.plot(train_df["signal"][::10],
        train_df["time"][::10],
        color = "red")


ax5 = fig.add_subplot(235)
ax5.set_title("time vs signal")
#plot each 100s point
ax5.plot(train_df["time"][::10],
        train_df["signal"][::10],
        color = "red")


ax5 = fig.add_subplot(236)
ax5.set_title("signal vs channel")
#plot each 100s point
ax5.plot(train_df["signal"][::10],
        train_df["open_channels"][::10],
        color = "red")



plt.show()



# In[ ]:


train_df['signal'].plot(kind='hist',
                     figsize=(15, 5),
                     bins=55, label='train', alpha=0.5)
test['signal'].plot(kind='hist',
                    bins=55,
                    label='test',
                    alpha=0.5,
                    title='Signal distribution in train vs test')
plt.legend()
plt.show()


# # Distrubtion for targets 

# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt

print('Histogram plot ')
sns.countplot('open_channels', data=train_df)
plt.title('open_channels(Target) size', fontsize=14)
plt.show()


# In[ ]:


# box and whisker plots train data 
train_df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histograms
train_df.hist()
plt.show()
# scatter plot matrix
#scatter_matrix(train_df)
#plt.show()


# In[ ]:


# box and whisker plots test data
print("box and whisker plots test data, the time data maybe 500ms off or delay")
test.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histograms
test.hist()
plt.show()
# scatter plot matrix
#scatter_matrix(train_df)
#plt.show()


# In[ ]:


fig = plt.figure()
#fig.suptitle("plot var corrlation and visualzation")

for i in range(11):
    c = "channel"+str(i)
    #print(c)
    c = train_df[train_df["open_channels"] == i]
    fig.add_subplot()
    sns.distplot(c['signal'])
    plt.title("distribution of signal for channel"+str(i))
    plt.show()
    


# plot all the hist together with better comparision view

# In[ ]:


color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
fig, axs = plt.subplots(4, 3, figsize=(15, 12))
axs = axs.flat
num = 0
for i, d in train_df.groupby('open_channels'):
    color_index = num
    if num > 6:
        color_index = num - 6
    d['signal'].plot(kind='hist',
                     ax=axs[num],
                     title=f'Distribution of Signal for {i} Open Channels',
                     bins=50, 
                    color=next(color_cycle))
    num += 1
plt.tight_layout()


# In[ ]:



num = 0
for i in range(11):
    channel_label = "channel"+str(i)
    #print(c)
    c = train_df[train_df["open_channels"] == i]
    # color
    palette = plt.get_cmap('Paired')
    num+=1
    # style
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    p1=sns.kdeplot(c['signal'], shade=False, color=palette(num),label=channel_label)
    
plt.show()


# practice of various way of plotting 

# In[ ]:


num = 0
for i, d in train_df.groupby('open_channels'):
    color_index = num
    if num > 6:
        color_index = num - 6
    d['signal'].plot(kind='hist',
                     title=f'Distribution of Signal for {i} Open Channels',
                     bins=50, 
                     figsize=(15, 5),
                     alpha=0.5,
                     color=next(color_cycle),
                     label=i)
plt.legend()
plt.show()

the data seem sperate collection in each 500,000 points (50s)
# In[ ]:


plt.scatter(train_df['time'], train_df['signal'], edgecolors='r')
plt.xlabel('time')
plt.ylabel('signal')
plt.title('time vs signal')
plt.show()


# In[ ]:


train_50s = train_df.loc[(train_df['time']<51) & (train_df['time']>49.9995)]
train_50s.head(20)


# In[ ]:


plt.scatter(train_50s['time'], train_50s['signal'], edgecolors='r')
plt.xlabel('time')
plt.ylabel('signal')
plt.title('time vs signal')
plt.show()


# In[ ]:


plt.figure(figsize=(6,6))
sns.distplot(train_df['signal'], bins=20)
sns.distplot(test['signal'], bins=20)
plt.title('Signal Distribution for Test and Train')
plt.legend(labels=['Train', 'Test'])


# # Break down the data to batch to do the analysis 

# In[ ]:


batch_split = 500000
plt.figure(figsize=(12,12))
for i in range(10):
    plt.subplot(5,2,i+1)
    idx = range(batch_split*i, batch_split*(i+1)-1)
    channel_count = train_df.loc[idx, 'open_channels'].value_counts()
    ax = sns.barplot(x=channel_count.index, y=channel_count.values)
    plt.subplots_adjust(hspace = 0.8)
    plt.title('channel count for batch ' + str(i))


# # channel and signal same plots 

# In[ ]:


def channel_sigal_plt(batch_df, title):
    plt.figure(figsize=(18,8))
    plt.plot(batch_df["time"], batch_df["signal"], color='b', label='Signal')
    plt.plot(batch_df["time"], batch_df["open_channels"], color='r', label='Open channel')
    plt.title(title, fontsize=24)
    plt.xlabel("Time (ms)", fontsize=20)
    plt.ylabel("Open Channel and Signal", fontsize=20)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


# In[ ]:


channel_sigal_plt(train_df[0:500000], "Channels and signal vs time train_df[0:500000] ")
channel_sigal_plt(train_df[500000:1000000], "Channels and signal vs time train_df[500000:1000000] ")

channel_sigal_plt(train_df[4500000:5000000], "Channels and signal vs time train_df[4500000:5000000] ")
channel_sigal_plt(train_df[7000:7500], "Channels and signal vs time train_df[7000:7500] ")


# it seems there is some parts has very clear signal relationship with open channel. Need some polish of the signal data.

# # pandas rolling method 
# more about the rolling https://www.youtube.com/watch?v=T2mQiesnx8s

# In[ ]:




roll_size = [10, 50, 100, 500, 1000, 5000, 10000, 25000,50000]

for roll in roll_size:
    train_df["rolling_mean_" + str(roll)] = train_df['signal'].rolling(window=roll).mean()
    train_df["rolling_std_" + str(roll)] = train_df['signal'].rolling(window=roll).std()
    train_df["rolling_var_" + str(roll)] = train_df['signal'].rolling(window=roll).var()
    train_df["rolling_min_" + str(roll)] = train_df['signal'].rolling(window=roll).min()
    train_df["rolling_max_" + str(roll)] = train_df['signal'].rolling(window=roll).max()
    
    train_df["rolling_min_max_ratio_" + str(roll)] = train_df["rolling_min_" + str(roll)] / train_df["rolling_max_" + str(roll)]
    train_df["rolling_min_max_diff_" + str(roll)] = train_df["rolling_max_" + str(roll)] - train_df["rolling_min_" + str(roll)]
    
    a = (train_df['signal'] - train_df['rolling_min_' + str(roll)]) / (train_df['rolling_max_' + str(roll)] - train_df['rolling_min_' + str(roll)])
    train_df["norm_" + str(roll)] = a * (np.floor(train_df['rolling_max_' + str(roll)]) - np.ceil(train_df['rolling_min_' + str(roll)]))
    
train_df = train_df.replace([np.inf, -np.inf], np.nan)
train_df.fillna(0, inplace=True)



# In[ ]:


train_df = reduce_mem_usage(train_df)


# In[ ]:


train_df


# In[ ]:


plt.figure(figsize=(15,10))
plt.grid(True)
#plt.plot(train_df['signal'],label='signal')
plt.plot(train_df['rolling_min_max_diff_25000'], label='rolling_min_max_diff_25000')
plt.plot(train_df['rolling_min_max_diff_500'], label='rolling_min_max_diff_500')
plt.legend(loc=2)


# In[ ]:


plt.figure(figsize=(15,10))
plt.grid(True)
plt.plot(train_df['signal'],label='signal')
#plt.plot(train_df['rolling_min_max_diff_25000'], label='rolling_min_max_diff_25000')
plt.plot(train_df['rolling_mean_500'], label='rolling_mean_500')
plt.plot(train_df['rolling_mean_5000'], label='rolling_mean_5000')
plt.legend(loc=2)


# In[ ]:


y = train_df['open_channels']
X = train_df.drop(["time","open_channels"], axis=1)
del train_df


# # sklearn.preprocessing.StandardScaler

# In[ ]:


scaler = StandardScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
del X


# In[ ]:


X_scaled.head()


# In[ ]:


# Do the same rolling and scaler for test data

roll_size = [10, 50, 100, 500, 1000, 5000, 10000, 25000,50000]

for roll in roll_size:
    test["rolling_mean_" + str(roll)] = test['signal'].rolling(window=roll).mean()
    test["rolling_std_" + str(roll)] = test['signal'].rolling(window=roll).std()
    test["rolling_var_" + str(roll)] = test['signal'].rolling(window=roll).var()
    test["rolling_min_" + str(roll)] = test['signal'].rolling(window=roll).min()
    test["rolling_max_" + str(roll)] = test['signal'].rolling(window=roll).max()
    
    test["rolling_min_max_ratio_" + str(roll)] = test["rolling_min_" + str(roll)] / test["rolling_max_" + str(roll)]
    test["rolling_min_max_diff_" + str(roll)] = test["rolling_max_" + str(roll)] - test["rolling_min_" + str(roll)]
    
    a = (test['signal'] - test['rolling_min_' + str(roll)]) / (test['rolling_max_' + str(roll)] - test['rolling_min_' + str(roll)])
    test["norm_" + str(roll)] = a * (np.floor(test['rolling_max_' + str(roll)]) - np.ceil(test['rolling_min_' + str(roll)]))
    
test = test.replace([np.inf, -np.inf], np.nan)
test.fillna(0, inplace=True)


# In[ ]:


test = test.drop(["time"], axis=1)


# In[ ]:


scaler = StandardScaler()
scaler.fit(test)
test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)
del test


# # feature define and data split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.22, random_state=1357)


# # Logistic Regression modellling 

# In[ ]:


model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)


# In[ ]:


# Showing Confusion Matrix
# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
def plot_confusion_matrix(y_true, y_pred_cm, title):
    figsize=(14,14)
    y_true = y_true.astype(int)
    y_pred_cm = y_pred_cm.astype(int)
    cm = confusion_matrix(y_true, y_pred_cm, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# # Accuracy and confusion matrix

# In[ ]:


# use the model to make predictions with the test data
y_pred = model_lr.predict(X_test)
# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
print(accuracy_score(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred, 'Confusion matrix for Logistic Regression')


# In[ ]:


del y_pred


# In[ ]:


submission_sample = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

submission_sample.shape


# In[ ]:


submission_sample.head()


# In[ ]:


preds_submission_channels = model_lr.predict(test_scaled)
test = pd.read_csv(INPUT + '/test.csv')
submission_lr = test.drop(['signal'], axis=1)
submission_lr['open_channels'] = np.round(np.clip(preds_submission_channels, 0, 10)).astype(int)
submission_lr.to_csv('submission_lr.csv', index=False, float_format='%.4f')


# # Code below is for the XGB
# 

# # release memory

# In[ ]:


del test
del submission_lr
del preds_submission_channels
del submission_sample


# In[ ]:


del count_misclassified


# In[ ]:


del X_scaled


# X_train = reduce_mem_usage(X_train)
# X_test = reduce_mem_usage(X_test)

# # XGB modelling 

# In[ ]:


#model_xgb = xgb.XGBRegressor(max_depth=3)
#model_xgb.fit(X_train, y_train)


# # use the model to make predictions with the test data
# y_pred_xgb_og = model_xgb.predict(X_test)
# # how did our model perform?
# y_pred_xgb = np.round(np.clip(y_pred_xgb_og, 0, 10)).astype(int)
# count_misclassified = (y_test != y_pred_xgb).sum()
# print('Misclassified samples: {}'.format(count_misclassified))
# accuracy = metrics.accuracy_score(y_test, y_pred_xgb)
# print('Accuracy: {:.2f}'.format(accuracy))
# print(accuracy_score(y_test, y_pred_xgb))
# #print(confusion_matrix(y_test, y_pred))
# 
# plot_confusion_matrix(y_test, y_pred_xgb, 'Confusion matrix for Logistic Regression')

# # Submission data prepare

# submission_sample = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
# 
# submission_sample.shape

# preds_submission_channels = model_xgb.predict(test_scaled)
# test = pd.read_csv(INPUT + '/test.csv')
# submission_xgb = test.drop(['signal'], axis=1)
# submission_xgb['open_channels'] = np.round(np.clip(preds_submission_channels, 0, 10)).astype(int)
# submission_xgb.to_csv('submission_lr.csv', index=False, float_format='%.4f')

# # Pleased comment this will encourage me !!!!!!

# In[ ]:




