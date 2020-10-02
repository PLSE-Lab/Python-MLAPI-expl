#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
plt.rcParams['figure.figsize'] = [20, 8]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import gc
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
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


def import_data(file, nrows):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, nrows=500000)
    df = reduce_mem_usage(df)
    return df


# More examples and documentation are on site: https://imbalanced-learn.readthedocs.io/en/stable/user_guide.html

# In[ ]:


nrows = 500000 # for faster calculations
train = import_data("../input/train.csv", nrows)


# In[ ]:


dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
dic1 = {'CA':0,'DA':1,'SS':3,'LOFT':4}
train["event"] = train["event"].apply(lambda x: dic[x])
train["event"] = train["event"].astype('int8')
train['experiment'] = train['experiment'].apply(lambda x: dic1[x])
y = train.event
train.drop(['event'], axis=1, inplace=True)


# In[ ]:


train = np.array(train)
y = np.array(y)


# In[ ]:


def plot_y(y, text):
    plt.hist(y)
    plt.title('Target')
    plt.ylabel('Count')
    plt.xlabel(text)

def plot_data (train,y, text):
    pca = PCA(n_components=2,copy=False)
    train_pca = pca.fit_transform(train)

    plt.scatter(train_pca[:,0], train_pca[:,1],c=y, edgecolor='none', alpha=0.9,
            cmap=plt.cm.get_cmap('seismic', 4))
    plt.title(text)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    del train, y


# In[ ]:


plt.hist(y)
plt.title('Target')
plt.ylabel('Count')
plt.xlabel('Target values')


# In[ ]:


get_ipython().run_cell_magic('time', '', "plot_data(train, y, 'Original Data')")


# In[ ]:


gc.collect()


# ****Over-sampling****

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from imblearn.over_sampling import SMOTE, ADASYN\ntrain_sm, y_sm = SMOTE().fit_resample(train, y)')


# In[ ]:


plot_y(y_sm, 'SMOTE')


# In[ ]:


plot_data(train_sm, y_sm, 'SMOTE')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from imblearn.over_sampling import SMOTE, ADASYN\ntrain_ad, y_ad = ADASYN().fit_resample(train, y)')


# In[ ]:


plot_y(y_ad, 'ADASYN')


# In[ ]:


plot_data(train_ad,y_ad ,'ADASYN')


# ****Under-sampling****

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from imblearn.under_sampling import RandomUnderSampler\ncc = RandomUnderSampler(random_state=42)\ntrain_cc, y_cc = cc.fit_resample(train, y)')


# In[ ]:


plot_y(y_cc, 'Random Under Sampler')


# In[ ]:


plot_data(train_cc, y_cc, 'Random Under Sampler')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from imblearn.under_sampling import RepeatedEditedNearestNeighbours\nrenn = RepeatedEditedNearestNeighbours()\ntrain_ren, y_ren= renn.fit_resample(train, y)')


# In[ ]:


plot_y(y_ren, 'Repeated Edited Nearest Neighbours')


# In[ ]:


plot_data(train_ren, y_ren, 'Repeated Edited Nearest Neighbours')

