#!/usr/bin/env python
# coding: utf-8

# From http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
dir = '../input'
print(os.listdir(dir))
# Any results you write to the current directory are saved as output.


# In[ ]:


from __future__ import print_function

import matplotlib as mpl
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

import seaborn as sns
sns.set(style="white", color_codes=True)

get_ipython().run_line_magic('matplotlib', 'inline')


import warnings
warnings.simplefilter('ignore')


# In[ ]:


dataset = pd.read_csv(dir + '/housing.csv')


# In[ ]:


dfX = pd.DataFrame(dataset['median_income'])
col = dfX['median_income'].values.reshape(-1, 1)

scalers = [
    #('Unscaled data', X),
    ('standard scaling', StandardScaler()),
    ('min-max scaling', MinMaxScaler()),
    ('max-abs scaling', MaxAbsScaler()),
    ('robust scaling', RobustScaler(quantile_range=(25, 75))),
    ('quantile transformation (uniform pdf)', QuantileTransformer(output_distribution='uniform')),
    ('quantile transformation (gaussian pdf)', QuantileTransformer(output_distribution='normal')),
    ('sample-wise L2 normalizing', Normalizer())
]

for scaler in scalers:
    dfX[scaler[0]] = scaler[1].fit_transform(col)
    
dfX.head()


# In[ ]:


orig = dfX['median_income']
orig_mean = orig.mean()
bins = 50
alpha=0.5

def plot_experiment(name):
    normalized = dfX[name]
    plt.figure(figsize=(10,5))
    plt.hist(orig, bins, alpha=alpha, label='Original')
    plt.axvline(orig_mean, color='k', linestyle='dashed', linewidth=1)

    plt.hist(normalized, bins, alpha=alpha, label=name)
    plt.axvline(normalized.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.legend(loc='upper right')

    plt.figure(figsize=(5,5))
    g = sns.jointplot(x="median_income", y=name, data=dfX, kind='hex', ratio=3)
    #sns.violinplot(x='median_income', data=dfX, )
    #sns.violinplot(x='standard scaling', data=dfX)
    #plt.boxplot(dfX['median_income'])
    #plt.boxplot(dfX['standard scaling'])
    plt.show()


# ## StandardScaler
# 
# StandardScaler removes the mean and scales the data to unit variance. However, the outliers have an influence when computing the empirical mean and standard deviation which shrink the range of the feature values.
# 
# StandardScaler therefore cannot guarantee balanced feature scales in the presence of outliers.

# In[ ]:


plot_experiment('standard scaling')


# ## MinMaxScaler
# 
# MinMaxScaler rescales the data set such that all feature values are in the range [0, 1]. 
# 
# As StandardScaler, MinMaxScaler is very sensitive to the presence of outliers.

# In[ ]:


plot_experiment('min-max scaling')


# ## MaxAbsScaler
# MaxAbsScaler differs from the previous scaler such that the absolute values are mapped in the range [0, 1]. On positive only data, this scaler behaves similarly to MinMaxScaler and therefore also suffers from the presence of large outliers.

# In[ ]:


plot_experiment('max-abs scaling')


# ## RobustScaler
# Unlike the previous scalers, the centering and scaling statistics of this scaler are based on percentiles and are therefore not influenced by a few number of very large marginal outliers. Consequently, the resulting range of the transformed feature values is larger than for the previous scalers and, more importantly, are approximately similar: most of the transformed values lie in a [-2, 3] range. Note that the outliers themselves are still present in the transformed data. If a separate outlier clipping is desirable, a non-linear transformation is required (see below).

# In[ ]:


plot_experiment('robust scaling')


# ## QuantileTransformer (uniform output)
# 
# QuantileTransformer applies a non-linear transformation such that the probability density function of each feature will be mapped to a uniform distribution. In this case, all the data will be mapped in the range [0, 1], even the outliers which cannot be distinguished anymore from the inliers.
# 
# As RobustScaler, QuantileTransformer is robust to outliers in the sense that adding or removing outliers in the training set will yield approximately the same transformation on held out data. But contrary to RobustScaler, QuantileTransformer will also automatically collapse any outlier by setting them to the a priori defined range boundaries (0 and 1).

# In[ ]:


plot_experiment('quantile transformation (uniform pdf)')


# ## QuantileTransformer (Gaussian output)
# QuantileTransformer has an additional output_distribution parameter allowing to match a Gaussian distribution instead of a uniform distribution. Note that this non-parametetric transformer introduces saturation artifacts for extreme values.

# In[ ]:


plot_experiment('quantile transformation (gaussian pdf)')


# ## Normalizer
# The Normalizer rescales the vector for each sample to have unit norm, independently of the distribution of the samples. It can be seen on both figures below where all samples are mapped onto the unit circle. In our example the two selected features have only positive values; therefore the transformed data only lie in the positive quadrant. This would not be the case if some original features had a mix of positive and negative values.

# In[ ]:


plot_experiment('sample-wise L2 normalizing')


# In[ ]:


dfX[['median_income', 'sample-wise L2 normalizing']].sample(20)

