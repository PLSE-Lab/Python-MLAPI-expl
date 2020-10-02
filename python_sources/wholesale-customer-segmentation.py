#!/usr/bin/env python
# coding: utf-8

# ## Customer Segmentation
# 
# In the following customers will be segmented according to their annual spending using KMeans.
# 
# * Data Exploration
# * Requirements Check
# * Data Preparation
# * Model development

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


wholesale_all = pd.read_csv('../input/wholesale-customers-data-set/Wholesale customers data.csv')

wholesale_all.info()


# ## Data Exploration

# In[ ]:


wholesale_all.head()


# In[ ]:


wholesale = wholesale_all.drop(['Channel','Region'], axis=1)
wholesale_all.groupby(['Channel', 'Region']).agg(['mean', 'std']).round(1)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,10))
sns.boxplot(x='variable', y='value', data=wholesale.melt())

plt.show()


# In[ ]:


sns.pairplot(wholesale, diag_kind='kde')

plt.show()


# ## Requirements Check

# As the distributions are skewed a transformation must be found to normalise the data.

# In[ ]:


from scipy.stats import boxcox, probplot, norm, shapiro

shapiro_test = {}
plt.figure(figsize=(15, 10))
for i in range(0,6):
    ax = plt.subplot(2,3,i+1)
    probplot(x = wholesale[wholesale.columns[i]], dist=norm, plot=ax)
    plt.title(wholesale.columns[i])
    shapiro_test[wholesale.columns[i]] = shapiro(wholesale[wholesale.columns[i]])
    
plt.show()

pd.DataFrame(shapiro_test, index=['Test Statistic', 'p-value']).transpose()


#  All the variable are statistically significant non normally distributed.
#  
#  Let's try the Logarithmic Transformation:

# In[ ]:


import numpy as np

wholesale_log = np.log(wholesale)

shapiro_test = {}

plt.figure(figsize=(15, 10))
for i in range(6):
    ax = plt.subplot(2,3,i+1)
    probplot(x = wholesale_log[wholesale_log.columns[i]], dist=norm, plot=ax)
    plt.title(wholesale_log.columns[i])
    shapiro_test[wholesale.columns[i]] = shapiro(wholesale[wholesale.columns[i]])
    
plt.show()

pd.DataFrame(shapiro_test, index=['Test Statistic', 'p-value']).transpose()


# The Log-Transformation is also not satisfactorily. Let's try BoxCox transformation:

# In[ ]:


from scipy.stats import boxcox

shapiro_test = {}
lambdas = {}

plt.figure(figsize=(15, 10))
plt.title('BoxCox Transformation')
for i in range(6):
    ax = plt.subplot(2,3,i+1)
    x, lbd = boxcox(wholesale[wholesale.columns[i]])
    probplot(x = x, dist=norm, plot=ax)
    plt.title(wholesale.columns[i])
    shapiro_test[wholesale.columns[i]] = shapiro(x)
    lambdas[wholesale.columns[i]] = lbd
    
plt.show()

pd.DataFrame(shapiro_test, index=['Test Statistic', 'p-value']).transpose()


# In[ ]:


pd.DataFrame.from_dict(lambdas, orient='index', columns=['lambda'])


# ## Data Preparation

# In[ ]:


from sklearn.preprocessing import PowerTransformer, StandardScaler

bc = PowerTransformer(method='box-cox')
wholesale_boxcox = bc.fit_transform(wholesale)

sc = StandardScaler()
wholesale_processed = sc.fit_transform(wholesale_boxcox)

wholesale_processed_df = pd.DataFrame(wholesale_processed, columns=wholesale.columns)
wholesale_processed_df


# In[ ]:


sns.pairplot(wholesale_processed_df, diag_kind='kde')

plt.show()


# ## Model Development

# In[ ]:


from sklearn.cluster import KMeans

sse = {}

for k in range(2,11):
    kmeans = KMeans(n_clusters = k, random_state=123)
    cluster_labels = kmeans.fit_predict(wholesale_processed_df)
    sse[k] = kmeans.inertia_
   
plt.figure(figsize=(10,5))
plt.title('Elbow Plot')
sns.pointplot(x = list(sse.keys()), y = list(sse.values()))

plt.show()


# The elbow is by 3 clusters so we try the kmeans with 3 and 4 clusters.

# In[ ]:


kmeans = KMeans(n_clusters=3, random_state=123)
kmeans.fit(wholesale_processed_df)

wholesale = wholesale.assign(segment = kmeans.labels_)

kmeans_3_means = wholesale.groupby('segment').mean()

kmeans = KMeans(n_clusters=4, random_state=123)
kmeans.fit(wholesale_processed_df)

wholesale = wholesale.assign(segment = kmeans.labels_)

kmeans_4_means = wholesale.groupby('segment').mean()

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.heatmap(kmeans_3_means.T, cmap='Blues')

plt.subplot(1,2,2)
sns.heatmap(kmeans_4_means.T, cmap='Blues')

plt.show()


# In[ ]:




