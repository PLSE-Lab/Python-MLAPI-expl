#!/usr/bin/env python
# coding: utf-8

# # Probabilistic generative model to discrete features (feat. PRML 4.2.3)

# ## 0. Outline

# In probabilistic generative model, we obtain posterior probability for class by Bayes theorem:
# 
# $$
# \begin{align}
# p(C_{k}|\mathbb{x})
# &=
# \frac{p(\mathbb{x}|C_{k})p(C_{k})}{p(\mathbb{x})}\\
# &=
# \frac{p(\mathbb{x}|C_{k})p(C_{k})}{\sum_{c}p(\mathbb{x}|c)p(c)}
# \end{align}
# $$

# Since this competition is a 2-class classification task, the posterior probability for class is given as this:
# $$
# p(C_{1}|\mathbb{x}) = \sigma(a) = \frac{1}{1+e^{-a}},
# $$
# 
# where
# $$
# a = \ln\frac{p(\mathbb{x}|C_{1})p(C_{1})}{p(\mathbb{x}|C_{0})p(C_{0})}
# $$

# Thus, all we need to do is to calculate class-conditional distribution and prior class probability:
# $$
# p(\mathbb{x}|C_{1}), \;\;p(\mathbb{x}|C_{0}), \;\;p(C_{1}), \;\;p(C_{0})
# $$

# Let input is a D-dimention vector of discrete components:
# $$ \mathbb{x}=(x_{1}, ..., x_{D}),\;\;x_{i}\in\{0,1,...,l_{i}-1\} $$

# Under Naive bayes assumption -- if we treat each feature as independent -- , we have this class-conditional distribution form:
# $$ p(\mathbb{x}|C_{k})=\prod_{i=1}^{D}\prod_{j=0}^{l_{i}-1}\mu_{kij}^{\delta_{x_{i},j}}$$
# 
# where
# 
# $$
# \delta_{x_{i},j} = 
# \begin{cases}
# 1 \;\;\mathrm{when}\;\; x_{i}=j \\
# 0 \;\;\mathrm{when}\;\; x_{i}\ne j \\
# \end{cases}
# \;\;,\;\;
# \sum_{j=0}^{l_{i}-1} \mu_{kij}=1
# $$

# Here, parameters to be determined are:
# 
# $$
# \mu_{kij}, \;\;p(C_{1}), \;\;p(C_{0})
# $$
# 
# With maximum likelihood estimate, they are determined as this:
# 
# $$
# \mu_{\mathrm{MLE},kij} = \frac{N(\mathbb{x}\in C_{k}, x_{i}=j)}{N(\mathbb{x}\in C_{k})} \\
# p_{\mathrm{MLE}}(C_{k})=\frac{N(\mathbb{x}\in C_{k})}{N(\mathbb{x})}
# $$
# 
# To supplement, with MAP estimate, they are determined as this:
# 
# $$
# \mu_{\mathrm{MAP},kij} =
# \frac{N(\mathbb{x}\in C_{k}, x_{i}=j)\bigl\{1+N(\mathbb{x}\in C_{k}, x_{i}=j)\bigr\}}
# {N(\mathbb{x}\in C_{k})\bigl\{1+N(\mathbb{x}\in C_{k})\bigr\}}
# \\
# p_{\mathrm{MAP}}(C_{k})
# =
# \frac{N(\mathbb{x}\in C_{k})\bigl\{1+N(\mathbb{x}\in C_{k})\bigr\}}
# {N(\mathbb{x})\bigl\{1+N(\mathbb{x})\bigr\}}
# $$

# Finally, we obtain posterior probability for class is given as this:
# $$
# p(C_{1}|\mathbb{x}) = \sigma(a) = \frac{1}{1+e^{-a}},
# $$
# 
# where
# 
# $$
# \begin{align}
# a &= \ln\frac{p(\mathbb{x}|C_{1})p(C_{1})}{p(\mathbb{x}|C_{0})p(C_{0})} \\
# &=
# \ln\prod_{i=1}^{D}\prod_{j=0}^{l_{i}-1}\mu_{\mathrm{MLE},1ij}^{\delta_{x_{i},j}}
# - \ln\prod_{i=1}^{D}\prod_{j=0}^{l_{i}-1}\mu_{\mathrm{MLE},0ij}^{\delta_{x_{i},j}}
# + \ln p_{\mathrm{MLE}}(C_{1})
# - \ln p_{\mathrm{MLE}}(C_{0}) \\
# &=
# \sum_{i=1}^{D}
# \sum_{j=0}^{l_{i}-1}
# \delta_{x_{i},j} \ln \frac{N(x\in C_{1}, x_{i}=j)}{N(x\in C_{0}, x_{i}=j)}
# + \ln\frac{N(x\in C_{1})}{N(x\in C_{0})}
# \end{align}
# $$

# By this probability, we can predict the class for new inputs with a threshold of 0.5:
# $$
# \begin{cases}
# \mathbb{x} \in C_{1} \;\; \mathrm{when} \; p(C_{1}|\mathbb{x}) > 0.5\\
# \mathbb{x} \in C_{0} \;\; \mathrm{when} \; p(C_{1}|\mathbb{x}) < 0.5
# \end{cases}
# $$

# For more details, refer to PRML book chapter 4.2.3:
# [https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)

# ## 1. Coding

# In[ ]:


import math
import numpy as np
import pandas as pd
import os
import random
        
import collections

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


TRAIN_PATH = '../input/cat-in-the-dat/train.csv'
TEST_PATH = '../input/cat-in-the-dat/test.csv'


# In[ ]:


df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)


# In[ ]:


df_train, df_valid = train_test_split(df_train, train_size=0.8, random_state=0)


# ### 1-1. prior class probability

# With maximum likelihood estimate, we obtain prior class probability:
# $$
# p_{\mathrm{MLE}}(C_{1})=\frac{N(C_{1})}{N}, \:\:\:
# p_{\mathrm{MLE}}(C_{0})=\frac{N(C_{0})}{N}
# $$
# 
# With MAP estimate:
# $$
# p_{\mathrm{MAP}}(C_{1})=\frac{(N(C_{1}))(1+N(C_{1}))}{N(1+N)}, \:\:\:
# p_{\mathrm{MAP}}(C_{0})=\frac{(N(C_{0}))(1+N(C_{0}))}{N(1+N)}
# $$

# In[ ]:


sns.countplot(x='target', data=df_train)
plt.title('target')
plt.xticks([0,1],[0,1])
plt.show()


# In[ ]:


n = df_train.shape[0]
nc1 = df_train.loc[:, 'target'].sum()
nc0 = n - nc1

pc1_mle = nc1 / n
pc0_mle = nc0 / n
pc1_map = nc1 * (1+nc1) / (n * (1+n))
pc0_map = nc0 * (1+nc0) / (n * (1+n))


# In[ ]:


pc1_mle


# In[ ]:


pc0_mle


# In[ ]:


pc1_map


# In[ ]:


pc0_map


# ### 1-2. feature encoding

# We need to convert non-numerical features into numerical features.

# In[ ]:


df_train


# #### (convert feature to numbers for bin_3, bin_4)

# In[ ]:


df_train.loc[:, 'bin_3'] = df_train.loc[:, 'bin_3'].replace({'F':0, 'T':1})
df_train.loc[:, 'bin_4'] = df_train.loc[:, 'bin_4'].replace({'N':0, 'Y':1})
df_train.loc[:, 'bin_0':'bin_4'] = df_train.loc[:, 'bin_0':'bin_4'].astype(np.uint8)


# #### (convert feature to numbers for nom_0 to nom_9)

# In[ ]:


replace_map = {}

for column in df_train.loc[:, 'nom_0':'nom_9'].columns:
    replace_map[column] = {}
    for i, key in enumerate(collections.Counter(df_train.loc[:, column]).keys()):
        replace_map[column][key] = i


# In[ ]:


for column in df_train.loc[:, 'nom_0':'nom_9'].columns:
    df_train.loc[:, column] = df_train.loc[:, column].replace(replace_map[column])
df_train.loc[:, 'nom_0':'nom_9'] = df_train.loc[:, 'nom_0':'nom_9'].astype(np.uint8)


# In[ ]:


df_train.loc[:, 'nom_0':'nom_9']


# ### 1-3. naive bayes assumption

# Little correletion is seen among features bin_0 to nom_9:

# In[ ]:


sns.heatmap(pd.concat([df_train.loc[:, 'bin_0':'nom_9'], df_train.loc[:, 'target']], axis=1).corr(), fmt='.2f', annot=True)


# Therefore, naive bayes assumption is valid -- we may treat features bin_0 to nom_9 as independent.

# ### 1-4. class-conditional distribution

# As naive bayes assumption is valid, we obtain class-conditional distribution as this:
# $$ 
# p(\mathbb{x}|C_{k})=\prod_{i=1}^{D}\prod_{j=0}^{l_{i}-1}\mu_{kij}^{\delta_{x_{i},j}}$$
# 
# where
# 
# $$
# \delta_{x_{i},j} = 
# \begin{cases}
# 1 \;\;\mathrm{when}\;\; x_{i}=j \\
# 0 \;\;\mathrm{when}\;\; x_{i}\ne j \\
# \end{cases}
# \;\;,\;\;
# \sum_{j=0}^{l_{i}-1} \mu_{kij}=1
# $$
# 
# Parameters can be determined by maximum likelihood estimation:
# $$
# \mu_{\mathrm{MLE},kij} = \frac{N(\mathbb{x}\in C_{k}, x_{i}=j)}{N(\mathbb{x}\in C_{k})}
# $$
# 
# Similarly, determined parameters with MAP estimation are:
# $$
# \mu_{\mathrm{MAP},kij} =
# \frac{N(\mathbb{x}\in C_{k}, x_{i}=j)\bigl\{1+N(\mathbb{x}\in C_{k}, x_{i}=j)\bigr\}}
# {N(\mathbb{x}\in C_{k})\bigl\{1+N(\mathbb{x}\in C_{k})\bigr\}}
# $$

# In[ ]:


m_mle = [[], []]
m_map = [[], []]


# In[ ]:


for column in df_train.loc[:, 'bin_0':'bin_4'].columns:
    class1 = df_train.loc[df_train.loc[:, 'target'] == 1, 'bin_0':'nom_9']
    class0 = df_train.loc[df_train.loc[:, 'target'] == 0, 'bin_0':'nom_9']
    mle1j = []
    mle0j = []
    map1j = []
    map0j = []
    for j in range(2):
        n1j = np.sum(class1.loc[:, column] == j)
        n0j = np.sum(class0.loc[:, column] == j)
        mle1j.append(n1j / class1.shape[0])
        mle0j.append(n0j / class0.shape[0])
        map1j.append((n1j * (1+n1j)) / (class1.shape[0] * (1+class1.shape[0])))
        map0j.append((n0j * (1+n0j)) / (class0.shape[0] * (1+class0.shape[0])))
    m_mle[1].append(mle1j)
    m_mle[0].append(mle0j)
    m_map[1].append(map1j)
    m_map[0].append(map0j)


for column in df_train.loc[:, 'nom_0':'nom_9'].columns:
    class1 = df_train.loc[df_train.loc[:, 'target'] == 1, 'bin_0':'nom_9']
    class0 = df_train.loc[df_train.loc[:, 'target'] == 0, 'bin_0':'nom_9']
    mle1j = []
    mle0j = []
    map1j = []
    map0j = []
    for j in range(len(replace_map[column])):
        n1j = np.sum(class1.loc[:, column] == j)
        n0j = np.sum(class0.loc[:, column] == j)
        mle1j.append(n1j / class1.shape[0])
        mle0j.append(n0j / class0.shape[0])
        map1j.append((n1j * (1+n1j)) / (class1.shape[0] * (1+class1.shape[0])))
        map0j.append((n0j * (1+n0j)) / (class0.shape[0] * (1+class0.shape[0])))
    m_mle[1].append(mle1j)
    m_mle[0].append(mle0j)
    m_map[1].append(map1j)
    m_map[0].append(map0j)


# ### 1-5. prediction

# Let us make a prediction!
# 
# Activation value of logistic sigmoid is obtained as this:
# 
# $$
# \begin{align}
# a &= \ln\frac{p(\mathbb{x}|C_{1})p(C_{1})}{p(\mathbb{x}|C_{0})p(C_{0})} \\
# &=
# \ln\prod_{i=1}^{D}\prod_{j=0}^{l_{i}-1}\mu_{1ij}^{\delta_{x_{i},j}}
# - \ln\prod_{i=1}^{D}\prod_{j=0}^{l_{i}-1}\mu_{0ij}^{\delta_{x_{i},j}}
# + \ln p(C_{1})
# - \ln p(C_{0})
# \end{align}
# $$

# Then we can predict class as this:
# $$
# y_{n} = \biggl\{
# \begin{array}{l}
# 1 \;\;\mathrm{when}\; \sigma(a)>0.5\\
# 0 \;\;\mathrm{when}\; \sigma(a)<0.5\\
# \end{array}
# $$

# In[ ]:


df_train.loc[:, 'y_mle_pred'] = np.log(pc1_mle) - np.log(pc0_mle)
df_train.loc[:, 'y_map_pred'] = np.log(pc1_map) - np.log(pc0_map)


# In[ ]:


def predict(df):
    df.loc[:, 'y_mle_pred'] = np.log(pc1_mle) - np.log(pc0_mle)
    df.loc[:, 'y_map_pred'] = np.log(pc1_map) - np.log(pc0_map)
    eps = 1e-8
    
    @np.vectorize
    def sigmoid(x):
        sigmoid_range = 34.538776394910684

        if x <= -sigmoid_range:
            return 1e-15
        if x >= sigmoid_range:
            return 1.0 - 1e-15

        return 1.0 / (1.0 + np.exp(-x))


    for i, column in enumerate(df.loc[:, 'bin_0':'nom_9'].columns):
        try:
            df.loc[:, 'y_mle_pred'] +=                 df.loc[:, column].apply(                    lambda j: np.log(max(eps, m_mle[1][i][j])) - np.log(max(eps, m_mle[0][i][j]))
                )
        except TypeError:
            j = random.randrange(0, len(replace_map[column]))
            df.loc[:, 'y_mle_pred'] +=                 np.log(max(eps, m_mle[1][i][j])) - np.log(max(eps, m_mle[0][i][j]))
        
    for i, column in enumerate(df.loc[:, 'bin_0':'nom_9'].columns):
        try:
            df.loc[:, 'y_map_pred'] +=                 df.loc[:, column].apply(                    lambda j: np.log(max(eps, m_map[1][i][j])) - np.log(max(eps, m_map[0][i][j]))
                )
        except TypeError:
            j = random.randrange(0, len(replace_map[column]))
            df.loc[:, 'y_map_pred'] +=                 np.log(max(eps, m_map[1][i][j])) - np.log(max(eps, m_map[0][i][j]))
            
    df.loc[:, 'y_mle_pred'] = sigmoid(df.loc[:, 'y_mle_pred'])
    df.loc[:, 'y_map_pred'] = sigmoid(df.loc[:, 'y_map_pred'])


# In[ ]:


predict(df_train)


# In[ ]:


df_train.loc[:, 'y_mle_pred':'y_map_pred']


# In[ ]:


roc_auc_score(df_train.loc[:, 'target'], df_train.loc[:, 'y_mle_pred'])


# In[ ]:


roc_auc_score(df_train.loc[:, 'target'], df_train.loc[:, 'y_map_pred'])


# In[ ]:


accuracy_score(df_train.loc[:, 'target'], (df_train.loc[:, 'y_mle_pred'] > 0.5) * 1)


# In[ ]:


accuracy_score(df_train.loc[:, 'target'], (df_train.loc[:, 'y_map_pred'] > 0.5) * 1)


# ## 2. Same procedure to sequential features, day and month

# For simplicity, we simply apply the same technique to sequential features (ord_0 to ord_5) and cyclic features (day, month).
# 
# That is, here we treat them like categorical features and ignore their sequence.

# In[ ]:


for column in df_train.loc[:, 'ord_0':'ord_5'].columns:
    replace_map[column] = {}
    for i, key in enumerate(collections.Counter(df_train.loc[:, column]).keys()):
        replace_map[column][key] = i


# In[ ]:


for column in df_train.loc[:, 'ord_0':'ord_5'].columns:
    df_train.loc[:, column] = df_train.loc[:, column].replace(replace_map[column])
df_train.loc[:, 'ord_0':'ord_5'] = df_train.loc[:, 'ord_0':'ord_5'].astype(np.uint8)


# In[ ]:


for column in df_train.loc[:, 'day':'month'].columns:
    replace_map[column] = {}
    for i, key in enumerate(collections.Counter(df_train.loc[:, column]).keys()):
        replace_map[column][key] = i


# In[ ]:


df_train.loc[:, 'day':'month'] = df_train.loc[:, 'day':'month'] - 1


# In[ ]:


for column in df_train.loc[:, 'ord_0':'month'].columns:
    class1 = df_train.loc[df_train.loc[:, 'target'] == 1, 'ord_0':'month']
    class0 = df_train.loc[df_train.loc[:, 'target'] == 0, 'ord_0':'month']
    mle1j = []
    mle0j = []
    map1j = []
    map0j = []
    for j in range(len(replace_map[column])):
        n1j = np.sum(class1.loc[:, column] == j)
        n0j = np.sum(class0.loc[:, column] == j)
        mle1j.append(n1j / class1.shape[0])
        mle0j.append(n0j / class0.shape[0])
        map1j.append((n1j * (1+n1j)) / (class1.shape[0] * (1+class1.shape[0])))
        map0j.append((n0j * (1+n0j)) / (class0.shape[0] * (1+class0.shape[0])))
    m_mle[1].append(mle1j)
    m_mle[0].append(mle0j)
    m_map[1].append(map1j)
    m_map[0].append(map0j)


# In[ ]:


def predict_all(df):
    df.loc[:, 'y_mle_pred'] = np.log(pc1_mle) - np.log(pc0_mle)
    df.loc[:, 'y_map_pred'] = np.log(pc1_map) - np.log(pc0_map)
    eps = 1e-8
    
    @np.vectorize
    def sigmoid(x):
        sigmoid_range = 34.538776394910684

        if x <= -sigmoid_range:
            return 1e-15
        if x >= sigmoid_range:
            return 1.0 - 1e-15

        return 1.0 / (1.0 + np.exp(-x))


    for i, column in enumerate(df.loc[:, 'bin_0':'month'].columns):
        try:
            df.loc[:, 'y_mle_pred'] +=                 df.loc[:, column].apply(                    lambda j: np.log(max(eps, m_mle[1][i][j])) - np.log(max(eps, m_mle[0][i][j]))
                )
        except TypeError:
            j = random.randrange(0, len(replace_map[column]))
            df.loc[:, 'y_mle_pred'] +=                 np.log(max(eps, m_mle[1][i][j])) - np.log(max(eps, m_mle[0][i][j]))
        
    for i, column in enumerate(df.loc[:, 'bin_0':'month'].columns):
        try:
            df.loc[:, 'y_map_pred'] +=                 df.loc[:, column].apply(                    lambda j: np.log(max(eps, m_map[1][i][j])) - np.log(max(eps, m_map[0][i][j]))
                )
        except TypeError:
            j = random.randrange(0, len(replace_map[column]))
            df.loc[:, 'y_map_pred'] +=                 np.log(max(eps, m_map[1][i][j])) - np.log(max(eps, m_map[0][i][j]))
            
    df.loc[:, 'y_mle_pred'] = sigmoid(df.loc[:, 'y_mle_pred'])
    df.loc[:, 'y_map_pred'] = sigmoid(df.loc[:, 'y_map_pred'])


# In[ ]:


predict_all(df_train)


# In[ ]:


df_train.loc[:, 'y_mle_pred':'y_map_pred']


# In[ ]:


roc_auc_score(df_train.loc[:, 'target'], df_train.loc[:, 'y_mle_pred'])


# In[ ]:


roc_auc_score(df_train.loc[:, 'target'], df_train.loc[:, 'y_map_pred'])


# In[ ]:


accuracy_score(df_train.loc[:, 'target'], (df_train.loc[:, 'y_mle_pred'] > 0.5) * 1)


# In[ ]:


accuracy_score(df_train.loc[:, 'target'], (df_train.loc[:, 'y_map_pred'] > 0.5) * 1)


# ## Validation

# In[ ]:


def preprocess(df):
    df.loc[:, 'bin_3'] = df.loc[:, 'bin_3'].replace({'F':0, 'T':1})
    df.loc[:, 'bin_4'] = df.loc[:, 'bin_4'].replace({'N':0, 'Y':1})
    df.loc[:, 'bin_0':'bin_4'] = df.loc[:, 'bin_0':'bin_4'].astype(np.uint8)
    
    for column in df.loc[:, 'nom_0':'nom_9'].columns:
        df.loc[:, column] = df.loc[:, column].replace(replace_map[column])
        
    for column in df.loc[:, 'ord_0':'ord_5'].columns:
        df.loc[:, column] = df.loc[:, column].replace(replace_map[column])
        
    df.loc[:, 'day':'month'] = df.loc[:, 'day':'month'] - 1


# In[ ]:


preprocess(df_valid)


# ### validation 1: Probabilistic generative model to discrete features (bin_0 to nom_9)

# In[ ]:


predict(df_valid)


# In[ ]:


df_valid.loc[:, 'y_mle_pred':'y_map_pred']


# In[ ]:


roc_auc_score(df_valid.loc[:, 'target'], df_valid.loc[:, 'y_mle_pred'])


# In[ ]:


roc_auc_score(df_valid.loc[:, 'target'], df_valid.loc[:, 'y_map_pred'])


# In[ ]:


accuracy_score(df_valid.loc[:, 'target'], (df_valid.loc[:, 'y_mle_pred'] > 0.5) * 1)


# In[ ]:


accuracy_score(df_valid.loc[:, 'target'], (df_valid.loc[:, 'y_map_pred'] > 0.5) * 1)


# ### validation 2: Apply one-hot encoding also to sequential features (ord_0 to ord_5)

# In[ ]:


predict_all(df_valid)


# In[ ]:


roc_auc_score(df_valid.loc[:, 'target'], df_valid.loc[:, 'y_mle_pred'])


# In[ ]:


roc_auc_score(df_valid.loc[:, 'target'], df_valid.loc[:, 'y_map_pred'])


# In[ ]:


accuracy_score(df_valid.loc[:, 'target'], (df_valid.loc[:, 'y_mle_pred'] > 0.5) * 1)


# In[ ]:


accuracy_score(df_valid.loc[:, 'target'], (df_valid.loc[:, 'y_map_pred'] > 0.5) * 1)


# # Submission

# The same procedure is available to test data.

# In[ ]:


preprocess(df_test)


# In[ ]:


predict_all(df_test)


# In[ ]:


df_test.loc[:, 'y_mle_pred':'y_map_pred']


# In[ ]:


df_test.loc[:, 'target'] = df_test.loc[:, 'y_map_pred']


# In[ ]:


submission = df_test.loc[:, ['id', 'target']]


# In[ ]:


submission


# In[ ]:


submission.to_csv('submission.csv', index=False)

