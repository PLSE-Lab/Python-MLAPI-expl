#!/usr/bin/env python
# coding: utf-8

# # Study if classes form separable clusters
# This kernel closely follows https://www.kaggle.com/mlisovyi/feature-engineering-lighgbm-with-f1-macro for data cleaning and feature engineering. 
# 
# # Short summary
# It seems that the class 4 = _"non vulnerable households"_ form clusters that can be separated from the rest of the clusters. The rest of classes are all mixed together

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# The following categorical mapping originates from [this kernel](https://www.kaggle.com/mlisovyi/categorical-variables-encoding-function).

# In[ ]:


from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    '''
    The function does not return, but transforms the input pd.DataFrame
    
    Encodes the Costa Rican Household Poverty Level data 
    following studies in https://www.kaggle.com/mlisovyi/categorical-variables-in-the-data
    and the insight from https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359631
    
    The following columns get transformed: edjefe, edjefa, dependency, idhogar
    The user most likely will simply drop idhogar completely (after calculating houshold-level aggregates)
    '''
    
    yes_no_map = {'no': 0, 'yes': 1}
    
    df['dependency'] = df['dependency'].replace(yes_no_map).astype(np.float32)
    
    df['edjefe'] = df['edjefe'].replace(yes_no_map).astype(np.float32)
    df['edjefa'] = df['edjefa'].replace(yes_no_map).astype(np.float32)
    
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])


# **There is also feature engineering magic happening here:**

# In[ ]:


def do_features(df):
    feats_div = [('children_fraction', 'r4t1', 'r4t3'), 
                 ('working_man_fraction', 'r4h2', 'r4t3'),
                 ('all_man_fraction', 'r4h3', 'r4t3'),
                 ('human_density', 'tamviv', 'rooms'),
                 ('human_bed_density', 'tamviv', 'bedrooms'),
                 ('rent_per_person', 'v2a1', 'r4t3'),
                 ('rent_per_room', 'v2a1', 'rooms'),
                 ('mobile_density', 'qmobilephone', 'r4t3'),
                 ('tablet_density', 'v18q1', 'r4t3'),
                 ('mobile_adult_density', 'qmobilephone', 'r4t2'),
                 ('tablet_adult_density', 'v18q1', 'r4t2')
                ]
    
    feats_sub = [('people_not_living', 'tamhog', 'tamviv'),
                 ('people_weird_stat', 'tamhog', 'r4t3')]

    for f_new, f1, f2 in feats_div:
        df['fe_' + f_new] = (df[f1] / df[f2]).astype(np.float32)       
    for f_new, f1, f2 in feats_sub:
        df['fe_' + f_new] = (df[f1] - df[f2]).astype(np.float32)
    
    # aggregation rules over household
    aggs_num = {'age': ['min', 'max', 'mean', 'count'],
                'escolari': ['min', 'max', 'mean', 'std']
               }
    aggs_cat = {'dis': ['mean']}
    for s_ in ['estadocivil', 'parentesco', 'instlevel']:
        for f_ in [f_ for f_ in df.columns if f_.startswith(s_)]:
            aggs_cat[f_] = ['mean']
    # aggregation over household
    for name_, df_ in [('18', df.query('age >= 18'))]:
        df_agg = df_.groupby('idhogar').agg({**aggs_num, **aggs_cat}).astype(np.float32)
        df_agg.columns = pd.Index(['agg' + name_ + '_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
        df = df.join(df_agg, how='left', on='idhogar')
        del df_agg
    df.fillna(0, inplace=True)
    # do something advanced above...
    
    # Drop SQB variables, as they are just squres of other vars 
    df.drop([f_ for f_ in df.columns if f_.startswith('SQB') or f_ == 'agesq'], axis=1, inplace=True)
    # Drop id's
    df.drop(['Id'], axis=1, inplace=True)
    # Drop repeated columns
    df.drop(['hhsize', 'female', 'area2'], axis=1, inplace=True)
    return df


# # Read in the data and clean it up

# In[ ]:


train = pd.read_csv('../input/train.csv')
#We do not need the test sample for this exercise
#test = pd.read_csv('../input/test.csv')


# In[ ]:


train.info()


# In[ ]:


def process_df(df_):
    # fix categorical features
    encode_data(df_)
    #fill in missing values based on https://www.kaggle.com/mlisovyi/missing-values-in-the-data
    for f_ in ['v2a1', 'v18q1', 'meaneduc', 'SQBmeaned']:
        df_[f_] = df_[f_].fillna(0)
    df_['rez_esc'] = df_['rez_esc'].fillna(-1)
    # do feature engineering and drop useless columns
    return do_features(df_)

train = process_df(train)
#test = process_df(test)


# In[ ]:


train.info()


# Note the change in the number of features of different type. What we did was:
# - encoded categorical variables appropreately into numerical values;
# - dropped a few irrelevant columns;
# - added several columns with household aggregates and cand-crafted ratio and subtraction features

# # VERY IMPORTANT
# > Note that ONLY the heads of household are used in scoring. All household members are included in test + the sample submission, but only heads of households are scored.

# In[ ]:


X = train.query('parentesco1==1')#.sample(frac=0.2)

# pull out the target variable
y = X['Target'] - 1
X = X.drop(['Target'], axis=1)


# In[ ]:


cols_2_drop=[]

X.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)
#test.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)


# # Scale the training data
# Data data has to be rescaled to mean of 0 and variance of 1, since most of the following algorithms use distance metric. We will initialise 3 different scalers: `StandardScaler`, `RobustScaler` and `MinMaxScaler` but only one will be used later on

# In[ ]:


from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler


# In[ ]:


rs = RobustScaler().fit(X)
ss = StandardScaler().fit(X)
mm = MinMaxScaler().fit(X)


# In[ ]:


X_rs = pd.DataFrame(rs.transform(X), columns=X.columns)
X_ss = pd.DataFrame(ss.transform(X), columns=X.columns)
X_mm = pd.DataFrame(mm.transform(X), columns=X.columns)


# Define helper functions  `transform_data` and `plot_transformed_data` that will be used by all following clustering algorithms.

# In[ ]:


import time
from sklearn.base import clone
def transform_data(tr_, X_, configs_, tr_name_):
    X_tr = {}
    for i,params in configs_.items():
        print('---------- {} -----------'.format(params))
        t_start = time.clock()
        X_tr[i] = clone(tr_).set_params(**params).fit_transform(X_ss)
        t_end = time.clock()
        print('{} fitted in {} sec'.format(tr_name_, t_end-t_start))
    return X_tr


# In[ ]:


colors=['r','b','y','g']
def plot_transformed_data(X_tr_, y_, configs_, tr_name_):
    for j, X_ in X_tr_.items():
        plt.figure(figsize=(6,4))
        for i in [3,0,1, 2]:
            plt.scatter(X_[y_==i,0], X_[y_==i,1], c=colors[i], s=5, label=i+1)
        plt.legend()
        plt.title('{}: {}'.format(tr_name_, configs_[j]))


# # t-SNE
# Define several variants to see dependence on the input parameters

# In[ ]:


from sklearn.manifold import TSNE
tsne_configs = {1: dict(init='random'),
                2: dict(init='pca'),
                3: dict(init='pca', n_iter=5000),
                4: dict(init='pca', n_iter=500),
                5: dict(init='pca', learning_rate=50),
                6: dict(init='pca', learning_rate=500),
                7: dict(init='pca', perplexity=15),
                8: dict(init='pca', perplexity=50)}
X_tsne = transform_data(TSNE(n_components=2, random_state=314), 
                        X_rs, 
                        tsne_configs, 
                        't-SNE')


# In[ ]:


plot_transformed_data(X_tsne, y, tsne_configs, 't-SNE')


# As one can see, the green dots (class 4 = _non vulnerable households_) can be separated from the rest, while the other three classes are closely mixed together

# # MDS
# Define several variants to see dependence on the input parameters

# In[ ]:


from sklearn.manifold import MDS

mds_configs = {1: dict(max_iter=100),
               2: dict(max_iter=300),
               3: dict(max_iter=500)}
X_mds = transform_data(MDS(n_components=2, n_init=2, n_jobs=1, random_state=314), 
                       X_rs, 
                       mds_configs, 
                       'MDS')


# In[ ]:


plot_transformed_data(X_mds, y, mds_configs, 'MDS')


# As one can see, the green dots (class 4 = _non vulnerable households_) can be separated from the rest (in particular, for `max_iter=500`), while the other three classes are closely mixed together

# # Isomap
# Define several variants to see dependence on the input parameters

# In[ ]:


from sklearn.manifold import Isomap

iso_configs = {1: dict(n_neighbors=20),
               2: dict(n_neighbors=50),
               3: dict(n_neighbors=100)}
X_isomap = transform_data(Isomap(n_components=2, n_jobs=4), 
                       X_rs, 
                       iso_configs, 
                       'Isomap')


# In[ ]:


plot_transformed_data(X_isomap, y, iso_configs, 'Isomap')


# As one can see, the green dots (class 4 = _non vulnerable households_) can be separated from the rest, while the other three classes are closely mixed together
