#!/usr/bin/env python
# coding: utf-8

# **...This kernel is still under further work...**<br><br><br>
# This kernel is part 7 of my work, listed below, in this competition. Any comments or suggestions you may have are greatly appreciated!
# 1. [EAD and Feature Engineering](https://www.kaggle.com/hoangnguyen719/1-eda-and-feature-engineering/notebook)
# 2. [ExtraTreesClassifier tuning](https://www.kaggle.com/hoangnguyen719/extratree-tuning)
# 3. [AdaboostClassifier tuning](https://www.kaggle.com/hoangnguyen719/adaboost-tuning)
# 4. [LGBMClassifier tuning](https://www.kaggle.com/hoangnguyen719/lightgbm-tuning)
# 5. [KNearestClassifier tuning](https://www.kaggle.com/hoangnguyen719/knn-tuning)
# 6. [StackingCVClassifier (use_probas tuning)](https://www.kaggle.com/hoangnguyen719/stacking-use-probas-tuning)
# 7. [Mis-Classified Inspection](https://www.kaggle.com/hoangnguyen719/mis-classified-inspection)
# <br><br>
# *Note*: Many of my EDA parts have been greatly inspired by previous kernels in the competition and I have been trying to give credits to the owners as much as I can. However, because (1) many kernels appear to have the same ideas (even codes), which makes it hard to trace back where the ideas originated from, and (2) I carelessly forgot to note down all the sources (this is totally my bad), sometimes the credit may not be given where it's due. I apologize beforehand, and please let me know in the comment section if you have any question or suggestions. Thank you!
# <br><br>
# **Outline of this notebook**<br>
# I. [Package and Data Loading](#I.-Package-and-Data-Loading)<br>
# II. [Feature Engineering](#II.-Feature-Engineering) <br>
# III. [Model Fitting](#III.-Model-Fitting) <br>
# IV. [Mis-classified Inspection](#IV.-Mis-classified-Inspection)

# # I. Package and Data Loading

# In[ ]:


import pandas as pd
import numpy as np
from itertools import product, combinations
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import gc

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.ensemble import (RandomForestClassifier
                              , RandomForestRegressor
                              , AdaBoostClassifier
                              , ExtraTreesClassifier
                             )
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


rand_state = 719

from time import time
def timer(t): # t = beginning of timing
    timing = time() - t
    if timing < 60:
        return str(round(timing,1)) + ' second(s)'
    elif timing < 3600:
        return str(round(timing / 60,1)) + ' minute(s)'
    else:
        return str(round(timing / 3600,1)) + ' hour(s)'


# In[ ]:


data_path = '/kaggle/input/learn-together/'
def reload(x):
    return pd.read_csv(data_path + x, index_col = 'Id')

train = reload('train.csv')
n_train = len(train)
test = reload('test.csv')
n_test = len(test)

index_test = test.index.copy()
y_train = train.Cover_Type.copy()

all_data = train.iloc[:,train.columns != 'Cover_Type'].append(test)
all_data['train'] = [1]*n_train + [0]*n_test

del train
del test


# # II. Feature Engineering
# ## 1. 0s imputation

# In[ ]:


questionable_0 = ['Hillshade_9am', 'Hillshade_3pm'] # Hillshade_3pm visualization looks weird
corr_cols = {'Hillshade_9am': ['Hillshade_3pm', 'Aspect', 'Slope', 'Soil_Type10', 'Wilderness_Area1',
                               'Wilderness_Area4', 'Vertical_Distance_To_Hydrology'],
             'Hillshade_3pm': ['Hillshade_9am', 'Hillshade_Noon', 'Slope', 'Aspect']
            }


# In[ ]:


rfr = RandomForestRegressor(n_estimators = 100, random_state = rand_state, verbose = 1, n_jobs = -1)
for col in questionable_0:
    print('='*20)
    print(col)
    all_data_0 = all_data[all_data[col] == 0].copy()
    all_data_non0 = all_data[all_data[col] != 0].copy()
    rfr.fit(all_data_non0[corr_cols[col]], all_data_non0[col])
    pred = rfr.predict(all_data_0[corr_cols[col]])
    pred_col = 'predicted_{}'.format(col)
    
    all_data[pred_col] = all_data[col].copy()
    all_data.loc[all_data_0.index, pred_col] = pred

for col in questionable_0:
    all_data['predicted_{}'.format(col)] = all_data['predicted_{}'.format(col)].apply(int)
del rfr
del corr_cols


# ## 2. Other Features
# ### 2.1. Aspect & Slope

# In[ ]:


def aspect_slope(df):
    df['AspectSin'] = np.sin(np.radians(df.Aspect))
    df['AspectCos'] = np.cos(np.radians(df.Aspect))
    df['AspectSin_Slope'] = df.AspectSin * df.Slope
    df['AspectCos_Slope'] = df.AspectCos * df.Slope
    df['AspectSin_Slope_Abs'] = np.abs(df.AspectSin_Slope)
    df['AspectCos_Slope_Abs'] = np.abs(df.AspectCos_Slope)
    df['Hillshade_Mean'] = df[['Hillshade_9am',
                              'Hillshade_Noon',
                              'Hillshade_3pm']].apply(np.mean, axis = 1)
    return df


# ### 2.2. Distances

# In[ ]:


def distances(df):
    horizontal = ['Horizontal_Distance_To_Fire_Points', 
                  'Horizontal_Distance_To_Roadways',
                  'Horizontal_Distance_To_Hydrology']
    
    df['Euclidean_to_Hydrology'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)
    df['EuclidHydro_Slope'] = df.Euclidean_to_Hydrology * df.Slope
    df['Elevation_VDH_sum'] = df.Elevation + df.Vertical_Distance_To_Hydrology
    df['Elevation_VDH_diff'] = df.Elevation - df.Vertical_Distance_To_Hydrology
    df['Elevation_2'] = df.Elevation**2
    df['Elevation_3'] = df.Elevation**3
    df['Elevation_log1p'] = np.log1p(df.Elevation) # credit: https://www.kaggle.com/evimarp/top-6-roosevelt-national-forest-competition/notebook
    
    for col1, col2 in combinations(zip(horizontal, ['HDFP', 'HDR', 'HDH']), 2):
        df['{0}_{1}_diff'.format(col1[1], col2[1])] = df[col1[0]] - df[col2[0]]
        df['{0}_{1}_sum'.format(col1[1], col2[1])] = df[col1[0]] + df[col2[0]]
    
    df['Horizontal_sum'] = df[horizontal].sum(axis = 1)
    return df


# ### 2.3. Soil_Type & Wilderness

# In[ ]:


def OHE_to_cat(df, colname, data_range): # data_range = [min_index, max_index+1]
    df[colname] = sum([i * df[colname + '{}'.format(i)] for i in range(data_range[0], data_range[1])])
    return df


# ### 2.4. Rockiness

# In[ ]:


soils = [
    [7, 15, 8, 14, 16, 17,
     19, 20, 21, 23], #unknow and complex 
    [3, 4, 5, 10, 11, 13],   # rubbly
    [6, 12],    # stony
    [2, 9, 18, 26],      # very stony
    [1, 24, 25, 27, 28, 29, 30,
     31, 32, 33, 34, 36, 37, 38, 
     39, 40, 22, 35], # extremely stony and bouldery
]
soil_dict = {}
for index, soil_group in enumerate(soils):
    for soil in soil_group:
        soil_dict[soil] = index

def rocky(df):
    df['Rocky'] = sum(i * df['Soil_Type' + str(i)] for i in range(1,41))
    df['Rocky'] = df['Rocky'].map(soil_dict)
    return df


# # III. Model fitting

# ## 1. Data Transform

# In[ ]:


t = time()
all_data = aspect_slope(all_data)
all_data = distances(all_data)
all_data = OHE_to_cat(all_data, 'Wilderness_Area', [1,5])
all_data = OHE_to_cat(all_data, 'Soil_Type', [1,41])
all_data = rocky(all_data)
all_data.drop(['Soil_Type7', 'Soil_Type15', 'train'] + questionable_0, axis = 1, inplace = True)

# Important columns: https://www.kaggle.com/hoangnguyen719/beginner-eda-and-feature-engineering
important_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology'
                  , 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways'
                  , 'Hillshade_Noon', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1'
                  , 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type3', 'Soil_Type4', 'Soil_Type10'
                  , 'predicted_Hillshade_9am', 'predicted_Hillshade_3pm', 'AspectSin', 'AspectCos'
                  , 'AspectSin_Slope', 'AspectCos_Slope', 'AspectSin_Slope_Abs', 'AspectCos_Slope_Abs'
                  , 'Hillshade_Mean', 'Euclidean_to_Hydrology', 'EuclidHydro_Slope'
                  , 'Elevation_VDH_sum', 'Elevation_VDH_diff', 'Elevation_2', 'Elevation_3'
                  , 'Elevation_log1p', 'HDFP_HDR_diff', 'HDFP_HDR_sum', 'HDFP_HDH_diff'
                  , 'HDFP_HDH_sum', 'HDR_HDH_diff', 'HDR_HDH_sum', 'Horizontal_sum'
                  , 'Wilderness_Area', 'Soil_Type', 'Rocky'
                 ]

all_data = all_data[important_cols]
print('Total data transforming time: {}'.format(timer(t)))


# ### Memory reduction

# In[ ]:


X_train = all_data.iloc[:n_train,:].copy()
X_test = all_data.iloc[n_train:, :].copy()
del all_data

def mem_reduce(df):
    # credit: https://www.kaggle.com/arateris/2-layer-k-fold-learning-forest-cover
    t = time()
    start_mem = df.memory_usage().sum() / 1024.0**2
    for col in df.columns:
        if df[col].dtype=='float64': 
            df[col] = df[col].astype('float32')
        if df[col].dtype=='int64': 
            if df[col].max()<1: df[col] = df[col].astype(bool)
            elif df[col].max()<128: df[col] = df[col].astype('int8')
            elif df[col].max()<32768: df[col] = df[col].astype('int16')
            else: df[col] = df[col].astype('int32')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Reduce from {0:.3f} MB to {1:.3f} MB (decrease by {2:.2f}%)'.format(start_mem, end_mem, 
                                                                (start_mem - end_mem)/start_mem*100))
    print('Total memory reduction time: {}'.format(timer(t)))
    return df

X_train = mem_reduce(X_train)
print('='*10)
X_test=mem_reduce(X_test)
gc.collect()


# ## 2. Model fitting

# In[ ]:


X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train
                                                                           , test_size = 0.3
                                                                           , random_state = rand_state
                                                                           )
del X_test


# In[ ]:


# RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 719
                             , max_depth = 464
                             , max_features = 0.3
                             , min_samples_split = 2
                             , min_samples_leaf = 1
                             , bootstrap = False
                             , verbose = 0
                             , random_state = rand_state
                            )
# ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators = 177
                          , max_depth = 794
                          , max_features = 0.9
                          , min_samples_leaf = 1
                          , min_samples_split = 2
                          , bootstrap = False
                          )

# AdaBoostClassifier
adac = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 16)
                         , n_estimators = 794
                         , learning_rate = 1
                         )

# LightGBMClassifier
lgbc = LGBMClassifier(num_leaves = 50
                      , max_depth = 15
                      , learning_rate = 0.1
                      , n_estimators = 1000
                      , reg_lambda = 0.1
                      , objective = 'multiclass'
                      , num_class = 7
                     )

# KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1
                           , n_jobs =-1)


# In[ ]:


meta_clf = RandomForestClassifier(n_estimators = 700
                                  , max_depth = 300
                                  , min_samples_split = 2
                                  , min_samples_leaf = 1
                                  , max_features = 1
                                  , bootstrap = False
                                 )

scc = StackingCVClassifier(classifiers = [rfc, etc, adac, lgbc, knn]
                           , meta_classifier = meta_clf
                           , cv = 3
                           , random_state = rand_state
                           , use_probas = True
                           , verbose = 0
                          )

t = time()
scc.fit(X_train_train, y_train_train)
print('Total training time: {}'.format(timer(t)))
y_train_predict = scc.predict(X_train_test)
y_train_predict = pd.Series(y_train_predict, index = X_train_test.index)


# In[ ]:


# OUTPUT prediction
train = reload('train.csv')
test = reload('test.csv')
train_train = train.iloc[X_train_train.index - 1,].copy()
train_test = train.iloc[X_train_test.index - 1,].copy() # because index=5864 equal position=5863
train_test['Actual'] = y_train_test
train_test['Predict'] = y_train_predict


# # IV. Mis-classified Inspection
# ## 1. Actual - Predicted distribution

# In[ ]:


errors = train_test.groupby('Actual')['Predict'].value_counts().sort_index().unstack(level=1).fillna(0)
for col in errors.columns:
    errors[str(col)+'_pct'] = round(errors[col] / errors.sum(axis=1),3)
plt.figure(figsize=(10,10))
sns.heatmap(errors[[str(col) + '_pct' for col in range(1,8)]], annot=True)
plt.show()


# `Cover_Type` 1 and 2 seem to be the most mistaken. Let's see what other types of mistake are also frequent (frequency threshold set at 0.05, meaning any noteworthy error types are those that occur 5% of more times).

# In[ ]:


# stacking the errors table above
errors = errors[[str(i)+'_pct' for i in range(1,8)]].stack().reset_index()
errors.columns = list(errors.columns[:2]) + ['Actual_Predict_rate']
errors['Actual_Predict'] = [str(actual)+'_'+pred[0] for actual,pred in zip(errors.Actual
                                                                           , errors.Predict
                                                                          )]

print('Number of noticeable errors (errors with Actual_Predict_rate >= 0.05): {}'.format(len(errors[errors.Actual_Predict_rate>= 0.05]) - 7))
print(*[x for x in errors[errors.Actual_Predict_rate >= 0.05].Actual_Predict if int(x[0]) != int(x[2])], sep=', ')

noticeable_errors = [x for x in errors[errors.Actual_Predict_rate >= 0.05].Actual_Predict if int(x[0]) != int(x[2])]


# Alright, so `Cover_Type` 1 and 2 are easily mistaken with one another, and so are type 3 and 6.

# In[ ]:


# add error rate in the dataset
train_test['Mis_Classified'] = [1 if x==False else 0 for x in train_test.Actual == train_test.Predict ]
train_test['Actual_Predict'] = train_test.Actual.astype(str) + '_' + train_test.Predict.astype(str)
train_test = train_test.merge(errors[['Actual_Predict_rate', 'Actual_Predict']], on='Actual_Predict', how='left')
del errors
train_test.head()


# In[ ]:





# ## 2. Distribution of Significant Error Types

# In[ ]:


numerical = ['Elevation', 'Horizontal_Distance_To_Hydrology',
             'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
             'Horizontal_Distance_To_Fire_Points',
             'Aspect', 'Slope', 
             'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

categorical = ['Soil_Type{}'.format(i) for i in range(1,41) if i!=7 and i!=15] + ['Wilderness_Area{}'.format(i) for i in range(1,5)]


# In[ ]:


desc = train_test[numerical].describe().T.join(train_train[numerical].describe().T
                                                     , how='left'
                                                     , lsuffix = '_predict'
                                                     , rsuffix = '_train'
                                                    )

desc = desc.reindex([measure+df for measure in ['min_', '50%_', 'mean_', 'std_', 'max_']
                    for df in ['predict', 'train']], axis = 1)
print('Test and Train dataset:')
desc


# In[ ]:


desc = train_test[train_test.Mis_Classified == 0][numerical].describe().T
desc = desc.join(train_test[train_test.Mis_Classified == 1][numerical].describe().T
                 , how='left'
                 , lsuffix = '_cor'
                 , rsuffix = '_mis'
                )

desc = desc.reindex([measure+df for measure in ['min_', '50%_', 'mean_', 'std_', 'max_']
                    for df in ['cor', 'mis']], axis = 1)
print('Correctly-classified and mis-classified:')
desc


# In[ ]:


def distplot(df, columns, colors=['red', 'green', 'blue', 'c', 'purple'], bins_num = None, hist = True, kde = False): 
    # df is either dataframe or list of ('name_df',df)
    # col is either string or list
    sns.set_style('whitegrid')
#### CONVERT INPUT DATA'S TYPE
    if type(df) != list: 
        df = [('df',df)] 
    if type(columns) == str: 
        columns = [columns]
    l_col = len(columns)
    l_df = len(df)
###### CALCULATE ROWS AND COLS OF GRAPHS
    c = min([l_col, 3]) # cols
    r = l_col//3 + sum([l_col%3!=0]) # rows
    fig = plt.figure(figsize=(c*7, r*6))
    
    for index in range(l_col):
        column = columns[index]
####### CALCULATE BINS OF HIST
        if bins_num == None: 
            combined_data = np.hstack(tuple([df[x][1][column] for x in range(l_df)])) 
            n_bins = min(50,len(np.unique(combined_data))) # number of bins: <= 50
            bins = np.histogram(combined_data, bins=n_bins)[1] # get "edge" of each bin
        bins = next(b for b in [bins_num, bins] if b is not None)
####### ADD SUBPLOT AND PLOT
        ax = fig.add_subplot(r,c,index+1) 
        for i in range(l_df):
            sns.distplot(df[i][1][column], bins=bins, hist = hist, kde=kde, color=colors[i], 
                         label=df[i][0], norm_hist=True, hist_kws={'alpha':0.4})
        plt.xlabel(column)
        if (l_df>1) & ((index+1) % c == 0): # legend at the graph on the right
            ax.legend()
    plt.tight_layout()
    plt.show() 


# In[ ]:


distplot([('correct',train_test[train_test.Mis_Classified == 0])
         , ('incorrect', train_test[train_test.Mis_Classified == 1])
         , ('full_train', train)
         , ('full_test', test)]
        , columns = numerical
         , hist = False
         , kde = True
        )


# Mis-classified often have
# - higher `Elevation` (~ 3,000)
# - `Horizontal_Distance_To_Roadways` and `Horizontal_Distance_To_Fire_Points` slightly less skew to the right
# - `Slope` skewer to the right
# - `Hillshade_3pm` skewer to the left
# <br><br>
# than the whole training set.

# In[ ]:


train_test = OHE_to_cat(train_test, 'Wilderness_Area', [1,5])
train_test = OHE_to_cat(train_test, 'Soil_Type', [1,41])
train_test


# ### 2.1. Elevation
# First, let's look at the distribution of `Elevation` among the 7 `Cover_Type`.

# In[ ]:


f = plt.figure(figsize=(8,6))
sns.boxplot(x='Cover_Type', y='Elevation', data=train)
plt.show()


# Let's look at `Elevation` distribution of `Cover_Type` with significant error rates (1, 2, 3 and 6)

# In[ ]:


train_test['types'] = np.where(train_test.Actual.isin([1,2]), 'Cover Type 1 or 2'
                                , np.where(train_test.Actual.isin([3,6]), 'Cover Type 3 or 6', 'Other Types'))
distplot([(x,train_test[(train_test['types'] == x) & (train_test.Mis_Classified == 1)]) for x in ['Cover Type 1 or 2'
                                                                                                     , 'Cover Type 3 or 6'
                                                                                                     , 'Other Types']]
        , columns = 'Elevation'
         , hist = False
         , kde = True
        )
train_test.drop(columns = 'types', inplace = True)


# It seems that the spike seen above was most likely due to the 1_2 and 2_1 error types. To confirm this, let's see the `Elevation` histogram by `Actual_Predict`.

# In[ ]:


train_test['errors'] = np.where(train_test.Actual_Predict.isin(['1_2','2_1'])
                                , '1_2 or 2_1'
                                , np.where(train_test.Actual_Predict.isin(['3_6', '6_3']), '3_6 or 6_3', 'other errors')
                               )
distplot([(x,train_test[(train_test['errors'] == x) & (train_test.Mis_Classified == 1)]) for x in ['1_2 or 2_1', '3_6 or 6_3', 'other errors']]
        , columns = 'Elevation'
         , hist = False
         , kde = True
        )
train_test.drop(columns = 'errors', inplace = True)


# From this, we can see that though `Elevation` is a powerful predictor of `Cover_Type`, it can still confuse between `Cover_Type1` and `Cover_Type2`.

# In[ ]:


distplot([(a_p,train_test[train_test.Actual_Predict == a_p]) for a_p in ['1_1', '1_2', '2_1', '2_2']]
        , columns = 'Elevation'
         , hist = False
         , kde = True
        )


# Let's dig deeper to in other features of `Actual_Predict` type1_2 and 2_1.

# In[ ]:


train_test['errors'] = np.where(train_test.Actual_Predict.isin(['1_2','2_1'])
                                , 'error 1-2'
                                , np.where(train_test.Actual_Predict.isin(['1_1', '2_2']), 'correct 1-2', 'others')
                               )

distplot([(e,train_test[train_test.errors == e]) for e in ['error 1-2', 'correct 1-2']]
        , columns = numerical
         , hist = False
         , kde = True
        )

train_test.drop(columns = 'errors', inplace=True)


# In[ ]:




