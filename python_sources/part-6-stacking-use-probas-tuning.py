#!/usr/bin/env python
# coding: utf-8

# This kernel is part 6 of my work in this competition with other work listed below. Any comments or suggestions you may have are greatly appreciated!
# 1. [EAD and Feature Engineering](https://www.kaggle.com/hoangnguyen719/1-eda-and-feature-engineering/notebook) (preliminary data exploration and features reduction)
# 2. [ExtraTreesClassifier tuning](https://www.kaggle.com/hoangnguyen719/extratree-tuning) (hyper-parameter tuning for ExtraTree model)
# 3. [AdaboostClassifier tuning](https://www.kaggle.com/hoangnguyen719/adaboost-tuning) (hyper-parameter tuning for AdaBoostClassifier model)
# 4. [LGBMClassifier tuning](https://www.kaggle.com/hoangnguyen719/lightgbm-tuning) (hyper-parameter tuning for LightGBM Classifier model)
# 5. [KNearestClassifier tuning](https://www.kaggle.com/hoangnguyen719/knn-tuning) (hyper-parameter tuning for KNearestNeighbor Classifier model)
# 6. [StackingCVClassifier (use_probas tuning)](https://www.kaggle.com/hoangnguyen719/stacking-use-probas-tuning) (stacking multiple classifier using StackingCV)
# 7. [Mis-Classified Inspection](https://www.kaggle.com/hoangnguyen719/mis-classified-inspection) (examining mis-classified instances)

# *Note*: Many of my EDA parts have been greatly inspired by previous kernels in the competition and I have been trying to give credits to the owners as much as I can. However, because (1) many kernels appear to have the same ideas (even codes), which makes it hard to trace back where the ideas originated from, and (2) I carelessly forgot to note down all the sources (this is totally my bad), sometimes the credit may not be given where it's due. I apologize beforehand, and please let me know in the comment section if you have any question or suggestions. Thank you!

# In[ ]:


import pandas as pd
import numpy as np
from itertools import product, combinations
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
from sklearn.model_selection import cross_val_score, GridSearchCV


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


# # I. Load Data

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
# ## 1. Impute "Fake" 0s
# @arateris discovered in [this notebook](https://www.kaggle.com/arateris/2-layer-k-fold-learning-forest-cover) that some 0 values in _Hillshade\_3pm_ were quite odd and actually could have been errors in data entry. He used `RandomForestRegressor` to impute these values which he believed should have been different than 0. 
# <br><br>
# In my [EAD notebook](https://www.kaggle.com/hoangnguyen719/eda-and-feature-engineering) I expanded the search to all other features that contain 0s and found _Hillshade\_9am_ to suspicious. The imputation section below, which also uses `RandomForestRegressor` therefore is applied on both _Hillshade\_9am_ and _Hillshade\_3pm_.

# In[ ]:


questionable_0 = ['Hillshade_9am', 'Hillshade_3pm']

corr_cols = {'Hillshade_9am': ['Hillshade_3pm', 'Aspect', 'Slope', 'Soil_Type10', 'Wilderness_Area1'
                               ,'Wilderness_Area4', 'Vertical_Distance_To_Hydrology']
            , 'Hillshade_3pm': ['Hillshade_9am', 'Hillshade_Noon', 'Slope', 'Aspect']
            }


# In[ ]:


rfr = RandomForestRegressor(n_estimators = 100, random_state = rand_state, verbose = 1, n_jobs = -1)

# for col in questionable_0: 
#     print('='*20)
#     scores = cross_val_score(rfr,
#                              all_data_non0[corr_cols[col]], 
#                              all_data_non0[col],
#                              n_jobs = -1)
#     print(col + ': {0:.4} (+/- {1:.4}) ## [{2}]'.format(scores.mean(), scores.std()*2, ', '.join(map(str, np.round(scores,4)))))

# ====================
# Hillshade_9am: 1.0 (+/- 0.00056) ## [0.9995, 0.9993, 0.9988]
# ====================
# Hillshade_3pm: 1.0 (+/- 0.0029) ## [0.9981, 0.9971, 0.9947]

## NEAR PERFECT SCORES FOR ALL => no need further feature engineering for questionable_0 predictions


# In[ ]:


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


# ## 2. Other Features
# ### 2.1. Aspect, Slope & Shadow

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


# ### 2.2. Distances & Elevation

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


# ### 2.3. Categorical

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


# ### 2.5. Combining everything

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


# ## III. Summary and Output

# ### 3.1 Reduce data memory
# Credit to @arateris in this __[notebook](https://www.kaggle.com/arateris/2-layer-k-fold-learning-forest-cover)__.

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


# ### 3.2. Set up models
# The hyperparameters are tuned and chosen from the following notebooks:
# - [ExtraTreesClassifier tuning](https://www.kaggle.com/hoangnguyen719/extratree-tuning)
# - [AdaBoostClassifier tuning](https://www.kaggle.com/hoangnguyen719/adaboost-tuning)
# - [LGBMClassifier tuning](https://www.kaggle.com/hoangnguyen719/lightgbm-tuning)
# - [KNearestNeighbors tuning](https://www.kaggle.com/hoangnguyen719/knn-tuning)
# - [RandomForestClassifier tuning](https://www.kaggle.com/joshofg/pure-random-forest-hyperparameter-tuning) (by @joshofg ).

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
adac = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 19)
                         , n_estimators = 794
                         , learning_rate = 1
                         )

# HistGradientBoostingClassifier
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


# ### 3.3. Hyper-params tuning for `StackingCVClassifier` 's `use_probas`
# Meta-classifier was chosen based on a separate notebook from `RandomForestClassifier`, `ExtraTreeClassifier` and `AdaboostClassifier`, among which `RandomForestClassifier` has the best performance. Due to limited kernel memory, the meta-classifier's hyper-parameters were chosen arbitrarily based on the `RandomForestClassifier` tuning result mentioned above.

# In[ ]:


meta_clf = RandomForestClassifier(n_estimators = 700
                                  , max_depth = 300
                                  , min_samples_split = 2
                                  , min_samples_leaf = 1
                                  , max_features = 1
                                  , bootstrap = False
                                 )
params = {'use_probas': [True, False]
         }
scc = StackingCVClassifier(classifiers = [rfc, etc, adac, lgbc, knn]
                           , meta_classifier = meta_clf
                           , cv = 3
                           , random_state = rand_state
                          )

grid = GridSearchCV(estimator = scc
                  , param_grid = params
                  , n_jobs = -1
                  , cv = 3
                  , scoring = 'accuracy'
                  , verbose = 1
                  )
t = time()
grid.fit(X_train, y_train)
print('Total training time: {}'.format(timer(t)))

predict = grid.predict(X_test)
print('Best hyper-parameters found:')
print(grid.best_params_)
print('\nFitting time:')
print(grid.refit_time_)
print('\Best score:')
print(grid.best_score_)


# Tuning's best 10 outputs

# In[ ]:


results = pd.DataFrame(grid.cv_results_)
results.sort_values(by=['rank_test_score'], inplace=True)
results[['params', 'mean_fit_time','mean_test_score', 'rank_test_score']].head(10)


# In[ ]:


print('Best 10 hyper-params combinations: ')
print(results[['params', 'mean_fit_time','mean_test_score', 'rank_test_score']].head(10))


# ### 3.4. Output for submission

# In[ ]:


output = pd.DataFrame({'Id': index_test
                       ,'Cover_Type': predict
                      })
output.to_csv('Submission.csv', index=False)

