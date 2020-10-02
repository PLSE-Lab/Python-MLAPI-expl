#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from itertools import product, combinations

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

rand_state = 719


# # Load Data

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


# In[ ]:


numerical = ['Elevation', 'Horizontal_Distance_To_Hydrology',
             'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
             'Horizontal_Distance_To_Fire_Points',
             'Aspect', 'Slope', 
             'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

categorical = ['Soil_Type{}'.format(i) for i in range(1,41)] + ['Wilderness_Area{}'.format(i) for i in range(1,5)]


# # Impute "Fake" 0s

# In[ ]:


questionable_0 = ['Hillshade_9am', 'Hillshade_3pm']

corr_cols = {'Hillshade_9am': ['Hillshade_3pm', 'Aspect', 'Slope', 'Soil_Type10', 'Wilderness_Area1',
                   'Wilderness_Area4', 'Vertical_Distance_To_Hydrology'],
             'Hillshade_3pm': ['Hillshade_9am', 'Hillshade_Noon', 'Slope', 'Aspect']
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
# Hillshade_Noon: 1.0 (+/- 0.00076) ## [0.9995, 0.9995, 0.9987] # Hillshade_Noon is removed
# ====================
# Hillshade_3pm: 1.0 (+/- 0.0029) ## [0.9981, 0.9971, 0.9947]
# ====================
# Slope: 0.9985 (+/- 0.001297) ## [0.9979, 0.9994, 0.9982]     # corr_cols['Slope'] with 12 highest corr columns has 0.85 accuracy

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


# # Other Features
# ### Aspect, Slope & Shadow

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


# ### Distances & Elevation

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


# ### Categorical

# In[ ]:


def OHE_to_cat(df, colname, data_range): # data_range = [min_index, max_index+1]
    df[colname] = sum([i * df[colname + '{}'.format(i)] for i in range(data_range[0], data_range[1])])
    return df


# ### Rockiness

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


# ## Summary and Output

# In[ ]:


all_data = aspect_slope(all_data)
all_data = distances(all_data)
all_data = OHE_to_cat(all_data, 'Wilderness_Area', [1,5])
all_data = OHE_to_cat(all_data, 'Soil_Type', [1,41])
all_data = rocky(all_data)
all_data.drop(['Soil_Type7', 'Soil_Type15', 'train'] + questionable_0, axis = 1, inplace = True)


# In[ ]:


X_train = all_data.iloc[:n_train,:].copy()
X_test = all_data.iloc[n_train:, :].copy()
del all_data


# In[ ]:


# NO NEED TO CHOOSE IMPORTANT COLS AS ADABOOST USES TREE STUMP
# clf_rf.fit(X_train, y_train)
# predict = clf_rf.predict(X_test)

# output = pd.DataFrame({'Id': test.index,
#                       'Cover_Type': predict})
# output.to_csv('Submission.csv', index=False)

# importances = pd.DataFrame({'Features': X_train.columns, 
#                                 'Importances': clf_rf.feature_importances_})
# fig = plt.figure(figsize=(10,16))
# sns.barplot(x='Importances', y='Features', data=importances.sort_values(by=['Importances']
#                                                                        , axis = 'index'
#                                                                        , ascending = False)
#            , orient="h")
# plt.xticks(rotation='vertical')
# plt.show()

# important_cols = importances[importances.Importances >= 0.003].Features


# In[ ]:


# base_est = [DecisionTreeClassifier(max_depth = i) for i in [1,3,5,7,10]]
# params = {'base_estimator': base_est
#           , 'n_estimators': np.logspace(1.5,3, 6).astype(int)
#           , 'learning_rate': [0.01, 0.03, 0.1, 0.3, 0.6, 1]
#          }
# adac = AdaBoostClassifier()
# grid = GridSearchCV(estimator = adac
#                   , param_grid = params
#                   , n_jobs = -1
#                   , cv = 3
#                   , scoring = 'accuracy'
#                   , verbose = 2
#                   )

# grid.fit(X_train, y_train)

# print('Best hyper-parameters found:')
# print(grid.best_params_)
# print('\nFitting time:')
# print(grid.refit_time_)
# print('\Best score:')
# print(grid.best_score_)

####### OUTPUT##########
# # Best hyper-parameters found:
# # {'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
# #                        max_features=None, max_leaf_nodes=None,
# #                        min_impurity_decrease=0.0, min_impurity_split=None,
# #                        min_samples_leaf=1, min_samples_split=2,
# #                        min_weight_fraction_leaf=0.0, presort=False,
# #                        random_state=None, splitter='best'), 'learning_rate': 1, 'n_estimators': 1000}

# # Fitting time:
# # 431.2910895347595
# # \Best score:
# # 0.7846560846560846


# In[ ]:


base_est = [DecisionTreeClassifier(max_depth = i) for i in [10,13,16,19]]
params = {'base_estimator': base_est
          , 'n_estimators': np.logspace(2,3.2, 5).astype(int) #array([ 100,  199,  398,  794, 1584])
          , 'learning_rate': [0.01, 0.03, 0.1, 0.3, 1]
         }
adac = AdaBoostClassifier()
grid = GridSearchCV(estimator = adac
                  , param_grid = params
                  , n_jobs = -1
                  , cv = 3
                  , scoring = 'accuracy'
                  , verbose = 2
                  )

grid.fit(X_train, y_train)

print('Best hyper-parameters found:')
print(grid.best_params_)
print('\nFitting time:')
print(grid.refit_time_)
print('\Best score:')
print(grid.best_score_)

predict = grid.predict(X_test)


# In[ ]:


results = pd.DataFrame(grid.cv_results_)
results.sort_values(by=['rank_test_score'], inplace=True)
results.head(10)


# In[ ]:


print(results.head(10))


# In[ ]:


# adac = AdaBoostClassifier(n_estimators = 163
#                , learning_rate = 0.003
#                )
# adac.fit(X_train, y_train) 
# predict = adac.predict(X_test)
output = pd.DataFrame({'Id': index_test
                       ,'Cover_Type': predict
                      })
output.to_csv('Submission.csv', index=False)

