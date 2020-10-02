#!/usr/bin/env python
# coding: utf-8

# # IDEAS
# * replace  missing values with mean of THE CLASS - causes overfitting
# 
# ### itay
# * fillna with mean is keeping the dtype of the column (ints are not becoming floats)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import SelectFromModel
from numpy import sort
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import plot_importance
from imblearn.pipeline import Pipeline
from tpot import TPOTClassifier
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tpot.builtins import ZeroCount, OneHotEncoder
from xgboost import XGBClassifier
import catboost
# consts 
sub_index = 68

train_file_path = '../input/saftey_efficay_myopiaTrain.csv'
test_file_path = '../input/saftey_efficay_myopiaTest.csv'
not_numeric_cols = ['D_L_Sex', 'D_L_Eye','D_L_Dominant_Eye','Pre_L_Contact_Lens','T_L_Laser_Type', 'T_L_Treatment_Type', 
                    'T_L_Cust._Ablation','T_L_Micro','T_L_Head','T_L_Therapeutic_Cont._L.','T_L_Epith._Rep.']
not_relevant_features = ['Pre_L_Pupil_Day']

good_cols=['D_L_Sex', 'D_L_Eye', 'Pre_L_Contact_Lens','T_L_Laser_Type', 'T_L_Treatment_Type','T_L_Cust._Ablation','T_L_Micro','T_L_Head','T_L_Therapeutic_Cont._L.','T_L_Epith._Rep.']

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read train and test sets
train_df_copy = pd.read_csv(train_file_path)
test_df_copy = pd.read_csv(test_file_path)
train_df_copy.replace(to_replace=[' ', 'nan'], value=np.nan, inplace=True)
train_df_copy.dropna(how="all", axis=0, inplace=True)


# In[ ]:


to_binarize = ['D_L_Eye', 'D_L_Sex', 'D_L_Dominant_Eye', 'Pre_L_Contact_Lens', 'T_L_Laser_Type', 'T_L_Treatment_Type', 'T_L_Cust._Ablation', 'T_L_Micro', 'T_L_Head','T_L_Epith._Rep.', 'T_L_Therapeutic_Cont._L.']
unknown_types = ['Pre_L_Pupil_Day', 'Pre_L_Cycloplegia_Sph', 'Pre_L_Cycloplegia_Cyl', 'Pre_L_Cycloplegia_Axis', 'T_L_Actual_AblDepth', 'T_L_PTK_mm', 'T_L_PTK_mmm']
level_2_features = ['Pre_L_Pupil_Day','Pre_L_Pupil_Night','Pre_L_Pachymetry','Pre_L_Average_K','Pre_L_Spherical_Equivalence']

train_df_copy = pd.get_dummies(train_df_copy, columns=to_binarize)
test_df_copy = pd.get_dummies(test_df_copy, columns=to_binarize)
# Edit when the means of groups are known
train_df_copy['Pre_L_Average_K_new_dist'] = train_df_copy.apply (lambda row: min(
    abs(row['Pre_L_Average_K']-42.980000), 
    abs(row['Pre_L_Average_K']-44.000000), 
    abs(row['Pre_L_Average_K']-45.050000)),axis=1)

test_df_copy['Pre_L_Average_K_new_dist'] = test_df_copy.apply (lambda row: min(
    abs(row['Pre_L_Average_K']-42.980000), 
    abs(row['Pre_L_Average_K']-44.000000), 
    abs(row['Pre_L_Average_K']-45.050000)),axis=1)

train_df_copy = train_df_copy.apply(lambda x : x.fillna(x.mean()), axis = 0)
test_df_copy = test_df_copy.apply(lambda x : x.fillna(x.mean()), axis = 0)

y = train_df_copy['Class']

common_cols = list(set(train_df_copy.columns) & set(test_df_copy.columns))
x, test_df_copy = train_df_copy[common_cols], test_df_copy[common_cols]
        
# remove constant columns in the training set
# x.drop(not_relevant_features, axis=1, inplace=True)
# test_df_copy.drop(not_relevant_features, axis=1, inplace=True)

for m in level_2_features:
    for n in level_2_features:
        if n!=m:
            x['{}_{}'.format(m,n)] = x[m]/x[n]
            test_df_copy['{}_{}'.format(m,n)] = test_df_copy[m]/test_df_copy[n]

x.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
test_df_copy.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
test_df_copy = test_df_copy.apply(lambda z : z.fillna(z.mean()), axis = 0)
x = x.apply(lambda z : z.fillna(z.mean()), axis = 0)

            
            
test_df = test_df_copy

print(x.shape)
print(test_df_copy.shape)


# In[ ]:


# Trying stacking - Adding classsifications as features
kf = KFold(n_splits=2)
for (train_index, test_index), (m,n) in zip(kf.split(x), [(catboost.CatBoostClassifier(custom_metric='AUC', iterations=100), 'cat0'),
                                                          (catboost.CatBoostClassifier(custom_metric='AUC', iterations=100), 'rf1'),
                                                         ]):
    X_train = x.values[train_index]
    y_train = y[train_index]
    rus = RandomUnderSampler(random_state=7, ratio=1)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    m.fit(X=X_train, y=y_train)
    x['{}_ans'.format(n)] = [a[1] for a in m.predict_proba(x)]
    test_df_copy['{}_ans'.format(n)] = [a[1] for a in m.predict_proba(test_df_copy)]


# # OLD VERSION
# to_binarize = ['D_L_Eye', 'D_L_Sex', 'D_L_Dominant_Eye', 'Pre_L_Contact_Lens', 'T_L_Laser_Type', 'T_L_Treatment_Type', 'T_L_Cust._Ablation', 'T_L_Micro', 'T_L_Head','T_L_Epith._Rep.', 'T_L_Therapeutic_Cont._L.']
# unknown_types = ['Pre_L_Pupil_Day', 'Pre_L_Cycloplegia_Sph', 'Pre_L_Cycloplegia_Cyl', 'Pre_L_Cycloplegia_Axis', 'T_L_Actual_AblDepth', 'T_L_PTK_mm', 'T_L_PTK_mmm']
# train_df_copy = pd.get_dummies(train_df_copy, columns=to_binarize)
# test_df_copy = pd.get_dummies(test_df_copy, columns=to_binarize)
# # Edit when the means of groups are known
# train_df_copy['Pre_L_Average_K_new_dist'] = train_df_copy.apply (lambda row: min(
#     abs(row['Pre_L_Average_K']-42.980000), 
#     abs(row['Pre_L_Average_K']-44.000000), 
#     abs(row['Pre_L_Average_K']-45.050000)),axis=1)
# 
# test_df_copy['Pre_L_Average_K_new_dist'] = test_df_copy.apply (lambda row: min(
#     abs(row['Pre_L_Average_K']-42.980000), 
#     abs(row['Pre_L_Average_K']-44.000000), 
#     abs(row['Pre_L_Average_K']-45.050000)),axis=1)
# 
# train_df_copy = train_df_copy.apply(lambda x : x.fillna(x.mean()), axis = 0)
# test_df_copy = test_df_copy.apply(lambda x : x.fillna(x.mean()), axis = 0)
# 
# y = train_df_copy['Class']
# 
# common_cols = list(set(train_df_copy.columns) & set(test_df_copy.columns))
# x, test_df_copy = train_df_copy[common_cols], test_df_copy[common_cols]
#         
# # remove constant columns in the training set
# # x.drop(not_relevant_features, axis=1, inplace=True)
# # test_df_copy.drop(not_relevant_features, axis=1, inplace=True)
# 
# test_df = test_df_copy
# 
# print(x.shape)
# print(test_df_copy.shape)

# In[ ]:


#x = train_df_copy[common_cols + ['Class']]
#info_1 = x[x['Class'] == 1]
#info_0 = x[x['Class'] == 0]
#
#info_1 = info_1.describe().transpose()
#info_0 = info_0.describe().transpose()
#
#info_join = info_1.join(info_0, lsuffix='_1', rsuffix='_0')
#info_join['delta_mean'] = abs(info_join['mean_1'] - info_join['mean_0'])
#info_join.sort_values('delta_mean', ascending=False)


# In[ ]:


params = {
    'classification__n_estimators': [100, 200, 300,500,550,600,1000,1050, 1100, 1150, 1200],
    'classification__criterion': ['gini', 'entropy'],
    'classification__max_depth': [None,4,5,6],
    'classification__min_samples_split': [0.1, 0.3, 0.5,  0.7, 0.9, 1.0 ],
}

aclf = RandomForestClassifier(n_jobs=-1)


# In[ ]:


params = {
    'classification__n_estimators': [100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050],
    'classification__base_estimator__criterion': ['gini', 'entropy'],
    'classification__base_estimator__max_depth': [None,3,5,8],
    'classification__base_estimator__min_samples_split': [0.1, 0.3, 0.5, 0.7, 0.9],
    'classification__algorithm': ['SAMME', 'SAMME.R']
}

aclf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())


# In[ ]:



params = {
    'classification__n_estimators': [100,150,200,300,400,500,550,600,700,750,800,850,900,1000, 1100, 1200],
    'classification__max_depth': [3, 5, 6, 8],
    'classification__subsample': [ 0.85, 0.9, 0.95],
    #'classification__learning_rate': [0.1, 0.05, 0.01],
    'classification__colsample_bytree': [0.3, 0.5, 0.7],
    'classification__gamma': [1,5,10]
}

aclf = xgb.XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          learning_rate=0.01,  
                          colsample_bytree = 0.4,
                          subsample = 0.95,
                          objective='binary:logistic', 
                          n_estimators=1000, 
                          reg_alpha = 0.3,
                          max_depth=3, 
                          gamma=10)


# In[ ]:


from matplotlib.legend_handler import HandlerLine2D

def generate_graph(n_ests):
    name, n_ests = n_ests
    train_results = []
    test_results = []
    rus = RandomUnderSampler(random_state=7)
    rus.fit(x, y)
    X_train_res, y_train_res = rus.fit_resample(x, y)
    
    x_train, x_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size=0.33)
    for n in n_ests:
        model =  GradientBoostingClassifier(n_estimators=n)
        # Cross validation
        
        
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = model.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
        
    line1, = plt.plot(n_ests, train_results, 'b', label='Train AUC')
    line2, = plt.plot(n_ests, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel(name)
    plt.show()
#generate_graph(('n_estimators', [50*(i+1) for i in range(20)]))


# scores = ['roc_auc'] #['precision', 'recall']
# 
# 
# model = Pipeline([
#         ('sampling', RandomUnderSampler(random_state=6)),
#         ('classification', aclf)
#     ])
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
# 
#     clf = GridSearchCV(estimator=model, param_grid=params, cv=3,
#                        scoring='%s' % score,
#                       verbose=1, n_jobs=-1)
#     
#     clf.fit(x.values, y.values)
# 
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
# 
# 
# 

# # Read train and test sets
# train_df_copy = pd.read_csv(train_file_path)
# test_df_copy = pd.read_csv(test_file_path)
# 
# # Create copies for debugging n' shit
# train_df = train_df_copy
# test_df = test_df_copy
# 
# # clear data and one hot encoding
#     # remove rows with no info
# train_df.replace(to_replace=[' ', 'nan'], value=np.nan, inplace=True)
# train_df.dropna(how="all", axis=0, inplace=True)
# 
#     # remove columns with more than 0.5 nans in the 1 class or if more than 0.95 nans in class 0
# train_df_ones = train_df[train_df['Class'] == 1]
# train_df_zeros = train_df[train_df['Class'] == 0]
# 
# 
# for col in train_df_ones.columns:
#     percent_missing = train_df_ones[col].isnull().sum() / len(train_df_ones[col])
#     if percent_missing >= 0.75:
#         not_relevant_features.append(col)
#         
# for col in train_df_zeros.columns:
#     percent_missing = train_df_zeros[col].isnull().sum() / len(train_df_zeros[col])
#     if percent_missing > 0.50:
#         not_relevant_features.append(col)
#         
# 
# # train_df_ones = train_df_ones.fillna(train_df_ones.mean())
# # train_df_zeros = train_df_zeros.fillna(train_df_zeros.mean())
# 
# # train_df = pd.concat([train_df_ones, train_df_zeros])
#         
# train_df.drop(not_relevant_features, inplace=True, axis=1)
# 
#     # one hot encoding
# train_df = pd.get_dummies(train_df)
# train_df = train_df.apply(lambda x : x.fillna(x.mean()), axis = 0)
# 
# y = train_df['Class']
# 
# test_df = pd.get_dummies(test_df)    
# test_df = test_df.apply(lambda x : x.fillna(x.mean()), axis = 0)
# 
# # normalize cols
# # cols = train_df.columns
# # for x in cols: 
# #     train_df[x]=(train_df[x]-train_df[x].min())/(train_df[x].max()-train_df[x].min())
# # cols = test_df.columns
# # for x in cols: 
# #     test_df[x]=(test_df[x]-test_df[x].min())/(test_df[x].max()-test_df[x].min())
# 
# # train_df['D_L_Age'] = pd.qcut(train_df['D_L_Age'], q=5, labels=[0,1,2,3,4])
# # test_df['D_L_Age'] = pd.qcut(test_df['D_L_Age'], q=5, labels=[0,1,2,3,4])
# 
# common_cols = list(set(train_df.columns) & set(test_df.columns))
# x, test_df = train_df[common_cols], test_df[common_cols]
# 
# colsToRemove = []
# for col in common_cols:
#     if x[col].std() == 0: 
#         colsToRemove.append(col)
#         
# # remove constant columns in the training set
# x.drop(colsToRemove, axis=1, inplace=True)
# test_df.drop(colsToRemove, axis=1, inplace=True)
# 
# print(x.shape)
# print(test_df.shape)
# print('ree')

# In[ ]:



# Does only damage for now...
def add_OtherAgg(train, test, features):
    train['Mean'] = train[features].mean(axis=1)
    train['Max'] = train[features].max(axis=1)
    train['Var'] = train[features].var(axis=1)
    train['Std'] = train[features].std(axis=1)

    test['Mean'] = test[features].mean(axis=1)
    test['Max'] = test[features].max(axis=1)
    test['Var'] = test[features].var(axis=1)
    test['Std'] = test[features].std(axis=1)

    return train, test

# print(x.shape)
# x, test_df = add_OtherAgg(x, test_df, ['Pre_L_Pachymetry', 'Pre_L_Steep_Axis_max', 'Pre_L_Steep_Axis_min','Pre_L_K_Minimum','Pre_L_Average_K'])
# print(x.shape)


# # model
# 
# model = TPOTClassifier(generations=10, population_size=50, cv=3, random_state=42, verbosity=3,n_jobs=-1, scoring='roc_auc', periodic_checkpoint_folder='.')
# 
# rus = RandomUnderSampler(random_state=7)
# rus.fit(x, y)
# X_train_res, y_train_res = rus.fit_resample(x, y)
# 
# X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size=0.33, random_state=7)
# 
# # fit model on all training data
# model.fit(X_train, y_train)
# # make predictions for test data and evaluate
# print(model.score(X_test, y_test))
# 
# model.export('tpot_exported_pipeline.py')

# In[ ]:


#file = open('tpot_exported_pipeline.py', 'r') 
#for line in file.readlines():
#    print(line)


# In[ ]:


model = xgb.XGBClassifier(silent=True, scale_pos_weight=1, learning_rate=0.01,colsample_bytree = 0.3, subsample = 0.95, objective='binary:logistic', n_estimators=850, reg_alpha = 0.5, max_depth=3, gamma=5)

model1 = catboost.CatBoostClassifier(custom_metric='AUC',iterations=200 )
sel_model = SelectFromModel(estimator=model1, max_features=25, threshold=-np.inf, )
# Cross validation
rus = RandomUnderSampler(random_state=7)
rus.fit(x, y)
X_train_res, y_train_res = rus.fit_resample(x, y)
#x_train, x_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size=0.33)
sel_model.fit(X_train_res, y_train_res)
X_reduced = sel_model.transform(X_train_res)

scores = cross_val_score(model, X_reduced, y_train_res, cv=5)
print("{} : {}".format(scores, sum(scores)/len(scores)))


# In[ ]:


# model
#model = xgb.XGBClassifier(silent=True, scale_pos_weight=1, learning_rate=0.01,colsample_bytree = 0.3, subsample = 0.95, objective='binary:logistic', n_estimators=850, reg_alpha = 0.5, max_depth=3, gamma=5)
model = catboost.CatBoostClassifier(custom_metric='AUC', iterations=100)
# Cross validation
rus = RandomUnderSampler(random_state=7)
rus.fit(x, y)
X_train_res, y_train_res = rus.fit_resample(x, y)
scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
print("{} : {}".format(scores, sum(scores)/len(scores)))


# In[ ]:





# In[ ]:


#'classification__colsample_bytree': 0.3, 'classification__gamma': 10, 'classification__max_depth': 8, 'classification__n_estimators': 1100, 'classification__subsample': 0.85
# get preds for test set
model1 = xgb.XGBClassifier(silent=True, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.3,#0.4,#0.3,
                      subsample = 0.85,#0.95,
                      objective='binary:logistic', 
                      n_estimators=1100,#100,#850, 
                      reg_alpha = 0.3,#0.5,
                      max_depth=8,#5,#3, 
                      gamma=10)#10)#5)
# 'classification__criterion': 'gini', 'classification__max_depth': None, 'classification__min_samples_split': 0.1, 'classification__n_estimators': 200
model2 = RandomForestClassifier(n_jobs=-1, n_estimators=200, max_depth=None, criterion='gini', min_samples_split=0.1)

# 'classification__algorithm': 'SAMME', 'classification__base_estimator__criterion': 'entropy',
#'classification__base_estimator__max_depth': 3, 'classification__base_estimator__min_samples_split': 0.9, 'classification__n_estimators': 100
model3 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=0.9, criterion='entropy', max_depth=3), n_estimators=100, algorithm='SAMME')
model = VotingClassifier(estimators=[('xgb', model1), ('rf', model2), ('ada', model3)], voting='soft')

model1 = xgb.XGBClassifier(silent=True, scale_pos_weight=1, learning_rate=0.01,colsample_bytree = 0.3, subsample = 0.95, objective='binary:logistic', n_estimators=850, reg_alpha = 0.5, max_depth=3, gamma=5)

model = catboost.CatBoostClassifier(custom_metric='AUC', iterations=200)
sel_model = SelectFromModel(estimator=model, max_features=35, threshold=-np.inf)

exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.1, sparse=False, threshold=10),
    XGBClassifier(learning_rate=0.001, max_depth=7, min_child_weight=15, n_estimators=100, nthread=1, subsample=0.6500000000000001)
)
model = exported_pipeline
model = xgb.XGBClassifier(silent=True, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.3,#0.4,#0.3,
                      subsample = 0.85,#0.95,
                      objective='binary:logistic', 
                      n_estimators=1100,#100,#850, 
                      reg_alpha = 0.3,#0.5,
                      max_depth=8,#5,#3, 
                      gamma=10)#10)#5)


rus = RandomUnderSampler(random_state=7, ratio=1)
rus.fit(x, y)
X_train_res, y_train_res = rus.fit_resample(x, y)
model.fit(X_train_res, y_train_res)
###################

##################
test_preds = model.predict_proba(test_df.values)
test_preds


# In[ ]:


# Make submission file
sub_name = "submission"+str(sub_index)
final_preds = [x[1] for x in test_preds]
final_id = [i for i in range(1, len(test_preds) + 1)]

sub = pd.DataFrame()
sub['id'] = final_id
sub['class'] = final_preds

sub.to_csv(sub_name+'.csv', index=False)


# In[ ]:


# inspect data
# for col in not_numeric_cols:
#     print("{} value: {} in train set   &   {} in test set".format(col,train_df_copy[col].unique(),test_df_copy[col].unique()))
deb = pd.read_csv(train_file_path)
print(good_cols)
deb

