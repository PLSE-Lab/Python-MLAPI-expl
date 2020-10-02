#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from datetime import datetime as dt\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom sklearn.base import BaseEstimator, TransformerMixin\nfrom sklearn.model_selection import GridSearchCV, cross_val_score\n\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.svm import SVC\nfrom sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier\nfrom xgboost import XGBClassifier\nfrom lightgbm import LGBMClassifier')


# In[ ]:


get_ipython().run_cell_magic('time', '', "### Read Data ###\ndf = pd.read_csv('../input/forest-cover-type-prediction/train.csv')\n#### Check Null Data ###\nif df[df.isnull().any(axis=1) == True].shape[0] != 0:\n    print('Warning, null data present')\n\n### Transform / Wrangle Data ###\nX_train = df.iloc[:, :-1]\nY_train = df.iloc[:, -1]\n\nX_test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')\nX_test_ids = X_test.iloc[:, 0]")


# # Helper Classes/Functions

# In[ ]:


class FeatureTransformer(TransformerMixin):
    '''
    Helper class for transforming input dataframes into desired input features. 
    Implements the feature engineering logic.
    '''
    def __init__(self):
        pass
    
    def fit(self, X):
        ignore_cols = ['Id']
        for col in X.columns:
            if X[col].std() == 0:
                print('Columns to drop: {}, std={}'.format(col, X[col].std()))
                ignore_cols.append(col)
        self.ignore_cols = ignore_cols
        return self
    
    def transform(self, X):
        X = X.copy()
        self.__clean_columns(X)
        return X

    def __clean_columns(self, X):
        drop_cols = self.ignore_cols
        for col in drop_cols:
            if col not in X.columns:
                drop_cols.remove(col)
        X.drop(labels=self.ignore_cols, axis=1, inplace=True)


# In[ ]:


def predict_results(estimator, X_test, X_test_ids):
    '''
    Helper function for predicting and saving test results
    '''
    Y_Pred = pd.DataFrame(estimator.predict(X_test), columns=['Cover_Type'])
    results = pd.concat([X_test_ids, Y_Pred], axis=1)
    results.to_csv('../input/forest-cover-type-prediction/submission.csv', index=False)


# In[ ]:


def get_feature_importances(estimator, X):
    return pd.DataFrame(
        np.array([X.columns, estimator.feature_importances_]).T, 
        columns=['Features', 'Importance']
    ).sort_values(by='Importance', ascending=False)


# # Feature Engineering

# In[ ]:


feature_transformer = FeatureTransformer()
X_train = feature_transformer.fit_transform(X_train)
X_test = feature_transformer.transform(X_test)


# In[ ]:


X_train.head()


# # Using of Plain Single Algorithm Approaches
# 1. LogisticRegression
# 2. SVC
# 3. ExtraTreesClassifier
# 4. RandomForestClassifier
# 5. LGBMClassifier
# 6. XGBClassifier

# ## 1. LogisticRegression
# <i>Approach not chosen as many iterations needed for convergence</i>

# In[ ]:


# %%time
# lrc = LogisticRegression()
# param_grid = [
#     {
#         'n_jobs': [2],
#         'solver': ['lbfgs', 'saga'],
#         'tol': [1e-4, 1e-5],
#         'C': [0.5, 1, 5],
#         'multi_class': ['auto']
#     }
# ]

# gscv = GridSearchCV(estimator=lrc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=2)
# gscv.fit(X_train, Y_train)
# print('Best Params: ', gscv.best_params_)
# print('Best Score: ', gscv.best_score_)

# lrc = gscv.best_estimator_


# ## 2. SVC

# In[ ]:


# %%time
# svc = SVC()
# param_grid = [
#     {
#         'kernel': ['linear', 'rbf'],
#         'tol': [1e-4, 0.001],
#         'C': [0.5, 1, 5],
#         'gamma': ['scale', 'auto']
#     }
# ]

# gscv = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=2)
# gscv.fit(X_train, Y_train)
# print('Best Params: ', gscv.best_params_)
# print('Best Score: ', gscv.best_score_)

# svc = gscv.best_estimator_


# ## 3. RandomForestClassifier

# In[ ]:


# %%time
# rfc = RandomForestClassifier()
# param_grid = [
#     {
#         'n_jobs': [2],
#         'criterion': ['gini', 'entropy'], 
#         'n_estimators': [200, 500, 700], 
#         'max_depth': [3, 15, 30, None],
#         'max_features': [0.3, 0.6, 'auto']
#     }
# ]

# gscv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=2)
# gscv.fit(X_train, Y_train)
# print('Best Params: ', gscv.best_params_)
# print('Best Score: ', gscv.best_score_)

# rfc = gscv.best_estimator_


# ## 4. ExtraTreesClassifier

# In[ ]:


# %%time
# etc = ExtraTreesClassifier()
# param_grid = [
#     {
#         'n_jobs': [2],
#         'criterion': ['gini', 'entropy'], 
#         'n_estimators': [200, 500, 700], 
#         'max_depth': [3, 15, 30, None],
#         'max_features': [0.3, 0.6, 'auto']
#     }
# ]

# gscv = GridSearchCV(estimator=etc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=2)
# gscv.fit(X_train, Y_train)
# print('Best Params: ', gscv.best_params_)
# print('Best Score: ', gscv.best_score_)

# etc = gscv.best_estimator_


# ## 5. LGBMClassifier

# In[ ]:


# %%time
# lgbmc = LGBMClassifier()
# param_grid = [
#     {
#         'n_jobs': [4],
#         'max_depth': [2, 3, -1], 
#         'n_estimators': [150, 200, 250], 
#         'num_leaves': [31, 45, 63, 67],
#         'learning_rate': [0.15, 0.2, 0.25],
#         'reg_lambda': [0, 1.5]
#     }
# ]

# gscv = GridSearchCV(estimator=lgbmc, param_grid=param_grid, n_jobs=4, scoring='accuracy', cv=5)
# gscv.fit(X_train, Y_train)
# print('Best Params: ', gscv.best_params_)
# print('Best Score: ', gscv.best_score_)

# lgbmc = gscv.best_estimator_


# ## 6. XGBClassifier
# <i>XGBoost Best Params:  {'max_depth': 2, 'n_estimators': 50, 'n_threads': 4, 'reg_lambda': 1.6, 'tree_method': 'hist'}
# XGBoost Best Score:  0.658531746031746</i>

# In[ ]:


# %%time
# xgbc = XGBClassifier()
# param_grid = [
#     {
#         'n_jobs': [4],
#         'max_depth': [2, 3, 10, len(X_train.columns)],
#         'n_estimators': [50, 100, 200], 
#         'reg_lambda': [0, 1.6]
#     }
# ]

# gscv = GridSearchCV(estimator=xgbc, param_grid=param_grid, n_jobs=4, scoring='accuracy', cv=5)
# gscv.fit(X_train, Y_train)
# print('Best Params: ', gscv.best_params_)
# print('Best Score: ', gscv.best_score_)

# xgbc = gscv.best_estimator_


# ## Best Classifiers of Each Algorithm Tested

# In[ ]:


get_ipython().run_cell_magic('time', '', "lrc = LogisticRegression(solver='lbfgs', multi_class='auto')\nsvc = SVC(gamma='scale')\nrfc = RandomForestClassifier(criterion='entropy', max_features=0.6, n_estimators=500, n_jobs=6)\netc = ExtraTreesClassifier(criterion='entropy', max_features=0.6, n_estimators=500, n_jobs=6)\nlgbmc = LGBMClassifier(learning_rate=0.2, n_estimators=200, num_leaves=63, n_jobs=6)\nxgbc = XGBClassifier(max_depth=2, n_estimators=50, reg_lambda=1.6, tree_method='hist', n_jobs=6)\n\nprint('LogisticRegression Accuracy: ', cross_val_score(estimator=lrc, X=X_train, y=Y_train, scoring='accuracy', cv=3))\nprint('SVC Accuracy: ', cross_val_score(estimator=svc, X=X_train, y=Y_train, scoring='accuracy', cv=3))\nprint('RandomForestClassifier Accuracy: ', cross_val_score(estimator=rfc, X=X_train, y=Y_train, scoring='accuracy', cv=3))\nprint('ExtraTreesClassifier Accuracy: ', cross_val_score(estimator=etc, X=X_train, y=Y_train, scoring='accuracy', cv=3))\nprint('LGBMClassifier Accuracy: ', cross_val_score(estimator=lgbmc, X=X_train, y=Y_train, scoring='accuracy', cv=3))\nprint('XGBClassifier Accuracy: ', cross_val_score(estimator=xgbc, X=X_train, y=Y_train, scoring='accuracy', cv=3))")


# <strong>ExtraTreesClassifier works best here</strong>

# In[ ]:


get_ipython().run_cell_magic('time', '', "etc = ExtraTreesClassifier(criterion='entropy', max_features=0.6, n_estimators=500, n_jobs=6)\n# Fitting best estimator\netc.fit(X_train, Y_train)\n# Predicting and getting output prediction file\npredict_results(estimator=etc, X_test=X_test, X_test_ids=X_test_ids)")


# Kaggle Score:
# * <strong style="color: green">ExtraTreesClassifier - 0.78275</strong>
# * RandomForestClassifier - 0.75646
# * LGBMClassifier - 0.76851
# * XGBClassifier - 0.58489

# In[ ]:


get_feature_importances(etc, X_train).head(10)


# # Extra Part: Exploring Ensemble Methods
# ### Variables are split and grouped in 4 segments:
# #### 1) Soil Group Vars --> RFC to get proba
# #### 2) Wilderness Area Group Vars --> RFC to get proba
# #### 3) Inclination Group Vars --> RFC or LGBMC to get proba
# #### 4) Spatial Group Vars --> LGBMC to get proba
# 

# ## Perform Study
# ### Prepare Group Vars

# In[ ]:


print(X_train.columns)
print(X_train.columns.shape)


# In[ ]:


X_soil = X_train.loc[:, ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']]
X_wild_area = X_train.loc[:, ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']]
X_incline = X_train.loc[:, ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']]
X_spatial = X_train.loc[:, ['Elevation', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 
                      'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']]

X_soil_test = X_test.loc[:, ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']]
X_wild_area_test = X_test.loc[:, ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']]
X_incline_test = X_test.loc[:, ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']]
X_spatial_test = X_test.loc[:, ['Elevation', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 
                      'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']]


# ## 1. Soil Group Vars

# In[ ]:


# %%time
# ### Soil Type RF Classifier ###
# rfc_soil = RandomForestClassifier()
# #### Perform GridSearchCV to optimize params ####
# rfc_soil_param_grid = [
#     {
#         'n_jobs': [6],
#         'n_estimators': [10, 100, 150],
#         'max_depth': [3, 5, None],
#         'criterion': ['gini', 'entropy']
#     }
# ]

# rfc_gscv = GridSearchCV(
#     estimator=rfc_soil, 
#     param_grid=rfc_soil_param_grid, 
#     scoring='neg_log_loss', 
#     cv=5, n_jobs=6
# )
# rfc_gscv.fit(X_soil, Y)

# print('RFC Soil Best Params: ', rfc_gscv.best_params_)
# print('RFC Soil Best Score: ', rfc_gscv.best_score_)

# #### Get best estimator and predict proba ####
# rfc_soil = rfc_gscv.best_estimator_
# Y_proba_soil_test = rfc_soil.predict_proba(X_soil_test)


# ## 2. Wilderness Area Group Vars

# In[ ]:


# %%time
# ### Wilderness Area RF Classifier ###
# rfc_wild_area = RandomForestClassifier()
# #### Perform GridSearchCV to optimize params ####
# rfc_wild_area_param_grid = [
#     {
#         'n_jobs': [6],
#         'n_estimators': [75, 100, 125],
#         'max_depth': [2, None],
#         'criterion': ['gini', 'entropy']
#     }
# ]

# rfc_wild_area_gscv = GridSearchCV(
#     estimator=rfc_wild_area, 
#     param_grid=rfc_wild_area_param_grid, 
#     scoring='neg_log_loss', 
#     cv=5, n_jobs=6
# )
# rfc_wild_area_gscv.fit(X_wild_area, Y)

# print('RFC Wilderness Best Params: ', rfc_wild_area_gscv.best_params_)
# print('RFC Wilderness Best Score: ', rfc_wild_area_gscv.best_score_)

# #### Get best estimator and predict proba ####
# rfc_wild_area = rfc_wild_area_gscv.best_estimator_
# Y_proba_wild_area_test = rfc_wild_area.predict_proba(X_wild_area_test)


# ## 3. Inclination Group Vars

# In[ ]:


# %%time
# ### Inclination RF Classifier ###
# rfc_incline = RandomForestClassifier()
# #### Perform GridSearchCV to optimize params ####
# rfc_incline_param_grid = [
#     {
#         'n_jobs': [6],
#         'n_estimators': [10, 100, 150, 200],
#         'max_depth': [3, 5, None],
#         'criterion': ['gini', 'entropy']
#     }
# ]

# rfc_incline_gscv = GridSearchCV(
#     estimator=rfc_incline, 
#     param_grid=rfc_incline_param_grid, 
#     scoring='neg_log_loss', 
#     cv=5, n_jobs=6
# )
# rfc_incline_gscv.fit(X_incline, Y)

# print('RFC Inclination Best Params: ', rfc_incline_gscv.best_params_)
# print('RFC Inclination Best Score: ', rfc_incline_gscv.best_score_)

# #### Get best estimator and predict proba ####
# rfc_incline = rfc_incline_gscv.best_estimator_
# Y_proba_incline_test = rfc_incline.predict_proba(X_incline_test)


# ## 4. Spatial Group Vars

# In[ ]:


# %%time
# ### Inclination RF Classifier ###
# lgbmc_spatial = LGBMClassifier()
# #### Perform GridSearchCV to optimize params ####
# lgbmc_spatial_param_grid = [
#     {
#         'n_jobs': [6],
#         'n_estimators': [200, 250, 275],
#         'learning_rate': [0.125, 0.15, 0.175, 0.2],
#         'num_leaves': [65, 67, 70]
#     }
# ]

# lgbmc_spatial_gscv = GridSearchCV(
#     estimator=lgbmc_spatial, 
#     param_grid=lgbmc_spatial_param_grid, 
#     scoring='neg_log_loss', 
#     cv=5, n_jobs=6
# )
# lgbmc_spatial_gscv.fit(X_spatial, Y)

# print('LGBMC Spatial Best Params: ', lgbmc_spatial_gscv.best_params_)
# print('LGBMC Spatial Best Score: ', lgbmc_spatial_gscv.best_score_)

# #### Get best estimator and predict proba ####
# lgbmc_spatial = lgbmc_spatial_gscv.best_estimator_
# Y_proba_spatial_test = lgbmc_spatial.predict_proba(X_spatial_test)


# ## Get final prediction
# ### Methods:
# * 1) Get weights for group vars, and use softmax to derive final probabilities + one-hot class for final prediction
# * 2) Build another ensemble estimator from the other estimators, and make final prediction
# * 3) TODO: find out ways to feed group vars outputs as intermediate inputs, and feed to another estimator for making final prediction

# ## 1) Weights for each output
# ### i) Prepare wrapper classes for each classifier

# In[ ]:


class SegmentClassifier(BaseEstimator, TransformerMixin):
    
    def __init__(self, classifier, columns):
        self.classifier = classifier
        self.columns = columns
    
    def fit(self, X, y):
        X = X.loc[:, self.columns]
        self.classifier.fit(X, y)
        return self
    
    def predict(self, X):
        X = X.loc[:, self.columns]
        return self.classifier.predict(X)
    
    def predict_proba(self, X):
        X = X.loc[:, self.columns]
        return self.classifier.predict_proba(X)


# In[ ]:


get_ipython().run_cell_magic('time', '', "soil_classifier = SegmentClassifier(\n    classifier=RandomForestClassifier(criterion='entropy', n_estimators=100, n_jobs=6),\n    columns=['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',\n       'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',\n       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16',\n       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',\n       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',\n       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',\n       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',\n       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',\n       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']\n)\n\nwild_area_classifier = SegmentClassifier(\n    classifier=RandomForestClassifier(criterion='gini', n_estimators=100, n_jobs=6),\n    columns=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']\n)\n\nincline_classifier = SegmentClassifier(\n    classifier=RandomForestClassifier(criterion='entropy', n_estimators=150, max_depth=5, n_jobs=6),\n    columns=['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']\n)\n\nspatial_classifier = SegmentClassifier(\n    classifier=LGBMClassifier(learning_rate=0.125, n_estimators=200, num_leaves=65, n_jobs=6),\n    columns=['Elevation', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', \n                      'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']\n)")


# In[ ]:


# ensemble_classifier = VotingClassifier(
#     estimators=[
#         ('soil_classifier', soil_classifier),
#         ('wild_area_classifier', wild_area_classifier),
#         ('incline_classifier', incline_classifier),
#         ('spatial_classifier', spatial_classifier)
#     ]
# )


# In[ ]:


# %%time
# param_grid = [
#     {
#         'voting': ['soft', 'hard'],
#         'weights': [[1,1,2,16], [1,2,3,4], [1,1,4,10]]
#     }
# ]
# gscv = GridSearchCV(ensemble_classifier, param_grid=param_grid, n_jobs=4, cv=5)
# gscv.fit(X_train, Y_train)
# print('Best Params: ', gscv.best_params_)
# print('Best Score: ', gscv.best_score_)


# In[ ]:


# predict_results(estimator=gscv.best_estimator_, X_test=X_test, X_test_ids=X_test_ids)


# Kaggle Score: 0.69541

# Score is lower than the single <strong>LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
#                importance_type='split', learning_rate=0.15, max_depth=-1,
#                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
#                n_estimators=250, n_jobs=6, num_leaves=70, objective=None,
#                random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
#                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)</strong>.
#                
# How about ensembling the single classifier with the ensemble classifier above?
# 
# * Method 1 => Use the 4 SegmentClassifiers + 1 LGBMClassifier in 1 VotingClassifier
# * Method 2 => Use a new VotingClassifier with VotingClassifier from the SegmentClassifers + LGBMClassifer as estimators

# In[ ]:


### Method 1 ###
ensemble_classifier = VotingClassifier(
    estimators=[
        ('soil_classifier', soil_classifier),
        ('wild_area_classifier', wild_area_classifier),
        ('incline_classifier', incline_classifier),
        ('spatial_classifier', spatial_classifier),
        ('original_classifier', etc)
    ]
)


# In[ ]:


get_ipython().run_cell_magic('time', '', "param_grid = [\n    {\n        'voting': ['soft'],\n        'weights': [[1,1,2,3,5], [1,2,3,4,5], [0,0,0,0,1], [0,0,0,2,5]]\n#         'weights': [[0,0,0,3,5], [0,0,0,2,5]]\n    }\n]\ngscv = GridSearchCV(ensemble_classifier, param_grid=param_grid, n_jobs=6, cv=5)\ngscv.fit(X_train, Y_train)\nprint('Best Params: ', gscv.best_params_)\nprint('Best Score: ', gscv.best_score_)")


# In[ ]:


predict_results(estimator=gscv.best_estimator_, X_test=X_test, X_test_ids=X_test_ids)


# Kaggle Score (from earlier score of params = {'voting': 'soft', 'weights': [0, 0, 0, 0, 1]}: 0.76769):
# * params = {'voting': 'soft', 'weights': [1, 1, 2, 3, 5]}: 0.76706
# * params = {'voting': 'soft', 'weights': [0, 0, 2, 3, 5]}: 0.76787
# * params = {'voting': 'soft', 'weights': [0, 0, 0, 3, 5]}: 0.76817
# * params = {'voting': 'soft', 'weights': [0, 0, 0, 2, 5]}: 0.76866

# To try more complex feature engineering as approach to increase score (since ensemble does not increase the score much)
# # Feature Engineering Approach

# Trying out approach by https://www.kaggle.com/jianyu/my-first-submission

# In[ ]:


class FeatureTransformer(TransformerMixin):
    '''
    Implementing __enhance_columns method to add more sophisticated features.
    '''
    def __init__(self):
        pass
    
    def fit(self, X):
        ignore_cols = ['Id']
        for col in X.columns:
            if X[col].std() == 0:
                print('Columns to drop: {}, std={}'.format(col, X[col].std()))
                ignore_cols.append(col)
        self.ignore_cols = ignore_cols
        return self
    
    def transform(self, X):
        X = X.copy()
        self.__clean_columns(X)
        self.__enhance_columns(X)
        return X

    def __clean_columns(self, X):
        drop_cols = self.ignore_cols
        for col in drop_cols:
            if col not in X.columns:
                drop_cols.remove(col)
        X.drop(labels=self.ignore_cols, axis=1, inplace=True)
        
    def __enhance_columns(self, X):
        X.loc[:, 'Distance_To_Hydrology'] = (X.loc[:, 'Horizontal_Distance_To_Hydrology'] ** 2 
            + X.loc[:, 'Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
        X.loc[:, 'Distance_To_Amenities_Avg'] = X.loc[:, [
            'Horizontal_Distance_To_Hydrology', 
            'Horizontal_Distance_To_Roadways', 
            'Horizontal_Distance_To_Fire_Points'
        ]].mean(axis=1)
        X.loc[:, 'Elevation_Minus_Disthy'] = X.loc[:, 'Elevation'] - X.loc[:, 'Vertical_Distance_To_Hydrology']
        X.loc[:, 'Elevation_Plus_Disthy'] = X.loc[:, 'Elevation'] + X.loc[:, 'Vertical_Distance_To_Hydrology']
        X.loc[:, 'Disthx_Minus_Distfx'] = X.loc[:, 'Horizontal_Distance_To_Hydrology'] - X.loc[:, 'Horizontal_Distance_To_Fire_Points']
        X.loc[:, 'Disthx_Plus_Distfx'] = X.loc[:, 'Horizontal_Distance_To_Hydrology'] + X.loc[:, 'Horizontal_Distance_To_Fire_Points']
        X.loc[:, 'Disthx_Minus_Distrx'] = X.loc[:, 'Horizontal_Distance_To_Hydrology'] - X.loc[:, 'Horizontal_Distance_To_Roadways']
        X.loc[:, 'Disthx_Plus_Distrx'] = X.loc[:, 'Horizontal_Distance_To_Hydrology'] + X.loc[:, 'Horizontal_Distance_To_Roadways']
        X.loc[:, 'Distfx_Minus_Distrx'] = X.loc[:, 'Horizontal_Distance_To_Fire_Points'] - X.loc[:, 'Horizontal_Distance_To_Roadways']
        X.loc[:, 'Distfx_Minus_Distrx'] = X.loc[:, 'Horizontal_Distance_To_Fire_Points'] - X.loc[:, 'Horizontal_Distance_To_Roadways']


# In[ ]:


get_ipython().run_cell_magic('time', '', "feature_transformer_new = FeatureTransformer()\nX_train = feature_transformer_new.fit_transform(X_train)\nX_test = feature_transformer_new.transform(X_test)\n\netc = ExtraTreesClassifier(criterion='entropy', max_features=0.6, n_estimators=500, n_jobs=6)\n# Fitting best estimator\netc.fit(X_train, Y_train)\n# Predicting and getting output prediction file\npredict_results(estimator=etc, X_test=X_test, X_test_ids=X_test_ids)")


# Kaggle Score: 0.80805

# In[ ]:


get_feature_importances(etc, X_train).head(10)

