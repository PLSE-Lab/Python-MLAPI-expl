#!/usr/bin/env python
# coding: utf-8

# # Probability of Default modeling
# 
# We are going to create a model that estimates a probability for a borrower to default her loan.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


ls /kaggle/input/credit-default-prediction-ai-big-data/


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('/kaggle/input/credit-default-prediction-ai-big-data/train.csv', 
                 index_col='Id')


# In[ ]:


df.columns = ['_'.join(col.split(' ')).lower() for col in df.columns]


# In[ ]:


df.T


# In[ ]:


years_dict = {'-1': -1, '10+ years': 10, '8 years': 8, '6 years': 6, 
              '7 years': 7, '5 years': 5, '1 year': 1, '< 1 year': 0, 
              '4 years': 4, '3 years': 3, '2 years': 2, '9 years': 9}

df['years_in_current_job'] = (df['years_in_current_job']
                              .fillna('-1')
                              .map(years_dict))


# In[ ]:


df['credit_default'].value_counts()


# In[ ]:


round(1e2 * df['credit_default'].value_counts()/len(df), 2)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.nunique()


# In[ ]:


(1e2 * df.isnull().sum()/len(df)).plot(kind='barh')
plt.xlim(0, 10**2)
plt.grid();


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(6, 5))
sns.heatmap(df.corr());


# ## Pipeline

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, 
                                   OneHotEncoder, LabelEncoder)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# In[ ]:


num_feat = df.drop('credit_default', axis=1).select_dtypes(include=np.number).columns
cat_feat = df.drop('credit_default', axis=1).select_dtypes(include=['object']).columns
X = df.drop('credit_default', axis=1)
y = df['credit_default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# In[ ]:


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    # ('imputer', KNNImputer()),
    ('scaler', StandardScaler()),
    # ('poly', PolynomialFeatures(2, include_bias=False)), 
    # ('scaler1', StandardScaler()),
    # ('pca', PCA(n_components=100))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('one_hot', OneHotEncoder(handle_unknown='ignore')),
])


# In[ ]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_feat),
        ('cat', categorical_transformer, cat_feat)
    ])


# ## Working Example

# In[ ]:


pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',  LogisticRegression())
])

model = pipe.fit(X_train, y_train)
target_names = y_test.unique().astype(str)
y_pred = model.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred, target_names=target_names))


# In[ ]:


print(round(pd.DataFrame(confusion_matrix(y_test, y_pred)/len(y_test)*1e2)))


# In[ ]:


f1_score(y_true=y_test, y_pred=y_pred)


# In[ ]:


from yellowbrick.classifier import DiscriminationThreshold

fig, ax = plt.subplots(figsize=(6, 6))
model_viz = DiscriminationThreshold(pipe)
model_viz.fit(X_train, y_train)
model_viz.poof();


# # Multiple Models

# In[ ]:


from time import time
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import (LogisticRegression, RidgeClassifier, 
                                  SGDClassifier, PassiveAggressiveClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              ExtraTreesClassifier, VotingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


# In[ ]:


results = pd.DataFrame(columns=['Name', 'f1', 'f1_test', 'StdDev(%)', 'Time(s)'])

for model in [
    DummyClassifier,
    LinearDiscriminantAnalysis,
    LogisticRegression, 
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
    GaussianNB,
    KNeighborsClassifier,
#     SVC,
    DecisionTreeClassifier,
    RandomForestClassifier, 
#     GradientBoostingClassifier,
    ExtraTreesClassifier,
#     MLPClassifier,
    XGBClassifier,
    LGBMClassifier
]:
    pipe = make_pipeline(preprocessor, model())
    start_time = time()
    kfold = StratifiedKFold(n_splits=4, random_state=1)
    scores = cross_val_score(pipe, X_train, y_train, scoring='f1', cv=kfold)
    pipe.fit(X_train, y_train)
    scores_test = f1_score(y_true=y_test, y_pred=pipe.predict(X_test))
    time_mod = time() - start_time
    results = results.append({
        'Name' : model.__name__, 
        'f1' : round(scores.mean(), 4), 
        'f1_test' : round(scores_test, 4), 
        'StdDev(%)' : round(1e2*scores.std(), 2), 
        'Time(s)': round(time_mod, 2)
    }, ignore_index=True)
    del pipe
    print('Analyzed {}.'.format(model.__name__))
print('Done!')

results = results.sort_values('f1', ascending=False)


# In[ ]:


results


# # Method 1

# In[ ]:


X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)


# In[ ]:


from mlxtend.classifier import StackingClassifier

clfs = [x for x in [GaussianNB(), 
                    XGBClassifier(), 
                    LGBMClassifier(),                     
                    DecisionTreeClassifier(), 
                    #RandomForestClassifier(), 
                    #ExtraTreesClassifier(), 
                    #SVC(probability=True), 
                    #LogisticRegression(), 
                    #KNeighborsClassifier()
                   ]]

stack = StackingClassifier(classifiers=clfs, meta_classifier=KNeighborsClassifier())

kfold = StratifiedKFold(n_splits=10, random_state=0)

cross_val_score(stack, X_train_prepared, y_train, scoring='f1', cv=kfold).mean()


# In[ ]:


from yellowbrick.classifier import DiscriminationThreshold

fig, ax = plt.subplots(figsize=(6, 6))
model_viz = DiscriminationThreshold(stack)
model_viz.fit(X_train_prepared, y_train)
model_viz.poof();


# In[ ]:


def threshold_optimizer(X_train, X_test, y_train, y_test):
    scores, thresholds = [], []
    for threshold in np.linspace(0, 1, 21):
        y_pred = np.zeros(len(y_test))

        all_models = [
            GaussianNB(), XGBClassifier(), 
            DecisionTreeClassifier(), LGBMClassifier(), 
            RandomForestClassifier(), ExtraTreesClassifier(), 
            SVC(probability=True), LogisticRegression(), 
            KNeighborsClassifier()
        ]

        for model in all_models:
            model.fit(X_train, y_train)
            y_pred += model.predict_proba(X_test)[:, 1]
#             y_pred += model.predict(X_test)


        y_pred /= len(all_models)
        y_pred = (y_pred > threshold) * 1

        thresholds.append(threshold)
        scores.append(f1_score(y_test, y_pred))
    return thresholds, scores


# In[ ]:


thresholds, scores = threshold_optimizer(X_train_prepared, X_test_prepared, y_train, y_test)


# In[ ]:


plt.plot(thresholds, scores)
plt.scatter(thresholds, scores)
plt.legend(['f1 Scores'])
plt.xlabel('Threshold')
plt.ylabel('f1 Score');
#plt.grid();


# In[ ]:


df_test = pd.read_csv('/kaggle/input/credit-default-prediction-ai-big-data/test.csv', 
                      index_col='Id')

df_test.columns = ['_'.join(col.split(' ')).lower() for col in df_test.columns]
df_test['years_in_current_job'] = (df_test['years_in_current_job']
                                   .fillna('-1')
                                   .map(years_dict))

X_prepared = preprocessor.fit_transform(X)
X_sub_prepared = preprocessor.transform(df_test)

all_models = [
    GaussianNB(), XGBClassifier(), 
    DecisionTreeClassifier(), LGBMClassifier(), 
    RandomForestClassifier(), ExtraTreesClassifier(), 
    SVC(probability=True), LogisticRegression(), 
    KNeighborsClassifier()
]

y_pred = np.zeros(len(df_test))

for model in all_models:
    model.fit(X_prepared, y)
    y_pred += model.predict_proba(X_sub_prepared)[:, 1]

y_pred /= len(all_models)
y_pred = (y_pred > .35) * 1

df_sub = pd.read_csv('/kaggle/input/credit-default-prediction-ai-big-data/sampleSubmission.csv', 
                     index_col='Id')

df_sub['Credit Default'] = y_pred


# # Grid Search

# In[ ]:


# X_train_prepared = preprocessor.fit_transform(X_train)
# X_test_prepared = preprocessor.transform(X_test)


# ### Note: Grid Search models are commented out, before submission.

# In[ ]:


# dtc = DecisionTreeClassifier()
# param_grid = [
#     {
#         'max_depth': [None, 2, 3, 5], 
#         'min_samples_split': [128,512, 2048], 
#         'min_samples_leaf': [2, 4, 6, 8], 
#         'max_features': ['auto', 'sqrt', 'log2', None]}
# ]
# grid_search = GridSearchCV(dtc, param_grid=param_grid, cv=5, scoring='f1')
# grid_search.fit(X_train_prepared, y_train)
# print(grid_search.best_estimator_)
# grid_search.score(X_test_prepared, y_test) # f1_score(y_test, grid_search.predict(X_test_prepared))


# In[ ]:


# dtc = XGBClassifier()
# param_grid = [
#     {
#         'eta': [0, .25, .5, .75, 1],
#         'gamma': [0, 10, 100, 1000], 
#         'max_depth': [2, 4, 6, 8, 10], 
#     }
# ]
# grid_search = GridSearchCV(dtc, param_grid=param_grid, cv=5, scoring='f1')
# grid_search.fit(X_train_prepared, y_train)
# print(grid_search.best_estimator_)
# grid_search.score(X_test_prepared, y_test)


# In[ ]:


# gnb = GaussianNB() 
# dtc = DecisionTreeClassifier(min_samples_leaf=6, min_samples_split=128)
# xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, eta=1, gamma=0, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=1, max_delta_step=0, max_depth=2,
#               min_child_weight=1, monotone_constraints='()',
#               n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#               tree_method='exact', validate_parameters=1, verbosity=None)
# lgb = LGBMClassifier()
# # svm = SVC(probability=True)
# # log = LogisticRegression()

# voting_clf = VotingClassifier(
#     estimators=[
#         ('gnb', gnb), 
#         ('dtc', dtc), 
#         ('xgb', xgb), 
# #         ('lgb', lgb)
#     ], 
#     voting='soft'
# )

# voting_clf.fit(X_train_prepared, y_train)

# for clf in (gnb, dtc, xgb, lgb, voting_clf):
#     clf.fit(X_train_prepared, y_train)
#     y_pred = clf.predict(X_test_prepared)
#     print(clf.__class__.__name__, f1_score(y_test, y_pred))


# In[ ]:


# for threshold in np.linspace(0, 1, 21):
#     y_pred = np.zeros(len(y_test))
#     y_pred = voting_clf.predict_proba(X_test_prepared)[:, 1]
#     y_pred = (y_pred > threshold) * 1
#     print('{:.4f}: {:.2f}'.format(f1_score(y_test, y_pred), threshold))


# In[ ]:


# df_test = pd.read_csv('/kaggle/input/credit-default-prediction-ai-big-data/test.csv', 
#                       index_col='Id')

# df_test.columns = ['_'.join(col.split(' ')).lower() for col in df_test.columns]
# df_test['years_in_current_job'] = (df_test['years_in_current_job']
#                                    .fillna('-99')
#                                    .map(years_dict))

# voting_clf = VotingClassifier(
#     estimators=[
#         ('gnb', gnb), 
#         ('dtc', dtc), 
#         ('xgb', xgb), 
# #        ('lgb', lgb)
#     ], 
#     voting='soft'
# )

# X_prepared = preprocessor.transform(X)
# voting_clf.fit(X_prepared, y)

# X_sub_test = df_test
# X_sub_test_prepared = preprocessor.transform(X_sub_test)

# df_sub = pd.read_csv('/kaggle/input/credit-default-prediction-ai-big-data/sampleSubmission.csv', 
#                      index_col='Id')

# df_sub['Credit Default'] = voting_clf.predict(X_sub_test_prepared)


# # Final Submission

# In[ ]:


df_sub.to_csv('submission.csv')

