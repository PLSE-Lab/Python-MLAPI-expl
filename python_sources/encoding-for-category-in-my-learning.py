#!/usr/bin/env python
# coding: utf-8

# ### Reference
# 
# 1. [Handling Categorical Variables:Encoding & Modeling](https://www.kaggle.com/vikassingh1996/handling-categorical-variables-encoding-modeling)
# 2. [An Overview of Encoding Techniques](https://www.kaggle.com/shahules/an-overview-of-encoding-techniques/notebook)
# 3. [Categorical Data encoding techniques](https://www.kaggle.com/ruchibahl18/categorical-data-encoding-techniques)
# 4. [Category Encoders Examples](https://www.kaggle.com/discdiver/category-encoders-examples)
# 5. [Entity embeddings to handle categories](https://www.kaggle.com/abhishek/entity-embeddings-to-handle-categories)
# 6. [Why Not Logistic Regression?](https://www.kaggle.com/peterhurford/why-not-logistic-regression)

# ### Backgroud

# * **Binary data**: A binary variable is a variable with only two values, like 1/0, such as bin_0,bin_1,bin_2.
# * **Categorical data** 
#     * **Ordinal data**: An ordinal variable is a categorical variable with a ordering. For low ordinal features, like ord_1, ord_2, ord_3, ord_4. For high ordinal data, like ord_5.
#     * **Nominal data**: Nominal variables contain two or more categories without a natural ordering. For low nominal features, like nom_0, nom_1, nom_2, nom_3, nom_4. For high-cardinality nominal features, like nom_5, nom_6, nom_7, nom_8, nom_9.
# * **Timeseries data**: Time series data, like day or moth, it seems to be a cyclical continuous features.

# ### Which Encoding methods is suitable to deal with above categorical features?

# I pick up some methods from excellent notebooks that have published in this kaggle competition, and then try it as the following.
# 
# * To binary data, I will do logical judgement.
# * To low ordial data, I will try LabelEncoder or simple replacement. But to high ordial data, I maybe use OrdinalEncoder.
# * To low nominal data, dummy variable is a common method. But to high nonmial data, I maybe use LeaveOneOutEncoder or FeatureHasher

# ### Import packages and data

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from mlxtend.classifier import StackingCVClassifier
import copy


# In[ ]:


train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col=['id'])
test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col=['id'])


# In[ ]:


train.dtypes


# In[ ]:


display(train.head())


# In[ ]:


X = train.drop("target", axis = 1)
y = train.loc[:,"target"]


# ### Encoding for catagory 

# #### Binary data

# In[ ]:


X.bin_3 = X.bin_3.apply(lambda x: 1 if x == "T" else 0)
X.bin_4 = X.bin_4.apply(lambda x: 1 if x == "Y" else 0)

print(X.columns)


# #### Nominal data

# In[ ]:


# h = FeatureHasher(input_type='string', n_features=1000)
# X[['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']].values
# hash_X = h.fit_transform(X[['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']].values)
# hash_X = pd.DataFrame(hash_X.toarray())

# hash_X.columns
# X = X.drop(["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"], axis=1).join(hash_X)

loo_encoder = LeaveOneOutEncoder(cols=["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"])
loo_X = loo_encoder.fit_transform(X[["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]], y)
X = X.drop(["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"], axis=1).join(loo_X)

X = X.drop(["nom_0", "nom_1", "nom_2", "nom_3", "nom_4"], axis=1)         .join(pd.get_dummies(X[["nom_0", "nom_1", "nom_2", "nom_3", "nom_4"]]))

print(X.columns)


# #### Ordinal data

# In[ ]:


X.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)
X.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)

for i in ["ord_3", "ord_4"]:
   le = LabelEncoder()
   X[[i]] = le.fit_transform(X[[i]])

oe = OrdinalEncoder(categories='auto')
X.ord_5 = oe.fit_transform(X.ord_5.values.reshape(-1,1))

print(X.columns)


# #### Timeseries data

# In[ ]:


def date_cyc_enc(df, col, max_vals):
   df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)
   df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)
   return df

X = date_cyc_enc(X, 'day', 7)
X = date_cyc_enc(X, 'month', 12)
X.drop(['day', 'month'], axis=1, inplace = True)

print(X.columns)


# ### Model refinement
# 
# #### Try it on linear model

# In[ ]:


# lr = LogisticRegression()
# scores_lr = cross_val_score(lr, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lr.mean(), scores_lr.std() * 2))


# In[ ]:


# rc = RidgeClassifier()
# scores_rc = cross_val_score(rc, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rc.mean(), scores_rc.std() * 2))


# In[ ]:


# lda = LinearDiscriminantAnalysis()
# scores_lda = cross_val_score(lda, X_new, y, cv=5, n_jobs=1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lda.mean(), scores_lda.std() * 2))


# In[ ]:


# linear_svm = LinearSVC(penalty="l2")
# scores_linear_svm = cross_val_score(linear_svm, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_linear_svm.mean(), scores_linear_svm.std() * 2))


# LogisticRegression, RidgeClassifier and LinearDiscriminantAnalysis revel better accuracy than LinearSVC.

# #### Try it on classification model

# In[ ]:


# fr = DecisionTreeClassifier(random_state=0)
# scores_dt = cross_val_score(fr, X, y, cv=5, n_jobs=2)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_dt.mean(), scores_dt.std() * 2))


# In[ ]:


# sgdc = SGDClassifier()
# scores_sgdc = cross_val_score(sgdc, X, y, cv=5, n_jobs=4)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_sgdc.mean(), scores_sgdc.std() * 2))


# In[ ]:


# ab = AdaBoostClassifier()
# scores_ab= cross_val_score(ab, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_ab.mean(), scores_ab.std() * 2))


# In[ ]:


# gbm = GradientBoostingClassifier()
# scores_gbm= cross_val_score(gbm, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_gbm.mean(), scores_gbm.std() * 2))


# In[ ]:


# rf = RandomForestClassifier()
# scores_rf= cross_val_score(rf, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))


# In[ ]:


# et = ExtraTreesClassifier()
# scores_et= cross_val_score(et, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_et.mean(), scores_et.std() * 2))


# In[ ]:


# xgb = XGBClassifier()
# scores_xgb= cross_val_score(xgb, X, y, cv=5, n_jobs=-1)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores_xgb.mean(), scores_xgb.std() * 2))


# We can see that GradientBoostingClassifier, XGBClassifier maybe have the best accuracy in above methods, which the accuracy of AdaBoostClassifier is also close to. Otherwise, others maybe performent not good.

# #### Grid Search for best params of XGBClassifier model

# In[ ]:


# params = {
#         'min_child_weight': [1, 5, 10, 13, 15],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5, 10, 20]
#         }
# xgb = XGBClassifier(silent=True, nthread=1)
# folds = 3
# param_comb = 5

# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

# random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, 
#                                    scoring='accuracy', n_jobs=-1, cv=skf.split(X, y), 
#                                    verbose=3, random_state=1001 )
# random_search.fit(X, y)

# print('\n All results:')
# print(random_search.cv_results_)
# print('\n Best estimator:')
# print(random_search.best_estimator_)
# print('\n Best hyperparameters:')
# print(random_search.best_params_)


# In[ ]:


# means = random_search.cv_results_['mean_test_score']
# stds = random_search.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, random_search.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))


# #### Grid Search for best params of GradientBoostingClassifier model

# In[ ]:


# params2 = {
#         'n_estimators': [50, 100, 300, 800],
#         'learning_rate': [0.01, 0.1, 0.5, 1],
#         'max_depth': [3, 10, 20, 50],
#         'min_samples_split': [100, 200, 500, 800],
#         'subsample': [0.2, 0.4, 0.6, 0.8, 1.0]
#         }
# gbm = GradientBoostingClassifier(random_state=1001)

# random_search2 = RandomizedSearchCV(gbm, param_distributions=params2, n_iter=param_comb, 
#                                    scoring='accuracy', n_jobs=-1, cv=skf.split(X, y), 
#                                    verbose=3, random_state=1001 )
# random_search2.fit(X, y)

# print('\n All results:')
# print(random_search2.cv_results_)
# print('\n Best estimator:')
# print(random_search2.best_estimator_)
# print('\n Best hyperparameters:')
# print(random_search2.best_params_)


# In[ ]:


# means = random_search2.cv_results_['mean_test_score']
# stds = random_search2.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, random_search2.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))


# #### Grid Search for best params of LogisticRegression model

# In[ ]:


# lr_params = {'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 
#              'C': [0.01, 0.1, 0.5, 1]
#             }
# lr = LogisticRegression(random_state=1001)

# random_search3 = RandomizedSearchCV(lr, param_distributions=lr_params, n_iter=param_comb, 
#                                    scoring='accuracy', n_jobs=-1, cv=skf.split(X, y), 
#                                    verbose=3, random_state=1001 )
# random_search3.fit(X, y)

# print('\n All results:')
# print(random_search3.cv_results_)
# print('\n Best estimator:')
# print(random_search3.best_estimator_)
# print('\n Best hyperparameters:')
# print(random_search3.best_params_)


# In[ ]:


# means = random_search3.cv_results_['mean_test_score']
# stds = random_search3.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, random_search3.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))


# ### Stacked model

# In[ ]:


# xgb_clf = XGBClassifier(booster='gbtree', gamma=5, colsample_bytree=0.8,
#                         learning_rate=0.1, max_depth=10, 
#                         min_child_weight=10, n_estimators=100, 
#                         silent=True, subsample=0.8)

# ab_clf = AdaBoostClassifier(n_estimators=200,
#                             base_estimator=DecisionTreeClassifier(
#                                 min_samples_leaf=2,
#                                 random_state=1001),
#                             random_state=1001)

# gbm_clf = GradientBoostingClassifier(n_estimators=300, min_samples_split=100,
#                                  max_depth=50, learning_rate=1, subsample=0.8,
#                                  random_state=1001)

# lr = LogisticRegression()

# stack = StackingCVClassifier(classifiers=[xgb_clf, gbm_clf, ab_clf], 
#                             meta_classifier=lr,
#                             cv=5,
#                             stratify=True,
#                             shuffle=True,
#                             use_probas=True,
#                             use_features_in_secondary=True,
#                             verbose=1,
#                             random_state=1001,
#                             n_jobs=-1)
# stack = stack.fit(X, y)


# ### Predictions

# In fact, as above analysis, I found the stack model in test data is very bad !
# 
# At least, I found if I use LogisticRegression model only or other linear model, the result is better. **For example, if I preprocess all features as above steps and perform LR model with default parameters, the score of this competition is 0.80266**
# 
# So I decided to preprocess the train data again with one-hot-encoding, and build model of LR. The result is shown below.

# #### Encoding for train/test data

# In[ ]:


X_train = train.drop("target", axis = 1)
y_train = train.loc[:,"target"]


# In[ ]:


X_train.bin_3 = X_train.bin_3.apply(lambda x: 1 if x == "T" else 0)
X_train.bin_4 = X_train.bin_4.apply(lambda x: 1 if x == "Y" else 0)

X_train.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)
X_train.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)

for i in ["ord_3", "ord_4"]:
   le = LabelEncoder()
   X_train[[i]] = le.fit_transform(X_train[[i]])

oe = OrdinalEncoder(categories='auto')
X_train.ord_5 = oe.fit_transform(X_train.ord_5.values.reshape(-1,1))


# In[ ]:


X_test = copy.deepcopy(test)


# In[ ]:


X_test.bin_3 = X_test.bin_3.apply(lambda x: 1 if x == "T" else 0)
X_test.bin_4 = X_test.bin_4.apply(lambda x: 1 if x == "Y" else 0)

X_test.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)
X_test.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)

for i in ["ord_3", "ord_4"]:
    le = LabelEncoder()
    X_test[[i]] = le.fit_transform(X_test[[i]])

oe = OrdinalEncoder(categories='auto')
X_test.ord_5 = oe.fit_transform(X_test.ord_5.values.reshape(-1,1))


# In[ ]:


data = pd.concat([X_train, X_test])
print(data.shape)


# In[ ]:


columns = data.columns
dummies = pd.get_dummies(data,
                         columns=columns,
#                          drop_first=True,
                         sparse=True)


# In[ ]:


print(dummies.shape)
print(X_train.shape[0])


# In[ ]:


X_train = dummies.iloc[:X_train.shape[0], :]
X_test = dummies.iloc[X_train.shape[0]:, :]


# In[ ]:


del dummies
del data
print (X_train.shape)
print(X_test.shape)


# In[ ]:


X_train = X_train.sparse.to_coo().tocsr()
X_test = X_test.sparse.to_coo().tocsr()

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")


# #### Making predictions

# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict_proba(X_test)
pred[:10,1]


# >**In fact, as above preprocess, I first transform some features with some normal methods ,and then I preprocess all features with get_dummies function, the score of competition drops from 0.80266 to 0.80184.**
# **However, if I adjust LR model with tuning parameters, the score will be upgrade to 0.80801.**
# **But I think these three scores are keeping in same level, so those scores can not indicate that which preprocess or encoding way is better.**

# In[ ]:


# lr = LogisticRegression(solver="lbfgs", C=0.1, max_iter=10000)
# lr.fit(X_train, y_train)
# pred2 = lr.predict_proba(X_test)
# pred2[:10,1]


# #### Export predictions and submission

# In[ ]:


predictions = pd.Series(pred[:,1], index=test.index, dtype=y.dtype)
predictions.to_csv("./submission.csv", header=['target'], index=True, index_label='id')

