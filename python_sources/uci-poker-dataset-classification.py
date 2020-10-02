#!/usr/bin/env python
# coding: utf-8

# # UCI Poker dataset classification with Pandas, Matplotlib and Scikit-learn
# 
# ***
# 
# We will be using UCI Poker dataset in this kernel.
# 
# Start by importing needed packages.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from xgboost import XGBClassifier


# # Load data
# Let's load data to Pandas dataframes and take a firts look at it.

# In[ ]:


test = pd.read_csv('../input/poker-hand-testing.data', header=None)


# In[ ]:


train = pd.read_csv('../input/poker-hand-training-true.data', header=None)


# In[ ]:


train.columns = ['S1', 'C1','S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5','Label']


# In[ ]:


test.columns = ['S1', 'C1','S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5','Label']


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# # Extract features and labels

# In[ ]:


X_train = train.loc[:,train.columns != 'Label']


# In[ ]:


X_test = test.loc[:,test.columns != 'Label']


# In[ ]:


Y_train = train['Label']


# In[ ]:


Y_test = test['Label']


# # Label distribution in train set

# In[ ]:


Y_train.groupby(Y_train).size()


# # Label distribution in test set

# In[ ]:


Y_test.groupby(Y_test).size()


# # Define helper functions

# In[ ]:


def preprocess_data(data:pd.DataFrame):
    df = data.copy()
    dfc = df[['C1', 'C2', 'C3', 'C4', 'C5']]
    dfc.values.sort()
    df[['C1', 'C2', 'C3', 'C4', 'C5']] = dfc
    df = df[['C1', 'C2', 'C3', 'C4', 'C5', 'S1', 'S2', 'S3', 'S4', 'S5', 'Label']]
    return df


# In[ ]:


def add_counts(df:pd.DataFrame):
    tmp = df[['C1', 'C2', 'C3', 'C4', 'C5']]
    df['Cnt_C1'] = tmp.apply(lambda x: sum(x==x[0]) ,axis=1)
    df['Cnt_C2'] = tmp.apply(lambda x: sum(x==x[1]) ,axis=1)
    df['Cnt_C3'] = tmp.apply(lambda x: sum(x==x[2]) ,axis=1)
    df['Cnt_C4'] = tmp.apply(lambda x: sum(x==x[3]) ,axis=1)
    df['Cnt_C5'] = tmp.apply(lambda x: sum(x==x[4]) ,axis=1)
    
    tmp = df[['S1', 'S2', 'S3', 'S4', 'S5']]
    df['Cnt_S1'] = tmp.apply(lambda x: sum(x==x[0]) ,axis=1)
    df['Cnt_S2'] = tmp.apply(lambda x: sum(x==x[1]) ,axis=1)
    df['Cnt_S3'] = tmp.apply(lambda x: sum(x==x[2]) ,axis=1)
    df['Cnt_S4'] = tmp.apply(lambda x: sum(x==x[3]) ,axis=1)    
    df['Cnt_S5'] = tmp.apply(lambda x: sum(x==x[4]) ,axis=1)


# In[ ]:


def add_diffs(df:pd.DataFrame):
    df['Diff1'] = df['C5'] - df['C4']
    df['Diff2'] = df['C4'] - df['C3']
    df['Diff3'] = df['C3'] - df['C2']
    df['Diff4'] = df['C2'] - df['C1']


# In[ ]:


def add_unique_count(df:pd.DataFrame):
    tmp = df[['S1', 'S2', 'S3', 'S4', 'S5']]
    df['UniqueS'] = tmp.apply(lambda x: len(np.unique(x)) , axis=1)


# In[ ]:


def cross_validation(alg, X_train, Y_train, folds=10):
    kf = KFold(n_splits = folds, shuffle=True)

    acc = []
    matrix = None
    first = True

    i = 1
    for train_index, test_index in kf.split(X_train, Y_train):
        print('{}-Fold'.format(i))
        fX_train, fX_test = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
        fy_train, fy_test = Y_train[train_index], Y_train[test_index]
        alg.fit(fX_train, fy_train)
        fy_pred = alg.predict(fX_test)
        curr = accuracy_score(fy_test, fy_pred, normalize=True)
        acc.append(curr)
        i = i+1

    acc = pd.Series(acc)
    return acc.mean()


# # First try with Decision tree classifier

# In[ ]:


alg = DecisionTreeClassifier(random_state=1)


# In[ ]:


alg.fit(X_train, Y_train)


# In[ ]:


y_pred = alg.predict(X_test)


# In[ ]:


accuracy_score(Y_test, y_pred, normalize=True)


# That's pretty low accuracy for a purely deterministic task as this one.

# # Preprocess data

# ## Sorting values
# If we take a look at poker hands in this dataset, we will see that cards in these hands are out of order.
# 
# First step at preprocessing is sorting cards in each hand, we will use function *preprocess_data* defined above.

# In[ ]:


X_train_pre = preprocess_data(train)


# In[ ]:


X_test_pre = preprocess_data(test)


# In[ ]:


X_train = X_train_pre.loc[:,X_train_pre.columns != 'Label']


# In[ ]:


X_test = X_test_pre.loc[:,X_test_pre.columns != 'Label']


# Let's give it another try.

# # 10-fold CV

# In[ ]:


alg = DecisionTreeClassifier(random_state=1, criterion='gini')
cross_validation(alg, X_train, Y_train)


# In[ ]:


alg = DecisionTreeClassifier(random_state=1, criterion='gini')
alg.fit(X_train, Y_train)
y_pred = alg.predict(X_test)
accuracy_score(Y_test, y_pred, normalize=True)


# 96% accuracy is pretty neat, we are almost two times better after sorting cards in hands.
# 
# Now we will take a look at cases where our classifier performs poorly.

# In[ ]:


pd.crosstab(y_pred, Y_test, rownames=['Predicted'], colnames=['True'], margins=True)


# In[ ]:


pred_series = pd.Series(y_pred).groupby(y_pred).size()
true_series = pd.Series(Y_test.values).groupby(Y_test).size()
pred_res = pd.DataFrame()
pred_res['TrueLabel'] = true_series
pred_res['PredictedLabel'] = pred_series


# In[ ]:


pred_res


# In[ ]:


f, ax = plt.subplots()
ax.set(yscale="log")
sns.barplot(data=pred_res.stack().reset_index().rename(columns={0: 'Count', 'level_1': 'Variable'}), x='Label', y='Count', hue='Variable')


# There is clearly problem with prediction of flushes and straight/royal flushes. We add new feature called *UniqueS* which contains number of unique suites of cards in hand.

# In[ ]:


add_unique_count(X_test)


# In[ ]:


add_unique_count(X_train)


# In[ ]:


alg = DecisionTreeClassifier(random_state=1, criterion='gini')
cross_validation(alg, X_train, Y_train)


# In[ ]:


alg = DecisionTreeClassifier(random_state=1, criterion='gini')
alg.fit(X_train, Y_train)
y_pred = alg.predict(X_test)
accuracy_score(Y_test, y_pred, normalize=True)


# In[ ]:


pred_series = pd.Series(y_pred).groupby(y_pred).size()
true_series = pd.Series(Y_test.values).groupby(Y_test).size()
pred_res = pd.DataFrame()
pred_res['TrueLabel'] = true_series
pred_res['PredictedLabel'] = pred_series
f, ax = plt.subplots()
ax.set(yscale="log")
sns.barplot(data=pred_res.stack().reset_index().rename(columns={0: 'Count', 'level_1': 'Variable'}), x='Label', y='Count', hue='Variable')


# CV accuracy is slightly better, although straight flushes get under-predicted, but problem with flushes is solved.
# 
# Now we will add new features with values of differences between consecutive cards in hand.

# In[ ]:


add_diffs(X_train)


# In[ ]:


add_diffs(X_test)


# In[ ]:


alg = DecisionTreeClassifier(random_state=1, criterion='gini')
cross_validation(alg, X_train, Y_train)


# In[ ]:


alg = DecisionTreeClassifier(random_state=1, criterion='gini')
alg.fit(X_train, Y_train)
y_pred = alg.predict(X_test)
accuracy_score(Y_test, y_pred, normalize=True)


# In[ ]:


pred_series = pd.Series(y_pred).groupby(y_pred).size()
true_series = pd.Series(Y_test.values).groupby(Y_test).size()
pred_res = pd.DataFrame()
pred_res['TrueLabel'] = true_series
pred_res['PredictedLabel'] = pred_series
f, ax = plt.subplots()
ax.set(yscale="log")
sns.barplot(data=pred_res.stack().reset_index().rename(columns={0: 'Count', 'level_1': 'Variable'}), x='Label', y='Count', hue='Variable')


# In[ ]:


pd.crosstab(y_pred, Y_test, rownames=['Predicted'], colnames=['True'], margins=True)


# Now we have really great accuracy of 99.99%, it's clear that out model somehow badly predicts straight flushes, but the rest is ok.
# 
# Decision tree performed better than expected, but we can try other classifiers as well.

# ## Random forest 10-fold CV

# In[ ]:


alg = RandomForestClassifier(criterion='gini', n_estimators=10, random_state=111, n_jobs=4)
cross_validation(alg, X_train, Y_train)


# In[ ]:


alg = RandomForestClassifier(criterion='entropy', n_estimators=51, random_state=111, n_jobs=4)
cross_validation(alg, X_train, Y_train)


# Random forest performs better with more estimators and entropy as split criterion, but decision tree is still better.

# ## Gradient Boosting Classifier 10-fold CV

# In[ ]:


alg = GradientBoostingClassifier(n_estimators=10, random_state=111)
cross_validation(alg, X_train, Y_train)


# ## Extreme Gradient Boosting Classifier 10-fold CV

# In[ ]:


alg = XGBClassifier(n_estimators=10, random_state=111)
cross_validation(alg, X_train, Y_train)


# # Test data evaluation
# 
# Let's do final test data evaluation with Decision tree classifier which had best accuracy among all tested classifiers.

# In[ ]:


alg = DecisionTreeClassifier(criterion='gini', random_state=111)


# In[ ]:


alg.fit(X_train, Y_train)


# In[ ]:


y_pred = alg.predict(X_test)


# In[ ]:


accuracy_score(y_pred=y_pred, y_true=Y_test, normalize=True)


# # Feature importances

# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(X_train.columns, alg.feature_importances_), key=lambda k: k[1], reverse=True))
feature_imp.columns = ['Feature', 'Importance']


# In[ ]:


f, ax = plt.subplots(figsize=(10, 7))
# ax.set(yscale="log")
plt.xticks(rotation=45)
sns.barplot(data=feature_imp, x='Feature', y='Importance')


# It's clear from the feature importance plot that classifier almost exclusively uses newly engineered features and original features like suites

# That's all folks!

# Your feedback is highly appreciated :)
