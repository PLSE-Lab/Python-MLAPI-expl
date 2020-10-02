#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualisation Library
import matplotlib.pyplot as plt
import seaborn as sns

#models

#
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.model_selection import train_test_split


get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.shape, test.shape


# In[ ]:


train.head()


# NO Dependencies could be seen through the correlation matrix now have to search for the normality of response variable

# In[ ]:


sns.set_style('white')
sns.set_color_codes(palette='deep')
f,ax= plt.subplots(figsize = (8,7))
sns.distplot(train['target'])


# In[ ]:


def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    
    return dict_x

missing = percent_missing(train)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
df_miss[0:10]


# In[ ]:


train.head()


# In[ ]:





# In[ ]:


train50 = train.iloc[:,0:50]
plt.figure(figsize=(12,10))
corr = train50.corr()
sns.heatmap(data = corr, annot=False)
#33,24


# In[ ]:


train75 = train.iloc[:,50:75]
train75['target'] = train['target']
corr = train75.corr()
plt.figure(figsize=(12,10))
sns.heatmap(data = corr, annot=False)
#65 #72 #59 #89,84,#101,105,114,119,130,183,176,164,199,201,221,226,272,289


# In[ ]:


newdf = train[['33','24','65', '72', '59', '89','84','101','105','114','119','130','183',
              '176','164','199','201','221','226','272','289']]
newdf['target'] = train['target']
corr = newdf.corr()
plt.figure(figsize=(12,12))
sns.heatmap(data = corr)


# In[ ]:


print('Distributions of top columns')
plt.figure(figsize=(26, 24))
for i, col in enumerate(list(newdf.columns)):
    plt.subplot(7, 4, i + 1)
    plt.hist(newdf[col])
    plt.title(col)


# In[ ]:


sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data= newdf , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)


# No Data missing is there now applying different models on it to check the performance 

# In[ ]:


# train.drop('id', axis=1)
# y = train['target']
# X = train.drop('target', axis =1)


# In[ ]:


y = newdf['target']
X= newdf.drop('target', axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


from lightgbm import LGBMClassifier


# In[ ]:


lgb = LGBMClassifier(
        boosting_type='gbdt',
        n_estimators=600,
        learning_rate=0.01,
        num_leaves=10,
        colsample_bytree=.3,
        subsample=.8,
        max_depth=-1,
        reg_alpha=.2,
        reg_lambda=.5,
        min_split_gain=.01)


# In[ ]:


lgbfit = lgb.fit(X_train, y_train)


# In[ ]:


preds = lgbfit.predict(X_test)


# In[ ]:


print(classification_report(y_test,preds))


# In[ ]:


test_Score = lgbfit.score(X_test,y_test)
print ("LGBM Score ",test_Score)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingCVClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score


# In[ ]:


# Setup cross validation folds
kf = KFold(n_splits=5, random_state=42, shuffle=True)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

#LR
logmodel = LogisticRegression()

#XGBoost

xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.01,
        booster='dart',
        num_leaves=10,
        gamma=0.2,
        reg_alpha=.002,
        reg_lambda=.06,
        min_split_gain=.01)

#Gradient Boosting

gbc = GradientBoostingClassifier(
    loss='deviance',
    learning_rate=0.01,
    n_estimators=3000,
    min_samples_split=2,
    min_samples_leaf=1,
    presort='auto',
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=0.0001)

#Random Forest

rf = RandomForestClassifier(
    n_estimators=1000,
    criterion='gini',
    max_depth=5,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None)

#Stacking up the models
stack_gen = StackingCVClassifier(classifiers=(xgb,gbc,rf,logmodel,lgb),
                                meta_classifier=logmodel)


# In[ ]:


scores = {}

score = cv_rmse(lgb)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())


# In[ ]:


score = cv_rmse(xgb)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())


# In[ ]:


score = cv_rmse(gbc)
print("GBC: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbc'] = (score.mean(), score.std())


# In[ ]:


score = cv_rmse(rf)
print("RF: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())


# In[ ]:


score = cv_rmse(logmodel)
print("LOG REG: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['logmodel'] = (score.mean(), score.std())


# In[ ]:


score = cv_rmse(lgb)
print("LGB: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())


# In[ ]:


print('xgb')
xgbfit= xgb.fit(X_train, y_train)


# In[ ]:


print('gbc')
gbcfit = gbc.fit(X_train,y_train)


# In[ ]:


print('rf')
rffit= rf.fit(X_train, y_train)


# In[ ]:


print('logmodel')
lgrfit= logmodel.fit(X_train, y_train)


# In[ ]:


print('lgb')
lgbfit= lgb.fit(X_train, y_train)


# In[ ]:


print("stack gen")
stack_genfit = stack_gen.fit(X_train,y_train)


# In[ ]:


preds = xgbfit.predict(X_test)
test_Score = xgbfit.score(X_test,y_test)
print ("XGB Score ",test_Score)
print(classification_report(y_test,preds))


# In[ ]:


preds = rffit.predict(X_test)
test_Score = rffit.score(X_test,y_test)
print(classification_report(y_test,preds))
print ("RF Score ",test_Score)


# In[ ]:


preds = lgrfit.predict(X_test)
test_Score = lgrfit.score(X_test,y_test)
print(classification_report(y_test,preds))
print ("LGR Score ",test_Score)


# In[ ]:


preds = gbcfit.predict(X_test)
test_Score = gbcfit.score(X_test,y_test)
print(classification_report(y_test,preds))
print ("GBC Score ",test_Score)


# In[ ]:


preds = stack_genfit.predict(X_test.as_matrix())
test_Score = stack_genfit.score(X_test.as_matrix(),y_test.as_matrix())
print(classification_report(y_test,preds))
print ("GBC Score ",test_Score)


# In[ ]:


# Plot the predictions for each model
sns.set_style("white")
fig = plt.figure(figsize=(24, 12))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.xlabel('Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)

plt.show()


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


submission.head()


# In[ ]:


testdf = test[['33','24','65', '72', '59', '89','84','101','105','114','119','130','183',
              '176','164','199','201','221','226','272','289']]


# In[ ]:


testdf.head()


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = logmodel.predict(testdf)

subxgb = pd.read_csv('../input/sample_submission.csv')
subxgb['target'] = xgbfit.predict(testdf)

subgbc = pd.read_csv('../input/sample_submission.csv')
subgbc['target'] = gbcfit.predict(testdf)

subrf = pd.read_csv('../input/sample_submission.csv')
subrf['target'] = rffit.predict(testdf)

sublgb = pd.read_csv('../input/sample_submission.csv')
sublgb['target'] = lgbfit.predict(testdf)


# In[ ]:


substack = pd.read_csv('../input/sample_submission.csv')
substack['target'] = stack_genfit.predict(testdf.as_matrix())


# In[ ]:


sub.to_csv('submissionLR.csv', index=False)
subxgb.to_csv('submissionXGB.csv', index=False)
subgbc.to_csv('submissionGBC.csv', index=False)
subrf.to_csv('submissionRF.csv', index=False)
sublgb.to_csv('submissionLGB.csv', index=False)
substack.to_csv('submissionStack.csv', index=False)


# In[ ]:




