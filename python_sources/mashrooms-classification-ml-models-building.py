#!/usr/bin/env python
# coding: utf-8

# I'll try some classification models on the dataset in the following code. Generally, we need two steps.
# ## 1. Data Preparation and analysis
# ## 2. Machine learning models training
# Let's begin!
# ## 1. Data Preparation and analysis
# Firstly, we need import necessary models.

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as pyo

from pprint import pprint
import json
import warnings
from time import time
from subprocess import check_output

warnings.filterwarnings('ignore', category=RuntimeWarning)


# In[ ]:


# set display options
np.set_printoptions(suppress=True, linewidth=300)
pd.options.display.float_format = lambda x: ('%0.6f' % x)
get_ipython().run_line_magic('matplotlib', 'inline')
pyo.init_notebook_mode(connected=True)
print(check_output(["ls", "../input"]).decode('utf-8'))


# In[ ]:


data_df = pd.read_csv('../input/mushrooms.csv')
data_df.info()


# In[ ]:


data_df.head()


# The following code is to visualize the positive percentage and quantity percentage by different single columns one by one. The columns that has only one unique value will be ignored.

# In[ ]:


data_df['y'] = data_df['class'].map({'p':1, 'e':0})
feature_columns = [c for c in data_df.columns if not c in ('class', 'y')]
stats_df = []
single_val_c = {}
for i, c in enumerate(feature_columns):
    if data_df[c].nunique()==1:
        single_val_c[c] = data_df[c].unique()[0]
        continue
    gb = data_df.groupby(c)
    m = gb['y'].mean()
    s = gb.size()
    df = pd.DataFrame(index=range(len(m)))
    df['col'] = c
    df['val'] = m.index.values
    df['positive_percentage'] = m.values
    df['quantity_percentage'] = s.values / s.sum()
    stats_df.append(df)
    trace_prate = go.Bar(x=df['val'], y=df['positive_percentage']*100, name='positive percentage')
    trace_cnt = go.Bar(x=df['val'], y=df['quantity_percentage']*100, name='quantity percentage')
    layout = go.Layout(xaxis=dict(title=c), yaxis=dict(title="positive and quantity percentage"))
    fig = go.Figure(data=[trace_prate, trace_cnt], layout=layout)
    pyo.iplot(fig)
stats_df = pd.concat(stats_df, axis=0)


# In[ ]:


stats_df.describe()


# In[ ]:


for c in single_val_c.keys():
    print("The column %s only has one unique value with %r." % (c, single_val_c[c]))
    print("It does work for the classification, which will be removed.")
    feature_columns.remove(c)
data_df = data_df[feature_columns+['y']].copy()


# ## 2. Machine Learning models building

# Looking over these above diagrams in the last part, it's easy to find that some unique values take a very low quantity percentage in some indivvidual columns, which could be taken as nosie. To build a machine learning model, one way is to use the original dataset directly and let the model hanld it automatically. Another way is to convert them into other similiar features set then train a model by using a the new dataset. I'll try the two way one by one.

# ### 1. Build model direcly

# Because the trining of a lot of machine models contain stochastic process more or less, the original data will be copied to two or more times to decrease unsteainess of the final model.

# In[ ]:


data_df = pd.concat((data_df, data_df), axis=0, ignore_index=True)
data_df.info()


# Firstly, we need convert features columns to one-hot code.

# In[ ]:


data_all = pd.get_dummies(data=data_df, columns=feature_columns, prefix=feature_columns)
data_all.info()


# In[ ]:


data_all.head()


# Let' train the model.

# In[ ]:


def grid_search(base_model, param_grid, X_train, y_train):
    gs_c = GridSearchCV(base_model, param_grid=param_grid, n_jobs=-1, cv=3)
    gs_c.fit(X_train, y_train)
    for param_name in sorted(gs_c.best_params_):
        print("The best value of param  %s is %r" % (param_name, gs_c.best_params_[param_name]))
    return gs_c

def ridge_model(X_train, y_train):
    r_c = Pipeline([
        ('poly', PolynomialFeatures(interaction_only=True)),
        ('clf', RidgeClassifier(random_state=1))
    ])
    params_pool = dict(poly__degree=[2], clf__alpha=[0.01, 0.03, 0.1, 0.3, 1])
    return grid_search(r_c, params_pool, X_train, y_train)

def randomForest_model(X_train, y_train):
    rf_c = RandomForestClassifier(random_state=1)
    params_pool = dict(max_depth=[5, 7, 9], max_features=[0.3, 0.5], n_estimators=[12, 20, 36, 50])
    return grid_search(rf_c, params_pool, X_train, y_train)

def gaussianNB_model(X_train, y_train):
    gnb = Pipeline([
        ('poly', PolynomialFeatures(interaction_only=True)),
        ('clf', GaussianNB())
    ])
    gnb.fit(X_train, y_train)
    return gnb

def multinomialNB_model(X_train, y_train):
    mnb = Pipeline([
        ('poly', PolynomialFeatures(interaction_only=True)),
        ('clf', MultinomialNB(alpha=0.00001))
    ])
    mnb.fit(X_train, y_train)
    return mnb

def do_model_train(model_name, X_train, y_train, X_test, y_test):
    bg = time()
    print("The model %s begin to train ..." % model_name)
    if 'Ridge' == model_name:
        model = ridge_model(X_train, y_train)
    elif 'RandomForest' == model_name:
        model = randomForest_model(X_train, y_train)
    elif 'GaussianNB' == model_name:
        model = gaussianNB_model(X_train, y_train)
    elif 'multinomialNB' == model_name:
        model = multinomialNB_model(X_train, y_train)
    print("Seconds spent on %s training: %0.3f" % (model_name, time() - bg))

    y_hat = model.predict(X_train)
    print("%s accuracy of train dataset: %0.3f%%" % (model_name, accuracy_score(y_train, y_hat) * 100))
    print("%s precision of train dataset: %0.3f%%" % (model_name, precision_score(y_train, y_hat) * 100))
    print("%s recall rate of train dataset: %0.3f%%" % (model_name, recall_score(y_train, y_hat) * 100))
    print("%s f1 score of train dataset: %0.3f%%" % (model_name, f1_score(y_train, y_hat) * 100))

    y_hat = model.predict(X_test)
    print("%s accuracy of test dataset: %0.3f%%" % (model_name, accuracy_score(y_test, y_hat) * 100))
    print("%s precision of test dataset: %0.3f%%" % (model_name, precision_score(y_test, y_hat) * 100))
    print("%s recall rate of test dataset: %0.3f%%" % (model_name, recall_score(y_test, y_hat) * 100))
    print("%s f1 score of test dataset: %0.3f%%" % (model_name, f1_score(y_test, y_hat) * 100))


# In[ ]:


X_all, y_all = data_all[[c for c in data_all.columns if c!='y']], data_all['y']
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33, random_state=1)
do_model_train('RandomForest', X_train, y_train, X_test, y_test)


# Great, we got 100%! Then I'll try to decrease the of number of training features.

# ### 2. Clean nosie then build model

# In[ ]:


def clean_data_df(df, threshold=0.02):
    feature_convert = dict()
    for col, sub in stats_df.groupby('col'):
        ns = sub[(sub.quantity_percentage<threshold)]
        n_ns = sub[(sub.quantity_percentage>=threshold)]
        for idx in ns.index:
            if ns.loc[idx, 'positive_percentage'] > 0.5:
                p_n_ns = n_ns[n_ns.positive_percentage > 0.5]
                if not p_n_ns.empty:
                    feature_convert.setdefault(col, []).append((ns.loc[idx, 'val'], p_n_ns['val'].values[0]))
                    df.loc[df[col]==ns.loc[idx, 'val'], col] = p_n_ns['val'].values[0]
            else:
                n_n_ns = n_ns[n_ns.positive_percentage <= 0.5]
                if not n_n_ns.empty:
                    feature_convert.setdefault(col, []).append((ns.loc[idx, 'val'], n_n_ns['val'].values[0]))
                    df.loc[df[col]==ns.loc[idx, 'val'], col] = n_n_ns['val'].values[0]
    return pd.get_dummies(data=df, columns=feature_columns, prefix=feature_columns),            feature_convert
cleaned_df, feature_convert = clean_data_df(data_df.copy())
cleaned_df.info()


# In[ ]:


pprint(feature_convert)


# In[ ]:


X_all, y_all = cleaned_df[[c for c in cleaned_df.columns if c!='y']], cleaned_df['y']
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33, random_state=1)
do_model_train('RandomForest', X_train, y_train, X_test, y_test)


# Great, we also got 100%. We also found that the param max_featrues grid seached is only 0.3 and we got only 12 trees. So probaly the necessary quantity of features could be much less. The following code will try to find really importance features.

# In[ ]:


def get_features_importance(x, y):
    rf = RandomForestClassifier(n_estimators=500, class_weight={0: 1, 1: 1 / np.sqrt(np.mean(y))},                                 max_features=0.75, n_jobs=-1, random_state=1)
    rf.fit(x, y)
    feature_importance = pd.DataFrame(data={"columns": x.columns, "importance": rf.feature_importances_})
    feature_importance.sort_values(by="importance", axis=0, ascending=False, inplace=True)
    feature_importance.loc[:, "cum_importance"] = feature_importance.importance.cumsum()
    return feature_importance

def get_features_corr(df, ycol):
    corr_y = df.corr()[ycol].map(np.abs)
    corr_y = corr_y[[c for c in corr_y.index if c!=ycol]]
    corr_y = corr_y / corr_y.sum()
    feature_importance = pd.DataFrame(data={"columns": corr_y.index.values, "importance": corr_y.values})
    feature_importance.sort_values(by="importance", axis=0, ascending=False, inplace=True)
    feature_importance.loc[:, "cum_importance"] = feature_importance.importance.cumsum()
    return feature_importance

data_all = pd.get_dummies(data=data_df, columns=feature_columns, prefix=feature_columns)
X_all, y_all = data_all[[c for c in data_all.columns if c!='y']], data_all['y']
fi = get_features_importance(X_all, y_all)
# fi = get_features_corr(data_all, 'y')
bg = time()
accuracyScores, precisionScores, recallScores, f1Scores = [], [], [], []
for i in range(len(fi)):
    cols = fi.iloc[:i+1]['columns'].values   
    model = Pipeline([
        ('poly', PolynomialFeatures(interaction_only=True, degree=2)),
        ('clf', GaussianNB())
    ])
    model.fit(X_all[cols], y_all)
    y_p = model.predict(X_all[cols])
    accuracyScores.append(accuracy_score(y_true=y_all, y_pred=y_p))
    precisionScores.append(precision_score(y_true=y_all, y_pred=y_p))
    recallScores.append(recall_score(y_true=y_all, y_pred=y_p))
    f1Scores.append(f1_score(y_true=y_all, y_pred=y_p))
    if accuracyScores[-1] == 1:
        break
print()
print('It took %.3f seconds.' % (time() - bg))
traces = [go.Bar(x=np.arange(len(fi))+1, y=fi['importance'][:i+1], name='importance', opacity=0.5,                     text=fi['columns']),
          go.Bar(x=np.arange(len(fi))+1, y=fi['cum_importance'][:i+1], name='left sum of importance', opacity=0.8, \
                    text=fi['columns']),
          go.Scatter(x=np.arange(len(fi))+1, y=accuracyScores, mode='markers+lines', name='accuracy Score'),
          go.Scatter(x=np.arange(len(fi))+1, y=precisionScores, mode='markers+lines', name='precision Score'),
          go.Scatter(x=np.arange(len(fi))+1, y=recallScores, mode='markers+lines', name='recall Score'),
          go.Scatter(x=np.arange(len(fi))+1, y=f1Scores, mode='markers+lines', name='F1 Score')]
layout=go.Layout(title='Feature importance/accuracy/precision/recall/F1 Score on different number of features',
                xaxis=dict(title='number of features'), yaxis=dict(title='importance/accuracy/precision/recall/F1 Score'))
pyo.iplot(go.Figure(data=traces, layout=layout))


# Looking over the score curves changed by feature number, we could find some features obviously change the score, which are really import and could be used to build a much simpler model.

# In[ ]:


col_idx = [0]
for i in range(len(f1Scores)):
    if i != 0:
        rst = False
        for l in (f1Scores, precisionScores, recallScores, accuracyScores):
            rst |= (l[i]!=l[i-1])
        if rst:
            col_idx.append(i)
x_cols = fi.iloc[col_idx]['columns'].values
print("%d features were found:" % len(x_cols))
pprint(x_cols)
X_all, y_all = data_all[x_cols], data_all['y']
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33, random_state=1)
do_model_train('RandomForest', X_train, y_train, X_test, y_test)


# Great! We got a prefect but much simpler model with only 16 features used.

# In[ ]:


col_vals = {}
no_pt = [col_vals.setdefault(it[0], []).append(it[1]) for it in[v.split('_') for v in x_cols]]
print("The avaliable columns and value are following:")
pprint(col_vals)


# The following code is to review the distribution and pairwise correlation of these selected features.

# In[ ]:


for c in x_cols:
    gb = data_all.groupby(c)
    m = gb['y'].mean()
    s = gb.size()
    df = pd.DataFrame(index=range(len(m)))
    df['col'] = c
    df['val'] = m.index.values
    df['val'] = df['val'].map({0: 'N', 1: 'Y'})
    df['positive_percentage'] = m.values
    df['quantity_percentage'] = s.values / s.sum()
    trace_prate = go.Bar(x=df['val'], y=df['positive_percentage']*100, name='positive percentage')
    trace_cnt = go.Bar(x=df['val'], y=df['quantity_percentage']*100, name='quantity percentage')
    layout = go.Layout(xaxis=dict(title='='.join(c.split('_'))+'?'), yaxis=dict(title="positive and quantity percentage"))
    fig = go.Figure(data=[trace_prate, trace_cnt], layout=layout)
    pyo.iplot(fig)


# In[ ]:


plt.figure(figsize=(18, 10))
corr = data_all[list(x_cols)+['y']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.01)


# The following code is a sklearn transform class to make feature transformation, which will be used to train a pipeline model as the first step.

# In[ ]:


class Feature_filter(BaseEstimator, TransformerMixin):
    
    def __init__(self, necessary_col_vals: dict):
        self.necessary_col_vals = necessary_col_vals
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The input X must be a pandas.DataFrame instance.")
        diff = set(self.necessary_col_vals.keys()).difference(X.columns)
        if diff:
            raise IndexError("The input X lacks of necessary columns: %r" % list(diff))
        self.X = X
        return self
    
    def transform(self, X):
        cols = list(self.necessary_col_vals.keys())
        XP = X[cols]
        XP = pd.get_dummies(data=XP, columns=cols, prefix=cols)
        n_cols = np.sum([[t[0]+'_'+ v for v in t[1]] for t in self.necessary_col_vals.items()])
        return XP[n_cols].copy()


# In[ ]:


fea_filter = Feature_filter(col_vals)
fea_filter.fit_transform(data_df).info()


# In[ ]:


def mushroom_lr_model(X_train, y_train, enough_col_vals):
    model = Pipeline([
        ('fillter', Feature_filter(enough_col_vals)),
        ('poly', PolynomialFeatures(interaction_only=True)),
        ('clf', LogisticRegressionCV(n_jobs=-1, random_state=1))
    ])
    print("Begin to fit...")
    bg = time()
    model.fit(X_train, y_train)
    print("Done fit and %.2f seconds was token." % (time() - bg))
    y_hat = model.predict(X_train)
    print("accuracy of train dataset: %0.3f%%" % (accuracy_score(y_train, y_hat) * 100))
    print("precision of train dataset: %0.3f%%" % (precision_score(y_train, y_hat) * 100))
    print("recall rate of train dataset: %0.3f%%" % (recall_score(y_train, y_hat) * 100))
    print("f1 score of train dataset: %0.3f%%" % (f1_score(y_train, y_hat) * 100))

    y_hat = model.predict(X_test)
    print("accuracy of test dataset: %0.3f%%" % (accuracy_score(y_test, y_hat) * 100))
    print("precision of test dataset: %0.3f%%" % (precision_score(y_test, y_hat) * 100))
    print("recall rate of test dataset: %0.3f%%" % (recall_score(y_test, y_hat) * 100))
    print("f1 score of test dataset: %0.3f%%" % (f1_score(y_test, y_hat) * 100))
X_all, y_all = data_df, data_df['y']
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33, random_state=1)
mushroom_lr_model(X_train, y_train, col_vals)


# Now, we get a very simple model to classify if a mushroom is poisonous. 
