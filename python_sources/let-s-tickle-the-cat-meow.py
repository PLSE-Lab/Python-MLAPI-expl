#!/usr/bin/env python
# coding: utf-8

# <img src='https://www.commonfloor.com/articles/wp-content/uploads/2012/07/dva1-300x199.jpg' width=500>
# <div align="center"><font size="2">Source: Google</font></div>

# <p>This is a Playground competition which will give you the opportunity to try different encoding schemes for different algorithms and compare how they perform.</p>
# 
# <p> This is the follow up competition to the previous categorical encoding challenge 1</p>
# 
# <p> The features are given as below </p> 
#     
#     
# * binary features
# * low- and high-cardinality nominal features
# * low- and high-cardinality ordinal features
# * (potentially) cyclical features

# > **Change of status:**
# > Update_v2: Stacking with Histgbm,catboost,logistic    
# 

# <font color='#000000' size=4>Objective</font><br>
# 
# * Exploration and finding interactions
# * Build models
# * Evaluate models

# **Let's start tickling!!**

# <font color='#088a5a' size=3>Kindly upvote the kernel if you like it!</font><br>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().system('pip install --upgrade scikit-learn')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.core.display import display, HTML
import pandas_profiling as pp
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
from IPython.display import Markdown
import scipy.stats as ss
import itertools
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing
from sklearn.metrics import auc,plot_roc_curve
import datetime
from time import time
from catboost import CatBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Any results you write to the current directory are saved as output.


# <font color='#088a5a' size=4>Data glimpse</font><br>

# In[ ]:


path='../input/cat-in-the-dat-ii/'

train=pd.read_csv(path+'train.csv')
test=pd.read_csv(path+'test.csv')
submission=pd.read_csv(path+'sample_submission.csv')


# In[ ]:


display(HTML(f"""
   
        <ul class="list-group">
          <li class="list-group-item disabled" aria-disabled="true"><h4>Shape of Train and Test Dataset</h4></li>
          <li class="list-group-item"><h4>Number of rows in Train dataset is: <span class="label label-primary">{ train.shape[0]:,}</span></h4></li>
          <li class="list-group-item"> <h4>Number of columns Train dataset is <span class="label label-primary">{train.shape[1]}</span></h4></li>
          <li class="list-group-item"><h4>Number of rows in Test dataset is: <span class="label label-success">{ test.shape[0]:,}</span></h4></li>
          <li class="list-group-item"><h4>Number of columns Test dataset is <span class="label label-success">{test.shape[1]}</span></h4></li>
        </ul>
  
    """))


# <font color='#0000ff' size=3>Panda's profiling</font><br>

# Taking a fraction of data(1%) for checking the over-view of data

# In[ ]:


sample_profile=train.sample(frac=0.01)

pp.ProfileReport(sample_profile)


# <font color='#088a5a' size=4>Data walk through</font><br>

# In[ ]:


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Total'] = df.count().values   
    summary['Missing Percentage']=(summary['Missing']/summary['Total'])*100
    summary['Uniques'] = df.nunique().values

    return summary


# In[ ]:


trainsum = resumetable(train)
trainsum


# In[ ]:


testsum = resumetable(test)
testsum


# Trivia:
# 
# * There seems to be around only 3% of missing values in both training and test data
# * Nominal features 5 to 9 has more unique values in both training and test data

# <font color='#0000ff' size=3>Target distribution</font><br>

# In[ ]:


## target distribution ##
cnt_srs=train.target.value_counts()

labels = (np.array(cnt_srs.index))
sizes = (np.array((cnt_srs / cnt_srs.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Target distribution',
    font=dict(size=15),
    width=500,
    height=500,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="usertype")


# Trivia:
# 
# * There is a huge imbalance between targets 0 (81.3%) and 1 (18.7%)

# In[ ]:


bin_features=[i for i in train.columns if i.split('_')[0]=='bin']
ord_features=[i for i in train.columns if i.split('_')[0]=='ord']
nom_features=[i for i in train.columns if i.split('_')[0]=='nom']

cyc_features=[i for i in train.columns if i in ['day','month']]


# Thanks to this wonderful [kernel](https://www.kaggle.com/vikassingh1996/handling-categorical-variables-encoding-modeling) for the below charts 

# <font color='#0000ff' size=3>Binary variables</font><br>

# In[ ]:


'''#2.Function for displaying bar lebels in relative scale.'''
def pct_bar_labels():
    font_size = 15
    plt.ylabel('Relative Frequency (%)', fontsize = font_size)
    plt.xticks(rotation = 0, fontsize = font_size)
    plt.yticks([]) 
    
    # Set individual bar lebels in proportional scale
    for x in ax1.patches:
        ax1.annotate(str(x.get_height()) + '%', 
        (x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 7), 
        textcoords = 'offset points', fontsize = font_size, color = 'black')
        
'''Display markdown formatted output like bold, italic bold etc.'''

def bold(string):
    display(Markdown(string))


# In[ ]:


'''Create a function that relative frequency of Target variable by a categorical variable. 
And then plots the relative frequency of target by a categorical variable.'''

def crosstab(cat, cat_target, color):
    '''cat = categorical variable, cat_target = our target categorical variable.'''
    global ax1
    fig_size = (18, 5)
    title_size = 18
    font_size = 15
    
    pct_cat_grouped_by_cat_target = round(pd.crosstab(index = cat, columns = cat_target, normalize = 'index')*100, 2)
       
    # Plot relative frequrncy of Target by a categorical variable
    ax1 = pct_cat_grouped_by_cat_target.plot.bar(color = color, title = 'Percentage Count of target by %s' %cat.name, figsize = fig_size)
    ax1.title.set_size(fontsize = title_size)
    pct_bar_labels()
    plt.xlabel(cat.name, fontsize = font_size)
    plt.show()


# In[ ]:


'''Plot the binary variables in relative scale'''

for i,val in enumerate(bin_features):
    bold(f'**Percentage Count of target by {val}:**')
    crosstab(train[val], train.target, color = ['g', 'b'])


# <font color='#0000ff' size=3>Ordinal variables</font><br>

# In[ ]:


'''Plot the ordinal variables in relative scale'''

for i,val in enumerate(ord_features[:3]):
    bold(f'**Percentage Count of target by {val}:**')
    crosstab(train[val], train.target, color = ['y', 'b'])


# <font color='#0000ff' size=3>Nominal variables</font><br>

# In[ ]:


'''Plot the nominal variables in relative scale'''

for i,val in enumerate(nom_features[:5]):
    bold(f'**Percentage Count of target by {val}:**')
    crosstab(train[val], train.target, color = ['r', 'g'])


# <font color='#0000ff' size=3>Cyclical variables</font><br>

# In[ ]:


'''Plot the cyclic variables in relative scale'''

for i,val in enumerate(cyc_features):
    bold(f'**Percentage Count of target by {val}:**')
    crosstab(train[val], train.target, color = ['y', 'b'])


# Trivia:
# 
# * There is a again a huge percentage differnce across all cat variables between targets

# <font color='#0000ff' size=3>Interaction between cat variables - Crammer's rule</font><br>

# In[ ]:


train_copy=train.dropna()


# In[ ]:


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

all_feat=bin_features+nom_features+ord_features+cyc_features


# In[ ]:


corrM = np.zeros((len(all_feat),len(all_feat)))

for col1, col2 in itertools.combinations(all_feat, 2):
    idx1, idx2 = all_feat.index(col1), all_feat.index(col2)
    corrM[idx1, idx2] = cramers_v(train_copy[col1], train_copy[col2])
    corrM[idx2, idx1] = corrM[idx1, idx2]

corr = pd.DataFrame(corrM, index=all_feat, columns=all_feat)
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(round(corr,2), annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");

del train_copy


# Trivia:
# 
# * Looks like there is no much of interactions between variables.

# <font color='#088a5a' size=4>Data encoding and cleaning</font><br>

# In[ ]:


# CREDITS : https://www.kaggle.com/caesarlupum/2020-20-lines-target-encoding

def encoding(train, test, smooth):
    print('Target encoding...')
    train.sort_index(inplace=True)
    target = train['target']
    test_id = test['id']
    train.drop(['target', 'id'], axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)
    cat_feat_to_encode = bin_features+nom_features+ord_features+cyc_features
    smoothing=smooth
    oof = pd.DataFrame([])
    
    for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(train, target):
        ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
        ce_target_encoder.fit(train.iloc[tr_idx, :], target.iloc[tr_idx])
        oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)
        
    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
    ce_target_encoder.fit(train, target)
    train = oof.sort_index()
    test = ce_target_encoder.transform(test)
    features = list(train)
    print('Target encoding done!')
    return train, test, test_id, features, target


# In[ ]:


# Encoding
train_encode, test_encode, test_id, features, target = encoding(train, test, 0.3)


# <font color='#088a5a' size=4>Modelling and inference- MEOW</font><br>

# In[ ]:


train_encode=pd.concat([train_encode,target],axis=1,ignore_index=True)
train_encode.columns=list(train.columns)+['target']


# In[ ]:


X, y = train_encode[all_feat], train_encode['target']


# In[ ]:


def make_classifier():
    clf = CatBoostClassifier(
                               loss_function='CrossEntropy',
                               eval_metric="AUC",
                               task_type="CPU",
                               learning_rate=0.05,
                               n_estimators =100,   #5000
                               early_stopping_rounds=10,
                               random_seed=2019,
                               silent=True
                              )
        
    return clf

#oof = np.zeros(len(X))


# In[ ]:


# preds = np.zeros(len(test_encode))
# oof = np.zeros(len(X))
# NFOLDS = 10

# folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2019)


# training_start_time = time()
# for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
#     start_time = time()
#     print(f'Training on fold {fold+1}')
#     clf = make_classifier()
#     clf.fit(X.loc[trn_idx, all_feat], y.loc[trn_idx], eval_set=(X.loc[test_idx, all_feat], y.loc[test_idx]),
#                           use_best_model=True, verbose=500)
    
#     preds += clf.predict_proba(test_encode)[:,1]/NFOLDS
#     oof[test_idx] = clf.predict_proba(X.loc[test_idx, all_feat])[:,1]
    
#     print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
    
# print('-' * 30)
# print('OOF',metrics.roc_auc_score(y, oof))
# print('-' * 30)


# <font color='#0000ff' size=3>Stacking classifiers</font><br>

# <p>Stacking is an ensemble learning technique to combine multiple classification models via a meta-classifier. Here we will choose catboost,histgbm as our base classifiers and logistic regression as final estimator
# </p>

# <img src='http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier_files/stackingclassification_overview.png' width=500>
# <div align="center"><font size="2">Source: Google</font></div>

# Trivia - HistGBM:
# *     It's based on LightGBM implementation and it's much faster than other GBM's. 
# *     It's still in experimental stage as of now

# <font color='#088a5a' size=4>Stacking</font><br>

# The below implementation is inspired from this [kernel](https://www.kaggle.com/nicapotato/whats-new-sklearn-0-22-1-cat-classifier-stack)

# In[ ]:


scoring = "roc_auc"

HistGBM_param = {
    'l2_regularization': 0.0,
    'loss': 'auto',
    'max_bins': 255,
    'max_depth': 15,
    'max_leaf_nodes': 31,
    'min_samples_leaf': 20,
    'n_iter_no_change': 50,
    'scoring': scoring,
    'tol': 1e-07,
    'validation_fraction': 0.15,
    'verbose': 0,
    'warm_start': False   
}

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
fold_preds = np.zeros([test_encode.shape[0],3])
oof_preds = np.zeros([X.shape[0],3])
results = {}

estimators = [
        ('histgbm', HistGradientBoostingClassifier(**HistGBM_param)),
        ('catboost', make_classifier())
    ]

# Fit Folds
f, ax = plt.subplots(1,3,figsize = [14,5])
for i, (trn_idx, val_idx) in enumerate(folds.split(X,y)):
    print(f"Fold {i} stacking....")
    clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            )
    clf.fit(X.loc[trn_idx,:], y.loc[trn_idx])
    tmp_pred = clf.predict_proba(X.loc[val_idx,:])[:,1]
    
    oof_preds[val_idx,0] = tmp_pred
    fold_preds[:,0] += clf.predict_proba(test_encode)[:,1] / folds.n_splits
        
    estimator_performance = {}
    estimator_performance['stack_score'] = metrics.roc_auc_score(y.loc[val_idx], tmp_pred)
    
    for ii, est in enumerate(estimators):
            model = clf.named_estimators_[est[0]]
            pred = model.predict_proba(X.loc[val_idx,:])[:,1]
            oof_preds[val_idx, ii+1] = pred
            fold_preds[:,ii+1] += model.predict_proba(test_encode)[:,1] / folds.n_splits
            estimator_performance[est[0]+"_score"] = metrics.roc_auc_score(y.loc[val_idx], pred)
            
    stack_coefficients = {x+"_coefficient":y for (x,y) in zip([x[0] for x in estimators], clf.final_estimator_.coef_[0])}
    stack_coefficients['intercept'] = clf.final_estimator_.intercept_[0]
        
    results["Fold {}".format(str(i+1))] = [
            estimator_performance,
            stack_coefficients
        ]

    plot_roc_curve(clf, X.loc[val_idx,:], y.loc[val_idx], ax=ax[i])
    ax[i].plot([0.0, 1.0])
    ax[i].set_title("Fold {} - ROC AUC".format(str(i)))

plt.tight_layout(pad=2)
plt.show()

f, ax = plt.subplots(1,2,figsize = [11,5])
sns.heatmap(pd.DataFrame(oof_preds, columns = ['stack','histgbm','catboost']).corr(),
            annot=True, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="magma",ax=ax[0])
ax[0].set_title("OOF PRED - Correlation Plot")
sns.heatmap(pd.DataFrame(fold_preds, columns = ['stack','histgbm','catboost']).corr(),
            annot=True, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="inferno",ax=ax[1])
ax[1].set_title("TEST PRED - Correlation Plot")
plt.tight_layout(pad=3)
plt.show()


# Let's take the stacked classifier output and submit the predictions

# In[ ]:


submission['target'] =fold_preds[:,0] #preds
submission.to_csv('submission.csv', index=None)
submission.head()


# In[ ]:


submission['target'].plot(kind='hist')


# <font color='#088a5a' size=4>Stay tuned!!</font><br>

# In[ ]:




