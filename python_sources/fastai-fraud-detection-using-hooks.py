#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this Notebook, we will build a fraud detection model which classifies normal credit card transactions and fraudelent transactions separately.
# 
# As it is expected normal transactions dominates the data with less than only 1% of transactions as fraudelent.
# 
# We have only used Training data for the purpose of building classifier. In the training data, we have around 250k records whereas only around 400 data recorded as fraudelent. Most of the features are masked and normalized.
# 
# We will use various packages to solve this challenge but primarily, we will use Fastai and Sklearn libraries.
# 
# In this Notebook, we will use following approaches to build the classifier:
# 
# * Using Fastai's Deep Learning Tabular Model
# * Using Pytorch's Hook functionality to extract the learnings from the trained model
# * Using Extra Trees Classifier to further train on these extracted learnings
# 
# In last, as an experiment, I used AutoEncoder method as well which performed really well.
# 
# So, let's dive in!

# # Loading Libraries and Data

# Let's first import all required libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# !pip install pretrainedmodels

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('pip install fastai==1.0.52')
import fastai

from fastai import *
from fastai.tabular import *

# from torchvision.models import *
# import pretrainedmodels

from utils import *
import sys

from fastai.callbacks.hooks import *

from fastai.callbacks.tracker import EarlyStoppingCallback
from fastai.callbacks.tracker import SaveModelCallback


# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from sklearn.manifold import TSNE


import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb

from scipy.special import erfinv
import matplotlib.pyplot as plt
import torch
from torch.utils.data import *
from torch.optim import *
from fastai.tabular import *
import torch.utils.data as Data
from fastai.basics import *
from fastai.callbacks.hooks import *
from tqdm import tqdm_notebook as tqdm


# # Custom Metrics

# We have created ROC AUC metrics which goes well with Fastai as a metric

# In[ ]:


from sklearn.metrics import roc_auc_score

def auroc_score(input, target):
    input, target = input.cpu().numpy()[:,1], target.cpu().numpy()
    return roc_auc_score(target, input)

class AUROC(Callback):
    _order = -20 #Needs to run before the recorder

    def __init__(self, learn, **kwargs): self.learn = learn
    def on_train_begin(self, **kwargs): self.learn.recorder.add_metric_names(['AUROC'])
    def on_epoch_begin(self, **kwargs): self.output, self.target = [], []
    
    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            self.output.append(last_output)
            self.target.append(last_target)
                
    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output)
            target = torch.cat(self.target)
            preds = F.softmax(output, dim=1)
            metric = auroc_score(preds, target)
            return add_metrics(last_metrics, [metric])


# # Normalizing Data

# This function will be used to normalize the data. Its called Rank - Gaussian Normalization technique. In very simple terms, for continious data in a column are sorted as per their values and ranks are determined. These ranks are then normalized using Gaussian distribution.
# 
# I found this technique of normalizing the continious data in dataset really helpful.

# In[ ]:


def to_gauss(x): return np.sqrt(2)*erfinv(x)  #from scipy

def normalize(data, exclude=None):
    # if not binary, normalize it
    norm_cols = [n for n, c in data.drop(exclude, 1).items() if len(np.unique(c)) > 2]
    n = data.shape[0]
    for col in norm_cols:
        sorted_idx = data[col].sort_values().index.tolist()# list of sorted index
        uniform = np.linspace(start=-0.99, stop=0.99, num=n) # linsapce
        normal = to_gauss(uniform) # apply gauss to linspace
        normalized_col = pd.Series(index=sorted_idx, data=normal) # sorted idx and normalized space
        data[col] = normalized_col # column receives its corresponding rank
    return data


# # Data

# Let's see how does our data look like

# In[ ]:


df_all = pd.read_csv('../input/creditcard.csv')


# In[ ]:


df_all.shape


# We can see that names of features are masked.

# In[ ]:


df_all.columns


# Let's see how does the distribution of normal and fraudelent transactions look like

# In[ ]:


df_all['Class'].value_counts()


# In[ ]:


plt.figure(figsize=(12,12))
sns.countplot(df_all['Class']).set_title('Dist of Class variables')


# In[ ]:


df_all.head()


# In[ ]:


df_all.describe()


# # Plots

# Let's see few plots.
# 
# First plot is to see how does the plot of time looks like when it comes to normal and fraudelent transactions
# 

# In[ ]:


class_0 = df_all.loc[df_all['Class'] == 0]["Time"]
class_1 = df_all.loc[df_all['Class'] == 1]["Time"]

hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
iplot(fig, filename='dist_only')


# In this plot, we will plot Boxplots to see how does the credit card transactions amount varies between normal and fraudelent transactions

# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=df_all, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=df_all, palette="PRGn",showfliers=False)
plt.show();


# Let's see the description of Normal and Fraudelent transactions

# In[ ]:


tmp = df_all[['Amount','Class']].copy()
class_0 = tmp.loc[tmp['Class'] == 0]['Amount']
class_1 = tmp.loc[tmp['Class'] == 1]['Amount']
class_0.describe()


# In[ ]:


class_1.describe()


# We can also see the heatmap (correlation plot) of the data

# In[ ]:


plt.figure(figsize = (14,14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = df_all.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()


# Here we can see that V20 and V7 are heavily positively correlated with Amount
# 
# and, V2 and V5 are heavily negatively correlated with Amount

# In[ ]:


s = sns.lmplot(x='V20', y='Amount',data=df_all, hue='Class', fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='V7', y='Amount',data=df_all, hue='Class', fit_reg=True,scatter_kws={'s':2})
plt.show()


# In[ ]:


s = sns.lmplot(x='V2', y='Amount',data=df_all, hue='Class', fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='V5', y='Amount',data=df_all, hue='Class', fit_reg=True,scatter_kws={'s':2})
plt.show()
gc.collect()


# In this plot, we can see for all features, how does the distribution of Normal and Fraudelent transactions look like

# In[ ]:


var = df_all.columns.values

i = 0
t0 = df_all.loc[df_all['Class'] == 0]
t1 = df_all.loc[df_all['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# In[ ]:


non_fraud = df_all[df_all['Class'] == 0].sample(2000)
fraud = df_all[df_all['Class'] == 1]

df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X = df.drop(['Class'], axis = 1).values
Y = df["Class"].values


# Let's see how does our data look like in 2D space using TSNE

# In[ ]:


from sklearn.manifold import TSNE

def tsne_plot(x1, y1, name="graph.png"):
    tsne = TSNE(n_components=2)
    X_t = tsne.fit_transform(x1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Non Fraud')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='Fraud')

    plt.legend(loc='best');
    plt.savefig(name);
    plt.show();
    
tsne_plot(X, Y, "original.png")


# In[ ]:


gc.collect()


# As we can see that classes are all jumbled up. Our job is to classify Red ones and Green ones separately.

# # Data preparation for Fastai

# In[ ]:


df_train = df_all


# In[ ]:


idx = df_train.columns.values[0:30]


# In[ ]:


df_train['sum'] = df_train[idx].sum(axis=1)  
df_train['min'] = df_train[idx].min(axis=1)
df_train['max'] = df_train[idx].max(axis=1)
df_train['mean'] = df_train[idx].mean(axis=1)
df_train['std'] = df_train[idx].std(axis=1)
df_train['skew'] = df_train[idx].skew(axis=1)
df_train['kurt'] = df_train[idx].kurtosis(axis=1)
df_train['med'] = df_train[idx].median(axis=1)


# In[ ]:


norm_data = normalize(df_train, exclude=['Class'])


# In[ ]:


df_train_new = norm_data.drop(['Class'], axis=1)
cont_names = df_train_new.columns
dep_var = 'Class'
procs = [FillMissing, Categorify]
cat_names=[]


# In[ ]:


data = (TabularList.from_df(norm_data, procs = procs, cont_names=cont_names, cat_names=cat_names)
        .split_by_rand_pct(0.3, seed=42)
        .label_from_df(cols=dep_var)
        .databunch(bs=1024))


# In[ ]:


# data.add_test(TabularList.from_df(df_test, cont_names=cont_names))


# In[ ]:


data.show_batch()


# In[ ]:


df_t = data.train_ds.inner_df
df_v = data.valid_ds.inner_df


# In[ ]:


df = df_t.append(df_v, ignore_index=True)


# In[ ]:


X = df.drop(['Class'], axis=1).values
Y = df['Class'].values


# In[ ]:


df.Class.value_counts()


# Before we see which features are the most important ones, lets see correlation of each feature with target variable and dendrogram

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

import numpy as np
import numpy as np
from pandas import *
import matplotlib.pyplot as plt
#from hcluster import pdist, linkage, dendrogram
from numpy.random import rand

X_ = df.T.values #Transpose values 
Y_ = pdist(X_)
Z_ = linkage(Y_)

plt.figure(figsize=(24,24))
#dendrogram(Z, labels = df.columns, orientation='bottom')
fig = ff.create_dendrogram(Z_, labels=df.columns, color_threshold=1.5)
fig.update_layout(width=1500, height=1000)
fig.show()


# In[ ]:


corr_df = pd.DataFrame(df.drop("Class", axis=1).apply(lambda x: x.corr(df.Class)))
corr_df.columns = ['corr']
corr_df.sort_values(by='corr')


# Now lets run ET Classifier to select top features of the data. Here, we have chosen threshold as Median

# In[ ]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

forest = SelectFromModel(ExtraTreesClassifier(bootstrap=True, criterion='gini', max_depth=16, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=7,
            min_samples_split=9, min_weight_fraction_leaf=0.0,
            n_estimators=300, n_jobs=1, oob_score=False, random_state=42,
            verbose=0, warm_start=False), threshold='median')

forest.fit(X, Y)


# In[ ]:


df_without_label = df.drop(['Class'], axis=1)
selected_feat= df_without_label.columns[(forest.get_support())]


# In[ ]:


print(selected_feat)


# Above are our top features which are contributing the most in the prediction task.
# 
# Now, lets plot the feature importance

# In[ ]:


importances = forest.estimator_.feature_importances_

data={'Feature_Name':df.drop(['Class'], axis=1).columns,
      'Feature_Importance': importances
     }

feature_df=pd.DataFrame(data)

feature_df.sort_values(by=['Feature_Importance'],ascending=False,inplace=True)

fig, ax = plt.subplots(figsize=(15,25))
sns.barplot(data=feature_df,y='Feature_Name',x='Feature_Importance')


# For our prediction task, we will only use those most important features and remove other unimportant features

# In[ ]:


df = norm_data[selected_feat]
df['Class'] = norm_data.Class


# In[ ]:


df.Class.value_counts()


# In[ ]:


cont_names = ['V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
       'V14', 'V16', 'V17', 'V18', 'V21', 'V27', 'min', 'med']


dep_var = 'Class'

procs = [FillMissing, Categorify, Normalize]

cat_names= []


# In[ ]:


data = (TabularList.from_df(df, procs = procs, cont_names=cont_names, cat_names=cat_names)
        .split_by_rand_pct(0.3, seed=42)
        .label_from_df(cols=dep_var)
        .databunch(bs=1024))


# In below code snippets, we will create Fastai Learner and train the model

# In[ ]:


from fastai.callbacks import *

learn = tabular_learner(data, layers=[200,100], metrics=accuracy,  ps=[0.2, 0.1], 
                        emb_drop=0.04)


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 1e-2
learn.fit_one_cycle(3, max_lr=lr,  pct_start=0.3, wd = 0.3)


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


lr=1e-5
learn.fit_one_cycle(3, max_lr=lr,  pct_start=0.3, wd = 0.2)


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


lr=1e-6
learn.fit_one_cycle(3, max_lr=lr,  pct_start=0.3, wd = 0.3)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('1st-round')
learn.load('1st-round')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8), dpi=60)


# In[ ]:


gc.collect()


# Usual training through Fastai methods have resulted quite well. Out of around 136 fraudelent records, model is predicting 113 correctly.

# # Fastai Hooks & Embeddings for Train data

# In below code snippets, we will first extract the learnings from the second last layer of the model and retrain on that using Extra Trees Classifier to improve the accuracy of the model

# In[ ]:


class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output): 
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self): 
        self.hook.remove()


# In[ ]:


sf = SaveFeatures(learn.model.layers[4])


# In[ ]:


_= learn.get_preds(data.train_ds)


# In[ ]:


label = [data.classes[x] for x in (list(data.train_ds.y.items))]


# In[ ]:


len(label)


# In[ ]:


df_new = pd.DataFrame({'label': label})


# In[ ]:


array = np.array(sf.features)


# In[ ]:


x=array.tolist()


# In[ ]:


df_new['img_repr'] = x


# In[ ]:


del df_train; gc.collect()


# In[ ]:


d2 = pd.DataFrame(df_new.img_repr.values.tolist(), index = df_new.index).rename(columns = lambda x: 'img_repr{}'.format(x+1))


# In[ ]:


df_new_2 = df_new.join(d2)


# In[ ]:


del d2; gc.collect()


# In[ ]:


df_new_2.drop(['img_repr'], axis=1, inplace=True)


# In[ ]:


non_fraud = df_new_2[df_new_2['label'] == 0].sample(2000)
fraud = df_new_2[df_new_2['label'] == 1]

df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X = df.drop(['label'], axis = 1).values
Y = df["label"].values


# Let's see how does our data look like now in 2D space after training through Fastai library. We can still see few Reds in the areas of Greens. Let's see if we can better this classification

# In[ ]:


tsne_plot(X, Y, "original.png")


# In[ ]:


# df_new_2.drop(['img_repr'], axis=1, inplace=True)

# from imblearn.over_sampling import SMOTE
# from imblearn.combine import SMOTETomek

# sm = SMOTE(ratio='minority', random_state=42)
# df_resampled, y_resampled = sm.fit_sample(df_new_2, df_new_2['label'])
# df_resampled = pd.DataFrame(df_resampled, columns = df_new_2.columns)
# df_new_2['label'].mean(), df_resampled['label'].mean()


# In[ ]:


# df_new_2 = df_resampled


# In[ ]:


# del df_resampled; gc.collect()


# # Embeddings for Valid Data

# In[ ]:


sf = SaveFeatures(learn.model.layers[4])


# In[ ]:


_=learn.get_preds(DatasetType.Valid)


# In[ ]:


label = [data.classes[x] for x in (list(data.valid_ds.y.items))]


# In[ ]:


df_new_valid = pd.DataFrame({'label': label})


# In[ ]:


array = np.array(sf.features)


# In[ ]:


x=array.tolist()


# In[ ]:


df_new_valid['img_repr'] = x


# In[ ]:


d2 = pd.DataFrame(df_new_valid.img_repr.values.tolist(), index = df_new_valid.index).rename(columns = lambda x: 'img_repr{}'.format(x+1))


# In[ ]:


df_new_valid_2 = df_new_valid.join(d2)


# In[ ]:


del d2; 
del sf.features
gc.collect()


# In[ ]:


df_new_valid_2.drop(['img_repr'], axis=1, inplace=True)


# In[ ]:


non_fraud = df_new_valid_2[df_new_valid_2['label'] == 0].sample(1000)
fraud = df_new_valid_2[df_new_valid_2['label'] == 1]

df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X = df.drop(['label'], axis = 1).values
Y = df["label"].values


# This is how data looks like in 2D space for Validation data

# In[ ]:


tsne_plot(X, Y, 'original_png')


# In[ ]:


gc.collect()


# # Extra Trees Classifer

# In[ ]:


X = df_new_2
y = df_new_2.label.copy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=42)


# In[ ]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[ ]:


X_train = X_train.drop("label", axis =1)
y_train = y_train

X_test = X_test.drop("label", axis =1)
y_test = y_test


# In[ ]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attributes_names].values


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# numerical pipeline

num_pipeline = Pipeline([
    
    ('select_data', DataFrameSelector(X_train.columns)),
    ('Std_Scaler', StandardScaler())
])

X_train_transformed = num_pipeline.fit_transform(X_train)
X_test_transformed = num_pipeline.fit_transform(X_test)


# In[ ]:


X_train_transformed.shape, X_test_transformed.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import time

start = time.time()

rf_clf = ExtraTreesClassifier(bootstrap=True, criterion='gini', max_depth=20, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=29,
            min_samples_split=7, min_weight_fraction_leaf=0.0,
            n_estimators=95, n_jobs=1, oob_score=False, random_state=42,
            verbose=0, warm_start=False)

rf_clf.fit(X_train_transformed, y_train)

end = time.time()

print("run_time:", (end-start)/(60*60))


# In[ ]:


from sklearn.model_selection import cross_val_predict

import time

start = time.time()

y_train_pred_rf = cross_val_predict(rf_clf, X_train_transformed, y_train, cv=3, verbose=5)

end = time.time()

print("run_time:", (end-start)/(60*60))


# Below is the Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_train, y_train_pred_rf)


# In[ ]:


type(y_train_pred_rf), type(X_train_transformed)


# In[ ]:


X = pd.DataFrame(X_train_transformed)
X['label'] = y_train_pred_rf


# In[ ]:


non_fraud = X[X['label'] == 0].sample(1000)
fraud = X[X['label'] == 1]

df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X = df.drop(['label'], axis = 1).values
Y = df["label"].values


# After training using ET Classifier, our classification of Reds and Greens looks cleaner now.

# In[ ]:


tsne_plot(X, Y, 'original_png')


# Let's see how does precision score, recall score, F1 score, Cohen Kappa score look like

# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, cohen_kappa_score

print(precision_score(y_train, y_train_pred_rf))
print(recall_score(y_train, y_train_pred_rf))
print(f1_score(y_train, y_train_pred_rf))
print(cohen_kappa_score(y_train, y_train_pred_rf))

print(classification_report(y_train, y_train_pred_rf))


# In[ ]:


# import scipy.stats as st
# from sklearn.model_selection import RandomizedSearchCV

# one_to_left = st.beta(10, 1)  
# from_zero_positive = st.expon(0, 50)

# params = {  
#     "n_estimators": st.randint(50, 300),
#     "max_depth": st.randint(3, 40),
#    "min_samples_leaf": st.randint(3, 40),
#     "min_samples_split": st.randint(3, 20),
#     'max_features': ['auto', 0.2, 0.5]
# }

# gs = RandomizedSearchCV(rf_clf, params, cv=3)


# In[ ]:


# gs.fit(X_train_transformed, y_train)  


# In[ ]:


# gs.best_params_


# In[ ]:


from sklearn.model_selection import cross_val_predict

y_probas_rf = cross_val_predict(rf_clf, X_train_transformed, y_train, cv=3, method="predict_proba", verbose=0)
y_scores_rf = y_probas_rf[:,1]


# Let's calculate ROC AUC score and plot ROC AUC curve along with Precision Recall curve

# In[ ]:


roc_score_rf = roc_auc_score(y_train, y_scores_rf)
roc_score_rf


# In[ ]:


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_rf)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds, recalls[:-1], "r-", label = "Recall")
    plt.xlabel("Thresholds")
    plt.legend(loc="upper left")
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# In[ ]:


from sklearn.metrics import roc_curve

fpr_rf, tpr_rf, threshold_rf = roc_curve(y_train, y_scores_rf)

def plot_roc_curve(fpr_rf, tpr_rf, figsize=(15,12), label = None):
    plt.plot(fpr_rf, tpr_rf, linewidth = 2, label = label)
    plt.plot([0,1], [0,1], "k--")
    plt.axis([0,1,0,1])
    plt.xlabel("False Pos Rate")
    plt.ylabel('True Neg Rate')
    
    
plot_roc_curve(fpr_rf, tpr_rf)


# In[ ]:


plt.plot(thresholds, recalls[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Thresholds')
plt.ylabel('Recall')
plt.show()


# # Performance on Unseen Test

# In[ ]:


y_pred_test_rf = rf_clf.predict(X_test_transformed)


# In[ ]:


confusion_matrix(y_test, y_pred_test_rf)


# In[ ]:


X = pd.DataFrame(X_test_transformed)
X['label'] = y_pred_test_rf


# In[ ]:


non_fraud = X[X['label'] == 0].sample(2000)
fraud = X[X['label'] == 1]

df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X = df.drop(['label'], axis = 1).values
Y = df["label"].values


# This is our data looks like in 2D space for unseen test dataset

# In[ ]:


tsne_plot(X, Y, 'org_png')


# Precision, Recall, F1 score, Cohen Kappa score on Test Data

# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, cohen_kappa_score

print(precision_score(y_test, y_pred_test_rf))
print(recall_score(y_test, y_pred_test_rf))
print(f1_score(y_test, y_pred_test_rf))
print(cohen_kappa_score(y_test, y_pred_test_rf))

print(classification_report(y_test, y_pred_test_rf))


# In[ ]:


y_probas_rf = rf_clf.predict_proba(X_test_transformed)
y_scores_rf = y_probas_rf[:,1]
roc_score_rf = roc_auc_score(y_test, y_scores_rf)
roc_score_rf


# In[ ]:


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores_rf)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds, recalls[:-1], "r-", label = "Recall")
    plt.xlabel("Thresholds")
    plt.legend(loc="upper left")
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# In[ ]:


from sklearn.metrics import roc_curve

fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, y_scores_rf)

def plot_roc_curve(fpr_rf, tpr_rf, figsize=(15,12), label = None):
    plt.plot(fpr_rf, tpr_rf, linewidth = 2, label = label)
    plt.plot([0,1], [0,1], "k--")
    plt.axis([0,1,0,1])
    plt.xlabel("False Pos Rate")
    plt.ylabel('True Neg Rate')
      
plot_roc_curve(fpr_rf, tpr_rf)


# # Performance on Valid Data

# Now, lets see how does our model perform on completely unseen validation data which our ET classifer never saw.

# In[ ]:


X = df_new_valid_2
y = df_new_valid_2.label.copy()


# In[ ]:


X_val = X.drop("label", axis =1)
y_val = y


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# numerical pipeline

num_pipeline = Pipeline([
    
    ('select_data', DataFrameSelector(X_val.columns)),
    ('Std_Scaler', StandardScaler())
])


X_val_transformed = num_pipeline.fit_transform(X_val)


# In[ ]:


y_pred_test_rf_val = rf_clf.predict(X_val_transformed)


# In[ ]:


confusion_matrix(y_val, y_pred_test_rf_val)


# In[ ]:


X = pd.DataFrame(X_val_transformed)
X['label'] = y_pred_test_rf_val 


# In[ ]:


non_fraud = X[X['label'] == 0].sample(1000)
fraud = X[X['label'] == 1]

df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X = df.drop(['label'], axis = 1).values
Y = df["label"].values


# In[ ]:


tsne_plot(X, Y, 'orig_ong')


# In[ ]:


gc.collect()


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, cohen_kappa_score

print(precision_score(y_val, y_pred_test_rf_val))
print(recall_score(y_val, y_pred_test_rf_val))
print(f1_score(y_val, y_pred_test_rf_val))
print(cohen_kappa_score(y_val, y_pred_test_rf_val))

print(classification_report(y_val, y_pred_test_rf_val))


# In[ ]:


y_probas_rf = rf_clf.predict_proba(X_val_transformed)
y_scores_rf = y_probas_rf[:,1]
roc_score_rf = roc_auc_score(y_val, y_scores_rf)
roc_score_rf


# In[ ]:


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores_rf)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds, recalls[:-1], "r-", label = "Recall")
    plt.xlabel("Thresholds")
    plt.legend(loc="upper left")
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# In[ ]:


from sklearn.metrics import roc_curve

fpr_rf, tpr_rf, threshold_rf = roc_curve(y_val, y_scores_rf)

def plot_roc_curve(fpr_rf, tpr_rf, figsize=(15,12), label = None):
    plt.plot(fpr_rf, tpr_rf, linewidth = 2, label = label)
    plt.plot([0,1], [0,1], "k--")
    plt.axis([0,1,0,1])
    plt.xlabel("False Pos Rate")
    plt.ylabel('True Neg Rate')
      
plot_roc_curve(fpr_rf, tpr_rf)


# This is quite nice result.
# 
# There are two options for the overall objective:
# 1. Either we want to reduce False Negatives i.e. cases which are normal but predicted as fraudelent OR
# 2. We want to reduce False Positives i.e. cases which are fraudelent but predicted as normal transactions

# # Light GBM

# In[ ]:


gc.collect()


# In[ ]:


import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')


# In[ ]:


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.20,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.20,
    'learning_rate': 1e-3,
    'max_depth': 10,  
    'metric':'auc',
    'min_data_in_leaf': 30,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 20,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1,
    'sub_feature':0.5
}


# In[ ]:


features_train = [c for c in df_new_2.columns if c not in ['label']]
target_train = df_new_2['label']


# In[ ]:


features_val = [c for c in df_new_valid_2.columns if c not in ['label']]
target_val = df_new_valid_2['label']


# In[ ]:


scaler = StandardScaler()
df_new_2[features_train] = scaler.fit_transform(df_new_2[features_train])
df_new_valid_2[features_val] = scaler.transform(df_new_valid_2[features_val])


# In[ ]:


trn_data = lgb.Dataset(df_new_2[features_train], label=target_train)
val_data = lgb.Dataset(df_new_valid_2[features_val], label=target_val)

num_round=50000
clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data,val_data], verbose_eval=500, early_stopping_rounds = 2000)
preds = clf.predict(df_new_valid_2[features_val], num_iteration=clf.best_iteration)   


# In[ ]:


for i in range(0,preds.shape[0]):
    if preds[i]>=.5:
        preds[i]=1
    else:  
        preds[i]=0


# In[ ]:


X = pd.DataFrame(df_new_valid_2)
X['label'] = preds 
X.head()


# In[ ]:


non_fraud = X[X['label'] == 0].sample(1000)
fraud = X[X['label'] == 1]

df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X = df.drop(['label'], axis = 1).values
Y = df["label"].values


# In[ ]:


tsne_plot(X, Y, 'lgb_png')


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure()
cm = confusion_matrix(target_val, preds)
labels = ['0', '1']
plt.figure(figsize=(8,6))
sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, cohen_kappa_score

print(precision_score(target_val, preds))
print(recall_score(target_val, preds))
print(f1_score(target_val, preds))
print(cohen_kappa_score(target_val, preds))

print(classification_report(target_val, preds))


# # Pytorch AutoEncoder

# In[ ]:


norm_data.shape


# In[ ]:


X = norm_data
y = norm_data.Class.copy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=42)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


df_new_2.shape


# In[ ]:


X_train = df_new_2[df_new_2.label == 0]
X_train = df_new_2.drop(['label'], axis=1)
X_test = df_new_valid_2.drop(['label'], axis=1)
y_test = df_new_valid_2.label

X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values
print(y_test.size)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(X_train.shape[1], 20),
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, 5),
            nn.LeakyReLU(),
            )
        
        self.decoder = nn.Sequential(
           nn.Linear(5, 10),
           nn.Tanh(),
           nn.Linear(10, 20),
           nn.Tanh(),
           nn.Linear(20, X_train.shape[1]),
           nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# In[ ]:


model = Autoencoder().double().cpu()


# In[ ]:


num_epochs = 150
minibatch_size = 32
learning_rate = 1e-3


# In[ ]:


import pandas as pd
import numpy as np
import pickle

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8


# In[ ]:


train_loader = data_utils.DataLoader(X_train, batch_size=minibatch_size, shuffle=True)


# In[ ]:


test_loader = data_utils.DataLoader(X_test, batch_size=1, shuffle=False)


# In[ ]:


criterion = nn.MSELoss()
optimizer = torch.optim.Adadelta(
model.parameters(), lr=learning_rate, weight_decay=1e-4)


# In[ ]:


history = {}
history['train_loss'] = []
history['test_loss'] = []


# In[ ]:


for epoch in range(num_epochs):
    h = np.array([])
    for data in train_loader:
#         print(type(data))
#         data = Variable(data).cpu()
#         print(type(data))
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        h = np.append(h, loss.item())
        
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    mean_loss = np.mean(h)
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, mean_loss))
    history['train_loss'].append(mean_loss)
    

torch.save(model.state_dict(), './credit_card_model.pth')


# In[ ]:


plt.plot(history['train_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[ ]:


model


# In[ ]:


pred_losses = {'pred_loss' : []}
model.eval()
with torch.no_grad():
    for data in test_loader:
        inputs = data
        outputs = model(inputs)
        loss = criterion(outputs, inputs).data.item()
        pred_losses['pred_loss'].append(loss)
        
        
reconstructionErrorDF = pd.DataFrame(pred_losses)
reconstructionErrorDF['Class'] = y_test


# In[ ]:


reconstructionErrorDF.describe()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = reconstructionErrorDF[(reconstructionErrorDF['Class']== 0) & (reconstructionErrorDF['pred_loss'] < 10)]
_ = ax.hist(normal_error_df.pred_loss.values, bins=10)


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = reconstructionErrorDF[(reconstructionErrorDF['Class']== 1) ]
_ = ax.hist(fraud_error_df.pred_loss.values, bins=10)


# In[ ]:


fpr, tpr, thresholds = roc_curve(reconstructionErrorDF.Class, reconstructionErrorDF.pred_loss)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[ ]:


threshold = 2
LABELS = ["Normal", "Fraud"]

y_pred = [1 if e > threshold else 0 for e in reconstructionErrorDF.pred_loss.values]
conf_matrix = confusion_matrix(reconstructionErrorDF.Class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", 
            cmap=plt.cm.get_cmap('Blues'));
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

