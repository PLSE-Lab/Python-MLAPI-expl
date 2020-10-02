#!/usr/bin/env python
# coding: utf-8

# ## Demonstration to Model Stacking in combination with Fastai2
# **This work is inspired from the Zachary Mueller demonstration of Ensembling in Fastai2** - [Link](https://github.com/muellerzr/Practical-Deep-Learning-for-Coders-2.0/tree/master/Tabular%20Notebooks)

# In[ ]:


get_ipython().system('pip install fastai2')
get_ipython().system('pip install rfpimp')


# In[ ]:


import numpy as np
import pandas as pd
import sys
import sklearn
import os
import pathlib
import fastai2
import numpy as np
import pandas as pd
from fastai2.tabular.all import *
from fastai2.basics import *
from rfpimp import *

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
print(fastai2.__version__)


# In[ ]:


path = pathlib.Path('/kaggle/input/cat-in-the-dat-ii/')
df =  pd.read_csv(path/'train.csv')
testdf = pd.read_csv(path/'test.csv')
test_id = testdf['id']


# In[ ]:


df.head(10).T


# In[ ]:


df.columns


# In[ ]:


df['target'] = df['target'].astype('category')


# In[ ]:


df.info()


# In[ ]:


df.nunique()


# In[ ]:


cat_names = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1',
       'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
       'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']
cont_names = ['ord_0']
procs = [FillMissing, Categorify, Normalize]
dep_var = 'target'
#block_y = CategoryBlock()
#splits = RandomSplitter()(range_of(df))
splits = TrainTestSplitter(test_size=0.20, stratify= df[dep_var])(range_of(df))


# In[ ]:


splits


# In[ ]:


pd.options.mode.chained_assignment=None


# In[ ]:


to = TabularPandas(df, procs, cat_names, cont_names, dep_var, y_block=CategoryBlock(),
                   splits=splits, inplace=True, reduce_memory=True)


# In[ ]:


config = tabular_config(embed_p = 0.04) #ps=[0.001,0.01]


# In[ ]:


import torch
from torch import nn

dls = to.dataloaders(bs = 512)
dls.c = 2
from fastai2.metrics import *
learn = tabular_learner(dls,
                        layers=[100,50],
                        config = config,
                        metrics=[accuracy, RocAuc(average='weighted'), F1Score(), Precision(), Recall()])


# In[ ]:


learn.summary()


# In[ ]:


lr_best, _ = learn.lr_find()


# In[ ]:


lr_best, _


# In[ ]:


learn.fit_one_cycle(30, lr_best,
                    cbs=[SaveModelCallback(),
                         EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=10)])
dl = learn.dls.test_dl(testdf)
raw_test_preds = learn.get_preds(dl=dl)
raw_test_preds[0].numpy()
test_preds = raw_test_preds[0].numpy().T[1]


# In[ ]:


class PermutationImportance():
    "Calculate and plot the permutation importance"
    def __init__(self, learn:Learner, df=None, bs=None):
        "Initialize with a test dataframe, a learner, and a metric"
        self.learn = learn
        self.df = df if df is not None else None
        bs = bs if bs is not None else learn.dls.bs
        self.dl = learn.dls.test_dl(self.df, bs=bs) if self.df is not None else learn.dls[1]
        self.x_names = learn.dls.x_names.filter(lambda x: '_na' not in x)
        self.na = learn.dls.x_names.filter(lambda x: '_na' in x)
        self.y = dls.y_names
        self.results = self.calc_feat_importance()
        self.plot_importance(self.ord_dic_to_df(self.results))

    def measure_col(self, name:str):
        "Measures change after column shuffle"
        col = [name]
        if f'{name}_na' in self.na: col.append(name)
        orig = self.dl.items[col].values
        perm = np.random.permutation(len(orig))
        self.dl.items[col] = self.dl.items[col].values[perm]
        metric = learn.validate(dl=self.dl)[1]
        self.dl.items[col] = orig
        return metric

    def calc_feat_importance(self):
        "Calculates permutation importance by shuffling a column on a percentage scale"
        print('Getting base error')
        base_error = self.learn.validate(dl=self.dl)[1]
        self.importance = {}
        pbar = progress_bar(self.x_names)
        print('Calculating Permutation Importance')
        for col in pbar:
            self.importance[col] = self.measure_col(col)
        for key, value in self.importance.items():
            self.importance[key] = (base_error-value)/base_error #this can be adjusted
        return OrderedDict(sorted(self.importance.items(), key=lambda kv: kv[1], reverse=True))

    def ord_dic_to_df(self, dict:OrderedDict):
        return pd.DataFrame([[k, v] for k, v in dict.items()], columns=['feature', 'importance'])

    def plot_importance(self, df:pd.DataFrame, limit=20, asc=False, **kwargs):
        "Plot importance with an optional limit to how many variables shown"
        df_copy = df.copy()
        df_copy['feature'] = df_copy['feature'].str.slice(0,25)
        df_copy = df_copy.sort_values(by='importance', ascending=asc)[:limit].sort_values(by='importance', ascending=not(asc))
        ax = df_copy.plot.barh(x='feature', y='importance', sort_columns=True, **kwargs)
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.4f}', ((p.get_width() * 1.005), p.get_y()  * 1.005))
            
imp = PermutationImportance(learn)


# ## XGBOOST

# In[ ]:


tst = dl.xs


# In[ ]:


type(tst), len(tst)
tst.head()


# In[ ]:


import xgboost as xgb
X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()


# In[ ]:



model = xgb.XGBClassifier(n_estimators = 100,
                          max_depth=10,
                          learning_rate=0.1,
                          subsample=0.5,
                          nthread = -1,
                          max_delta_step = 5
                         )
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model = model.fit(X_train, y_train,
                      eval_metric=["error", "logloss"],
                      eval_set=eval_set, verbose=True,
                      early_stopping_rounds=10)


# In[ ]:


# retrieve performance metrics
   results = xgb_model.evals_result()
   epochs = len(results['validation_0']['logloss'])
   x_axis = range(0, epochs)
   
   # plot log loss
   fig, ax = plt.subplots(figsize=(8,8))
   ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
   ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
   ax.legend()
   
   plt.ylabel('Log Loss')
   plt.title('XGBoost Log Loss')
   plt.show()


# In[ ]:


from xgboost import plot_importance
plot_importance(xgb_model)


# In[ ]:


xgb_valid = xgb_model.predict_proba(X_test)
print(accuracy(tensor(xgb_valid), tensor(y_test)))


# In[ ]:


# xgb_preds = xgb_model.predict_proba(X_test)
xgb_preds = xgb_model.predict_proba(tst)[:, 1]
print(xgb_preds)
print(xgb_preds.shape)


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
tree = RandomForestClassifier(n_estimators=100)
tree.fit(X_train, y_train)
imp = importances(tree, X_test, to.valid.ys)


# In[ ]:


plot_importances(imp)


# In[ ]:


type(X_test), type(y_test)


# In[ ]:


tree_valid = np.argmax(tree.predict_proba(X_test), axis = 1)
tree_valid


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
print(accuracy_score(tree_valid, y_test))


# In[ ]:


confusion_matrix(tree_valid, y_test)


# In[ ]:


#forest_preds = tree.predict_proba(X_test)
forest_preds = tree.predict_proba(tst)[:, 1]


# ## Categorical NaiveBayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)


# In[ ]:


nb_valid = np.argmax(clf.predict_proba(X_test), axis = 1)
print(accuracy_score(nb_valid, y_test))
print(confusion_matrix(nb_valid, y_test))
print(roc_auc_score(nb_valid, y_test))


# In[ ]:


#NB_preds = clf.predict_proba(X_test)
NB_preds = clf.predict_proba(tst)[:, 1]


# # KNN

# In[ ]:


from sklearn import neighbors
n_neighbors = 4
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X_train, y_train)
KNN4_preds = clf.predict_proba(tst)[:, 1]
KNN4_preds
knn_valid = np.argmax(clf.predict_proba(X_test), axis = 1)
print(accuracy_score(knn_valid, y_test))
print(confusion_matrix(knn_valid, y_test))
print(roc_auc_score(knn_valid, y_test))


# In[ ]:


n_neighbors = 8
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X_train, y_train)


# In[ ]:


knn_valid = np.argmax(clf.predict_proba(X_test), axis = 1)
print(accuracy_score(knn_valid, y_test))
print(confusion_matrix(knn_valid, y_test))
print(roc_auc_score(knn_valid, y_test))


# In[ ]:


KNN8_preds = clf.predict_proba(tst)[:, 1]
KNN8_preds


# In[ ]:


n_neighbors = 16
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X_train, y_train)
KNN32_preds = clf.predict_proba(tst)[:, 1]
KNN32_preds


# In[ ]:


knn_valid = np.argmax(clf.predict_proba(X_test), axis = 1)
print(accuracy_score(knn_valid, y_test))
print(confusion_matrix(knn_valid, y_test))
print(roc_auc_score(knn_valid, y_test))


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X_train, y_train)
LR_preds = clf.predict_proba(tst)[:, 1]

LR_valid = np.argmax(clf.predict_proba(X_test), axis = 1)
print(accuracy_score(LR_valid, y_test))
print(confusion_matrix(LR_valid, y_test))
print(roc_auc_score(LR_valid, y_test))


# ## Deep Learning Model

# In[ ]:


data = {
        'KNN4':KNN4_preds,
        'KNN8':KNN8_preds,
        'LR':LR_preds,
        'NB':NB_preds,
        'RF':forest_preds,
        'XGB':xgb_preds,
        'NN': test_preds
       } 
result = pd.DataFrame(data)


# In[ ]:


result.head()


# In[ ]:


result['Avg'] = (result.KNN8 + result.NB + result.RF + result.LR +
                 result.XGB + result.NN)/len(result.columns)
result.head()


# In[ ]:


print("Saving submission file")
submission = pd.DataFrame.from_dict({
    'id': test_id,
    'target': result['Avg']
})
submission.to_csv("submission_stacking_v3.csv", index=False)


# In[ ]:





# In[ ]:




