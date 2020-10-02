#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# This notebook deals with models for predicting whether ceramic glazes will be glossy or matte when fired, using the [Glazy](https://glazy.org/) glaze database. The primary target audience is potters with some understanding of glaze calculation. Although familiarity with standard data science techniques would be helpful in understanding the construction of the models, this is not required to understand the predictions. Regardless of your background, please feel free to ask questions or make suggestions.
# 
# Some of the factors that can affect whether or not a glaze will be glossy are the chemical composition, the firing profile (particularly the cooling phase), the particle size of the ingredients, the clay-body to which the glaze was applied, and whether the person who measured the glaze batch was paying attention or not. The data in the Glazy library contains fields for the chemical composition, expressed in terms of oxides, and an upper and lower bound for the firing [cone](https://wiki.glazy.org/t/measuring-heatwork-temperature-with-orton-cones/509), but there are no fields for the other factors. Can this limited information be used to predict whether a glaze will be matte or glossy?

# The first part of this notebook involves data preparation. If you're only interested in the predictions, skip ahead by using the menu on the left. The code cells have been hidden, but can be viewed by clicking the `Code` button.

# ## Data preparation

# Load Python modules:

# In[ ]:


# import modules
import numpy as np
import pandas as pd
import json

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mpl_colour
import seaborn as sns

import lightgbm as lgbm
import catboost as cat
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, MetaEstimatorMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, confusion_matrix, make_scorer

# Own scripts:
from split import splitter, split, WeightedGroupKFold, StratifiedWeightedGroupKFold
from add_value_labels import add_value_labels

from hyperopt import hp, STATUS_OK, fmin, tpe, Trials
from hyperopt.pyll import scope 

import eli5
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import iter_shuffled

from functools import partial
import collections
import itertools
import os
#print(os.listdir('../input/glazy-data-june-2019'))


# Construct the indices for the features that will go into the models. Since the oxides of potassium (K<sub>2</sub>O) and sodium (Na<sub>2</sub>O) are largely interchangeable, we only use their sum, denoted KNaO.

# In[ ]:


# Find out what the columns are
data0 = pd.read_csv("../input/glazy-data-june-2019/glazy_data_june_2019.csv", nrows=0)
"""for i, col in enumerate(list(data0.columns)):
    print(i, col)"""
percent_col_indices = list(range(18, 79))
umf_col_indices = list(range(79, 140))[:31]
xumf_col_indices = list(range(140, 201))
percent_mol_col_indices = list(range(262, 323))[:31]
ox_umf = data0.columns[umf_col_indices].tolist()[:31]  # Don't select the rare earths
ox_umf.remove('K2O_umf')   # KNaO is a feature, so we'll drop both 'K2O_umf' and 'Na2O_umf'
ox_umf.remove('Na2O_umf')
ox_percent_mols = data0.columns[percent_mol_col_indices].tolist()


# Load the data from the CSV file into a pandas DataFrame, omitting features we're not interested in.

# In[ ]:


data = pd.read_csv("../input/glazy-data-june-2019/glazy_data_june_2019.csv", 
                   usecols=[0,1,3,10,     12,13,14,15] + umf_col_indices + list(np.arange(323, 326)))  # Only selecting percent mol
data.drop(columns=['Na2O_umf', 'K2O_umf'], inplace=True)
#data['transparency_type'].astype('str').value_counts().sort_index().plot.bar()
plt.show()
#print(data.shape)
#display(data.head())   # Show the first 5 rows

# Drop analyses and primitive materials:
data = data[(~(data["is_analysis"]==1)) & ~(data["is_primitive"]==1)]    
data.drop(columns=["is_analysis","is_primitive"], inplace=True)


# Drop recipes that are not glazes.

# In[ ]:


# Use the glaze id as the index
data.set_index("id", inplace=True)

# Select glaze recipes
data = data.loc[data["material_type_id"].between(460, 1170)]   

# Drop name and material type
data.drop(columns=['name', 'material_type_id'], inplace=True)

#display(data.head())


# Drop recipes with no upper or no lower cone bounds

# In[ ]:


data = data[~data.from_orton_cone.isnull()]
data = data[~data.to_orton_cone.isnull()]


# Plot the frequency of each surface type:

# In[ ]:


"""
surface_types = data['surface_type'].unique().astype('str')
print('Surface types:')
for s in sorted(surface_types):
    print(s)"""
data['surface_type'].astype('str').value_counts().sort_index().plot.bar()
plt.show()


# For some glazes, the surface type hasn't been recorded (these have the value 'nan'), so we'll drop them. To simplify the classification task, we'll just use 2 labels; the 'Glossy' and 'Glossy - Semi' surface types will be indexed by 1 (glossy), and the rest by 0 (matte).

# In[ ]:


labelled_data = data[~data.surface_type.isnull()]
labelled_data.loc[:,'surface_type'] = labelled_data['surface_type'].isin(['Glossy', 'Glossy - Semi'])  # 1 = Glossy, 0 = Matte
unlabelled_data = data[data.surface_type.isnull()]
#print(labelled_data.shape[0], 'recipes with surface type labels,', 
#      unlabelled_data.shape[0], 'recipes without surface type labels')
num_glossy = labelled_data['surface_type'].sum()
#print(num_glossy, 'glossy recipes,', labelled_data.shape[0] - num_glossy, 'matte recipes')
#display(train.head())
labels = 'Glossy', 'Unlabelled', 'Matte' 
sizes = [num_glossy, unlabelled_data.shape[0], labelled_data.shape[0] - num_glossy]
colours = [(0.993248, 0.906157, 0.143936, 1.0), 'lightgrey', (0.267004, 0.004874, 0.329415, 0.6)]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colours, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# Many recipes in the Glazy dataset are duplicates or slight variations of other recipes. This creates a problem when deciding which model has the best predictive ability, for the following reason. A model's predictive ability is determined by training it on a subset of the data (the 'training set'), and then measuring the accuracy of its predictions on data that it was not trained on (the 'validation set'). If there is a substantial overlap between the training and validation sets, models which simply memorise the training set will do well on the validation set, even if they perform poorly on data not in the training set. To mitigate this, we group the data into clusters of similar recipes, and ensure that recipes in the same cluster don't occur in both the training and validation sets. 
# 
# A related problem is that recipes which have many duplicates have a disproportionate effect on the training and evaluation of models. To counteract this, we weight each recipe in a cluster by 1/n, where n is the number of recipes in the cluster. This means that no matter how many times a recipe is duplicated, the combined weight of all the duplicates is still 1.
# 
# We use the [K-Means](https://en.wikipedia.org/wiki/K-means_clustering) algorithm to cluster the recipes, and select the best number of clusters by maximising the average [silhouette score](https://en.wikipedia.org/wiki/Silhouette_(clustering%29).

# In[ ]:


get_ipython().run_cell_magic('time', '', 'cluster_candidates0 = labelled_data[ox_umf] #[~data.surface_type.isnull()][ox_umf]\n# The clustering will be done in two stages. The first stage is to identify the clusters with only a single recipe\n# (the singletons). These are then removed, and the remaining recipes are re-clustered. This is because the silhouette\n# score of a singleton is defined to be 0.\n# In the code below, we iterate through a list of length 1, since this has already been optimised. For a different \n# data set, multiple values should be checked.\nfor n_clusters in [2400]: #range(2496,2505,1):\n    best_silhouette_avg = 0\n    for state in np.arange(0,5):  # KMeans depends on a random initialisation, so we take the best of 5 runs\n        clusterer = KMeans(n_clusters=n_clusters, random_state=state)\n        cluster_labels = clusterer.fit_predict(cluster_candidates0)\n        silhouette_avg = silhouette_score(cluster_candidates0, cluster_labels)\n        #print(silhouette_avg)\n        if silhouette_avg > best_silhouette_avg:\n            best_silhouette_avg = silhouette_avg \n            cluster_labels0 = cluster_labels\n    print("For", n_clusters, "clusters, the best average silhouette score is", best_silhouette_avg)\n    \nfreq = collections.Counter(cluster_labels0)\ncounts = list(dict(freq).values())\nweights = pd.Series([1/freq[l] for l in cluster_labels0], index=cluster_candidates0.index)\nsingletons = weights==1\nweights0 = weights[singletons]\n#print((weights==1).sum())\n#plt.hist(counts, bins=24, range=(2,26))\ncluster_candidates1 = cluster_candidates0[~singletons]')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'bestest_silhouette_avg = 0\nfor n_clusters in range(1100,1101,10):\n    best_silhouette_avg = 0\n    for state in np.arange(5, 10):\n        clusterer = KMeans(n_clusters=n_clusters, random_state=state)\n        cluster_labels = clusterer.fit_predict(cluster_candidates1)\n        silhouette_avg = silhouette_score(cluster_candidates1, cluster_labels)\n        if silhouette_avg > best_silhouette_avg:\n            best_silhouette_avg = silhouette_avg \n            cluster_labels1 = cluster_labels\n    if best_silhouette_avg > bestest_silhouette_avg:\n        bestest_silhouette_avg = best_silhouette_avg \n        best_cluster_labels1 = cluster_labels1\n    print("For", n_clusters, "clusters on the reduced dataset, the best average silhouette score is", best_silhouette_avg)\nprint("Overall best average silhouette score is", bestest_silhouette_avg)')


# Construct labels for the clusters, and calculate the weights. From now on, any quantity involving the glazes will be calculated using the weights, even if this is not explicitly stated. For example, averages will be weighted averages, and sums will be weighted sums.

# In[ ]:


final_cluster_labels = pd.Series(0, index=cluster_candidates0.index)
final_cluster_labels[~singletons] = best_cluster_labels1
final_cluster_labels[singletons] = np.arange(0, singletons.sum()) + best_cluster_labels1.max() + 1
#display(final_cluster_labels)
#print(final_cluster_labels.sort_values())
freq = collections.Counter(final_cluster_labels)
#counts = list(dict(freq).values())
weights = pd.Series([1/freq[l] for l in final_cluster_labels], index=final_cluster_labels.index)
plt.hist(list(dict(freq).values()), bins=80, range=(0,80), log=True)
plt.title('Histogram of cluster sizes (log scale)')
plt.show()


# Construct the input data, which will consist of the firing range, the UMF ([unity molecular formula](https://juxtamorph.com/what-is-a-segerunity-formula/)) of the glaze, the silica-alumina ratio, and the sum of the UMF alkali metals (R<sub>2</sub>O). Since `RO_umf` is equal to `1 - R2O_umf`, this is not included. A number of oxides, which only occur in non-trivial amounts in a small number of glazes, are omitted.

# In[ ]:


X = labelled_data.drop(columns=['surface_type', 'RO_umf'])
X.drop(columns=['MnO_umf', 'Cr2O3_umf', 'FeO_umf', 'ZrO_umf', 'NiO_umf', 'CdO_umf', 
                'Cu2O_umf', 'BeO_umf', 'PbO_umf', 'F_umf', 'V2O5_umf'],
      inplace=True)
cone_dict = {'05 &#189;':-5.5, '5 &#189;':5.5}
for k in range(10):
    cone_dict['0'+str(k)] = -k
X[['from_orton_cone', 'to_orton_cone']] = X[['from_orton_cone', 'to_orton_cone']].replace(to_replace=cone_dict).astype('float16')

Y = labelled_data['surface_type'].astype('int')
print('Features to use for prediction:')
print(list(X.columns))

#X = X[X.index < 150]
#Y = Y[Y.index < 150]


# ## Training and Evaluation

# We'll use the [XGBoost](https://xgboost.readthedocs.io/en/latest/) algorithm to predict whether a glaze is glossy or matte. This is a type of [boosting](https://towardsdatascience.com/boosting-algorithms-explained-d38f56ef3f30) algorithm, which falls under the umbrella of [supervised machine learning](https://en.wikipedia.org/wiki/Supervised_learning) methods. Briefly, the goal in supervised machine learning is to produce a function that can correctly predict an outcome based on some input data. In our case, we want to produce a function that predicts either 'glossy' or 'matte', given the UMF data and firing range of the glaze. This function is constructed (or 'learnt') from a list of inputs together with the corresponding list of outcomes. Once the function has been constructed, you can feed it any input data of the appropriate form, and it will produce an outcome, or prediction.
# 
# To determine how well it performs, we'll first train it on a subset of the data, then evaluate its predictions on the remaining data. These two subsets are called the *training* and *validation* subsets. Different choices of training / validation splits may lead to different results, so we do this multiple times, in a process called [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html). This gives multiple evaluations, which we'll summarise by reporting the mean (average) and standard deviation.
# 
# Machine learning algorithms usually depend on a number of parameters (called *hyper*parameters in this context), and XGBoost is no exception. Models with hyperparameters can be thought of as a family of algorithms, indexed by the hyperparameters. Instead of picking one such algorithm by using a fixed choice of hyperparameters, during the training step we'll try multiple hyperparameters, and use cross-validation on the training set to select the ones with the highest average cross-validation score. The hyperparameters we try are determined using [Bayesian optimisation](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f).
# 
# There are a number of ways of measuring the predictive ability of an algorithm that predicts one of two outcomes (a *binary classifier*). The most intuitive is probably the accuracy (proportion of correct predictions), but we'll use something called the [Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) (MCC), which has better properties than the accuracy. The key thing to know about the MCC is that it takes values in the interval from -1 to 1, with 1 being all predictions correct, -1 being all predictions incorrect, and 0 the expected value if predictions are made randomly.

# In[ ]:


seeds = np.arange(0,200)

strat_wt_kfold = StratifiedWeightedGroupKFold(n_splits=5, 
                                              n_repeats=3, 
                                              shuffle=True, 
                                              random_states=seeds,
                                              weights=weights)


# In[ ]:


# https://medium.com/vantageai/bringing-back-the-time-spent-on-hyperparameter-tuning-with-bayesian-optimisation-2e21a3198afb

# helper function
def convert_param_types(param_dist, params):
    return {k: eval(param_dist[k].name)(v) for k, v in params.items()}

#from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class BOestimator(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class=LogisticRegression, param_space={}, fixed_params={}, num_eval=20, groups=None, scorer=None):
        self.model_class = model_class
        self.param_space = param_space
        self.fixed_params = fixed_params
        self.num_eval = num_eval
        self.groups=groups
        self.scorer = scorer

    def fit(self, X, y=None, sample_weight=None, cv=5, seed=None):
        
        if type(cv) is int:
            pass
        else:
            try:
                cv.weights = sample_weights
            except:
                pass
            cv=cv.split(X, y, groups=self.groups[X.index])   # iterator of splits
        def objective_function(params):
            clf = self.model_class(**params, **self.fixed_params)
            score = cross_val_score(clf, 
                                    X, 
                                    y, 
                                    #groups=self.groups[X.index], # do we still need this?
                                    fit_params={'sample_weight' : sample_weight},
                                    cv=cv, 
                                    scoring=self.scorer).mean()
            return {'loss': -score, 'status': STATUS_OK}

        self.trials_ = Trials()
        best_param = fmin(objective_function, 
                          self.param_space, 
                          algo=tpe.suggest, 
                          max_evals=self.num_eval, 
                          trials=self.trials_,
                          rstate= np.random.RandomState(seed))
    
        self.best_param_ = convert_param_types(self.param_space, best_param)
        self.clf_best_ = self.model_class(**self.best_param_, **self.fixed_params)   
        self.clf_best_.fit(X, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X):
    
        return self.clf_best_.predict(X)
    
    def predict_proba(self, X):
        
        return self.clf_best_.predict_proba(X)
    
    def score(self, X, y):
        
        return make_scorer(matthews_corrcoef)(self.clf_best_, X, y, sample_weight=weights[X.index])  # should make this more general


# Perform 5-fold cross-validation 3 times to estimate the performance of the algorithm.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'lgbm_params= {\n    \'learning_rate\': hp.loguniform(\'learning_rate\', np.log(0.01), np.log(0.2)),\n    \'max_depth\': scope.int(hp.quniform(\'max_depth\', 5, 10, 1)),\n    \'n_estimators\': scope.int(hp.quniform(\'n_estimators\', 1, 300, 1)),\n    \'num_leaves\': scope.int(hp.quniform(\'num_leaves\', 2, 40, 1)),\n    #\'boosting_type\': hp.choice(\'boosting_type\', [\'gbdt\', \'dart\']),\n    \'colsample_bytree\': hp.uniform(\'colsample_bytree\', 0.3, 1),\n    \'reg_lambda\': hp.uniform(\'reg_lambda\', 0.5, 2.0),\n}\n\nxgb_params = {#\'gamma\':[0], \n              \'colsample_bytree\': hp.uniform(\'colsample_bytree\', 0.4, 1),\n              #\'min_child_weight\': scope.int(hp.quniform(\'min_child_weight\', 1, 1, 1)),\n              \'learning_rate\': hp.loguniform(\'learning_rate\', np.log(0.01), np.log(0.1)),\n               \'max_depth\': scope.int(hp.quniform(\'max_depth\', 5, 10, 1)),\n              \'subsample\': hp.uniform(\'subsample\', 0.6, 1),\n              \'n_estimators\': scope.int(hp.quniform(\'n_estimators\', 100, 300, 1)),\n              }\n\ncatboost_params = {\'learning_rate\': hp.loguniform(\'learning_rate\', np.log(0.01), np.log(0.2)), \n              \'iterations\':scope.int(hp.quniform(\'iterations\', 1, 3, 1)),\n              \'depth\':scope.int(hp.quniform(\'depth\', 5, 10, 1)),\n              \'l2_leaf_reg\': scope.int(hp.quniform(\'l2_leaf_reg\', 1, 10, 1))} # default 3\n\nnum_eval=50\nmcc_scorer = lambda model, X, y: make_scorer(matthews_corrcoef)(model, X, y, sample_weight=weights[X.index])\nacc_scorer = lambda model, X, y: make_scorer(accuracy_score)(model, X, y, sample_weight=weights[X.index])\n\nesti = BOestimator(model_class=xgb.XGBClassifier, #cat.CatBoostClassifier, #lgbm.LGBMClassifier, \n                   param_space=xgb_params, #catboost_params, #lgbm_params, \n                   #fixed_params={\'verbose\':0}, #for CatBoost\n                   num_eval=num_eval, \n                   groups=final_cluster_labels, \n                   scorer=mcc_scorer)\n\nstrat_wt_kfold1 = StratifiedWeightedGroupKFold(n_splits=5, \n                                              n_repeats=3, \n                                              shuffle=True, \n                                              random_states=seeds,\n                                              weights=weights)\n\ncv_results = []\nn_splits = strat_wt_kfold.n_splits\nn_iter = 4  # Number of iterations for permutation importance\nfor i, s in list(enumerate(strat_wt_kfold.split(X, Y, groups=final_cluster_labels))):\n    i_tr, i_te = s\n    X_tr, Y_tr, X_te, Y_te = X.iloc[i_tr], Y.iloc[i_tr], X.iloc[i_te], Y.iloc[i_te]\n    print("Step %s"%(i+1), "of %s"%n_splits)\n    esti.fit(X_tr, Y_tr, sample_weight=weights.iloc[i_tr], cv=strat_wt_kfold, seed=i)\n    train_score = esti.score(X_tr, Y_tr)\n    val_score = esti.score(X_te, Y_te)\n    \n    # Evaluations for permutation importance\n    scores = []\n    for j in range(n_iter):\n        col_shuffle_iterator = iter_shuffled(X_te.values, pre_shuffle=True, random_state=j)\n        for X_shuffled in col_shuffle_iterator:\n            scores.append(esti.score(pd.DataFrame(X_shuffled, index=X_te.index, columns=X.columns), Y_te))\n    score_decrease = val_score - np.array(scores).reshape(n_iter, X.columns.size)\n    \n    # Calculated probabilities\n    pred_proba = pd.DataFrame(esti.predict_proba(X_te), index=X_te.index)   # n x 2 array with sum of each row equal to 1\n    pred_proba.iloc[:,0] = Y_te\n    \n    cv_results.append((esti.trials_, \n                       esti.best_param_,\n                       train_score,\n                       val_score,\n                       acc_scorer(esti.clf_best_, X_te, Y_te),\n                       score_decrease, \n                       pred_proba\n                       ))\n    print("")')


# In[ ]:


trials, best_params, train_scores, val_scores, acc_scores, score_decreases, proba_list = zip(*cv_results)   # unzipping


# In[ ]:


results = pd.DataFrame(best_params)
results['Best cv scores'] = [-min([x['result']['loss'] for x in t.trials]) for t in trials]
results['Training scores'] = train_scores
results['Validation scores'] = val_scores
results['Validation accuracy'] = acc_scores
display(results)


# The scores on the validation sets are generally significantly lower than those on the training sets. However, on average they are comparable with the cross-validation scores produced during the training step, although they have a much wider spread.

# In[ ]:


def compare_scores(df, x, y):  
    scores = df[[x, y]]
    print(scores.describe().loc[['mean','std']].T)
    min_val = scores.values.min()
    max_val = scores.values.max()
     
    scores.plot.scatter(x, y, color='red')
    
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='lightgrey', 
             linestyle=(0, (5, 5)) #'dashed' #
            )

    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
    
compare_scores(results, 'Training scores', 'Validation scores')
compare_scores(results, 'Best cv scores', 'Validation scores')


# The XGBoost classifier predicts an outcome by first producing probabilities of being matte or glossy, and then choosing the surface type with the higher probability. Below, we plot histograms of these calculated probabilities for matte and for glossy glazes (actually, we only use the probability `P` of being glossy, but the probability of being matte is just `1 - P`).

# In[ ]:


proba_array = pd.concat(proba_list)
proba_array.columns = ['outcome', 'probability']
y = proba_array['outcome']
pa = proba_array['probability']
wt = weights[proba_array.index].values
def make_hist(bins, hide=False):
    hist = plt.hist([pa[y==0], pa[y==1]], 
                 bins=bins, 
                 weights=[wt[y==0], wt[y==1]],  
                 label=['Matte', 'Glossy'],
                 color=[(0.267004, 0.004874, 0.329415, 0.7), (0.993248, 0.906157, 0.143936, 1.0)],
                 edgecolor='black', linewidth=0.5)
    
    if hide:
        plt.clf()
        
    return hist
hist = make_hist(30)
plt.title('Weighted histogram of calculated probabilities')
plt.xlabel('P = Calculated probability of being glossy')
plt.ylabel('Weighted number of recipes')
plt.legend()
plt.show()


# In general, the probabilities produced by XGBoost are not true probabilities, in the sense that if one looks at all the data points with, say, a 90% chance of having a given classification, it is not necessarily the case that 90% of them actually have that classification. However, we can check whether the probabilities produced by XGBoost match up with the empirical ones by working out the proportion of glossy glazes in each bin of the histogram above.

# In[ ]:


probs = np.array(hist[0])
bins = hist[1]
bin_midpoints = (bins[1:] + bins[:-1])/2
plt.plot(bin_midpoints, probs[1]/probs.sum(axis=0), 'o')
ax = plt.gca()    # get current axes
plt.plot([0,1], [0,1], linestyle='dashed', color='orange')
ax.set_aspect('equal')
plt.xlabel('Calculated probability')
plt.ylabel('Empirical probability')
plt.title('Probability of being glossy')
plt.show()


# The empirical probabilities are fairly close to those produced by XGBoost, so we won't attempt to recalibrate them.
# 
# To simplify things, we'll split the predictions into 4 groups, depending on whether `P` is in the interval `[0, 0.25]`, `(0.25, 0.5]`, `(0.5, 0.75]`, or `(0.75, 1]`, and work out the proportions of glossy glazes in each group. This lets us work out the proportion of correct predictions in each group.

# In[ ]:


grouped_probs = proba_array.copy()
grouped_probs['bins'] = pd.cut(pa, bins=[0, 0.25, 0.5, 0.75, 1], labels=[1,2,3,4])
grouped_probs['weights'] = weights[proba_array.index]
glossy_proportions = grouped_probs.groupby('bins').apply(lambda x: np.average(x.outcome, weights=x.weights))
glossy_proportions = glossy_proportions.values  # convert to numpy array
bin_accuracies = np.concatenate([1-glossy_proportions[:2], glossy_proportions[2:]])
hist = make_hist(4, hide=True)
probs = np.array(hist[0])
bins = hist[1]
bin_midpoints = (bins[1:] + bins[:-1])/2

colours = [plt.cm.get_cmap("viridis")(c) for c in [0, 1/3, 2/3, 0.9999]]
plt.bar(bin_midpoints, bin_accuracies, width=0.24, color=colours)
plt.xticks([0, 0.25, 0.5, 0.75, 1])
plt.yticks(np.linspace(0,1,11))

plt.xlabel('P = Calculated probability of being glossy')
plt.ylabel('Proportion of correct predictions')
plt.title('Average prediction accuracy for probability bins')
plt.ylim(0,1.1)

label_dict = {0:'0 \u2264 P \u2264 0.25 (Predicted matte)',
              1:'0.25 < P \u2264 0.5 (Predicted matte)',
              2:'0.5 < P \u2264 0.75 (Predicted glossy)', 
              3:'0.75 < P \u2264 1 (Predicted glossy)'}
patches = [ mpatches.Patch(color=colours[i], label=label_dict[i] ) for i in range(4) ]
 # put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
ax = plt.gca()
add_value_labels(ax, dec_pt=3)
    
plt.show()


# It should be emphasised that the accuracies above are for a fixed set of glazes. While they may be rough estimates for the accuracies of the model on collections of glazes with a similar distribution of oxides and firing cones, this is *not* true in general.

# ### Feature importance

# We can get an idea of the importance of a feature by working out how much the measure of accuracy drops when the input data is corrupted by randomly permuting the values of that feature. A large drop in accuracy is intuitively a sign that the feature is very important, so the relative drops in accuracy are used as measures of the importance of features. Since there is randomness involved, the importance of a feature is a distribution of values, rather than a single number. Below, we print a list of means and standard deviations, and show plot box and whisker plots.

# In[ ]:


sd = np.array(score_decreases).reshape(-1, X.columns.size)
pi_df_summary = pd.DataFrame({'Mean' : sd.mean(axis=0), 'Std' : sd.std(axis=0)}, index=X.columns)
pi_df_summary.sort_values(by='Mean', ascending=False, inplace=True)
print(pi_df_summary)
pi_df = pd.DataFrame(sd, columns=X.columns)
pi_df = pi_df[pi_df_summary.index.values[::-1]]
boxplot = pi_df.boxplot(rot=0, figsize=(8,6), vert=False)
boxplot.set_title("Cross-validation feature importance")
plt.show()


# The most important feature is `SiO2_Al2O3_ratio_umf`, followed by `B2O3_umf`, which is not too surprising. These are by far the most important. Next in line are `SiO2_umf` and `MgO_umf`, followed by `from_orton_cone` and `to_orton_cone`. Again, this seems reasonable. Surprisingly, `P2O5_umf` is ranked relatively high up, considering that for most glazes it only occurs as an impurity. On the other hand, `TiO2_umf`, which one might expect to be more important, is ranked fairly low. However, note that this measure of importance depends on both the set of glazes, and the *model* used to make predictions. So if a feature is deemed unimportant, it just means that the model doesn't make much use of it. It's possible that a better model utilises this feature more effectively. 
# 
# As a side note, since `SiO2_umf`, `Al2O3_umf` and `SiO2_Al2O3_ratio_umf` are related, it's possible to construct a model that gives the same predictions for all glazes as our model does, but where the permutation importance of `SiO2_Al2O3_ratio_umf` is zero. (Just add a preprocessing step where the given value of `SiO2_Al2O3_ratio_umf` is replaced by `SiO2_umf` / `Al2O3_umf`)

# In[ ]:


def show_matte_glossy_hist(feature, hist_range, n_bins=25):
    values = X[feature]
    wt = weights[X.index][values.index]
    plt.hist([values[Y==0], values[Y==1]], 
             bins=n_bins, 
             weights=[wt[Y==0], wt[Y==1]], 
             range=hist_range, 
             label=['Matte', 'Glossy'],
             color=[(0.267004, 0.004874, 0.329415, 0.7), (0.993248, 0.906157, 0.143936, 1.0)],
             edgecolor='black', linewidth=0.5)
    plt.title('Weighted histogram of '+feature+' values')
    plt.legend()
    plt.show()


# Given its importance, let's look at the histograms of the silica : alumina ratio for glossy and matte glazes:

# In[ ]:


show_matte_glossy_hist('SiO2_Al2O3_ratio_umf', [0, 20], n_bins=20)


# The matte glazes clearly peak at a smaller silica : alumina ratio than the glossy ones. 

# In[ ]:


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Simple classifier that classifies a data point based on whether a particular feature 
    is above or below a threshold value. There is no training involved, and only one 
    hyperparameter, the threshold (if we regard the feature as fixed, otherwise this is also 
    a hyperparameter)"""
    
    def __init__(self, feature=0, threshold=0.5):
        
        self.feature = feature
        self.threshold = threshold

    def fit(self, X, y=None, sample_weight=None, cv=5, feature=0):
        
        return self
    
    def predict(self, X):
    
        return (X[self.feature] > self.threshold).astype('int')
    
    def score(self, X, y):
        
        return make_scorer(matthews_corrcoef)(self, X, y, sample_weight=weights[X.index])  # should make this more gener

threshold_params = {'threshold': hp.uniform('threshold', 5, 9)}

besti = BOestimator(ThresholdClassifier, 
                   threshold_params,
                   {'feature': 'SiO2_Al2O3_ratio_umf'},
                   num_eval=50, 
                   groups=final_cluster_labels[X.index], 
                   scorer=mcc_scorer)

cv_results = []
n_splits = strat_wt_kfold.n_splits
for i, s in list(enumerate(strat_wt_kfold.split(X, Y, groups=final_cluster_labels))):
    i_tr, i_te = s
    X_tr, Y_tr, X_te, Y_te = X.iloc[i_tr], Y.iloc[i_tr], X.iloc[i_te], Y.iloc[i_te]
    print("Step %s"%(i+1), "of %s"%n_splits)
    besti.fit(X_tr, Y_tr, sample_weight=weights.iloc[i_tr], cv=strat_wt_kfold, seed=i)
    #print(besti.best_param_, [t['misc']['vals']['threshold'] for t in besti.trials_])
    train_score = besti.score(X_tr, Y_tr)
    val_score = besti.score(X_te, Y_te)
    cv_results.append((train_score, val_score, besti.best_param_['threshold']))
    


# If you're interested, click the `output` button to see the results of training a model that uses only `SiO2_Al2O3_ratio_umf` and a threshold to decide whether a glaze is matte or glossy. The average validation MCC of 0.339 is lower than the average of 0.454 for the model we use above, but is pretty good for such a simple model.

# In[ ]:


train_scores, val_scores, thresholds = zip(*cv_results)
results = pd.DataFrame({'Training scores': train_scores, 'Validation scores': val_scores, 'Thresholds':thresholds})
compare_scores(results, 'Training scores', 'Validation scores')
print("Average cross-validation threshold:", np.average(thresholds))
#boxplot = plt.boxplot(thresholds, vert=False)
#plt.show()


# For boron, the histograms are as follows:

# In[ ]:


show_matte_glossy_hist('B2O3_umf',[0, 0.8], n_bins=24)
#show_matte_glossy_hist('MgO_umf',[0, 0.8])
#show_matte_glossy_hist('P2O5_umf',[0.005, 0.5])


# Even though glossy glazes outnumber matte glazes by more than 30%, there are more matte glazes than glossy glazes among glazes with at most 0.1 B<sub>2</sub>O<sub>3</sub>.

# ## Predictions

# Train the models on all the recipes where the surface type has been given.

# In[ ]:


#xgb_model.fit(X, Y, groups=final_cluster_labels, sample_weight=weights)
#lgbm_model.fit(X, Y, sample_weight=weights)
esti.fit(X, Y, sample_weight=weights, cv=strat_wt_kfold1, seed=42)
print(esti.best_param_)


# In[ ]:


def plot_predicted_biaxial(model, x_dict, y_dict, base_dict, aspect=1):
    base =  pd.Series(0., index = X.columns)
    for k in set(x_dict.keys()).union(set(y_dict.keys())).intersection(set(base_dict.keys())):
        del base_dict[k]
    for k, v in base_dict.items():
        base[k]= v
        if v!=0:
            print(k+':', round(v,3))
    
    x_vals = list(x_dict.values())
    y_vals = list(y_dict.values())
    n_x = x_vals[0].shape[0]
    n_y = y_vals[0].shape[0]
    arr = np.tile(base.values[:, np.newaxis], n_x*n_y)
    biax = pd.DataFrame(arr.T, columns = base.index)

    #y_vals = np.linspace(1, 0, num) #np.array([-6,-5.5,-5,-4,-3,-2,-1,1,2,3,4,5,5.5,6,7,8,9,10])[::-1]

    for k, v in x_dict.items():
        biax[k] = np.tile(v, n_y)
    for k, v in y_dict.items():
        biax[k] = np.repeat(v[::-1], n_x)

    biax['SiO2_Al2O3_ratio_umf'] = biax['SiO2_umf'] / biax['Al2O3_umf']
    biax['R2O_umf'] = biax['KNaO_umf'] + biax['Li2O_umf']

    try:
        stull_pred = model.predict_proba(biax)[:,1] #model.predict(biax) #
    except:
        stull_pred = np.array([mod.predict(biax) for mod in model]).mean(axis=0)
    stull_pred = (np.ceil(stull_pred*4) - 1)/3.0

    fig, ax = plt.subplots()
    a, b = x_vals[0][0], x_vals[0][-1]
    c, d = y_vals[0][0], y_vals[0][-1]
    im = plt.imshow(stull_pred.reshape(n_y, n_x), 
                    extent=[a, b, c, d], 
                    norm=mpl_colour.Normalize(vmin=0, vmax=1))
    ax.set_xlabel(list(x_dict.keys())[0])
    ax.set_ylabel(list(y_dict.keys())[0])
    ax.set_aspect(aspect*(a-b)/(c-d))
    
    # Next section modified from https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
    # get the colours of the values, according to the colormap used by imshow
    colours = [im.cmap(value/3.) for value in [0, 1, 2, 3]]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colours[i], label={0:'Matte',1:'Matte (?)', 2:'Glossy (?)', 3:'Glossy'}[i] ) for i in [0, 1, 2, 3] ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    plt.show()
    
models = esti #cat_model.best_estimator_  #[lgbm_model, cat_model, xgb_model]


# We can use the model to predict regions of matte and glossy glazes in biaxial grids. The model produces probabilities of being matte or glossy, and these are shown as follows
# * purple: 75% - 100% probability of being matte
# * blue: 50% - 75% probability of being matte
# * green: 50% - 75% probability of being glossy
# * yellow: 75% - 100% probability of being glossy
# 
# However, these probabilities are for glazes drawn from the same distribution as the glazes used to train the model, and since the collections of glazes represented below are not representative of Glazy glazes, we can't expect that these probabilities hold for them. Nevertheless, for regions where Glazy recipes are well-represented, this gives a rough idea of how confident we can be in the predictions.
# 
# We'll begin by plotting a silica-alumina biaxial, using the same parameters as the [Stull chart](https://wiki.glazy.org/t/influences-of-variable-silica-and-alumina-on-porcelain-glazes/766). Oxide proportions are expressed in terms of the UMF, so for example 0.7 CaO means the number of CaO molecules in the glaze is 0.7 of the total number of flux molecules.

# In[ ]:


num = 200

base = {}
base['from_orton_cone'] = 11
base['to_orton_cone'] = 11
base['SiO2_umf'] = 3
base['Al2O3_umf'] = 0.5
base['KNaO_umf'] = 0.3
base['CaO_umf'] = 0.7

plot_predicted_biaxial(models,
                       {'SiO2_umf': np.linspace(0, 7, num)}, 
                       {'Al2O3_umf': np.linspace(0.01, 1, num)}, 
                       base)

"""base['SiO2_umf'] = 2.8
base['Al2O3_umf'] = 0.4
cones = np.linspace(-4, 10, num) #np.array([-6,-5.5,-5,-4,-3,-2,-1,1,2,3,4,5,5.5,6,7,8,9,10])
plot_predicted_biaxial(models,
                       {'B2O3_umf': np.linspace(0, 0.5, num)}, 
                       {'from_orton_cone':cones, 'to_orton_cone': cones}, 
                       base)"""
text = "This is just here so the commented out section above doesn't get printed"


# The boundary between the matte and glossy regions is very close to the SiO<sub>2</sub> : Al<sub>2</sub>O<sub>3</sub> = 6 line, unlike the Stull chart, where the boundary is closer to the SiO<sub>2</sub> : Al<sub>2</sub>O<sub>3</sub> = 5 line. Note, however, that Glazy contains relatively few cone 11 recipes, so the predictions probably reflect the state of glazes at cones 9 and 10.
# 
# Glazes in the lower right-hand corner will be underfired, hence matte. The reason none of the models predict this is presumably because there are no such glazes in Glazy. In general, since Glazy contains no recipes in the top left or bottom right corners of the SiO<sub>2</sub> - Al<sub>2</sub>O<sub>3</sub> chart, predictions for these regions should not be relied upon.
# 
# Next, we'll see what the model predicts when we fire at cone 6 instead of 11.

# In[ ]:


base['from_orton_cone'] = 6
base['to_orton_cone'] = 6

plot_predicted_biaxial(models, 
                       {'SiO2_umf': np.linspace(0, 6, num)}, 
                       {'Al2O3_umf': np.linspace(0.01, 1, num)}, 
                       base)


# The glossy region has shrunk, presumably because some glazes that fired glossy at cone 11 are underfired mattes at cone 6.
# 
# Now let's drop to cone 1:

# In[ ]:


base['from_orton_cone'] = 1
base['to_orton_cone'] = 1

plot_predicted_biaxial(models, 
                       {'SiO2_umf': np.linspace(0, 6, num)}, 
                       {'Al2O3_umf': np.linspace(0.01, 1, num)}, 
                       base)


# The model doesn't predict glossy glazes with any great confidence. 
# 
# What happens if we add boron? We'll add 0.1, 0.2, and 0.3 B<sub>2</sub>O<sub>3</sub>.

# In[ ]:


base['from_orton_cone'] = 1
base['to_orton_cone'] = 1
base['B2O3_umf'] = 0.2
for boron in [0.1, 0.2, 0.3]:
    base['B2O3_umf'] = boron
    plot_predicted_biaxial(models, 
                       {'SiO2_umf': np.linspace(0, 6, num)}, 
                       {'Al2O3_umf': np.linspace(0.01, 1, num)}, 
                       base)


# The glossy (or potentially glossy) region grows steadily as boron is added. While this is expected, Glazy doesn't contain many cone 1 glazes, so these predictions should regarded with some caution.
# 
# However, a similar result can be seen at cone 6, which is the most common firing cone among Glazy glazes.

# In[ ]:


base['from_orton_cone'] = 6
base['to_orton_cone'] = 6
for boron in [0.1, 0.2, 0.3]:
    base['B2O3_umf'] = boron
    plot_predicted_biaxial(models, 
                       {'SiO2_umf': np.linspace(0, 6, num)}, 
                       {'Al2O3_umf': np.linspace(0.01, 1, num)}, 
                       base)


# So far, we've just looked at glazes with 0.7 CaO and 0.3 KNaO. In the next three charts, we'll go back to cone 11, with no boron, and replace some CaO with progressively more MgO, in steps of 0.1.

# In[ ]:


base['from_orton_cone'] = 11
base['to_orton_cone'] = 11
base['B2O3_umf'] = 0.
for magnesium in [0.1, 0.2, 0.3]: #[0.15, 0.3, 0.45, 0.6]:
    base['MgO_umf'] = magnesium
    base['CaO_umf'] = 0.7 - magnesium
    plot_predicted_biaxial(models, 
                       {'SiO2_umf': np.linspace(0, 6, num)}, 
                       {'Al2O3_umf': np.linspace(0.01, 1, num)}, 
                       base)


# The chart with 0.1 MgO is similar to the Stull chart, except that the boundary between matte and glossy glazes is now roughly SiO<sub>2</sub> : Al<sub>2</sub>O<sub>3</sub> = 7. Further increases in MgO at the expense of CaO lead to a reduction in the extent of the glossy region.
# 
# Next, we'll examine the effect of varying the proportions of CaO and KNaO. We'll plot a series of silica - alumina biaxials at cone 11, where the fluxes range from CaO : KNaO = 1 : 0 to CaO : KNaO = 0 : 1, in increments of 0.1.

# In[ ]:


base['MgO_umf'] = 0
for KNaO in np.linspace(0,1,11): #[0.15, 0.3, 0.45, 0.6]:
    base['KNaO_umf'] = KNaO
    base['CaO_umf'] = 1 - KNaO
    plot_predicted_biaxial(models, 
                       {'SiO2_umf': np.linspace(0, 6, num)}, 
                       {'Al2O3_umf': np.linspace(0.01, 1, num)}, 
                       base)


# Be aware that the joint distribution of CaO and KNaO is highly non-uniform. In particular, predictions for KNaO values above 0.5 are based on relatively few data points, so should be used with caution.

# In[ ]:


fig, ax = plt.subplots(figsize=(6,6))
#fig.set_figsize((10,10))
plt.hexbin(X['KNaO_umf'], X['CaO_umf'], C=weights, reduce_C_function=np.sum, gridsize=(20,20), 
           cmap=plt.get_cmap('Greys'))
ax.set_aspect(1.)
ax.set_xlabel('KNaO')
ax.set_ylabel('CaO')
ax.set_title('Density')
plt.show()


# We can also plot predictions for charts where oxides other than SiO<sub>2</sub> and Al<sub>2</sub>O<sub>3</sub> vary. For example, in the chart below, we fix SiO<sub>2</sub> at 3, and vary Al<sub>2</sub>O<sub>3</sub> and B<sub>2</sub>O<sub>3</sub>.

# In[ ]:


base['SiO2_umf'] = 3
base['CaO_umf'] = 0.7
base['KNaO_umf'] = 0.3
plot_predicted_biaxial(models, 
                   {'B2O3_umf': np.linspace(0, 0.4, num)}, 
                   {'Al2O3_umf': np.linspace(0.01, 1, num)}, 
                   base)


# As B<sub>2</sub>O<sub>3</sub> increases from 0 to 0.3, there's an increase in the upper Al<sub>2</sub>O<sub>3</sub> bound for glossy glazes, from about 0.5 to 0.6. Since SiO<sub>2</sub> is fixed at 3, this corresponds to a decrease in the SiO<sub>2</sub> : Al<sub>2</sub>O<sub>3</sub> ratio from 6 to 5.
# 
# It would be interesting to see whether this prediction can be confirmed experimentally, and whether the upper Al<sub>2</sub>O<sub>3</sub> bound for glossy glazes continues to increase as more B<sub>2</sub>O<sub>3</sub> is added.

# In[ ]:




