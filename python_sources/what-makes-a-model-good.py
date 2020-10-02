#!/usr/bin/env python
# coding: utf-8

# Industry applications for Machine Learning aim to balance objectives in model compression, inference speed, model bias and variance and model stability across random initialization and updates as part of their design considerations.  While traditional methods in hyperparamter optimization focus on optimally sampling from the universe of model architectures and hyperparameters to improve only a single metric for model performance, multi-objective methods in hyper-parameter optimization have been used select models which are optimal across one or more criteria.  In this notebook I aim to investigate the applications of Data Envelopment Analysis (DEA) as method for multi-objective hyper-parameter search and model selection in constrained environments. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from time import time
from typing import List, Tuple, Union
from functools import partial

from scipy.optimize import linprog
import scipy.stats as stats
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import holoviews as hv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.hv.extension('bokeh')
hv.extension('bokeh')


# 
# Applications of Machine Learning at the Edge face many design constraints around the size, speed and accuracy of Machine Learning pipelines.  In order to handle such constraints, practitioners have looked to two primary areas of study: model compression and acceleration, which aims to shrink existing models to fit on device, and multi-objective optimization, which aim to optimally search for new efficient models.  
# 
# While these two approaches can and have been used in conjunction with one-another, neither method addresses the problem of scoring and selecting from within the large space of models which lie of the Pareto-optimal front. As such, these methods rely on domain expertise in evaluating these trade-off and provide limited signal to their underlying optimization procedures on the efficiency of particular models compared to their peers.  
# 
# Data Envelopment Analysis (DEA) is a non-parametric method in Operations Research for estimating production possibility frontiers. Inside of finance and quantitative management, DEA is typically used as a method by which to access the productive efficiency of fund managers or business units. Using DEA, one aims to assess a decision making unit (DMU) against their peers based an their optimal weighting of inputs, such as their allocated funds and number of staff, and outputs, such as their performance against a benchmark and fund value-at-risk (VAR).  In the equations below, we show the dual formulation of DEA which provides a means by which to compute an efficiency
# measure, $E$, to score DMU's.  This score represents the largest ratio of inputs used by a weighted sum of non-reference DMU's to the inputs of a reference DMU at some minimum level of output.  
# 
# $$
# \begin{array}{ll@{}ll}
# \text{Minimise}  & \displaystyle E  &\\
# \text{subject to}& \displaystyle E x_{i, 0} -& \sum\limits_{k=1}^{n}   \lambda_{k}x_{i, k} \geq 0  & \forall i \in 1 ,..., m \\
# \text{}  &
# \displaystyle &\sum\limits_{i=1}^{m}   \lambda_{k}y_{r, k}  \geq y_{r, 0}  &\forall r \in 1 ,..., s\\
#                  &                                                & \lambda_{k} \geq 0
# \end{array}
# $$
# 
# To date, Data  Envelopment  Analysis  (DEA) has seen little study in it's application to model scoring, selection and optimization.  By considering model size, memory usage, training and inference time as inputs and measures like Accuracy, Precision and Recall as outputs, one can devise a approach using DEA by which to score models' efficiency based on their performance across criteria and resource utilization. 

# In[ ]:


def _unit_shadow_prices(
    model_metrics: pd.Series, peer_metrics: pd.DataFrame, greater_is_better: List[bool], compute_primal: bool = False
) -> np.ndarray:
    peer_metrics = (peer_metrics.where(lambda x: x.ne(model_metrics, 0), np.nan)
    .dropna())

    greater_is_better_weight = np.where(greater_is_better, 1, -1)
    inputs_outputs = greater_is_better_weight * np.ones_like(peer_metrics)

    # outputs - inputs
    A_ub = inputs_outputs * peer_metrics
    b_ub = np.zeros(A_ub.shape[0])

    # \sum chosen model inputs = 1
    A_eq = np.where(greater_is_better_weight < 0.0, model_metrics, 0).reshape(1, -1)
    b_eq = np.array(1.0).reshape(1, -1)

    # max outputs == min -outputs
    c = np.where(greater_is_better_weight >= 0.0, model_metrics, 0.0).reshape(1, -1)

    # compute dual
    dual_A_ub = np.vstack((A_ub, A_eq)).T
    dual_c = np.hstack((b_ub, b_eq.reshape(-1,))).T
    dual_b_ub = c.T

    dual_result = linprog(
        dual_c,
        A_ub=-dual_A_ub,
        b_ub=-dual_b_ub,
        bounds=[(0, None) for _ in range(dual_A_ub.shape[1] -1 )] + [(None, None)],
    )

    return dual_result.fun


def data_envelopment_analysis(
    validation_metrics: Union[pd.DataFrame, np.ndarray], greater_is_better: List = []
) -> pd.DataFrame:
    """
    :param validation_metrics: Metrics produced by __SearchCV
    :param greater_is_better: Whether that metric are to be considered inputs to decrease or outputs to increase
    :return: Shadow prices for comparing a model to is peers & Hypothetical Comparison Units to compare units
    """
    partialed_unit_shadow_scores = partial(
        _unit_shadow_prices,
        peer_metrics=validation_metrics,
        greater_is_better=greater_is_better,
    )
    efficiency_scores = pd.DataFrame(validation_metrics).apply(
        partialed_unit_shadow_scores, axis=1
    )

    return efficiency_scores


# # Experiment
# In this notebook we have taken two approaches to investigating model performance using the Efficiency Scores of Data Envelopment Analysis.  The first approach uses the results from a randomized search of hyper-parameters for a shallow neural network classifier trained using Stochastic Gradient Descent (SGD) on the Breast Cancer Wisconsin (Diagnostic) Data Set, and the second  approach investigates a public dataset of ImageNet model metrics provided by the [Papers with Code](https://paperswithcode.com/sota/image-classification-on-imagenet). 

# ## Neural Network Classifier
# For our neural network model, we have opted to perform DEA on the results of random search accross 100 models. Accross these models we have explored neural networks of varying depth, regularization and learning rate.  

# In[ ]:


# get some data
X, y = load_breast_cancer(return_X_y=True)
# build a classifier
clf = MLPClassifier()
# specify parameters and distributions to sample from
param_dist = {
    "hidden_layer_sizes": [(10, 5, ), (10, ), (10, 5, 3, )],
    "learning_rate_init": stats.uniform(0, 1),
    "alpha": stats.uniform(1e-4, 1e0),
}
# run randomized search
n_iter_search = 100
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score),  'F1': make_scorer(f1_score)}
random_search = RandomizedSearchCV(
    clf, param_distributions=param_dist, 
    scoring = scoring,
    n_iter=n_iter_search, 
    n_jobs=-1,
    refit = 'AUC',
    return_train_score=True
)
random_search.fit(X, y)
cv_results_df = pd.DataFrame(random_search.cv_results_)


# %%
metrics = [
    "mean_fit_time",
    "mean_score_time",
    "mean_test_Accuracy",
    "mean_test_AUC",
    "mean_test_F1",
]
metrics_greater_is_better = [False, False, True, True, True]
efficiency_scores = data_envelopment_analysis(
    validation_metrics=cv_results_df[metrics],
    greater_is_better=metrics_greater_is_better,
)


# In the plot below, we can easily see the pareto front which emergences for all paired metrics. In red, I have flagged the models which DEA has scores as beng in the top 25% of models with the greatest efficiency. Using DEA, have have attempted to find models which for a given budget of compute provide the best AUC,F1 and Acurracy Scores. 

# In[ ]:


table = hv.Dataset(cv_results_df.loc[:, metrics])
matrix = hv.operation.gridmatrix(table)

top = hv.Dataset(cv_results_df.loc[:, metrics]
                   .assign(efficiency_scores=efficiency_scores)
                   .where(lambda df: df.efficiency_scores >= df.efficiency_scores.quantile(0.75))
                   .drop(columns=['efficiency_scores']))
best = hv.operation.gridmatrix(top)
(matrix * best).opts(title='Top 25% in Red', width=800, height=600)


# As we can see from our results, our best models appear to be our shallow models with high leels of regularization and low learning rates. This is to be expected for such a dataset and provides confidence in our approach taken. 

# In[ ]:


(cv_results_df.loc[:, ['param_alpha', 'param_hidden_layer_sizes', 'param_learning_rate_init'] + metrics]
 .assign(efficiency_scores=efficiency_scores)
 .nlargest(3, 'efficiency_scores'))


# Our worst models tend to be larger models with higher learning rates and lower levels of regularization. 

# In[ ]:


(cv_results_df.loc[:, ['param_alpha', 'param_hidden_layer_sizes', 'param_learning_rate_init'] + metrics]
 .assign(efficiency_scores=efficiency_scores)
 .nsmallest(3, 'efficiency_scores'))


# # Papers with Code

# While simulated data is valuable in any study, I thought it interesting to investigate the use of real-world data for our analysis. Here I used a dateset from the platform Papers with Code, which details the scores for state-of-the-art models on the ImageNet image classification problem through time. Here we will be comparing the efficiency of models based on their inpus of NUMBER OF PARAMS (in millions) to their predictive accuracy metrics, TOP 1 ACCURACY and TOP 5 ACCURACY. 

# In[ ]:


columns = ['RANK', 'METHOD', 'TOP 1 ACCURACY', 'TOP 5 ACCURACY', 'NUMBER OF PARAMS', 'PAPER TITLE', 'YEAR']
imagenet = pd.read_csv('/kaggle/input/papers-with-code-imagenet-rankings/efficiency_results (1).csv', usecols=columns)
imagenet.loc[:, 'NUMBER OF PARAMS'] = imagenet.loc[:, 'NUMBER OF PARAMS'].apply(lambda s: float(s[:-1]))
imagenet.loc[:, 'TOP 5 ACCURACY'] = imagenet.loc[:, 'TOP 5 ACCURACY'].apply(lambda s: float(s[:-1])/100).astype(float)
imagenet.loc[:, 'TOP 1 ACCURACY'] = imagenet.loc[:, 'TOP 1 ACCURACY'].apply(lambda s: float('.'.join(s[:-1].split(',')))/100).astype(float)

imagenet


# I again opted to mark out top 25% best models, based on efficiency scores, in red. What is interesting to note is now much model architecture plays into efficiency. This is definately an artefact of how DEA contracts input and outputs, but it worth noting. 

# In[ ]:


imagenet_metrics = ['NUMBER OF PARAMS', 'TOP 1 ACCURACY','TOP 5 ACCURACY']
imagenet_metrics_greater_is_better = [False, True, True]
                    
imagenet_efficiency_scores = data_envelopment_analysis(
    validation_metrics=imagenet.loc[:, imagenet_metrics],
    greater_is_better=imagenet_metrics_greater_is_better,
)

table = hv.Dataset(imagenet.loc[:, imagenet_metrics])
matrix = hv.operation.gridmatrix(table)

top = hv.Dataset(imagenet.loc[:, imagenet_metrics]
                   .assign(efficiency_scores=imagenet_efficiency_scores)
                   .where(lambda df: df.efficiency_scores >= df.efficiency_scores.quantile(0.75))
                   .dropna()
                   .drop(columns=['efficiency_scores']))
best = hv.operation.gridmatrix(top)
(matrix * best).opts(title='Top 25% in Red', width=800, height=600)


# Unsurprisingly our most efficient models were many of our models designed for the edge.  Here, these models demonstrate high levels of Accurancy for their given number of parameters. 

# In[ ]:


(imagenet
 .assign(efficiency_scores=imagenet_efficiency_scores)
 .nlargest(3, 'efficiency_scores'))


# The worst performers were our larger models with some having over 800 million parameters. 

# In[ ]:


(imagenet
 .assign(efficiency_scores=imagenet_efficiency_scores)
 .nsmallest(3, 'efficiency_scores'))


# # Conclusion
# This is an idea I have been playing around with with a while now. I initially wrote a blog post on this idea about two years ago after taking an advanced course in Operations Research. I explored the idea again about 9 months ago when I did some benchmarking for my [super_spirals github project](https://github.com/marcusinthesky/super-spirals) which I collaberated with one of my classmates on and worked on an early proposal around this concept. I am still a little skeptical about the approach myself, but really like the idea and believe despite some reaching assumptions it may make a great heuristic for model selection in some settings. 
