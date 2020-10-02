#!/usr/bin/env python
# coding: utf-8

# # Intro
# - In my work on trying to classify forest cover so far, I've found that K-nearest neighbors (KNN) performs the best with little feature engineering: not even scaling. Why is it so good?
# - KNN is highly non-linear, even non-monotonic, and obviously exploits clustering in the data that is related to forest cover type. But it's basically impossible to interpret parsimoniously, especially in the case of a 1-nearest neighbor predictor. 
# - This has me wondering if I can use [Neighborhood Components Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html#sklearn.neighbors.NeighborhoodComponentsAnalysis) (NCA)  to help me 
#     1. understand why KNN has performed so well 
#     2. do some feature engineering so that I can get a better-performing interpretable model: e.g. better understand nonlinearity and scaling in the data
#     3. identify clusters and/or outliers in the data that I can interpret
# - I think that this question is relevant because I don't think that there would be much use developing an uninterpretable model to predict forest cover type in a < 1 hectare patch
#     + Given that we require soil samples and survey measurements to make our predictions, the marginal cost of asking the person who is gathering those to classify the forest type seems minimal
#     + On the other hand, supposing this is some kind of remote-operated or autonomous vehicle that has a good soil lab built in, our efforts here might be valuable
#     + Even then, we might like to have some idea of what's going on so we could troubleshoot
#     + Some reasons we might want to predict forest cover type: allocation of conservation budget, number of hunting/fishing licenses for a particular species, number of camping/mushroom gathering/mining/construction permits, number of fire lookouts/rangers, among others.
# - I also want to learn more about KNN and especially about NCA, which I've just found out about.
# - Sorry about the language in this kernel: I tend to be a little bit stilted in the way its used, which implies a lot more certainty than I would like. _Everything_ in here is questionable.

# # Assumptions
# - NCA is a linear transformation of the data, so: 
#     - The NCA process can be thought of as looking at each pair of features at a time, then rotating the plane and squishing/stretching the axes for each NCA dimension
# 

# # Hypotheses 
# 1. The value of a feature's coefficients across all NCA dimensions reflects this process
#     - if they are uniformly 0, then the feature does not influence the accuracy of a KNN-based classifier much
#     - if they are positive and much larger than the feature's standard deviation, then the more influential they are with respect the KNN predictor fit
#     - if they are negative and much larger than the feature's standard deviation, then patches of forest with similar values of the feature are _less_ likely to have the same cover and the association is likely spurious (this may be an outlier flag)
# 2. If a feature is influential on the KNN fit due to its large scale, it will have large, positive NCA coefficients in a $d$-dimensional unscaled fit but a pattern of one or two positive NCA coefficients, but mostly coefficients of alternating signs in a scaled fit.
#     - NCA coefficients are informative about whether to scale the data before modeling.
# 3. When the $K$ used in the nearest neighbor fit far exceeds that implied by the NCA transform (see paper linked in documentation) then there will be overfitting, but not likely severe overfitting. 
#     - Using the smallest $d$ NCA model with adequate performance in the validation set will lead to the best model in the test set

# In[ ]:


import numpy as np # linear algebra
from scipy.stats import pearsonr
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import union_categoricals
from matplotlib import pyplot as plt 
import seaborn as sns
from pprint import pprint
from time import time

from os import listdir
from os import path

## Much code copied from the Faces example taken from scikit 
## (https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html)
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
print(__doc__)
## Display progress logs on stdout
logging.basicConfig(level = logging.INFO,
                   format = '%(asctime)s %(message)s')
random_state = 3412


# # Design of the comparison
# 1. Split training data into training/validation sets
# 2. Use NCA to reduce the input data to $d  = \left\{1,\, 2,\, 5\right\}$ dimensions. 
# 3. For each value of $d$: 
#     - Fit predictors based on $K = \left\{1,\, 3,\, 5,\, 9,\, 17,\,\right\}$-nearest neighbors.
#     - Fit a logistic regression classifier with multinomial loss function
# 4. Compare these $3 + 3\times 5 = 18$ predictive models on the validation set using weighted average of f1-score across cover types

# In[ ]:


train_data = pd.read_csv(path.join("..", "input", "learn-together", "train.csv"))
test_data = pd.read_csv(path.join("..", "input", "learn-together", "test.csv"))


# In[ ]:


X = train_data.drop(['Cover_Type', 'Id'], axis = 1)
X_test = test_data.drop(['Id'], axis = 1)

y = train_data['Cover_Type']
test_Id = test_data.Id

target_names = np.array(['Spruce/Fir',
                             'Lodgepole Pine',
                             'Ponderosa Pine',
                             'Cottonwood/Willow',
                             'Aspen',
                             'Douglas-fir',
                             'Krummholz'])


# In[ ]:


# Quantitative evaluation of the predictions using matplotlib
d = [2,4]
K = [1,5]

knn_pipeline = Pipeline([
    ('scale', StandardScaler(with_mean = True)),
    ('nca', NeighborhoodComponentsAnalysis(max_iter = 50, random_state = random_state)),
    ('clf', KNeighborsClassifier())
])
    
knn_params = {
    'scale__with_std': (True, False),
    'nca__n_components': d,
    'clf__n_neighbors': K
}

knn_grid = GridSearchCV(knn_pipeline, knn_params, cv=5,
                               n_jobs=1, verbose=1, scoring = 'f1_weighted', return_train_score = True)

print("Performing grid search...")
print("pipeline:", [name for name, _ in knn_pipeline.steps])
print("parameters:")
pprint(knn_params)
t0 = time()
knn_grid.fit(X, y)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % knn_grid.best_score_)
print("Best parameters set:")
best_parameters = knn_grid.best_estimator_.get_params()
for param_name in sorted(knn_params.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[ ]:


logit_pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('nca', NeighborhoodComponentsAnalysis(max_iter = 50, random_state = random_state)),
    ('clf', LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs' ))
])
    
logit_params = {
    'scale__with_mean': (True, False),
    'scale__with_std': (True, False),
    'nca__n_components': d,
    'clf__n_neighbors': K
}

logit_grid = GridSearchCV(logit_pipeline, logit_params, cv=5,
                               n_jobs=1, verbose=1, scoring = 'f1_weighted', return_train_score = True)

print("Performing grid search...")
print("pipeline:", [name for name, _ in logit_pipeline.steps])
print("parameters:")
pprint(logit_params)
t0 = time()
logit_grid.fit(X, y)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % logit_grid.best_score_)
print("Best parameters set:")
best_logit_parameters = logit_grid.best_estimator_.get_params()
for param_name in sorted(logit_params.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

