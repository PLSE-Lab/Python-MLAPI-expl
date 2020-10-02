#!/usr/bin/env python
# coding: utf-8

# # Bayesian Hyperparameters Tuning

# The goal of this notebook is to present a simple implementation of bayesian hyperparameters tuning using the [hyperopt](https://github.com/hyperopt/hyperopt) library. Additionally, we compare bayesian tuning with grid search and random search optimization approaches in terms of performance and precision. The comparison is performed for the following estimators:
# 
# * Logistic Regression;
# * Decision Tree Classifier;
# * Suporting Vector Classifier.

# # Import Libraries

# In[ ]:


#Basic imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

#Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

#General Sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

#Hyperopt imports
import hyperopt as hp
from hyperopt import fmin, tpe, Trials, hp, STATUS_OK, space_eval
from hyperopt.pyll.base import scope

#Classification imports
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[ ]:


#Ignore warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# # Load and Treat Data

# For this study, we are going to use the Heart Disease UCI dataset. Since our goal is not to evaluate the data preprocessing steps required, we quickly prepare the dataset for the classification algorithms.

# In[ ]:


#Load dataset
data = pd.read_csv('../input/heart-disease-uci/heart.csv')


# In[ ]:


#Encode multiclass categorical features
to_encode = ['cp', 'restecg', 'slope', 'ca', 'thal']
for col in to_encode:
    new_feat = pd.get_dummies(data[col])
    new_feat.columns = [col+str(x) for x in range(new_feat.shape[1])]
    new_feat = new_feat.drop(columns = new_feat.columns.values[0])
    
    data[new_feat.columns.values] = new_feat
    data = data.drop(columns = col)


# In[ ]:


#Split data into attributes and target
atr = data.drop(columns = 'target')
target = data['target']


# In[ ]:


#Scale dataset
scaler = MinMaxScaler()
atr = scaler.fit_transform(atr)


# # Preliminar Modeling

# Before we proceed to actually tuning the classification algorithms, a simple Naive-Bayes model is created in order to provides us a reference accuracy value for the posterior runs.

# In[ ]:


#Preliminar modeling
pre_score = cross_val_score(estimator = GaussianNB(),
                            X = atr, 
                            y = target,
                            scoring = 'accuracy',
                            cv = 10,
                            verbose = 0)

print('Naive-Bayes mean score: %5.3f' %np.mean(pre_score))


# # Hyperparameters Tuning

# Here, we finally compare bayesian, grid search and random search tuning methods. However, it is important to run each of the chosen algorithms before any optimization takes place. This way, we can have a better ideia of how each model benefits from each tuning approach.

# In[ ]:


#Compare algorithms in their default configurations
models = [LogisticRegression(), DecisionTreeClassifier(), SVC()]
model_names = [type(x).__name__ for x in models]

std_score = []
for m in tqdm(models):
    std_score.append(cross_val_score(estimator = m,
                                 X = atr,
                                 y = target,
                                 scoring = 'accuracy',
                                 cv = 10).mean())
    
pd.Series(data = std_score, index = model_names)


# ## Bayesian Tuning

# In[ ]:


#Bayesian hyperparameters tuning: Define function
def bayes_tuning(estimator, xdata, ydata, cv, space, max_it):
    
    #Define objective function
    def obj_function(params):
        model = estimator(**params)
        score = cross_val_score(estimator = model, X = xdata, y = ydata,
                                scoring = 'accuracy',
                                cv = cv).mean()
        return {'loss': -score, 'status': STATUS_OK}
    
    start = time.time()
    
    #Perform tuning
    hist = Trials()
    param = fmin(fn = obj_function, 
                 space = space,
                 algo = tpe.suggest,
                 max_evals = max_it,
                 trials = hist,
                 rstate = np.random.RandomState(1))
    param = space_eval(space, param)
    
    #Compute best score
    score = -obj_function(param)['loss']
    
    return param, score, hist, time.time() - start


# In[ ]:


#Define hyperparameters spaces for Bayesian tuning
lr_params = {'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-2)),
             'C': hp.uniform('C', 0.1, 2.0),
             'fit_intercept': hp.choice('fit_intercept', [True, False]),
             'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
             'max_iter': scope.int(hp.quniform('max_iter', 50, 500, 20))
}

dt_params = {'criterion': hp.choice('criterion', ['gini', 'entropy']),
             'splitter': hp.choice('splitter', ['best', 'random']),
             'max_depth': scope.int(hp.quniform('max_depth', 3, 50, 1)),
             'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 50, 1)),
             'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 50, 1)),
             'max_features': hp.choice('max_features', ['auto', 'log2', None])
}

sv_params = {'C': hp.uniform('C', 0.1, 2.0),
             'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
             'degree': scope.int(hp.quniform('degree', 2, 5, 1)),
             'gamma': hp.choice('gamma', ['auto', 'scale']),
             'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-2)),
             'max_iter': scope.int(hp.quniform('max_iter', -1, 100, 1))
}


# In[ ]:


#Apply bayesian tuning
models = [LogisticRegression, DecisionTreeClassifier, SVC]
model_params = [lr_params, dt_params, sv_params]

bayes_score, bayes_time, bayes_hist = [], [], []
for m, par in tqdm(zip(models, model_params)):
    param, score, hist, dt = bayes_tuning(m, atr, target, 10, par, 150)
    bayes_score.append(score)
    bayes_time.append(dt)
    bayes_hist.append(hist)


# In[ ]:


#Print bayesian tuning results
bayes_df = pd.DataFrame(index = model_names)
bayes_df['Accuracy'] = bayes_score
bayes_df['Time'] = bayes_time

print(bayes_df)


# ## Grid Search Tuning

# In[ ]:


#Define function for grid search tuning
def grid_tuning(estimator, xdata, ydata, cv, space):
    
    start = time.time()
    
    #Perform tuning
    grid = GridSearchCV(estimator = estimator,
                        param_grid = space,
                        scoring = 'accuracy',
                        cv = 10)
    grid.fit(xdata, ydata)
    
    return grid.best_params_, grid.best_score_, time.time() - start


# In[ ]:


#Define hyperparameters spaces for grid seach tuning
lr_params = {'tol': [1e-5, 1e-3, 1e-2],
             'C': [0.1, 0.5, 1.0, 2.0],
             'fit_intercept': [True, False],
             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
             'max_iter': [50, 100, 250, 500]
}

dt_params = {'criterion': ['gini', 'entropy'],
             'splitter': ['best', 'random'],
             'max_depth': [3, 10, 25, 40, 50],
             'min_samples_split': [2, 10, 25, 50, 50],
             'min_samples_leaf': [1, 10, 25, 50, 50],
             'max_features': ['auto', 'log2', None]
}

sv_params = {'C': [0.1, 0.5, 1.0, 2.0],
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
             'degree': [2, 3, 5],
             'gamma': ['auto', 'scale'],
             'tol': [1e-5, 1e-3, 1e-2],
             'max_iter': [-1, 50, 100]
}


# In[ ]:


#Apply grid seach tuning
models = [LogisticRegression(), DecisionTreeClassifier(), SVC()]
model_params = [lr_params, dt_params, sv_params]

grid_score, grid_time = [], []
for m, par in tqdm(zip(models, model_params)):
    _, score, dt = grid_tuning(m, atr, target, 10, par)
    grid_score.append(score)
    grid_time.append(dt)


# In[ ]:


#Print grid search tuning results
grid_df = pd.DataFrame(index = model_names)
grid_df['Accuracy'] = grid_score
grid_df['Time'] = grid_time

print(grid_df)


# # Random Search Tuning

# In[ ]:


#Define function for random search tuning
def random_tuning(estimator, xdata, ydata, cv, space, max_iter):
    
    start = time.time()
    
    #Perform tuning
    rand = RandomizedSearchCV(estimator = estimator,
                              param_distributions = space,
                              n_iter = max_iter,
                              scoring = 'accuracy',
                              cv = 10,
                              random_state = np.random.RandomState(1))
    rand.fit(xdata, ydata)
    
    return rand.best_params_, rand.best_score_, rand.cv_results_['mean_test_score'], time.time() - start


# In[ ]:


#Define hyperparameters spaces for random search tuning
lr_params = {'tol': list(np.logspace(np.log(1e-5), np.log(1e-2), num = 10, base = 3)),
             'C': list(np.linspace(0.1, 2.0, 20)),
             'fit_intercept': [True, False],
             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
             'max_iter': list(range(50, 501))
}

dt_params = {'criterion': ['gini', 'entropy'],
             'splitter': ['best', 'random'],
             'max_depth': list(range(3, 51)),
             'min_samples_split': list(range(2, 50)),
             'min_samples_leaf': list(range(1, 50)),
             'max_features': ['auto', 'log2', None]
}

sv_params = {'C': list(np.linspace(0.1, 2.0, 10)),
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
             'degree': list(range(2, 6)),
             'gamma': ['auto', 'scale'],
             'tol': list(np.logspace(np.log(1e-5), np.log(1e-2), num = 10, base = 3)),
             'max_iter': list(range(-1, 101))
}


# In[ ]:


#Apply random seach tuning
models = [LogisticRegression(), DecisionTreeClassifier(), SVC()]
model_params = [lr_params, dt_params, sv_params]

rand_score, rand_time, rand_hist = [], [], []
for m, par in tqdm(zip(models, model_params)):
    _, score, hist, dt = random_tuning(m, atr, target, 10, par, 150)
    rand_score.append(score)
    rand_time.append(dt)
    rand_hist.append(hist)


# In[ ]:


#Print random search tuning results
rand_df = pd.DataFrame(index = model_names)
rand_df['Accuracy'] = rand_score
rand_df['Time'] = rand_time

print(rand_df)


# # Compare Tuning Approaches

# As mentioned before, we want to evaluate the three tuning methods. This is done first comparing the cross-validation accuracy and then the processing time for each approach. The plots in this notebook use the Plotly library. So, we take the time to set up the approppriate imports.

# In[ ]:


#Install plotly
get_ipython().system('pip install plotly')


# In[ ]:


#Plotly imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ## Compare Accuracy

# In[ ]:


#Compare accuracy
tuning_acc = pd.DataFrame(index = model_names)
tuning_acc['Bayes'] = bayes_score
tuning_acc['Grid'] = grid_score
tuning_acc['Random'] = rand_score

fig = go.Figure(data = [
    go.Bar(name = 'Bayes Tuning', x = tuning_acc.index, y = tuning_acc['Bayes']),
    go.Bar(name = 'Grid Tuning', x = tuning_acc.index, y = tuning_acc['Grid']),
    go.Bar(name = 'Random Tuning', x = tuning_acc.index, y = tuning_acc['Random'])
])

fig.update_layout(barmode = 'group', 
                  title = 'Accuracy Comparison',
                  xaxis_title = 'Estimator',
                  yaxis_title = 'Cross-validation accuracy (%)',
                  yaxis = dict(range = [0.75, 0.9]))
fig.show()


# As we can see, the three tuning approaches produce similar accuracy. If we are dealing with Kaggle competitions, however, the slighest improvement is of great importance. On that page, the Bayesian generates the best results for both the DecisionTreeClassifier and SVC. For the LogisticRegression algorithm, the Grid Search and Random Search methods present a nearly neglectable advantage.
# 
# Now, let's try to undertand these results. Why would the results present such differences given the hyperparameters space limits were the same for all tuning procedures? First of all, it is easy to understand the Random Search results. As the name goes, the tuning procedure is completely random. Applying it several times for the same hyperparameters space will likely yield a different result every time. 
# 
# As for the Grid Search tuning, these results can be explained by the fact only specific values for each hyperparameter are tested. The other methods, on the other hand, test the parameters values within a range. It is, of course, possible to work around this limitation by providing several more values for each hyperparameter. This, however, is not advisable, since the processing time will greatly increase, rendering the procedure too ineficient.
# 
# Finally, the Bayesian tuning, given its nature, will eventually find a good, not necessarily the best, result if the hyperparameter space provided is wide and the number of trials is large enough. Of course, these concepts are arbitrary and depend on the problem at hand.

# ## Compare Performance

# In[ ]:


#Compare performance
tuning_time = pd.DataFrame(index = model_names)
tuning_time['Bayes'] = bayes_time
tuning_time['Grid'] = grid_time
tuning_time['Random'] = rand_time

fig = go.Figure(data = [
    go.Bar(name = 'Bayes Tuning', x = tuning_time.index, y = tuning_time['Bayes']),
    go.Bar(name = 'Grid Tuning', x = tuning_time.index, y = tuning_time['Grid']),
    go.Bar(name = 'Random Tuning', x = tuning_time.index, y = tuning_time['Random'])
])

fig.update_layout(barmode = 'group',
                  title = 'Performance Comparison',
                  xaxis_title = 'Estimator',
                  yaxis_title = 'Computation time (sec)')
fig.show()


# The performance results, given the accuracy data, show the Grid Search tuning approach takes much more time to find the best set of hyperparameters while producing comparable results. In this case, we are dealing with a small dataset, so the tuning times are not really a matter of concern. However, this can easily become an issue for larger datasets contaning a larger number of attributes.
# 
# Now, the comparison between the Bayesian and Random Search tuning procedures shows us the latter is sistematically faster then the former. The explanation here is simple. The Bayesian tuning method takes some extra time to evaluate the next hyperparameters set to be tested. As we are running both methods with the same amount of iterations, this difference accounts exactly to this extra step. 

# ## Compare Tuning Progression

# Now, we want to understand how the Bayesian and Random Search tunings progressed to their final results. In other words, we want to know how many iterations it took for both approaches to get their best accuracy score.

# In[ ]:


#Compare Bayesian and Random
bayes_best = dict()
random_best = dict()
for i,model in enumerate(model_names):
    dummy = [-x['loss'] for x in bayes_hist[i].results]
    bayes_best[model] = np.maximum.accumulate(dummy)
    
    dummy = [x for x in rand_hist[i]]
    random_best[model] = np.maximum.accumulate(dummy)


# In[ ]:


#Logistic Regression
fig = go.Figure()

fig.add_trace(go.Scatter(x = list(range(150)), y = bayes_best['LogisticRegression'], name = 'Bayes (Hyperopt)'))
fig.add_trace(go.Scatter(x = list(range(150)), y = random_best['LogisticRegression'], name = 'Random search'))

fig.update_layout(title = 'Logistic Regression Tuning progression',
                  xaxis_title = 'Iteration',
                  yaxis_title = 'Cross-validation accuracy (%)')
fig.show()


# In[ ]:


#DecisionTreeClassifier
fig = go.Figure()

fig.add_trace(go.Scatter(x = list(range(150)), y = bayes_best['DecisionTreeClassifier'], name = 'Bayes (Hyperopt)'))
fig.add_trace(go.Scatter(x = list(range(150)), y = random_best['DecisionTreeClassifier'], name = 'Random search'))

fig.update_layout(title = 'Decision Tree Tuning progression',
                  xaxis_title = 'Iteration',
                  yaxis_title = 'Cross-validation accuracy (%)')
fig.show()


# In[ ]:


#SVC
fig = go.Figure()

fig.add_trace(go.Scatter(x = list(range(150)), y = bayes_best['SVC'], name = 'Bayes (Hyperopt)'))
fig.add_trace(go.Scatter(x = list(range(150)), y = random_best['SVC'], name = 'Random search'))

fig.update_layout(title = 'SVC Tuning progression',
                  xaxis_title = 'Iteration',
                  yaxis_title = 'Cross-validation accuracy (%)')
fig.show()


# Let us now understand the results in the plots above individually:
# 
# * Logistic Regression:
# 
# The Random Search tuning rapidly reaches its best accuracy value, while the Bayesin tuning takes much more steps to find its optimal value.
# 
# * Decision Tree:
# 
# The Bayesian tuning begins at a smaller accuracy value, but soon improves and reachs its highest values several iterations before the Random Search method.
# 
# * SVC:
# 
# The Bayesian method takes more iterations to find its best accuracy value than the Random Search approach. However, it present the superior value at every iteration level.

# # Conclusion

# The hyperopt Bayesin tuning presents a good alternative for hyperparameters optimization. It greatly outperforms the Grid Search approach, while producing similar results. Additionally, it presents comparable performance to the Random Search method, speacilly considering it can achieve similar results with a smaller number of iterations. 

# # References

# 1. [An Introductory Example of Bayesian Optimization in Python with Hyperopt](https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0)
# 
# 2. [Hyperparameter optimisation with Hyperopt](https://github.com/MBKraus/Hyperopt/blob/master/Hyperopt.ipynb)
