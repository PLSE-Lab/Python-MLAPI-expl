#!/usr/bin/env python
# coding: utf-8

# ---
# # IBM Employee Attrition: CatBoost vs XGBoost vs LigthtGBM Comparison
# **[Nicholas Holloway](https://github.com/nholloway)**
# 
# ---
# ### Mission
# We are going to explore the accuracy, speed, and ease of use of the three most popular gradient boosting algorithms. Gradient boosted decision trees (GBDTs) are one of the most important machine learning models. Understanding how to quickly implement and fine-tune a gradient boosting tree will allow us to solve a variety of problems and iterate quickly. This kernel will walk through a basic implementation of each gradient boosted algorithm, use **bayesian optimization** for hyperparameter search, and conclude with a comparison that we hope provides insight.
# 
# ### Table of Contents
# 1. [Introduction](#introduction)
# 2. [Data Exploration](#data exploration)
# 3. [Bayesian Hyperparameter Tuning](#bayesian)
# 4. [XGBoost](#xgboost)
# 5. [LightGBM](#lightgbm)
# 6. [CatBoost](#catboost)
# 7. [Results](#results)

# In[ ]:


import time
import math
import seaborn as sns
import pandas as pd
import numpy as np
import scipy as sci
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt import fmin

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight') 
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='introduction'></a>
# # Comparing Gradient Boosted Decision Trees (GBDTs)
# ---
# Gradient Boosted Decision Trees may be the most widely used model on Kaggle and are a stapple for classification and regression problems. GBDTs originate from AdaBoost, an algorithm that ensembled weak learners and used the majority vote, weighted by their individual accuracy, to solve binary classification problems. The weak learners in this case are decision trees with a single split, called decision stumps. 
# 
# Gradient boosted trees developed into a statistical framework where boosting was a numerical optimization problem predicated on reducing the model loss by adding weak learners. How those weak learners were added, how the learners were constructed, and what kind of loss functions were used expanded as boosted trees became more popular. There are many resources that go into greater depth about boosting algorithms so we'll move to the present- where there are three widely used gradient boosted decision trees: XGBoost, CatBoost, and LightGBM. The focus of this kernel is applying each of these algorithms, seeing how they perform and how we might choose between them in the future. 

# In[ ]:


ibm_df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
description = pd.DataFrame(index=['observations(rows)', 'percent missing', 'dtype', 'range'])
numerical = []
categorical = []
for col in ibm_df.columns:
    obs = ibm_df[col].size
    p_nan = round(ibm_df[col].isna().sum()/obs, 2)
    num_nan = f'{p_nan}% ({ibm_df[col].isna().sum()}/{obs})'
    dtype = 'categorical' if ibm_df[col].dtype == object else 'numerical'
    numerical.append(col) if dtype == 'numerical' else categorical.append(col)
    rng = f'{len(ibm_df[col].unique())} labels' if dtype == 'categorical' else f'{ibm_df[col].min()}-{ibm_df[col].max()}'
    description[col] = [obs, num_nan, dtype, rng]

numerical.remove('EmployeeCount')
numerical.remove('StandardHours')
pd.set_option('display.max_columns', 100)
display(description)
display(ibm_df.head())


# <a id='data exploration'></a>
# # Data Exploration
# ---
# There isn't much data cleaning to be done. Here's what we want to look for: 
# - Missing data
# - Consistent data type across a feature
# - Outliers or inconsistencies in data columns
# - Observe the columns and look for features that may be related. These features may be used later in feature engineering.
# 
# Luckily, the data isn't missing any values and we can spend more time on the point of this kernel, comparing different gradient boosted tree algorithms. First, we want to get a sense of our data:
# - What features have the most divergent distributions based on target class
# - Do we have a target label imbalance
# - How our independent variables are distributed relative to our target label
# - Are there features that have strong linear or monotonic relationships, making correlation heatmaps makes it easy to identify possible colinearity

# In[ ]:


p_col = 2
fig, ax = plt.subplots(12, p_col, figsize=(10, 30))

for idx, feature in enumerate(numerical): 
    col = idx%p_col
    row = math.floor(idx/p_col)
    ibm_df.boxplot(column=feature, by='Attrition', ax = ax[row][col])
    
plt.tight_layout()


# In[ ]:


count_yes = ibm_df.Attrition[ibm_df.Attrition == 'Yes'].size
count_no = ibm_df.Attrition[ibm_df.Attrition == 'No'].size

plt.bar(['Leave', 'Stay'], [count_yes, count_no])
plt.title('IBM Attrition Label Imbalance')
plt.xlabel('Whether IBM Employee Left')
plt.ylabel('Count of Employees')
plt.show()


# In[ ]:


features = ['MonthlyIncome', 'Attrition', 'JobLevel', 'TotalWorkingYears', 'YearsAtCompany', 'YearsWithCurrManager']
pairplot = sns.pairplot(ibm_df[features], diag_kind='kde', hue='Attrition')
plt.show()


# In[ ]:


trace1 = go.Heatmap(
    z = ibm_df[numerical].astype(float).corr().values,
    x = ibm_df[numerical].columns.values,
    y = ibm_df[numerical].columns.values,
    colorscale = 'Portland', 
    reversescale = False, 
    opacity = 1.0)
        
data = [trace1]
layout = go.Layout(
    title = 'Correlation Among IBM Employee Attrition Numerical Features',
    xaxis = dict(ticks = '', nticks = 36),
    yaxis = dict(ticks = ''),
    width = 700, height = 700
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# ### What the data is showing us
# One of the first problems in our data is the imbalance of target labels. We have about a 6:1 *No* attrition label compared to *Yes*. The effect of the imbalance really shows up in the pairplots where the *yes* markers in the scatter plots are all but drowned out, though this would be less of a problem if our classes were more distinct. When we begin to test our model a smart thing to do would be to look at the confusion matrix and see how well our model performs on the minority class, *yes*. If we're seeing problems classifying *yes* then we can oversample the *yes* class with a technique like SMOTE, or- and what I'd prefer to do is used Tomek Links to undersample the majority class and ideally create a cleaner boundary in our scatterplots. 
# 
# In the pairplots we have a smattering of variables that were chosen based on our boxplots. In the boxplots those features seemed to show the greatest disparity in distributions between our target label and therefore seemed most interesting to plot. From the diagonal distributions it seems that there are no features that have drastically different distributions between our classes, luckily there are many features in this data- which bodes well for our decision trees. 
# 
# Lastly, we plot the correlations between our features to look for colinear relationships. These are usually a problem for GBDTs but if we have many features with high correlation we might look to delete all but one or do some feature engineering.

# <a id='xgboost'></a>
# # XGBoost
# ---
# XGBoost is a workhorse gradient boosted decision tree algorithm. Its been around since 2014 and has come to dominate the Kaggle and data science community. XGB introduced gradinet boosting where new models are fit to the residuals of prior models and then added together, using a gradient descent algorithm to minimize the loss. 
# 
# XGBoost's parameters are broken up into three categories: general, tree booster, and learning task parameters. General and learning parameters are mostly determined by what we are modelling so most of our time will be spent with the tree boost parameters. Our GBDT algorithms have common parameters and reviewing them can help us understand the different levers we can pull as data scientists to adapt to problems we come across in our models. 
# 
# ### General Parameters
# 1. Booster (default=gbtree): Type of model to run at each iteration (gbtree= tree-based, gblinear=linear model)
# 2. silent (defualt=0): Set to True to aviod printing updates with each cycle.
# 3. nthread (default= max available): Number of threads for parallel processing. 
# 
# ### Tree Booster Parameters
# 1. eta (default=0.3): Aliased as learning rate, typical values- 0.01-0.3
# 2. min_child_weight (default=1): Defines the minimum sum of the weights of all observations required in child. Used to control overfitting, high values can lead to underfitting. 
# 3. max_depth (default=6): Higher depth allows learned relations very specific to a particular sample. Typical values are 3-10.
# 4. gamma (default=0): Minimum loss reduction required to partition on a leaf node of a tree- the larger gamma the more conservative the algorithm.
# 5. max_delta_step (default=0): Maximum delta step we allow each leaf output to be. This is generally not used but might help in logistic regression when the class is extremely imbalanced. 
# 6. subsample (default=1): Subsample ratio of the training instances, if 0.5 XGBoost would randomly sample half of the training data prior to growing trees, and this will prevent overfitting. 
# 7. colsample_bytree (default=1): Denotes the fraction of features to use.
# 8. colsample_bylevel (defualt=1): Denotes the fraction of features but in terms of the level rather than tree. 
# 9. lambda (default=1): L2 regularization parameter. 
# 10. alpha (defualt=0): : L1 regularization parameter.  
# 11. scale_pos_weight (default=1): Controls the balance of positive and negative weights, its useful for unbalanced classes. 
# 
# ### Learning Task Parameters
# These parameters define the optimization objective to be calculated at each step.
# 1. objective (default=linear): The loss function, logistic, softmax, and softprob (same as softmax but returns predicted probabilites not predicted classes). 
# 2. eval_metric (default based on objective): The metric used to validate data like: rmse, mae, logloss, error(binary classification), merror (multiclass error), mlogloss, auc (area under the curve). 
# 
# **num_boosting_rounds**: This is passed to `.train()` and `.cv()`. For each round in the results we will see our mean and standard dev of the evaluation metric. Fitting a model with `fit()` will instead use `n_estimators`. 

# <a id='bayesian'></a>
# # Hyperparameter Tuning 
# ---
# There are tutorials online for how to tune our tree boost parameters in dependency-based groups if we are using a gridsearch or random search approach, but for this kernel we will use a Bayesian optimization approach. In a Bayesian optimization approach we treat the possible parameter space as an optimization problem.
# 
# ### Hyperopt 
# Hyperopt is a python library for Bayesian optimization of hyperparameters. There are two big reasons I like hyperopt: 
# 1. It has better overall performance and takes less time than grid search and random search methods.
# 2. I prefer the api for hyperopt to sklearn's gridsearch. I think it's more flexible and easy to integrate with various machine learning models. 
# 
# Hyperopt works by treating the search of hyperparameters as an optimization task where it uses the returned loss from the objective function, *fit_model(parameters)*, to decide what the next set of hyperparameters should be. Hyperparameter search spaces are generally large multi-dimensional spaces so hyperopt is a big improvement over grid and random search, especially as the search space expands. 
# 
# Hyperopt has a few components:
# 1. Objective function. This is where we put our ml model and what we pass our hyperparameters into. The objective function will return at a minimum, the loss- more complex designs will return an object which can have various fields.
# 2. Feature space. The feature space is an object that gets passed to our objective function. The `hp` object in hyperopt has many functions for constructing ranges and distributions for our features. 
# 3. `Trials` object. This is not strictly necessary but the `Trials` object allows us to store all the values that we returned from our experiment. 
# 4. `fmin` function. This is the function that minimizes our objective. `fmin` takes our object, space, trials, and the max amount of evaluations, `max_evals`. 

# In[ ]:


def org_results(trials, hyperparams, model_name):
    fit_idx = -1
    for idx, fit  in enumerate(trials):
        hyp = fit['misc']['vals']
        xgb_hyp = {key:[val] for key, val in hyperparams.items()}
        if hyp == xgb_hyp:
            fit_idx = idx
            break
            
    train_time = str(trials[-1]['refresh_time'] - trials[0]['book_time'])
    acc = round(trials[fit_idx]['result']['accuracy'], 3)
    train_auc = round(trials[fit_idx]['result']['train auc'], 3)
    test_auc = round(trials[fit_idx]['result']['test auc'], 3)

    results = {
        'model': model_name,
        'parameter search time': train_time,
        'accuracy': acc,
        'test auc score': test_auc,
        'training auc score': train_auc,
        'parameters': hyperparams
    }
    return results


# In[ ]:


xgb_data = ibm_df.copy()
xgb_dummy = pd.get_dummies(xgb_data[categorical], drop_first=True)
xgb_data = pd.concat([xgb_dummy, xgb_data], axis=1)
xgb_data.drop(columns = categorical, inplace=True)
xgb_data.rename(columns={'Attrition_Yes': 'Attrition'}, inplace=True)

y_df = xgb_data['Attrition'].reset_index(drop=True)
x_df = xgb_data.drop(columns='Attrition')
train_x, test_x, train_y, test_y = train_test_split(x_df, y_df, test_size=0.20)

def xgb_objective(space, early_stopping_rounds=50):
    
    model = XGBClassifier(
        learning_rate = space['learning_rate'], 
        n_estimators = int(space['n_estimators']), 
        max_depth = int(space['max_depth']), 
        min_child_weight = space['m_child_weight'], 
        gamma = space['gamma'], 
        subsample = space['subsample'], 
        colsample_bytree = space['colsample_bytree'],
        objective = 'binary:logistic'
    )

    model.fit(train_x, train_y, 
              eval_set = [(train_x, train_y), (test_x, test_y)],
              eval_metric = 'auc',
              early_stopping_rounds = early_stopping_rounds,
              verbose = False)
     
    predictions = model.predict(test_x)
    test_preds = model.predict_proba(test_x)[:,1]
    train_preds = model.predict_proba(train_x)[:,1]
    
    xgb_booster = model.get_booster()
    train_auc = roc_auc_score(train_y, train_preds)
    test_auc = roc_auc_score(test_y, test_preds)
    accuracy = accuracy_score(test_y, predictions) 

    return {'status': STATUS_OK, 'loss': 1-test_auc, 'accuracy': accuracy,
            'test auc': test_auc, 'train auc': train_auc
           }

space = {
    'n_estimators': hp.quniform('n_estimators', 50, 1000, 25),
    'max_depth': hp.quniform('max_depth', 1, 12, 1),
    'm_child_weight': hp.quniform('m_child_weight', 1, 6, 1),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'learning_rate': hp.loguniform('learning_rate', np.log(.001), np.log(.3)),
    'colsample_bytree': hp.quniform('colsample_bytree', .5, 1, .1)
}

trials = Trials()
xgb_hyperparams = fmin(fn = xgb_objective, 
                 max_evals = 150, 
                 trials = trials,
                 algo = tpe.suggest,
                 space = space
                 )

xgb_results = org_results(trials.trials, xgb_hyperparams, 'XGBoost')
display(xgb_results)


# <a id='lightgbm'></a>
# # LightGBM
# ---
# LightGBM is an open-source GBDT framework created by Microsoft as a fast and scalable alternative to XGB and GBM. By default LightGBM will train a Gradient Boosted Decision Tree (GBDT), but it also supports random forests, Dropouts meet Multiple Additive Regression Trees (DART), and Gradient Based One-Side Sampling (Goss). 
# 
# ### Tree Parameters
# 1. max_depth: The maximum depth of the tree, this is used to handle model overfitting. If the model is overfitting lowering max_depth should be the first thing to change. 
# 2. min_data_in_leaf (default=20): The mininmum number of records a leaf may have, helps with overfitting. Setting it to a large value can avoid growing too deep a tree, but may cause underfitting. Setting it to hundreds or thousands is enough in a large dataset. 
# 3. feature_fraction: The fraction of the features randomly sampled from when building trees, a familiar parameter in GBDts. 
# 4. bagging_fraction: Specifies the fraction of data to be used for each iteration and is generally used to speed up the training and avoid overfitting. 
# 5. early_stopping_round: Model will stop training if the validation metric doesn't improve for consecutive rounds.
# 6. lambda: Specifies regularization, typical values 0-1. 
# 7. min_gain_to_split: This parameter will describe the minimum gain to make a split.
# 
# ### General Parameters
# 1. Application:
#     - regression: Regression
#     - binary: Binary classification
#     - multiclass: Multiclass classification
# 2. Boosting:
#     - gdbt: Gradient Boosted Decision Tree
#     - rf: Random Forest
#     - dart: Dropouts meet multiple regression trees
#     - goss: Gradient-based One-Side Sampling
# 3. num_boost_round: Number of boosting iterations, typically 100+
# 4. learning_rate: This determines the impact of each tree on the final outcome. Typical values 0.1, 0.001, 0.003
# 5. num_leaves (default=31): Number of leaves in full tree. This is the main parameter to control the complexity of the model. Ideally, the value of num_leaves should be less than or equal to 2^ (max_depth). Value more than this will result in overfitting. 
# 6. Metric Parameter:
#     - Mae
#     - Mse
#     - Binary_logloss
#     - Multi_logloss 
# 7. categorical_feature: It denotes the index of categorical features. 

# In[ ]:


lgb_data = ibm_df.copy()
lgb_dummy = pd.get_dummies(lgb_data[categorical], drop_first=True)
lgb_data = pd.concat([lgb_dummy, lgb_data], axis=1)
lgb_data.drop(columns = categorical, inplace=True)
lgb_data.rename(columns={'Attrition_Yes': 'Attrition'}, inplace=True)

y_df = lgb_data['Attrition'].reset_index(drop=True)
x_df = lgb_data.drop(columns='Attrition')
train_x, test_x, train_y, test_y = train_test_split(x_df, y_df, test_size=0.20)

def lgb_objective(space, early_stopping_rounds=50):
    
    lgbm = LGBMClassifier(
        learning_rate = space['learning_rate'],
        n_estimators= int(space['n_estimators']), 
        max_depth = int(space['max_depth']),
        num_leaves = int(space['num_leaves']),
        colsample_bytree = space['colsample_bytree'],
        feature_fraction = space['feature_fraction'],
        reg_lambda = space['reg_lambda'],
        reg_alpha = space['reg_alpha'],
        min_split_gain = space['min_split_gain']
    )
    
    lgbm.fit(train_x, train_y, 
            eval_set = [(train_x, train_y), (test_x, test_y)],
            early_stopping_rounds = early_stopping_rounds,
            eval_metric = 'auc',
            verbose = False)
    
    predictions = lgbm.predict(test_x)
    test_preds = lgbm.predict_proba(test_x)[:,1]
    train_preds = lgbm.predict_proba(train_x)[:,1]
    
    train_auc = roc_auc_score(train_y, train_preds)
    test_auc = roc_auc_score(test_y, test_preds)
    accuracy = accuracy_score(test_y, predictions)  

    return {'status': STATUS_OK, 'loss': 1-test_auc, 'accuracy': accuracy,
            'test auc': test_auc, 'train auc': train_auc
           }

trials = Trials()
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
    'n_estimators': hp.quniform('n_estimators', 50, 1200, 25),
    'max_depth': hp.quniform('max_depth', 1, 15, 1),
    'num_leaves': hp.quniform('num_leaves', 10, 150, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0), 
    'feature_fraction': hp.uniform('feature_fraction', .3, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'min_split_gain': hp.uniform('min_split_gain', 0.0001, 0.1)
}

lgb_hyperparams = fmin(fn = lgb_objective, 
                 max_evals = 150, 
                 trials = trials,
                 algo = tpe.suggest,
                 space = space
                 )

lgb_results = org_results(trials.trials, lgb_hyperparams, 'LightGBM')
display(lgb_results)


# <a id='catboost'></a>
# # CatBoost
# ---
# Catboost was released in 2017 by Yandex, showing, by their benchmark to be faster in prediction, better in accuracy, and easier to use for categorical data across a series of GBDT tasks. 
# 
# Catboost introduces *ordered boosting* as a better gradient boosting algorithm but the greatest innovation of catboost is how it deals with categorical data. Categorical data introduces several challenges because it has to have a numerical encoding. We could use dummy variables that split the column of n categories into n columns of one-hot encoded features but this can explode our feature set. One tactic is to use *target mean encoding*, where we assign our category value to the mean of the target variable for that category. Catboost uses a variation on target encoding that calculates the target encoding with available history and a random permutation to encode and process our categorical data. Catboost uses the available history instead of the mean because a model running in real time would not know the true mean for its target. In training we can, of course, calculate the mean because we have all the data and this leads to *target leakage* where the training accuracy for our model is inflated in comparison to its accuracy in production.  
# 
# Additional capabilities of catboost include plotting feature interactions and object (row) importance. 
# 
# ### CatBoost Parameters
# Several catboost parameters have aliases so keep that in mind if looking at various sources of parameter information. Catboost parameters are broken up into 8 groups, we'll be focusing on a smaller set for our optimization.
# 1. Common parameters (32)
# 2. Overfitting detection parameters (4) 
# 3. Binarization parameters (2) 
# 4. Multiclassification parameters (1)
# 5. Performance parameters (1)
# 6. Processing Unit parameters (2)
# 7. Output parameters (9)
# 8. Ctr parameters (8)- Categorical to numerical feature encoding
# 
# ### Important Hyperparameters
# 1. learning_rate (defualt=0.03): Dictates reduction in gradient step.
# 2. iterations (default=1000): Max number of trees that can be built when solving machine learning problems.
# 3. l2_leaf_reg (defualt=3): L2 regularization coefficient for leaf calculation, any positive values are allowed. 
# 4. depth (default=6): Depth of trees, like in XGB it has a high impact on accuracy and training time. It can be any integer up to 32 but a good range is 1-10.
# 5. random_strength: The amount of randomness to use for scoring splits, used to avoid overfitting. 
# 6. bagging_temperature: Defines the settings of the Bayesian bootstrap, `bootstrap_type` defines the sampling of the weights, Bayesian is the defualt. If set to 1 weights are sampled from an exponential distribution, if 0 then all weights are equal to 1. 
# 7. border_count(default=254 for cpu, 128 for gpu): The number of splits for numerical features, integers between 1 and 255 are allowed. 
# 8. ctr_border_count: The number of splits for categorical features. All values are integers from 1 to 255. 

# In[ ]:


cbo_data = ibm_df.copy()

for cat in categorical:
    cbo_data[cat] = cbo_data[cat].astype('category').cat.codes

y_df = cbo_data['Attrition'].reset_index(drop=True)
x_df = cbo_data.drop(columns='Attrition')

cboost_cat = categorical[1:]
train_x, test_x, train_y, test_y = train_test_split(x_df, y_df, test_size=0.20)
cat_dims = [train_x.columns.get_loc(name) for name in cboost_cat]     
    
def cat_objective(space, early_stopping_rounds=30):
    
    cboost = CatBoostClassifier(
    eval_metric  = 'AUC', 
    learning_rate = space['learning_rate'],
    iterations = space['iterations'],
    depth = space['depth'],
    l2_leaf_reg = space['l2_leaf_reg'],
    border_count = space['border_count']
    )
    
    cboost.fit(train_x, train_y, 
              eval_set = [(train_x, train_y), (test_x, test_y)],
              early_stopping_rounds = early_stopping_rounds,
              cat_features = cat_dims, 
              verbose = False)
    
    predictions = cboost.predict(test_x)
    test_preds = cboost.predict_proba(test_x)[:,1]
    train_preds = cboost.predict_proba(train_x)[:,1]    

    train_auc = roc_auc_score(train_y, train_preds)
    test_auc = roc_auc_score(test_y, test_preds)
    accuracy = accuracy_score(test_y, predictions)
    
    return {'status': STATUS_OK, 'loss': 1-test_auc, 'accuracy': accuracy,
            'test auc': test_auc, 'train auc': train_auc}
    
trials = Trials()
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
    'iterations': hp.quniform('iterations', 25, 1000, 25),
    'depth': hp.quniform('depth', 1, 16, 1),
    'border_count': hp.quniform('border_count', 30, 220, 5), 
    'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 10, 1)
}

cboost_hyperparams = fmin(fn = cat_objective, 
                 max_evals = 150, 
                 trials = trials,
                 algo = tpe.suggest,
                 space = space
                 )

cbo_results = org_results(trials.trials, cboost_hyperparams, 'CatBoost')
display(cbo_results)


# <a id='results'></a>
# # Results 
# ---
# There is no clear winner. It seems that each gradient boosted algorithm excells in its own way. XGBoost is still a wonderful algorithm with great documentation and many examples from years of use. XGB had good accuracy and consistently had a fairly small gap between the train and test auc scores- which gives me confidence it isn't overfitting our training set. LightGBM was clearly the fastest algorithm, often being 10x faster than XGBoost. In terms of accuracy the test and train dataset often had different balances of minority and majority class so 1-to-1 comparisons aren't perfect, but LightGBM consistently performed in concert with the other algorithms. CatBoost was the algorithm I was most interested in using because of the supposed innovations in working with categorical data. Despite good accuracy however, I ran into several problems. First, the documentation, while pretty, was less readable than boilerplate documentation sites and the api lacked good methods for feature importance plotting- where LightGBM was really strong. Second, CatBoost seemed significantly slower than the other algorithms, it seemed to stall on some evaluations but  I lowered the `early_stopping_rounds` from 50 to 30, which helped. I would want to do more experimentation with CatBoost and data that had lots of categorical features to get a better feel for when it may perform best.

# In[ ]:


final_results = pd.DataFrame([xgb_results, lgb_results, cbo_results])
display(final_results)

