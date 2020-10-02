#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Import-libraries" data-toc-modified-id="Import-libraries-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import libraries</a></span></li><li><span><a href="#Locally-defined-functions" data-toc-modified-id="Locally-defined-functions-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Locally defined functions</a></span><ul class="toc-item"><li><span><a href="#Metrics" data-toc-modified-id="Metrics-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Metrics</a></span></li><li><span><a href="#Display-functions" data-toc-modified-id="Display-functions-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Display functions</a></span></li><li><span><a href="#Define-features" data-toc-modified-id="Define-features-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Define features</a></span></li></ul></li><li><span><a href="#Global-options" data-toc-modified-id="Global-options-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Global options</a></span></li><li><span><a href="#Load--data" data-toc-modified-id="Load--data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Load  data</a></span><ul class="toc-item"><li><span><a href="#Training-data" data-toc-modified-id="Training-data-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Training data</a></span></li><li><span><a href="#Out-of-sample-data-(to-predict)" data-toc-modified-id="Out-of-sample-data-(to-predict)-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Out of sample data (to predict)</a></span></li></ul></li><li><span><a href="#Feature-exploration" data-toc-modified-id="Feature-exploration-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Feature exploration</a></span><ul class="toc-item"><li><span><a href="#Categorical-features" data-toc-modified-id="Categorical-features-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Categorical features</a></span></li><li><span><a href="#Numerical-features" data-toc-modified-id="Numerical-features-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Numerical features</a></span></li><li><span><a href="#Pair-plot" data-toc-modified-id="Pair-plot-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Pair plot</a></span></li></ul></li><li><span><a href="#Feature-generation" data-toc-modified-id="Feature-generation-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Feature generation</a></span></li><li><span><a href="#Feature-selection" data-toc-modified-id="Feature-selection-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Feature selection</a></span><ul class="toc-item"><li><span><a href="#Drop-features-(optional)" data-toc-modified-id="Drop-features-(optional)-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Drop features (optional)</a></span></li></ul></li><li><span><a href="#ML-data-preparation" data-toc-modified-id="ML-data-preparation-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>ML data preparation</a></span><ul class="toc-item"><li><span><a href="#Categorical-feature-encoding" data-toc-modified-id="Categorical-feature-encoding-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Categorical feature encoding</a></span></li><li><span><a href="#Train-test-split" data-toc-modified-id="Train-test-split-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Train test split</a></span></li></ul></li><li><span><a href="#Machine-learning-model" data-toc-modified-id="Machine-learning-model-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Machine learning model</a></span><ul class="toc-item"><li><span><a href="#Model-definition" data-toc-modified-id="Model-definition-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Model definition</a></span></li><li><span><a href="#ML-model-training" data-toc-modified-id="ML-model-training-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>ML model training</a></span></li></ul></li><li><span><a href="#Model-evaluation" data-toc-modified-id="Model-evaluation-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Model evaluation</a></span><ul class="toc-item"><li><span><a href="#Train,-test-predictions" data-toc-modified-id="Train,-test-predictions-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>Train, test predictions</a></span></li><li><span><a href="#Regression-coefficients/Feature-importance" data-toc-modified-id="Regression-coefficients/Feature-importance-10.2"><span class="toc-item-num">10.2&nbsp;&nbsp;</span>Regression coefficients/Feature importance</a></span></li><li><span><a href="#Metrics" data-toc-modified-id="Metrics-10.3"><span class="toc-item-num">10.3&nbsp;&nbsp;</span>Metrics</a></span></li><li><span><a href="#Model-Performance-plots" data-toc-modified-id="Model-Performance-plots-10.4"><span class="toc-item-num">10.4&nbsp;&nbsp;</span>Model Performance plots</a></span></li><li><span><a href="#Optionally-retrain-on-the-whole-data-set" data-toc-modified-id="Optionally-retrain-on-the-whole-data-set-10.5"><span class="toc-item-num">10.5&nbsp;&nbsp;</span>Optionally retrain on the whole data set</a></span></li></ul></li><li><span><a href="#Apply-model-to-OOS-data" data-toc-modified-id="Apply-model-to-OOS-data-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Apply model to OOS data</a></span><ul class="toc-item"><li><span><a href="#Subset-to-relevant-columns" data-toc-modified-id="Subset-to-relevant-columns-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span>Subset to relevant columns</a></span></li><li><span><a href="#Apply-categorical-encoding" data-toc-modified-id="Apply-categorical-encoding-11.2"><span class="toc-item-num">11.2&nbsp;&nbsp;</span>Apply categorical encoding</a></span></li><li><span><a href="#Apply-model-and-produce-output" data-toc-modified-id="Apply-model-and-produce-output-11.3"><span class="toc-item-num">11.3&nbsp;&nbsp;</span>Apply model and produce output</a></span></li></ul></li></ul></div>

# # Import libraries

# # Locally defined functions

# In[ ]:


# this may need to be installed separately with
# !pip install category-encoders
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# python general
import pandas as pd
import numpy as np
from collections import OrderedDict

#scikit learn

import sklearn
from sklearn.base import clone

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE

# ML models
from sklearn import tree
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from catboost import CatBoostRegressor

# error metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import cross_val_score

# plotting and display
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from IPython.display import display
pd.options.display.max_columns = None

# widgets and widgets based libraries
import ipywidgets as widgets
from ipywidgets import interact, interactive


# ## Metrics

# In[ ]:


def rmse(y_true, y_pred):
    res = np.sqrt(((y_true - y_pred) ** 2).mean())
    return res

def mape(y_true, y_pred):
    y_val = np.maximum(np.array(y_true), 1e-8)
    return (np.abs(y_true -y_pred)/y_val).mean()


# In[ ]:


metrics_dict_res = OrderedDict([
            ('mean_absolute_error', mean_absolute_error),
            ('median_absolute_error', median_absolute_error),
            ('root_mean_squared_error', rmse),
            ('mean abs perc error', mape)
            ])


# In[ ]:


def regression_metrics_yin(y_train, y_train_pred, y_test, y_test_pred,
                           metrics_dict, format_digits=None):
    df_results = pd.DataFrame()
    for metric, v in metrics_dict.items():
        df_results.at[metric, 'train'] = v(y_train, y_train_pred)
        df_results.at[metric, 'test'] = v(y_test, y_test_pred)

    if format_digits is not None:
        df_results = df_results.applymap(('{:,.%df}' % format_digits).format)

    return df_results


# ## Display functions

# In[ ]:


def describe_col(df, col):
    display(df[col].describe())

def val_count(df, col):
    display(df[col].value_counts())

def show_values(df, col):
    print("Number of unique values:", len(df[col].unique()))
    return display(df[col].value_counts(dropna=False))


# In[ ]:


def plot_distribution(df, col, bins=100, figsize=None, xlim=None, font=None, histtype='step'):
    if font is not None:
        mpl.rc('font', **font)

    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=(10, 6))
    dev = df[col]    
    dev.plot(kind='hist', bins=bins, density=True, histtype=histtype, color='b', lw=2,alpha=0.99)
    print('mean:', dev.mean())
    print('median:', dev.median())
    if xlim is not None:
        plt.xlim(xlim)
    return plt.gca()


# In[ ]:


def plot_feature_importances(model, feature_names=None, n_features=20):
    if feature_names is None:
        feature_names = range(n_features)
    
    importances = model.feature_importances_
    importances_rescaled = 100 * (importances / importances.max())
    xlabel = "Relative importance"

    sorted_idx = np.argsort(-importances_rescaled)

    names_sorted = [feature_names[k] for k in sorted_idx]
    importances_sorted = [importances_rescaled[k] for k in sorted_idx]

    pos = np.arange(n_features) + 0.5
    plt.barh(pos, importances_sorted[:n_features], align='center')

    plt.yticks(pos, names_sorted[:n_features])
    plt.xlabel(xlabel)

    plt.title("Feature importances")

    return plt.gca()


# In[ ]:


def plot_act_vs_pred(y_act, y_pred, scale=1, act_label='actual', pred_label='predicted', figsize=None, xlim=None,
                     ylim=None, font=None):
    
    if font is not None:
        mpl.rc('font', **font)

    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=(10, 6))
    plt.scatter(y_act/scale, y_pred/scale)
    x = np.linspace(0, y_act.max()/scale, 10)
    plt.plot(x, x)
    plt.xlabel(act_label)
    plt.ylabel(pred_label)
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim([0, 1e2])
    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim([0, 1e2])
    return plt.gca()


# In[ ]:


def compute_perc_deviation(y_act, y_pred, absolute=False):
    dev = (y_pred - y_act)/y_act * 100
    if absolute:
        dev = np.abs(dev)
        dev.name = 'abs % error'
    else:
        dev.name = '% error'
    return dev

def plot_dev_distribution(y_act, y_pred, absolute=False, bins=100, figsize=None, xlim=None, font=None):
    if font is not None:
        mpl.rc('font', **font)

    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=(10, 6))
    dev = compute_perc_deviation(y_act, y_pred, absolute=absolute)
    dev.plot(kind='hist', bins=bins, density=True)
    print('mean % dev:', dev.mean())
    print('median % dev:', dev.median())
    # plt.vlines(dev.mean(), 0, 0.05)
    plt.title('Distribution of errors')
    plt.xlabel('% deviation')
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim([-1e2, 1e2])
    return plt.gca()


# ## Define features

# In[ ]:


categorical_features = [
    'Body_Type',
    'Driven_Wheels',
    'Global_Sales_Sub-Segment',
    'Brand',
    'Nameplate',
    'Transmission',
    'Turbo',
    'Fuel_Type',
    'PropSysDesign',
    'Plugin',
    'Registration_Type',
    'country_name'
]

numeric_features = [
    'Generation_Year',
    'Length',
    'Height',
    'Width',
    'Engine_KW',
    'No_of_Gears',
    'Curb_Weight',
    'CO2',
    'Fuel_cons_combined',
    'year'
]

all_numeric_features = list(numeric_features)
all_categorical_features = list(categorical_features)

target = [
    'Price_USD'
]

target_name = 'Price_USD'


# # Global options

# In[ ]:


#ml_model_type = 'Linear Regression'
#ml_model_type = 'Decision Tree'
#ml_model_type = 'Random Forest'
#ml_model_type = 'Gradient Boosting Regressor'
#ml_model_type = 'AdaBoost'
#ml_model_type = 'XGBoost'
ml_model_type = 'CatBoost'

regression_metric = 'mean abs perc error'

do_grid_search_cv = False
scoring_greater_is_better = False  # THIS NEEDS TO BE SET CORRECTLY FOR CV GRID SEARCH

do_retrain_total = True
write_predictions_file = True

# relative size of test set
test_size = 0.2
random_state = 33


# # Load  data
# 

# ## Training data

# In[ ]:


df = pd.read_csv('/kaggle/input/ihsmarkit-hackathon-june2020/train_data.csv',index_col='vehicle_id')
df['date'] = pd.to_datetime(df['date'])
#g = df['Brand'].value_counts()
#df['Brand'] = np.where(df['Brand'].isin(g.index[g >= 200]), df['Brand'], 'Other')


# In[ ]:


# basic commands on a dataframe
# df.info()
df.head(5)
# df.shape
# df.head()
# df.tail()


# In[ ]:


df['country_name'].value_counts()


# In[ ]:


df.groupby(['year', 'country_name'])['date'].count()


# ## Out of sample data (to predict)

# In[ ]:


df_oos = pd.read_csv('/kaggle/input/ihsmarkit-hackathon-june2020/oos_data.csv', index_col='vehicle_id')
df_oos['date'] = pd.to_datetime(df_oos['date'])
df_oos['year'] = df_oos['date'].map(lambda d: d.year)
#g_oos = df_oos['Brand'].value_counts()
#df_oos['Brand'] = np.where(df_oos['Brand'].isin(g.index[g >= 200]), df_oos['Brand'], 'Other')


# In[ ]:


# df_oos.shape
df_oos.head()


# In[ ]:


df_oos.groupby(['year', 'country_name'])['date'].count()


# # Feature exploration

# ## Categorical features

# In[ ]:


# unique values, categorical variables
for col in all_categorical_features:
    print(col, len(df[col].unique()))


# In[ ]:


interactive(lambda col: show_values(df, col), col=all_categorical_features)


# ## Numerical features

# In[ ]:


# summary statistics
df[numeric_features + target].describe()


# In[ ]:


figsize = (16,12)
sns.set(style='whitegrid', font_scale=2)

bins = 1000
bins = 40
#xlim = [0,100000]
xlim = None
price_mask = df['Price_USD'] < 100000
interactive(lambda col: plot_distribution(df[price_mask], col, bins=bins, xlim=xlim), col=sorted(all_numeric_features + target))
#interactive(lambda col: plot_distribution(df, col, bins=bins, xlim=xlim), col=sorted(all_numeric_features + target))


# ## Pair plot

# In[ ]:


# this is quite slow
sns.set(style='whitegrid', font_scale=1)
# sns.pairplot(df[numeric_features[:6] + target].iloc[:10000])
#sns.pairplot(df[['Engine_KW'] + target].iloc[:10000])
price_mask = df['Price_USD'] < 100000
df_temp = df[price_mask].copy()
sns.pairplot(df_temp[['Engine_KW'] + target])


# # Feature generation

# In[ ]:


additional_numeric_features = []


# # Feature selection
# 
# You can read about feature selection here
# https://scikit-learn.org/stable/modules/feature_selection.html#

# The dataset is fairly clean with a moderate number of features available. For a start, I did not have to worry too much about missing values in the dataset. A few things I tried out:
# - I tried dropping Nameplate because it has too many categories and when combined with one hot encoding, could make your dataset really sparse. Dropping Nameplate certainly results in superior results when paired with Gradient Boosting algorithms, though in the end I add the feature back in since I decided to use Catboost, which takes care of categorical features really well.
# - I tried setting year and generation year to be categorical instead of numerical, as they should be. However, I did not notice any improvement. 
# - There are many highly correlated features which should/could be dropped to reduce the number of dimensions. I have not tried going down the PCA path but would expect an improvement at least for Linear Regression.

# ## Drop features (optional)

# In[ ]:


features_drop = []

if ml_model_type == 'Linear Regression':
    features_drop = categorical_features + numeric_features
    features_to_use = ['Engine_KW']
    # features_to_use = ['country_name', 'Engine_KW']
    for feature in features_to_use:
        features_drop.remove(feature)
#else:
    #features_drop = categorical_features + numeric_features
    #features_to_use = ['Brand', 'country_name', 'Engine_KW']
    #features_to_use = ['country_name', 'Engine_KW']
    #for feature in features_to_use:
        #features_drop.remove(feature)
    #features_drop = ['Nameplate']
    

categorical_features = list(filter(lambda f: f not in features_drop, categorical_features))
numeric_features = list(filter(lambda f: f not in features_drop, numeric_features))


# In[ ]:


features = categorical_features + numeric_features + additional_numeric_features
model_columns = features + [target_name]
len(model_columns)


# #  ML data preparation

# In[ ]:


#dataframe for further processing
df_proc = df[model_columns].copy()
df_proc.shape


# ## Categorical feature encoding

# I did not use one hot encoding, and instead let Catboost take care of the categorical variables.

# In[ ]:


# One-hot encoding
#encoder = ce.OneHotEncoder(cols=categorical_features, handle_unknown='value', 
                           #use_cat_names=True)
#encoder.fit(df_proc)
#df_comb_ext = encoder.transform(df_proc)
#features_ext = list(df_comb_ext.columns)
#features_ext.remove(target_name)


# In[ ]:


#del df_proc
#df_comb_ext.head()


# In[ ]:


# df_comb_ext.memory_usage(deep=True).sum()/1e9
#features_model
#df_comb_ext.shape


# ## Train test split

# In[ ]:


#X_train, X_test, y_train, y_test = train_test_split(df_comb_ext[features_ext], df_comb_ext[target_name], 
                                                    #test_size=test_size, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(df_proc[features], df_proc[target_name], 
                                                    test_size=test_size, random_state=random_state)
print(X_train.shape)
print(X_test.shape)


# Scaling did not seem to improve any performance, which I think could be because I was using a tree-based model, where scaling isn't exactly a deal breaker.

# In[ ]:


#scaler = MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.fit_transform(X_test)


# # Machine learning model
# 
# Supervised learning
# 
# https://scikit-learn.org/stable/supervised_learning.html
# 
# Ensemble methods in scikit learn
# 
# https://scikit-learn.org/stable/modules/ensemble.html
# 
# 
# Decision trees
# 
# https://scikit-learn.org/stable/modules/tree.html
# 

# Before deciding on a model to use, I try running the dataset through a handful of out of the box sklearn algorithms. Uncomment this cell if you want to check out the results. Gradient Boosting gave me the best result i.e. lowest MSE.

# In[ ]:


#pipelines = []
#pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
#pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
#pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
#pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
#pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

#results = []
#names = []
#for name, model in pipelines:
    #cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    #results.append(cv_results)
    #names.append(name)
    #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #print(msg)


# ##  Model definition

# In[ ]:


if ml_model_type == 'Linear Regression':
    model_hyper_parameters_dict = OrderedDict(fit_intercept=True, normalize=False)
    regressor =  LinearRegression(**model_hyper_parameters_dict)

if ml_model_type == 'Decision Tree':
    model_hyper_parameters_dict = OrderedDict(max_depth=3, random_state=random_state)
    regressor =  DecisionTreeRegressor(**model_hyper_parameters_dict)
      
if ml_model_type == 'Random Forest':

    model_hyper_parameters_dict = OrderedDict(n_estimators=10, 
                                              max_depth=4, 
                                              min_samples_split=2, 
                                              max_features='sqrt',
                                              min_samples_leaf=1, 
                                              random_state=random_state, 
                                              n_jobs=4)
   
    regressor = RandomForestRegressor(**model_hyper_parameters_dict)

if ml_model_type == 'Gradient Boosting Regressor':
    model_hyper_parameters_dict = OrderedDict(learning_rate=0.1,
                                              max_depth=6,
                                              subsample=0.8,
                                              max_features=0.2,
                                             n_estimators=200,
                                             random_state=random_state)
    regressor = GradientBoostingRegressor(**model_hyper_parameters_dict)
    

if ml_model_type == 'AdaBoost':
    model_hyper_parameters_dict = OrderedDict(n_estimators=180,
                                             random_state=random_state)
    regressor = AdaBoostRegressor(**model_hyper_parameters_dict)
    
    
    
if ml_model_type == 'XGBoost':
    model_hyper_parameters_dict = OrderedDict(learning_rate=0.01,
                                              colsample_bytree=0.3,
                                              max_depth=3,
                                              subsample=0.8,
                                              n_estimators=1000,
                                              seed=random_state)
    
    regressor = xgb.XGBRegressor(**model_hyper_parameters_dict)
    

if ml_model_type == 'CatBoost':
    model_hyper_parameters_dict = OrderedDict(iterations=4000,
                                              early_stopping_rounds=50,
                                              learning_rate=0.05,
                                              depth=12,
                                              one_hot_max_size=40,
                                              colsample_bylevel=0.5,
                                              bagging_temperature=12,
                                              random_strength=0.7,
                                              reg_lambda=1.0,
                                              eval_metric='RMSE',
                                              logging_level='Silent',
                                              random_seed = random_state)
    
    regressor = CatBoostRegressor(**model_hyper_parameters_dict)    
    
base_regressor = clone(regressor)
    
if do_grid_search_cv:
    
    scoring = make_scorer(metrics_dict_res[regression_metric], greater_is_better=scoring_greater_is_better)
    
    if ml_model_type == 'Random Forest':
        

        grid_parameters = [{'n_estimators': [10], 'max_depth': [3, 5, 10], 
                             'min_samples_split': [2,4], 'min_samples_leaf': [1]} ]
        
    if ml_model_type == 'XGBoost':
        
        grid_parameters = [{'colsample_bytree':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]}]
        
    if ml_model_type == 'CatBoost':
        
        grid_parameters = [{'learning_rate': [0.1, 0.3, 0.5, 0.8]}]
        
    n_splits = 10
    n_jobs = 4
    cv_regressor = GridSearchCV(regressor, grid_parameters, cv=n_splits, scoring=scoring, return_train_score=True,
                                refit=True, n_jobs=n_jobs)    


# In[ ]:


# Use XGBoost API Learner. Comment the whole cell out if you want to use other models
#DM_train = xgb.DMatrix(data=X_train,label=y_train)
#DM_test =  xgb.DMatrix(data=X_test,label=y_test)
#params = {"booster":"gblinear", "objective":"reg:linear"}
#xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)


# ## ML model training

# In[ ]:


if do_grid_search_cv:
    cv_regressor.fit(X_train, y_train, cat_features=categorical_features)
    regressor_best = cv_regressor.best_estimator_
    model_hyper_parameters_dict = cv_regressor.best_params_
    train_scores = cv_regressor.cv_results_['mean_train_score']
    test_scores = cv_regressor.cv_results_['mean_test_score']
    test_scores_std = cv_regressor.cv_results_['std_test_score']
    cv_results = cv_regressor.cv_results_

elif ml_model_type == 'CatBoost':
    regressor.fit(X_train, y_train, cat_features=categorical_features)
    
else:
    regressor.fit(X_train, y_train)


# In[ ]:


if do_grid_search_cv:
    # print(cv_results)
    print(model_hyper_parameters_dict)
    plt.plot(-train_scores, label='train')
    plt.plot(-test_scores, label='test')
    plt.xlabel('Parameter set #')
    plt.legend()
    regressor = regressor_best


# # Model evaluation

# ## Train, test predictions

# In[ ]:


y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)
#y_train_pred = xg_reg.predict(DM_train)
#y_test_pred = xg_reg.predict(DM_test)


# ## Regression coefficients/Feature importance

# In[ ]:


if ml_model_type == 'Linear Regression':
    df_reg_coef = (pd.DataFrame(zip(['intercept'] + list(X_train.columns), 
                               [regressor.intercept_] + list(regressor.coef_)))
                 .rename({0: 'feature', 1: 'coefficient value'}, axis=1))
    display(df_reg_coef)


# In[ ]:


if hasattr(regressor, 'feature_importances_'):
    sns.set(style='whitegrid', font_scale=1.5)
    plt.figure(figsize=(12,10))
    plot_feature_importances(regressor, features_ext, n_features=np.minimum(20, X_train.shape[1]))


# ## Metrics

# In[ ]:


df_regression_metrics = regression_metrics_yin(y_train, y_train_pred, y_test, y_test_pred,
                                               metrics_dict_res, format_digits=3)

df_output = df_regression_metrics.copy()
df_output.loc['Counts','train'] = len(y_train)
df_output.loc['Counts','test'] = len(y_test)
df_output


# ##  Model Performance plots

# In[ ]:


figsize = (16,10)
xlim = [0, 250]
font={'size': 20}
sns.set(style='whitegrid', font_scale=2.5)
act_label = 'actual price [k$]'
pred_label='predicted price [k$]'
plot_act_vs_pred(y_test, y_test_pred, scale=1000, act_label=act_label, pred_label=pred_label, 
                 figsize=figsize, xlim=xlim, ylim=xlim, font=font)
print()


# In[ ]:


figsize = (14,8)
xlim = [0, 100]
#xlim = [-100, 100]
# xlim = [-50, 50]
#xlim = [-20, 20]

font={'size': 20}
sns.set(style='whitegrid', font_scale=1.5)

p_error = (y_test_pred - y_test)/y_test *100
df_p_error = pd.DataFrame(p_error.values, columns=['percent_error'])
#display(df_p_error['percent_error'].describe().to_frame())

bins=1000
bins=500
#bins=100
absolute = True
#absolute = False
plot_dev_distribution(y_test, y_test_pred, absolute=absolute, figsize=figsize, 
                      xlim=xlim, bins=bins, font=font)
print()


# ## Optionally retrain on the whole data set

# In[ ]:


if do_retrain_total:
    cv_opt_model = clone(base_regressor.set_params(**model_hyper_parameters_dict))
    # train on complete data set
    #X_train_full = df_comb_ext[features_ext].copy()
    #y_train_full = df_comb_ext[target_name].values
    X_train_full = df_proc[features].copy()
    y_train_full = df_proc[target_name].values
    #cv_opt_model.fit(X_train_full, y_train_full) 
    cv_opt_model.fit(X_train_full, y_train_full, cat_features=categorical_features) 
    regressor = cv_opt_model


# # Apply model to OOS data

# In[ ]:


# df_oos.head()


# ## Subset to relevant columns

# In[ ]:


df_proc_oos = df_oos[model_columns[:-1]].copy()
df_proc_oos[target_name] = 1


# In[ ]:


df_proc_oos


# In[ ]:


df_proc_oos.drop(target_name, axis=1, inplace=True)


# ## Apply categorical encoding

# Again, not needed as I eventually go with Catboost.

# In[ ]:


#df_comb_ext_oos = encoder.transform(df_proc_oos)


# In[ ]:


#df_comb_ext_oos.drop(target_name, axis=1, inplace=True)
#df_comb_ext_oos = scaler.fit_transform(df_comb_ext_oos)


# ## Apply model and produce output

# In[ ]:


#y_oos_pred = regressor.predict(df_comb_ext_oos)
y_oos_pred = regressor.predict(df_proc_oos)


# In[ ]:


id_col = 'vehicle_id'
df_out = (pd.DataFrame(y_oos_pred, columns=[target_name], index=df_proc_oos.index)
            .reset_index()
            .rename({'index': id_col}, axis=1))


# In[ ]:


df_out.head()


# In[ ]:


df_out.shape


# In[ ]:


if write_predictions_file:
    df_out.to_csv('submission.csv', index=False)


# ## Conclusion

# All in all, I was very impressed with CatBoost's performance. I did not have a chance to try out LightGBM, but comparing to XGBoost and Stochastic Gradient Boosting, CatBoost gave superior results with pretty consistent results between training and testing data, and performance is comparable (albeit I did not track closely). 
# 
# Next step would be to play around with the hyperparameters tuning -- With the time constraints, I did not have time to try to tinker all the parameters. My three cents:
# - CatBoost deals with categorical features pretty well and I encourage you to make use of that. The one_hot_max_size parameter also allows you to define what is the maximum number of categories. Too many categories will decrease performance of the algorithm.
# - Strike a balance between iterations and learning rate. My suggestion would be to start with a high number of iterations (so you could make use of early stopping) and small learning rate, and then slowly increase learning rate while reducing iterations.

# In[ ]:




