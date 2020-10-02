#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Import-libraries" data-toc-modified-id="Import-libraries-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import libraries</a></span></li><li><span><a href="#Locally-defined-functions" data-toc-modified-id="Locally-defined-functions-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Locally defined functions</a></span><ul class="toc-item"><li><span><a href="#Metrics" data-toc-modified-id="Metrics-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Metrics</a></span></li><li><span><a href="#Display-functions" data-toc-modified-id="Display-functions-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Display functions</a></span></li><li><span><a href="#Define-features" data-toc-modified-id="Define-features-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Define features</a></span></li></ul></li><li><span><a href="#Global-options" data-toc-modified-id="Global-options-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Global options</a></span></li><li><span><a href="#Load--data" data-toc-modified-id="Load--data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Load  data</a></span><ul class="toc-item"><li><span><a href="#Training-data" data-toc-modified-id="Training-data-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Training data</a></span></li><li><span><a href="#Out-of-sample-data-(to-predict)" data-toc-modified-id="Out-of-sample-data-(to-predict)-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Out of sample data (to predict)</a></span></li></ul></li><li><span><a href="#Feature-exploration" data-toc-modified-id="Feature-exploration-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Feature exploration</a></span><ul class="toc-item"><li><span><a href="#Categorical-features" data-toc-modified-id="Categorical-features-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Categorical features</a></span></li><li><span><a href="#Numerical-features" data-toc-modified-id="Numerical-features-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Numerical features</a></span></li><li><span><a href="#Pair-plot" data-toc-modified-id="Pair-plot-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Pair plot</a></span></li></ul></li><li><span><a href="#Feature-generation" data-toc-modified-id="Feature-generation-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Feature generation</a></span></li><li><span><a href="#Feature-selection" data-toc-modified-id="Feature-selection-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Feature selection</a></span><ul class="toc-item"><li><span><a href="#Drop-features-(optional)" data-toc-modified-id="Drop-features-(optional)-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Drop features (optional)</a></span></li></ul></li><li><span><a href="#ML-data-preparation" data-toc-modified-id="ML-data-preparation-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>ML data preparation</a></span><ul class="toc-item"><li><span><a href="#Categorical-feature-encoding" data-toc-modified-id="Categorical-feature-encoding-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Categorical feature encoding</a></span></li><li><span><a href="#Train-test-split" data-toc-modified-id="Train-test-split-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Train test split</a></span></li></ul></li><li><span><a href="#Machine-learning-model" data-toc-modified-id="Machine-learning-model-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Machine learning model</a></span><ul class="toc-item"><li><span><a href="#Model-definition" data-toc-modified-id="Model-definition-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Model definition</a></span></li><li><span><a href="#ML-model-training" data-toc-modified-id="ML-model-training-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>ML model training</a></span></li></ul></li><li><span><a href="#Model-evaluation" data-toc-modified-id="Model-evaluation-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Model evaluation</a></span><ul class="toc-item"><li><span><a href="#Train,-test-predictions" data-toc-modified-id="Train,-test-predictions-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>Train, test predictions</a></span></li><li><span><a href="#Regression-coefficients/Feature-importance" data-toc-modified-id="Regression-coefficients/Feature-importance-10.2"><span class="toc-item-num">10.2&nbsp;&nbsp;</span>Regression coefficients/Feature importance</a></span></li><li><span><a href="#Metrics" data-toc-modified-id="Metrics-10.3"><span class="toc-item-num">10.3&nbsp;&nbsp;</span>Metrics</a></span></li><li><span><a href="#Model-Performance-plots" data-toc-modified-id="Model-Performance-plots-10.4"><span class="toc-item-num">10.4&nbsp;&nbsp;</span>Model Performance plots</a></span></li><li><span><a href="#Optionally-retrain-on-the-whole-data-set" data-toc-modified-id="Optionally-retrain-on-the-whole-data-set-10.5"><span class="toc-item-num">10.5&nbsp;&nbsp;</span>Optionally retrain on the whole data set</a></span></li></ul></li><li><span><a href="#Apply-model-to-OOS-data" data-toc-modified-id="Apply-model-to-OOS-data-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Apply model to OOS data</a></span><ul class="toc-item"><li><span><a href="#Subset-to-relevant-columns" data-toc-modified-id="Subset-to-relevant-columns-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span>Subset to relevant columns</a></span></li><li><span><a href="#Apply-categorical-encoding" data-toc-modified-id="Apply-categorical-encoding-11.2"><span class="toc-item-num">11.2&nbsp;&nbsp;</span>Apply categorical encoding</a></span></li><li><span><a href="#Apply-model-and-produce-output" data-toc-modified-id="Apply-model-and-produce-output-11.3"><span class="toc-item-num">11.3&nbsp;&nbsp;</span>Apply model and produce output</a></span></li></ul></li></ul></div>

# # Import libraries

# In[ ]:


# this may need to be installed separately with
# !pip install category-encoders
import category_encoders as ce

# python general
import pandas as pd
import numpy as np
from collections import OrderedDict

#scikit learn

import sklearn
from sklearn.base import clone
from sklearn.datasets import load_iris

#feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import make_regression
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel


# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# ML models
from sklearn import tree
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# error metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

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


# # Locally defined functions

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
ml_model_type = 'XG Boost'

regression_metric = 'mean abs perc error'

do_grid_search_cv = True
scoring_greater_is_better = False  # Because the scoring is being done for loss (error) function 

do_retrain_total = True
write_predictions_file = True

# relative size of test set
test_size = 0.20
random_state = 33


# # Load  data
# 

# ## Training data

# In[ ]:


df = pd.read_csv('/kaggle/input/ihsmarkit-hackathon-june2020/train_data.csv',index_col='vehicle_id')
df['date'] = pd.to_datetime(df['date'])
#df['Age'] = df['year'] - df['Generation_Year']
#len(df[df['Age'] < 0])


# In[ ]:


#df[df['Age'] < 0].shape


# In[ ]:


# basic commands on a dataframe
#df.info()
#df.head(5)
#df.shape
# df.head()
df.nunique(axis=0)
# df.tail()


# In[ ]:


df.tail(5)


# In[ ]:


df['country_name'].value_counts()
df['Nameplate'].value_counts()
df['Body_Type'].value_counts()


# In[ ]:


df[df['Brand'] == 'volkswagen'].groupby(['Brand', 'Nameplate'])['date'].count()


# ## Out of sample data (to predict)

# In[ ]:


df_oos = pd.read_csv('/kaggle/input/ihsmarkit-hackathon-june2020/oos_data.csv', index_col='vehicle_id')
df_oos['date'] = pd.to_datetime(df_oos['date'])
df_oos['year'] = df_oos['date'].map(lambda d: d.year)
#df_oos['Age'] = df_oos['year'] - df_oos['Generation_Year']


# In[ ]:


# df_oos.shape
df_oos.head()


# In[ ]:


df_oos.nunique(axis=0)


# In[ ]:


df_oos[df_oos['Brand'] == 'volkswagen'].groupby(['Brand', 'Nameplate'])['date'].count().sort_values(ascending=False)


# In[ ]:


df['country_name'].value_counts()
df_oos['Brand'].value_counts()
#df_oos['Body_Type'].value_counts()


# In[ ]:


count = df_oos.groupby(['Brand', 'Nameplate'])['date'].count()
count.sort_values(ascending=False)


# In[ ]:


df_oos.groupby(['year', 'country_name'])['date'].count()
c=df_oos.groupby(['Body_Type', 'No_of_Gears'])['date'].count()
c.sort_values(ascending=False)


# In[ ]:


df_oos.loc[~df_oos["Nameplate"].isin(df["Nameplate"])]['Nameplate'].unique()
df_oos.loc[~df_oos["Body_Type"].isin(df["Body_Type"])]['Body_Type'].unique()


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


df.sort_values('Price_USD',ascending=False).head(10)


# In[ ]:


# data an outlier with high price as seen from the summary above where mean and median of Price_USD are far apart 

df[df.Brand == 'koenigsegg']
df = df[df.Brand != 'koenigsegg']
df.shape

#How can car be generated in 2018 when exact price is being captured in 2016. Negative differences between age and year are unexpected so remove noisy

df = df[(df.year - df.Generation_Year) >= 0 ]
df.shape 

#There are duplicates!
df = df.drop_duplicates()
df.shape


# In[ ]:


figsize = (16,12)
sns.set(style='whitegrid', font_scale=2)

bins = 1000
bins = 40
#xlim = [0,100000]
xlim = None
#price_mask = df['Price_USD'] < 50000000
#interactive(lambda col: plot_distribution(df[price_mask], col, bins=bins, xlim=xlim), col=sorted(all_numeric_features + target))
interactive(lambda col: plot_distribution(df, col, bins=bins, xlim=xlim), col=sorted(all_numeric_features + target))


# ## Pair plot

# In[ ]:


# this is quite slow
sns.set(style='whitegrid', font_scale=1)
sns.pairplot(df[numeric_features[0:10] + target].iloc[:10000])
#sns.pairplot(df[['Engine_KW'] + target].iloc[:10000])


# * Sup thoughts : Length and Width are quite correlated , can choose to drop width if using regression . Curb weight?
# * CO2 & Fuel_cons_combined coorelated ofcourse former is just city.

# In[ ]:


price_mask = df['Price_USD'] < 100000
df_temp = df[price_mask].copy()
sns.pairplot(df_temp[['Generation_Year'] + target])


# In[ ]:


plt.scatter(df['No_of_Gears'],df[target])
plt.show()


# In[ ]:


plt.scatter(df['year'],df[target])
plt.show()


# In[ ]:


price_mask = df['Price_USD'] < 100000
df_temp = df[price_mask].copy()
#sns.pairplot(df_temp[['Generation_Year'] + target])
#plt.scatter(df_temp['Age'],df_temp[target])
#plt.show()


# In[ ]:


plt.scatter(df['Body_Type'],df['Global_Sales_Sub-Segment'])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.scatter(df['Body_Type'],df[target])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.scatter(df['PropSysDesign'],df[target])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.scatter(df['Transmission'],df[target])
plt.xticks(rotation=90)
plt.show()


# In[ ]:



#df_temp = df[price_mask].copy()
#plt.scatter(df_temp['Generation_Year'],df_temp['Nameplate'])
#plt.xticks(rotation=90)
#plt.show()
#'Nameplate','Body_Type'


# In[ ]:


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# # Feature generation

# In[ ]:


#additional_numeric_features = ['Age']
additional_numeric_features = []


# # Feature selection
# 
# You can read about feature selection here
# https://scikit-learn.org/stable/modules/feature_selection.html#

# In[ ]:


fs = SelectKBest(score_func=f_regression, k=6)
# apply feature selection
X_selected = fs.fit_transform(df[numeric_features], df[target_name])
print(X_selected.shape)


# In[ ]:


print(X_selected)


# In[ ]:


print(df[numeric_features].head())


# In[ ]:





# ## Drop features (optional)

# In[ ]:


features_drop = []

if ml_model_type == 'Linear Regression':
    features_drop = categorical_features + numeric_features
    #features_to_use = ['Engine_KW']
    features_to_use = ['country_name', 'Engine_KW']
    for feature in features_to_use:
        features_drop.remove(feature)
else:
    features_drop = categorical_features + numeric_features
    features_to_use = ['Generation_Year','Turbo','Global_Sales_Sub-Segment','Brand', 'country_name', 'Length','Height','Width','Curb_Weight','Engine_KW','Body_Type','Fuel_cons_combined','No_of_Gears','Transmission','PropSysDesign']
    #features_to_use = ['Generation_Year','Length','Height','Width','Engine_KW','No_of_Gears','Curb_Weight','Fuel_cons_combined','year']
    for feature in features_to_use:
        features_drop.remove(feature)
    # features_drop = ['Nameplate']
    

categorical_features = list(filter(lambda f: f not in features_drop, categorical_features))
numeric_features = list(filter(lambda f: f not in features_drop, numeric_features))


# In[ ]:


numeric_features


# In[ ]:


categorical_features


# #  ML data preparation

# In[ ]:


features = categorical_features + numeric_features + additional_numeric_features
model_columns = features + [target_name]
model_columns


# In[ ]:


#dataframe for further processing
df_proc = df[model_columns].copy()
df_proc.shape


# ## Categorical feature encoding

# In[ ]:


# One-hot encoding
encoder = ce.OneHotEncoder(cols=categorical_features, handle_unknown='value', 
                           use_cat_names=True)
encoder.fit(df_proc)
df_comb_ext = encoder.transform(df_proc)
features_ext = list(df_comb_ext.columns)
features_ext.remove(target_name)


# In[ ]:


#del df_proc
df_comb_ext.head()


# In[ ]:


#df_comb_ext.memory_usage(deep=True).sum()/1e9
#features_model
df_comb_ext.shape


# ## Train test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_comb_ext[features_ext], df_comb_ext[target_name], 
                                                    test_size=test_size, random_state=random_state)

print(X_train.shape)
print(X_test.shape)


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

# ##  Model definition

# In[ ]:


#xgb cv for model params
#import xgboost as xgb
#from sklearn.metrics import mean_absolute_error
#fixed_params = {'max_depth': 10,
#                     'min_child_weight': 1, 'learning_rate' : 0.3}
#dtrain = xgb.DMatrix(X_train, label=y_train)
#dtest = xgb.DMatrix(X_test, label=y_test)

#params = {'max_depth': 10,'min_child_weight': 1, 'learning_rate' : 0.2, 'colsample_bytree': 0.8, 'subsample': 1}
#params['eval_metric'] = "mae"

#cvresult = xgb.train(params, dtrain, num_boost_round=999,early_stopping_rounds=10,evals=[(dtest, "Test")])
#cvresult


# In[ ]:


#print("Best MAE: {:.2f} with {} rounds".format(
#                 cvresult.best_score,
#                 cvresult.best_iteration+1))


# In[ ]:


#cv_results['test-mae-mean'].min()


# In[ ]:


if ml_model_type == 'Linear Regression':
    model_hyper_parameters_dict = OrderedDict(fit_intercept=True, normalize=False)
    regressor =  LinearRegression(**model_hyper_parameters_dict)

if ml_model_type == 'Decision Tree':
    model_hyper_parameters_dict = OrderedDict(max_depth=3, random_state=random_state)
    regressor =  DecisionTreeRegressor(**model_hyper_parameters_dict)
      
if ml_model_type == 'Random Forest':

    model_hyper_parameters_dict = OrderedDict(n_estimators=30, 
                                              max_depth=10, 
                                              min_samples_split=2, 
                                              max_features='sqrt',
                                              min_samples_leaf=5, 
                                              random_state=random_state, 
                                              n_jobs=2)
   
    regressor = RandomForestRegressor(**model_hyper_parameters_dict)
        
if ml_model_type == 'XG Boost':
    
    model_hyper_parameters_dict = OrderedDict(n_estimators=500,
                                              max_depth=10,
                                              learning_rate=0.3, min_child_weight=1)
    
    regressor = XGBRegressor(**model_hyper_parameters_dict)
    
base_regressor = clone(regressor)
     
     
if do_grid_search_cv:
    
    scoring = make_scorer(metrics_dict_res[regression_metric], greater_is_better=scoring_greater_is_better)
    
    if ml_model_type == 'Random Forest':
        

        grid_parameters = [{'n_estimators': [10,30,100,1000], 'max_depth': [5, 10], 
                             'min_samples_split': [1,2], 'min_samples_leaf': [1,5],
                              'max_features' : ['sqrt','auto']}]

    if ml_model_type == 'XG Boost':  
        
        grid_parameters = [{"n_estimators" : [500],'max_depth': [10],
                             'min_child_weight': [1], 'learning_rate' : [0.3]}]
   
        
    n_splits = 5
    n_jobs = 4
    cv_regressor = GridSearchCV(regressor, grid_parameters, cv=n_splits, scoring=scoring, return_train_score=True,
                                refit=True, n_jobs=n_jobs)    
        


# ## ML model training

# In[ ]:


if do_grid_search_cv:
    cv_regressor.fit(X_train, y_train)
    regressor_best = cv_regressor.best_estimator_
    model_hyper_parameters_dict = cv_regressor.best_params_
    train_scores = cv_regressor.cv_results_['mean_train_score']
    test_scores = cv_regressor.cv_results_['mean_test_score']
    test_scores_std = cv_regressor.cv_results_['std_test_score']
    cv_results = cv_regressor.cv_results_
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

# In[ ]:


#params = { "n_estimators" : 100, 'max_depth': 10,
#                             'min_child_weight': 1, 'learning_rate' : 0.3, 
#                             'colsample_bytree' : 0.4,
#                             'alpha' : 5}

#cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
 #                   num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

#print((cv_results["test-rmse-mean"]).tail(1))


# ## Train, test predictions

# In[ ]:


y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)


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
    X_train_full = df_comb_ext[features_ext].copy()
    y_train_full = df_comb_ext[target_name].values
    cv_opt_model.fit(X_train_full, y_train_full) 
    regressor = cv_opt_model


# # Apply model to OOS data

# In[ ]:


df_oos.head()


# ## Subset to relevant columns

# In[ ]:


df_proc_oos = df_oos[model_columns[:-1]].copy()
df_proc_oos[target_name] = 1


# ## Apply categorical encoding

# In[ ]:


df_comb_ext_oos = encoder.transform(df_proc_oos)


# In[ ]:


df_comb_ext_oos.drop(target_name, axis=1, inplace=True)


# ## Apply model and produce output

# In[ ]:


y_oos_pred = regressor.predict(df_comb_ext_oos)


# In[ ]:


id_col = 'vehicle_id'
df_out = (pd.DataFrame(y_oos_pred, columns=[target_name], index=df_comb_ext_oos.index)
            .reset_index()
            .rename({'index': id_col}, axis=1))


# In[ ]:


df_out.head()


# In[ ]:


df_out.shape


# In[ ]:


if write_predictions_file:
    df_out.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




