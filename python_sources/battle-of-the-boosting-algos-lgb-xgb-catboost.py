#!/usr/bin/env python
# coding: utf-8

# # The Goal
# 
# ## What're we doing?
# We're going to let XGBoost, LightGBM and Catboost battle it out in 3 rounds:
# 
# - **Classification:** Classify images in the Fashion MNIST (60,000 rows, 784 features)
# 
# - **Regression:** Predict NYC Taxi fares (60,000 rows, 7 features)
# 
# - **Massive Dataset:** Predict NYC Taxi fares (2 million rows, 7 features)
# 
# 
# ## How're we doing it?
# In each round here are the steps we'll follow:
# 1. Train baseline models of XGBoost, Catboost, LightGBM (trained using the same paramaters for each model)
# 2. Train fine-tuned models of XGBoost, Catboost, LightGBM using GridSearchCV
# 3. Measure performance on the following metrics:
#     - training and prediction times
#     - prediction score
#     - interpretability (feature importance, shap values, visualize trees)
# 
# ## What does this all mean?
# A detailed analysis of the results can be found on my blog at: https://lavanya.ai/2019/06/27/battle-of-the-boosting-algorithms/
# 
# I hope you find this analysis useful, I would love to hear your feedback on improving it! I encourage you to fork this kernel, and play with the code.
# 
# If you like this kernel, please give it an upvote. Thank you!

# # Setup

# In[ ]:


# Essentials
import numpy as np
import pandas as pd
import random
import time
import gc
import os
from datetime import datetime

# Plots
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
import seaborn as sns
from matplotlib.pylab import rcParams
##set up the parameters
rcParams['figure.figsize'] = 80,60

# Models
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from mlxtend.classifier import StackingCVClassifier
from mlxtend.regressor import StackingCVRegressor
from lightgbm.plotting import plot_importance
import lightgbm
import xgboost as xgb
import catboost
from xgboost import plot_tree

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


# Misc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from dateutil import tz
from geopy import distance
import shap

pd.set_option('display.max_columns', None)

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

print(os.listdir("../input/"))


# In[ ]:


def show_time(diff):
   m, s = divmod(diff, 60)
   h, m = divmod(m, 60)
   s,m,h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
   print("Execution Time: " + "{0:02d}:{1:02d}:{2:02d}".format(h, m, s))


# In[ ]:


# Takes in a classifier, calculates the training + prediction times and accuracy score, returns a model
def Train(clf, X, y, X_predict, y_predict, type='classification'):
    # Train
    start = time.time()
    model = clf.fit(X,y)
    end = time.time()
    print('Training time: ')
    show_time(end - start)
    training_times.append(end - start)

    # Predict
    start = time.time()
    if(type=='classification'):
        scores.append(accuracy_score(y_predict, model.predict(X_predict)))
    else:
        scores.append(rmse(y_test, model.predict(X_test)))
    end = time.time()
    prediction_times.append(end - start)
    print('\nPrediction time: ')
    show_time(end - start)
    return model

# Takes in a classifier, calculates the training + prediction times and accuracy score, returns a model
def GridSearch(clf, params, X, y, X_predict, y_predict, type='classification'):
    # Train
    start = time.time()
    if(type=='classification'):
        model = GridSearchCV(clf, params, scoring='accuracy', n_jobs=-1, cv=5).fit(X,y).best_estimator_
    else:
        model = GridSearchCV(clf, params, scoring='r2', n_jobs=-1, cv=5).fit(X,y).best_estimator_
    end = time.time()
    print('Training time: ')
    show_time(end - start)
    training_times.append(end - start)

    # Predict
    start = time.time()
    if(type=='classification'):
        scores.append(accuracy_score(y_predict, model.predict(X_predict)))
    else:
        scores.append(rmse(y_test, model.predict(X_test)))
    end = time.time()
    prediction_times.append(end - start)
    print('Prediction time: ')
    show_time(end - start)
    return model


# In[ ]:


# Takes in model scores and plots them on a bar graph
def plot_metric(model_scores, score='Accuracy'):
    # Set figure size
    rcParams['figure.figsize'] = 7,5
    plt.bar(model_scores['Model'], height=model_scores[score])
    xlocs, xlabs = plt.xticks()
    xlocs=[i for i in range(0,6)]
    xlabs=[i for i in range(0,6)]
    if(score != 'Prediction Times'):
        for i, v in enumerate(model_scores[score]):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.xlabel('Model')
    plt.ylabel(score)
    plt.xticks(rotation=45)
    plt.show()


# In[ ]:


# Takes in training data and a model, and plots a bar graph of the model's feature importances
def feature_importances(df, model, model_name, max_num_features=10):
    feature_importances = pd.DataFrame(columns = ['feature', 'importance'])
    feature_importances['feature'] = df.columns
    feature_importances['importance'] = model.feature_importances_
    feature_importances.sort_values(by='importance', ascending=False, inplace=True)
    feature_importances = feature_importances[:max_num_features]
    # print(feature_importances)
    plt.figure(figsize=(12, 6));
    sns.barplot(x="importance", y="feature", data=feature_importances);
    plt.title(model_name+' features importance:');

# Takes in training data and a model, and plots a bar graph of SHAP values
def shap_values(df, model, model_name):
    shap_values = shap.TreeExplainer(model).shap_values(df)
    shap_values[:5]
    shap.summary_plot(shap_values, df.iloc[:1000,:])


# # Round 1: Classification: Classify images in the Fashion MNIST

# ## A. Explore the Fashion MNIST dataset (60000 rows, 784 features)

# In[ ]:


# Read in dataset
fetch_from = '../input/fashionmnist/fashion-mnist_train.csv'
train = pd.read_csv(fetch_from)

fetch_from = '../input/fashionmnist/fashion-mnist_test.csv'
test = pd.read_csv(fetch_from)


# In[ ]:


# Perform train-test split
X_train, y_train, X_test, y_test = train.iloc[:,1:], train['label'], test.iloc[:,1:], test['label']
X_train.head()


# In[ ]:


X_train.shape, X_test.shape
# Each image is 28*28(=784) pixels, hence the 784 features


# In[ ]:


# Sample some images in the dataset
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
plt.figure(figsize=(9,9))
example_images = X_train[:100]
plot_digits(example_images.values, images_per_row=10)
plt.show()


# ## B. Baseline models

# In[ ]:


prediction_times = []
training_times = []
scores = []
# training_times
# prediction_times


# ### XGBoost

# In[ ]:


xgboost = Train(XGBClassifier(n_estimators=50, max_depth=5), X_train, y_train, X_test, y_test)


# ### LightGBM

# In[ ]:


lgb = Train(LGBMClassifier(n_estimators=50, max_depth=5), X_train, y_train, X_test, y_test)


# ### CatBoost

# In[ ]:


cat = Train(CatBoostClassifier(n_estimators=50, verbose=False, max_depth=6), X_train, y_train, X_test, y_test)


# ## C. Fine-tuned models

# ### XGBoost

# In[ ]:


# XGBoost with GridSearch
param_grid=[{'learning_rate':[0.05,0.1,0.15],
            'max_depth':[5,10,20,50],
            'min_child_weight' : [1,3,6],
            'n_estimators':[200],
            'colsample_bytree':[0.7,0.85]}]
xgboost_gs = GridSearch(XGBClassifier(random_state=42), param_grid, X_train, y_train, X_test, y_test)


# ### LightGBM

# In[ ]:


# LightGBM with GridSearch
param_grid=[{'learning_rate':[0.05,0.1,0.15],
            'max_depth':[5,10,20,50],
            'min_child_weight' : [1,3,6],
            'num_leaves': [100,500,1200],
            'n_estimators':[200],
            'colsample_bytree':[0.7,0.85]}]
lgb_gs = GridSearch(LGBMClassifier(random_state=42), param_grid, X_train, y_train, X_test, y_test)


# ### CatBoost

# In[ ]:


# CatBoost with GridSearch
start = time.time()
param_grid=[{'learning_rate':[0.05,0.1,0.15],
            'depth': [5,10,20,50],
            'n_estimators':[200],
            'iterations': [500]
            'l2_leaf_reg': [1,4,9],
            'rsm':[0.5,0.8]}]
cat_gs = GridSearch(CatBoostClassifier(random_state=42, silent = True,
                        bootstrap_type = 'Bernoulli'), param_grid, X_train, y_train, X_test, y_test)


# ## D. The Results

# In[ ]:


# free up memory be deleting dataframes no longer needed
del [[y_train, X_test, y_test, train, test]]


# In[ ]:


models = [('XGBoost', xgboost),
         ('LightGBM', lgb),
         ('CatBoost', cat),
         ('XGBoost GridSearch', xgboost_gs),
         ('LightGBM GridSearch', lgb_gs),
         ('CatBoost GridSearch', cat_gs)]


# ### Accuracy Scores

# In[ ]:


model_scores = pd.DataFrame({ 'Model': [name for name, _ in models], 'Accuracy': scores })
model_scores.sort_values(by='Accuracy',ascending=False,inplace=True)
plot_metric(model_scores, score='Accuracy')


# ### Training and Prediction Times

# In[ ]:


training_times = [round(time,2) for time in training_times]
model_train_times = pd.DataFrame({ 'Model': [name for name, _ in models], 'Training Times': training_times })
plot_metric(model_train_times, score='Training Times')


# In[ ]:


prediction_times = [round(time,2) for time in prediction_times]
model_train_times = pd.DataFrame({ 'Model': [name for name, _ in models], 'Prediction Times': prediction_times })
plot_metric(model_train_times, score='Prediction Times')


# ### Interpretability

# A model's prediction score only paints a partial picture of its predictions. We also want to know *why* the model is making its predictions.
# 
# Here we plot the model's feature importances, SHAP values and draw an actual decision tree to get a firmer understanding of the model's predictions.

# #### Feature Importances

# In[ ]:


# XGBoost
feature_importances(X_train, xgboost, 'XGBoost')


# In[ ]:


# CatBoost
feature_importances(X_train, cat, 'CatBoost')


# In[ ]:


# LightGBM
feature_importances(X_train, lgb, 'LightGBM')


# #### SHAP Values

# Reference table for understanding which class names the class indexes refer to in the graphs below:
# 
# | Class | Name |
# | --- | --- |
# | 0 | T-shirt/top |
# | 1 | Trouser |
# | 2 | Pullover |
# | 3 | Dress |
# | 4 | Coat |
# | 5 | Sandal |
# | 6 | Shirt |
# | 7 | Sneaker |
# | 8 | Bag |
# | 9 | Ankle boot |

# In[ ]:


# XGBoost
shap_values(X_train.iloc[:500,:], xgboost, 'XGBoost')


# In[ ]:


# LightGBM
shap_values(X_train.iloc[:500,:], lgb, 'LightGBM')


# **CatBoost**
# CatBoost doesn't work out of the box with shap_values() and results in the kernel crashing.

# #### Visualize Trees

# In[ ]:


# Set figure size for decision tree plots
rcParams['figure.figsize'] = 80,50


# In[ ]:


# LightGBM
lightgbm.plot_tree(lgb);


# In[ ]:


# XGBoost
xgb.plot_tree(xgboost);


# **CatBoost**
# 
# CatBoost ships with no plotting function for its trees. If you really need to visualize CatBoost results, a work-around is proposed here: https://blog.csdn.net/l_xzmy/article/details/81532281

# In[ ]:


# Clear memory before moving onto the next round
import gc
gc.collect()


# In[ ]:


del [[X_train]]


# # Round 2: Regression on a medium Dataset: Predict NYC Taxi fares

# ## A. Explore the NYC Taxi dataset (60,000 rows, 6 features)

# In[ ]:


# Get data from New York City Taxi Fare Prediction
n = 60000
train = pd.read_csv('../input/nyctaxi/train.csv', nrows=n)
test = pd.read_csv('../input/nyctaxi/test.csv')
train.head(5)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


# Feature Engineering
# this cell was adapted from https://www.kaggle.com/mahtieu/nyc-taxi-fare-prediction-data-expl-xgboost
def feature_engineering(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    #Drop rows with null values
    df = df.dropna(how = 'any', axis = 'rows')
    #Free rides, negative fares and passenger count filtering
    df = df[df.eval('(fare_amount > 0) & (passenger_count <= 6)')]
    # Coordinates filtering - Pickup and dropoff locations should be within the limits of NYC
    df = df[(df.pickup_longitude >= -77) &
                  (df.pickup_longitude <= -70) &
                  (df.dropoff_longitude >= -77) &
                  (df.dropoff_longitude <= 70) &
                  (df.pickup_latitude >= 35) &
                  (df.pickup_latitude <= 45) &
                  (df.dropoff_latitude >= 35) &
                  (df.dropoff_latitude <= 45)]

    df.pickup_datetime = df.pickup_datetime.dt.tz_localize('UTC')
    df.pickup_datetime = df.pickup_datetime.dt.tz_convert(tz.gettz('America/New_York'))

    # Fares may change every year
    df['year'] = df.pickup_datetime.dt.year

    # Different fares during weekdays and weekends
    df['dayofweek'] = df.pickup_datetime.dt.dayofweek

    # Different fares during public holidays
    df['dayofyear'] = df.pickup_datetime.dt.dayofyear

    # Different fares in peak periods and off-peak periods
    df['hourofday'] = df.pickup_datetime.dt.hour

    df = df.drop('pickup_datetime', axis=1)

    # Computes the distance (in miles) between the pickup and the dropoff locations
    df['distance'] = df.apply(
        lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (x.dropoff_latitude, x.dropoff_longitude)).miles,
        axis = 1)

    df = df[df.eval('(distance > 0) & (distance < 150)')]
    fare_distance_ratio = (df.fare_amount/df.distance)
    fare_distance_ratio.describe()

    (fare_distance_ratio[fare_distance_ratio < 45]).hist()

    # Drop incoherent fares
    df = df[fare_distance_ratio < 45]
    del fare_distance_ratio

    # Coordinates of the 3 airpots of NYC
    airports = {'jfk': [40.6441666, -73.7822222],
                'laguardia': [40.7747222, -73.8719444],
                'newark': [40.6897222, -74.175]}

    # Computes the distance between the pickup location and the airport
    pickup = df.apply(lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (airports.get('jfk'))).miles, axis=1)
    # Computes the distance between the dropoff location and the airport
    dropoff = df.apply(lambda x: distance.distance((x.dropoff_latitude, x.dropoff_longitude), (airports.get('jfk'))).miles, axis=1)
    # Selects the shortest distance
    df['to_jfk'] = pd.concat((pickup, dropoff), axis=1).min(axis=1)

    pickup = df.apply(lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (airports.get('laguardia'))).miles, axis=1)
    dropoff = df.apply(lambda x: distance.distance((x.dropoff_latitude, x.dropoff_longitude), (airports.get('laguardia'))).miles, axis=1)
    df['to_laguardia'] = pd.concat((pickup, dropoff), axis=1).min(axis=1)

    pickup = df.apply(lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (airports.get('newark'))).miles, axis=1)
    dropoff = df.apply(lambda x: distance.distance((x.dropoff_latitude, x.dropoff_longitude), (airports.get('newark'))).miles, axis=1)
    df['to_newark'] = pd.concat((pickup, dropoff), axis=1).min(axis=1)
    del pickup, dropoff
    return df

def remove_sparse(df):
    features = [x for x in df.columns]
    for feature in features:
        if len(np.unique(df[feature]))<2:
            df.drop(feature, axis=1, inplace=True)
    return df


# In[ ]:


train = remove_sparse(train)
test = remove_sparse(test)
train = feature_engineering(train)
test = feature_engineering(test)
y_train = train.fare_amount
X_train = train.drop('fare_amount', axis=1)
y_test = test.fare_amount
X_test = test.drop('fare_amount', axis=1)


# In[ ]:


X_train.head(5)


# In[ ]:


X_test.head(5)


# ## B. Baseline models

# In[ ]:


prediction_times = []
training_times = []
scores = []
# training_times
# prediction_times


# In[ ]:


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# ### XGBoost

# In[ ]:


# XGBoost
xgboost = Train(XGBRegressor(n_estimators=50,
                        max_depth = 9,
                        boosting_type = 'gbdt',
                        learning_rate = 0.05,
                        subsample = 0.85,
                        colsample_bytree = 0.85,
                        reg_alpha = 1e-4,
                        silent = True,
                        n_jobs = -1), X_train, y_train, X_test, y_test, type='reg')


# ### LightGBM

# In[ ]:


# LightGBM
lgb = Train(LGBMRegressor(n_estimators=50,
                    max_depth = 9,
                    boosting_type = 'gbdt',
                    learning_rate = 0.05,
                    subsample = 0.85,
                    colsample_bytree = 0.85,
                    reg_alpha = 1e-4,
                    silent = True,
                    n_jobs = -1), X_train, y_train, X_test, y_test, type='reg')


# ### CatBoost

# In[ ]:


# Catboost
cat = Train(CatBoostRegressor(n_estimators=50,
                        max_depth = 9,
                        loss_function = 'RMSE',
                        eval_metric = 'RMSE',
                        learning_rate = 0.05,
                        boosting_type = 'Plain',
                        bootstrap_type = 'Bernoulli',
                        subsample = 0.85,
                        silent = True), X_train, y_train, X_test, y_test, type='reg')


# ## C. Fine-tuned models

# ### XGBoost

# In[ ]:


# XGBoost with GridSearch
param_grid = [{'n_estimators': [400],
               'num_leaves': [100,500,1200],
               'min_child_weight' : [1,3,6],
               'max_depth':[5,10,20,50],
               'colsample_bytree': [0.8, 0.9],
               'learning_rate':[0.05,0.1,0.15],
               'boosting_type': ['gbdt'],
               'reg_alpha': [1e-4]
               }]
xgboost_gs = GridSearch(XGBRegressor(), param_grid, X_train[:4000], y_train[:4000], X_test, y_test, type='reg')


# ### LightGBM

# In[ ]:


# LightGBM with GridSearch
param_grid = [{'n_estimators': [400],
               'num_leaves': [100,500,1200],
               'min_child_weight' : [1,3,6],
               'max_depth':[5,10,20,50],
               'colsample_bytree': [0.8, 0.9],
               'learning_rate':[0.05,0.1,0.15],
               'boosting_type': ['gbdt'],
               'reg_alpha': [1e-4]
               }]
lgb_gs = GridSearch(LGBMRegressor(), param_grid, X_train[:4000], y_train[:4000], X_test, y_test, type='reg')


# ### CatBoost

# In[ ]:


# CatBoost with GridSearch
param_grid = [{'learning_rate':[0.05,0.1,0.15],
                'depth': [5,10,20,50],
                'n_estimators':[200],
                'iterations': [500]
                'l2_leaf_reg': [1,4,9],
                'subsample': [0.8, 0.9]
               }]
cat_gs = GridSearch(CatBoostRegressor(loss_function = 'RMSE',
                        eval_metric = 'RMSE',
                        boosting_type = 'Plain',
                        bootstrap_type = 'Bernoulli',
                        silent = True), param_grid, X_train[:4000], y_train[:4000], X_test, y_test, type='reg')


# ## D. The Results

# In[ ]:


models = [('XGBoost', xgboost),
         ('LightGBM', lgb),
         ('CatBoost', cat),
         ('XGBoost GridSearch', xgboost_gs),
         ('LightGBM GridSearch', lgb_gs),
         ('CatBoost GridSearch', cat_gs)]


# ### R2 Scores

# In[ ]:


scores = [round(score) for score in scores]
model_scores = pd.DataFrame({ 'Model': [name for name, _ in models], 'R2': scores })
model_scores.sort_values(by='R2',ascending=False,inplace=True)
plot_metric(model_scores, score='R2')


# ### Training and Prediction Times

# In[ ]:


training_times = [round(time,2) for time in training_times]
model_train_times = pd.DataFrame({ 'Model': [name for name, _ in models], 'Training Times': training_times })
plot_metric(model_train_times, score='Training Times')


# In[ ]:


model_train_times = pd.DataFrame({ 'Model': [name for name, _ in models], 'Prediction Times': prediction_times })
plot_metric(model_train_times, score='Prediction Times')


# ### Interpretability

# A model's prediction score only paints a partial picture of its predictions. We also want to know *why* the model is making its predictions.
# 
# Here we plot the model's feature importances, SHAP values and draw an actual decision tree to get a firmer understanding of the model's predictions.

# #### Feature Importances

# In[ ]:


# XGBoost
feature_importances(X_train, xgboost, 'XGBoost')


# In[ ]:


# CatBoost
feature_importances(X_train, cat, 'CatBoost')


# In[ ]:


# LightGBM
feature_importances(X_train, lgb, 'LightGBM')


# #### SHAP Values

# In[ ]:


# XGBoost
shap_values(X_train.iloc[:500,:], xgboost, 'XGBoost')


# In[ ]:


# LightGBM
shap_values(X_train.iloc[:500,:], lgb, 'LightGBM')


# **CatBoost**
# CatBoost doesn't work out of the box with shap_values() and results in the kernel crashing.

# #### Visualize Trees

# In[ ]:


# Set figure size for decision tree plots
rcParams['figure.figsize'] = 80,50


# In[ ]:


# LightGBM
lightgbm.plot_tree(lgb);


# In[ ]:


# XGBoost
xgb.plot_tree(xgboost);


# **CatBoost**
# 
# CatBoost ships with no plotting function for its trees. If you really need to visualize CatBoost results, a work-around is proposed here: https://blog.csdn.net/l_xzmy/article/details/81532281

# In[ ]:


# Clear memory before moving onto the next round
import gc
gc.collect()


# # Round 3: Regression on a massive Dataset: Predict NYC Taxi fares (2 million rows, 7 features)

# ## NYC Taxi dataset (2 million rows, 6 features)

# We're using the same dataset as above, but this time instead of training on 60,000 rows, we're going to train the models on 2 million rows and see if they're up to the challenge!

# In[ ]:


# Get data from New York City Taxi Fare Prediction
# allocate 1000 rows for test set, the rest for training set
n=2000000
train = pd.read_csv('../input/nyctaxi/train_20mil.csv', nrows=n)
test = pd.read_csv('../input/nyctaxi/train_20mil.csv', skiprows=n)
train.head(5)


# In[ ]:


train.shape, test.shape


# ## A. Baseline models

# In[ ]:


prediction_times = []
training_times = []
scores = []


# In[ ]:


train = remove_sparse(train)
test = remove_sparse(test)
train = feature_engineering(train)
test = feature_engineering(test)
y_train = train.fare_amount
X_train = train.drop('fare_amount', axis=1)
y_test = test.fare_amount
X_test = test.drop('fare_amount', axis=1)


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# ### XGBoost

# In[ ]:


# XGBoost
xgboost = Train(XGBRegressor(n_estimators=3,
                        max_depth = 9,
                        boosting_type = 'gbdt',
                        learning_rate = 0.05,
                        subsample = 0.85,
                        colsample_bytree = 0.8,
                        reg_alpha = 1e-4,
                        silent = True,
                        n_jobs = -1), X_train, y_train, X_test, y_test, type='reg')


# ### LightGBM

# In[ ]:


# LightGBM
lgb = Train(LGBMRegressor(n_estimators=3,
                    max_depth = 9,
                    boosting_type = 'gbdt',
                    learning_rate = 0.05,
                    subsample = 0.85,
                    colsample_bytree = 0.8,
                    reg_alpha = 1e-4,
                    silent = True,
                    n_jobs = -1), X_train, y_train, X_test, y_test, type='reg')


# ### CatBoost

# In[ ]:


# Catboost
cat = Train(CatBoostRegressor(n_estimators=3,
                        max_depth = 9,
                        loss_function = 'RMSE',
                        eval_metric = 'RMSE',
                        boosting_type = 'Plain',
                        bootstrap_type = 'Bernoulli',
                        learning_rate = 0.05,
                        subsample = 0.85,
                        silent = True), X_train, y_train, X_test, y_test, type='reg')


# ## B. The Results

# In[ ]:


models = [('XGBoost', xgboost),
         ('LightGBM', lgb),
         ('CatBoost', cat)]


# ### R2 Scores

# In[ ]:


scores = [round(score) for score in scores]
model_scores = pd.DataFrame({ 'Model': [name for name, _ in models], 'R2': scores })
model_scores.sort_values(by='R2',ascending=False,inplace=True)
plot_metric(model_scores, score='R2')


# ### Training and Prediction Times

# In[ ]:


training_times = [round(time,2) for time in training_times]
model_train_times = pd.DataFrame({ 'Model': [name for name, _ in models], 'Training Times': training_times })
plot_metric(model_train_times, score='Training Times')


# In[ ]:


model_train_times = pd.DataFrame({ 'Model': [name for name, _ in models], 'Prediction Times': prediction_times })
plot_metric(model_train_times, score='Prediction Times')


# ### Feature Importances

# In[ ]:


# XGBoost
feature_importances(X_train, xgboost, 'XGBoost')


# In[ ]:


# CatBoost
feature_importances(X_train, cat, 'CatBoost')


# In[ ]:


# LightGBM
feature_importances(X_train, lgb, 'LightGBM')


# #### SHAP Values

# In[ ]:


# XGBoost
shap_values(X_train.iloc[:500,:], xgboost, 'XGBoost')


# In[ ]:


# LightGBM
shap_values(X_train.iloc[:500,:], lgb, 'LightGBM')


# **CatBoost**
# CatBoost doesn't work out of the box with shap_values() and results in the kernel crashing.

# #### Visualize Trees

# In[ ]:


# Set figure size for decision tree plots
rcParams['figure.figsize'] = 80,50


# In[ ]:


# LightGBM
lightgbm.plot_tree(lgb);


# In[ ]:


# XGBoost
xgb.plot_tree(xgboost);


# **CatBoost**
# 
# CatBoost ships with no plotting function for its trees. If you really need to visualize CatBoost results, a work-around is proposed here: https://blog.csdn.net/l_xzmy/article/details/81532281

# In[ ]:


# Clear memory before moving onto the next round
import gc
gc.collect()


# # The Analysis
# 
# I hope you find this analysis useful! I encourage you to fork this kernel, and play with the code.
# 
# A detailed analysis of the results obtained in this kernel can be found on my blog at: https://lavanya.ai/2019/06/27/battle-of-the-boosting-algorithms/
# 
# If you like this kernel, please give it an upvote. Thank you!
