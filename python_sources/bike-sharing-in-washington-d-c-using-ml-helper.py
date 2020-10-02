#!/usr/bin/env python
# coding: utf-8

# # Bike Sharing in Washington D.C.
# 
# Two datasets from [Bike Sharing in Washington D.C.](https://www.kaggle.com/marklvl/bike-sharing-dataset/home) containing information about the Bike Sharing service in Washington D.C. "Capital Bikeshare" are provided.
# 
# One dataset contains hourly data and the other one has daily data from the years 2011 and 2012.
# 
# The following variables are included in the data:
# 
# * instant: Record index
# * dteday: Date
# * season: Season (1:springer, 2:summer, 3:fall, 4:winter)
# * yr: Year (0: 2011, 1:2012)
# * mnth: Month (1 to 12)
# * hr: Hour (0 to 23, only available in the hourly dataset)
# * holiday: whether day is holiday or not (extracted from Holiday Schedule)
# * weekday: Day of the week
# * workingday: If day is neither weekend nor holiday is 1, otherwise is 0.
# * weathersit: (extracted from Freemeteo)
#     1: Clear, Few clouds, Partly cloudy, Partly cloudy
#     2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# * temp: Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
# * atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
# * hum: Normalized humidity. The values are divided to 100 (max)
# * windspeed: Normalized wind speed. The values are divided to 67 (max)
# * casual: count of casual users
# * registered: count of registered users
# * cnt: count of total rental bikes including both casual and registered (Our target variable)
# 
# We will build a predictive model that can determine how many people will use the service on an hourly basis. We will use the first 5 quarters of the data for our training dataset and the last quarter of 2012 will be the holdout against which we perform our validation. Since that data was not used for training, we are sure that the evaluation metric that we get for it (R2 score) is an objective measurement of its predictive power.
# 
# ### Outline
# 
# We separate the project in 3 steps:
# 
# Data Loading and Exploratory Data Analysis: Load the data and analyze it to obtain an accurate picture of it, its features, its values (and whether they are incomplete or wrong), its data types among others. Also, the creation of different types of plots in order to help us understand the data and make the model creation easier.
# 
# Feature Engineering / Modeling and Pipeline: Once we have the data, we create some features and then the modeling stage begins, making use of different models (and ensembles) and a strong pipeline with different transformers, we will hopefully produce a model that fits our expectations of performance. Once we have that model, a process of tuning it to the training data would be performed.
# 
# Results and Conclusions: Finally, with our tuned model, we  predict against the test set we decided to separate initially, then we review those results against their actual values to determine the performance of the model, and finally, outlining our conclusions.
# 
# ### Helpers
# 
# **To run this code you must install my package called ml-helper**, which is a set of helpers to speed up the the machine learning process and provide a formal structure for it. These helpers are the basis for my package ML-Helper and they can be used in your own projects by downloading the package at [Pypi](https://pypi.org/project/ml-helper/) ```(pip install ml-helper)```.
# 
# If you wish to see a working example and explanation of what the package does, without using the package directly, please see my [kernel "Employee Attrition"](https://www.kaggle.com/akoury/employee-attrition-helpers-to-speed-up-ml-process/) or take a look at the code [at my GitHub](https://github.com/akoury/ml-helper)

# In[ ]:


get_ipython().system('pip install ml_helper')


# In[ ]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tempfile import mkdtemp
from sklearn.base import clone
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from ml_helper.helper import Helper
from imblearn import FunctionSampler
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from gplearn.genetic import SymbolicTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score as metric_scorer
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, PowerTransformer, OneHotEncoder, FunctionTransformer

warnings.filterwarnings('ignore')


# ### Setting Key Values
# 
# The following values are used throught the code, this cell gives a central source where they can be managed. We also create a helper object from the package ML-Helper while passing along these keys, they will help the package run a few functions under the hood.

# In[ ]:


MEMORY = mkdtemp()

KEYS = {
    'SEED': 1,
    'DATA_H': '../input/bike-sharing-dataset/hour.csv',
    'DATA_D' : '../input/bike-sharing-dataset/day.csv',
    'DATA_P': 'https://gist.githubusercontent.com/akoury/6fb1897e44aec81cced8843b920bad78/raw/b1161d2c8989d013d6812b224f028587a327c86d/precipitation.csv',
    'TARGET': 'cnt',
    'METRIC': 'r2',
    'TIMESERIES': True,
    'SPLITS': 3,
    'ESTIMATORS': 150,
    'MEMORY': MEMORY
}

hp = Helper(KEYS)


# ### Data Loading
# 
# Here we load the necessary data, print its first rows and describe its contents.

# In[ ]:


def read_data(input_path):
    return pd.read_csv(input_path, parse_dates=[1])

data = read_data(KEYS['DATA_H'])
data_daily = read_data(KEYS['DATA_D'])

data.head()


# In[ ]:


data.describe()


# ### Precipitation Data
# 
# In order to generate our model, we will add precipitation data obtained from the [National Climatic Data Center.](https://www.ncdc.noaa.gov/cdo-web/datasets)
# 
# However, since most of the values are 0, we will convert them to a boolean that determines if rain was present or not at that specific hour.

# In[ ]:


precipitation = read_data(KEYS['DATA_P'])
data = pd.merge(data, precipitation,  how='left', on=['dteday','hr'])
data['precipitation'].fillna(0, inplace=True)
data['precipitation'][data['precipitation'] > 0] = 1
data['precipitation'] = data['precipitation'].astype(int).astype('category')

data_hourly = data.copy()
data_hourly = data_hourly[data_hourly['dteday'].isin(pd.date_range('2011-01-01','2012-09-30'))]


# ### Data types
# 
# We review the data types for each column.

# In[ ]:


data.dtypes


# ### Missing Data
# 
# We check if there is any missing data.

# In[ ]:


hp.missing_data(data)


# ### Converting columns to their true categorical type
# Now we convert the data types of numerical columns that are actually categorical.

# In[ ]:


data = hp.convert_to_category(data, data.iloc[:,2:10])

data.dtypes


# ## Exploratory Data Analysis
# 
# Here we will perform all of the necessary data analysis, with different plots that will help us understand the data and therefore, create a better model.
# 
# We must specify that all of this analysis is performed only on the training data, so that we do not incur in any sort of bias when modeling.
# 
# To start we define some color palettes to be used.

# In[ ]:


palette_tot_cas_reg = ['darkgreen', 'darkred', 'darkblue']

palette_cas = ['darkred', 'salmon']
palette_reg = ['darkblue', 'skyblue']


# ### Overall distribution of the target variable

# In[ ]:


plt.figure(figsize=(16, 8))
sns.distplot(data['cnt'])


# ### Usage over time, by type of user and total

# In[ ]:


data_daily = data_daily[data_daily['dteday'].isin(pd.date_range('2011-01-01','2012-09-30'))]
data_daily = hp.convert_to_category(data_daily, data_daily.iloc[:,2:9])
data_daily.set_index('dteday')

plt.figure(figsize=(16, 5))

ax = sns.lineplot(data = data_daily, x = 'dteday', y = 'cnt', color='darkgreen', size = 1,label = 'count')
ax = sns.lineplot(data = data_daily, x = 'dteday', y = 'casual', color='darkred', size = 1, label = 'casual')
ax = sns.lineplot(data = data_daily, x = 'dteday', y = 'registered', color='darkblue', size = 1, label = 'registered')

handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles[0:1]+handles[3:4]+handles[6:7], labels[0:1]+labels[3:4]+labels[6:7], loc=2)
plt.xlabel('Date')
plt.ylabel('Users')


# They all seem to be increasing and have some seasonality.

# ### Monthly average usage

# In[ ]:


df_month = pd.DataFrame(data_daily.groupby("mnth")[["cnt", 'casual', 'registered']].mean()).reset_index()
months = pd.Series(["January","February","March","April","May","June","July","August","September","October","November","December"]).rename("months")
df_month = pd.concat([df_month, months], axis = 1)


plt.figure(figsize=(12, 5))
ax = sns.pointplot(data = df_month, x = "months", y = "cnt", color = 'darkgreen')
ax = sns.pointplot(data = df_month, x = "months", y = "casual", color = 'darkred')
ax = sns.pointplot(data = df_month, x = "months", y = "registered", color = 'darkblue')

plt.xlabel('')
plt.ylabel('Users')


# No considerable differences in trends between casual and registered.
# 
# ### Weekly trend usage

# In[ ]:


df_week = pd.DataFrame(data_daily.groupby("weekday")[["cnt", 'casual', 'registered']].mean()).reset_index()
df_week = pd.melt(df_week, id_vars = ['weekday'], value_vars = ['cnt', 'casual', 'registered'], var_name = 'type', value_name = 'users')

plt.figure(figsize=(12, 5))
ax = sns.lineplot(data = df_week, x = "weekday", y = "users", hue = "type", palette = palette_tot_cas_reg)
plt.xlabel('Weekday')
plt.ylabel('Users')


# Casual and registered users follow exactly opposed trends throughout the week.
# 
# ### Daily trend
# 
# By type of users:

# In[ ]:


data_hourly = hp.convert_to_category(data_hourly, data_hourly.iloc[:,2:9])
data_hourly.set_index('dteday')

df_day = pd.DataFrame(data_hourly.groupby("hr")[["cnt", 'casual', 'registered']].mean()).reset_index()
df_day = pd.melt(df_day, id_vars = ['hr'], value_vars = ['cnt', 'casual', 'registered'], var_name = 'type', value_name = 'users')

plt.figure(figsize=(12, 5))
sns.lineplot(data = df_day, x = "hr", y = "users", hue = "type", palette = palette_tot_cas_reg)

plt.xlabel('Hour')
plt.ylabel('Users')


# ### Weekends compared with working days

# In[ ]:


plt.figure(figsize=(12, 5))
sns.lineplot(data = data_hourly, x = "hr", y = "casual", hue = 'workingday', palette = palette_cas)
sns.lineplot(data = data_hourly, x = "hr", y = "registered", hue = 'workingday', palette = palette_reg)
plt.xlabel('Hour')
plt.ylabel('Users')


# ### Temperature effect on casual users

# In[ ]:


atemp_binned = pd.cut(x = data_hourly['atemp'], bins = 4).rename('atemp_binned')
data_hourly_binned = pd.concat([data_hourly, atemp_binned], axis = 1)

df_day_by_day_atemp = pd.DataFrame(data_hourly_binned.groupby(["hr", "atemp_binned"])[["cnt", 'casual', 'registered']].mean()).reset_index()
df_day_by_day_atemp.head()

plt.figure(figsize=(12, 5))
sns.lineplot(data = df_day_by_day_atemp, x = 'hr', y = 'casual', hue = 'atemp_binned', palette = 'husl')


# ### Temperature effect on registered users

# In[ ]:


plt.figure(figsize=(12, 5))
sns.lineplot(data = df_day_by_day_atemp, x = 'hr', y = 'registered', hue = 'atemp_binned', palette = 'husl')


# As you can see, the effect is much more pronounced on the casual users.

# ### Temperature Vs. Usage on working days for casual users

# In[ ]:


sns.scatterplot(data = data_daily, x = 'atemp', y = 'casual', hue = 'workingday', alpha = .3)
plt.title('Casual Users')


# ### Temperature Vs. Usage on working days for registered users

# In[ ]:


sns.scatterplot(data = data_daily, x = 'atemp', y = 'registered', hue = 'workingday', alpha = .3)
plt.title('Registered Users')


# As we can see, temperature does affect casual user usage, but registered users do not seem to mind the temperature.

# ### Precipitation Vs. Usage throughout the day

# In[ ]:


plt.figure(figsize=(12, 5))
ax = sns.lineplot(data = data_hourly, x = "hr", y = "casual", hue = 'precipitation', palette = palette_cas, label = 'casual')

ax = sns.lineplot(data = data_hourly, x = "hr", y = "registered", hue = 'precipitation', palette = palette_reg, label = 'registered')

handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles[0:2]+handles[5:7], labels[0:2]+labels[5:7], loc=2)
plt.xlabel('Hour')
plt.ylabel('Users')


# ### Boxplot of Numerical Variables
# 
# We review the distribution of scaled numerical data through a boxplot for each variable.

# In[ ]:


hp.boxplot(data, ['instant'])


# From this we know that we should try some outlier treatment

# ### Coefficient of Variation
# 
# The coefficient of variation is a dimensionless meassure of dispersion in data, the lower the value the less dispersion a feature has. We will select columns that have a variance of less than 0.05 since they would probably perform poorly.

# In[ ]:


invariant = hp.coefficient_variation(data, threshold = 0.05, exclude = ['instant'])


# ## Baseline
# 
# A basic linear model is created in order to set a baseline, further models will be compared against its results.

# In[ ]:


base_holdout = data[data['dteday'].isin(pd.date_range('2012-10-01','2012-12-31'))].copy()
base_holdout = hp.drop_columns(base_holdout, ['dteday', 'casual', 'registered'])
base_data = data[data['dteday'].isin(pd.date_range('2011-01-01','2012-09-30'))].copy()
base_data = hp.drop_columns(data, ['dteday', 'casual', 'registered'])

y, predictions = hp.predict(base_data, base_holdout, LinearRegression())
base_score = metric_scorer(y, predictions)
print('Baseline score: ' + str(base_score))


# ### Data Correlation
# 
# Now we will analyze correlation in the data for both numerical and categorical columns and plot them, using a threshold of 90%.

# In[ ]:


training_data = data[data['dteday'].isin(pd.date_range('2011-01-01','2012-09-30'))].copy()
correlated_cols = hp.correlated(training_data, 0.9)


# ### Underrepresented Features
# 
# Now we determine underrepresented features, meaning those that in more than 97% of the records are composed of a single value.

# In[ ]:


under_rep = hp.under_represented(data, 0.97)


# ### Principal Component Analysis (PCA)
# 
# We plot PCA component variance to define the number of components we wish to consider in the pipeline.

# In[ ]:


hp.plot_pca_components(data.drop('dteday', axis=1))


# ### Feature Importance
# 
# Here we plot feature importance using a random forest in order to get a sense of which features have the most importance.

# In[ ]:


hp.feature_importance(hp.drop_columns(data, ['dteday', 'registered', 'casual']), RandomForestRegressor(n_estimators=KEYS['ESTIMATORS'], random_state = KEYS['SEED'], n_jobs = -1), convert = True)


# ### Defining Holdout Set for Validation
# 
# The first 5 quarters of the data will be used to train our model, while the remaining quarter will be used later on to validate the accuracy of our model.

# In[ ]:


holdout = data[data['dteday'].isin(pd.date_range('2012-10-01','2012-12-31'))].copy().reset_index()
holdout_final_plots = holdout.copy() # we will use this for plots at the end
train_data = data[data['dteday'].isin(pd.date_range('2011-01-01','2012-09-30'))]


# ## Feature Engineering / Pipeline / Modeling
# 
# A number of different combinations of feature engineering steps and transformations will be performed in a pipeline with different models, each one will be cross validated to review the performance of the model.
# 
# **Some of the steps are commented, the point is for the user to comment/uncomment the steps they wish to try and those pipelines and scores will be saved for later use**, that way you can see what improves the score and what decreases it.
# 
# Overall, we try removing unneeded columns, clustering, removing outliers through isolation forests, quantile binning, polynomial combinations, genetic transformations, one hot encoding, rebalancing techniques, recursive feature elimination, feature selection, PCA and more.
# 
# The pipeline uses the cross evaluation function, which handles time series splits for fold creation (instead of Kfolds which does not work for time series) while also setting a holdout to perform after the cross validation.

# In[ ]:


def day(df):
    df = df.copy()
    df['day'] = df['dteday'].dt.day
    df = hp.convert_to_category(df, ['day'])
    
    return df

def drop_features(df, cols):
    return df[df.columns.difference(cols)]

def kmeans(df, clusters = 3):
    clusterer = KMeans(clusters, random_state=KEYS['SEED'])
    cluster_labels = clusterer.fit_predict(df)
    df = np.column_stack([df, cluster_labels])
    
    return df

def outlier_rejection(X, y):
    model = IsolationForest(random_state=KEYS['SEED'], behaviour='new', n_jobs = -1)
    model.fit(X)
    y_pred = model.predict(X)
    
    return X[y_pred == 1], y[y_pred == 1]

num_pipeline = Pipeline([ 
    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize = True))
])

categorical_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

pipe = Pipeline([
    ('day', FunctionTransformer(day, validate=False)),
    ('drop_features', FunctionTransformer(drop_features, kw_args={'cols': ['dteday','casual', 'registered'] + correlated_cols + under_rep}, validate=False)),
    ('column_transformer', ColumnTransformer([
        ('numerical_pipeline', num_pipeline, ['hum', 'temp', 'windspeed']),
        ('categorical_pipeline', categorical_pipeline, ['day', 'hr', 'mnth', 'precipitation', 'season', 'weathersit', 'weekday', 'yr']),
    ], remainder='passthrough'))
])

models = [
    {'name':'linear_regression', 'model': LinearRegression()},
    {'name':'random_forest', 'model': RandomForestRegressor(n_estimators = KEYS['ESTIMATORS'], random_state = KEYS['SEED'], n_jobs = -1)},
    {'name':'xgb', 'model': XGBRegressor(random_state = KEYS['SEED'])}
]


# ## Scores
# 
# Here you can see all of the scores throughout the entire cross validation process for each pipeline. To begin, we run three initial models with basic transformations and then add/remove transformers to see how the score moves. In certain cases errors can happen (for example when a certain fold contains a sparse matrix), therefore you may see errors marked as such in the score.

# In[ ]:


all_scores = hp.pipeline(train_data, models, pipe)


# After running the initial pipeline, the point of using the package is to try different combinations, therefore you can feel free to uncomment the steps you wish to try and run them here. Since this is not possible in Kaggle, we will edit the pipeline separately.
# 
# ### Binning And Polynomial Features
# Now we try adding binning and polynomial features to our pipeline and see how it performs.

# In[ ]:


num_pipeline = Pipeline([ 
    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize = True)),
    ('binning', KBinsDiscretizer(n_bins = 5, encode = 'onehot-dense')),
    ('polynomial', PolynomialFeatures(degree = 2, include_bias = False)),
])

categorical_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

pipe = Pipeline([
    ('day', FunctionTransformer(day, validate=False)),
    ('drop_features', FunctionTransformer(drop_features, kw_args={'cols': ['dteday','casual', 'registered'] + correlated_cols + under_rep}, validate=False)),
    ('column_transformer', ColumnTransformer([
        ('numerical_pipeline', num_pipeline, ['hum', 'temp', 'windspeed']),
        ('categorical_pipeline', categorical_pipeline, ['day', 'hr', 'mnth', 'precipitation', 'season', 'weathersit', 'weekday', 'yr']),
    ], remainder='passthrough')),
])

all_scores = hp.pipeline(train_data, models, pipe, all_scores)


# ### PCA
# We try doing Principal Component Analysis and see how it performs.

# In[ ]:


num_pipeline = Pipeline([ 
    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize = True)),
])

categorical_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

pipe = Pipeline([
    ('day', FunctionTransformer(day, validate=False)),
    ('drop_features', FunctionTransformer(drop_features, kw_args={'cols': ['dteday','casual', 'registered'] + correlated_cols + under_rep}, validate=False)),
    ('column_transformer', ColumnTransformer([
        ('numerical_pipeline', num_pipeline, ['hum', 'temp', 'windspeed']),
        ('categorical_pipeline', categorical_pipeline, ['day', 'hr', 'mnth', 'precipitation', 'season', 'weathersit', 'weekday', 'yr']),
    ], remainder='passthrough')),
    ('pca', PCA(n_components = 6))
])

all_scores = hp.pipeline(train_data, models, pipe, all_scores)


# ### Feature Selection
# We perform feature selection based on feature importance (using a random forest) and setting a threshold of 0.005

# In[ ]:


num_pipeline = Pipeline([ 
    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize = True)),
])

categorical_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

pipe = Pipeline([
    ('day', FunctionTransformer(day, validate=False)),
    ('drop_features', FunctionTransformer(drop_features, kw_args={'cols': ['dteday','casual', 'registered'] + correlated_cols + under_rep}, validate=False)),
    ('column_transformer', ColumnTransformer([
        ('numerical_pipeline', num_pipeline, ['hum', 'temp', 'windspeed']),
        ('categorical_pipeline', categorical_pipeline, ['day', 'hr', 'mnth', 'precipitation', 'season', 'weathersit', 'weekday', 'yr']),
    ], remainder='passthrough')),
    ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators = KEYS['ESTIMATORS'], random_state = KEYS['SEED'], n_jobs = -1), threshold = 0.005)),
])

all_scores = hp.pipeline(train_data, models, pipe, all_scores)


# ### Outlier Removal Through Isolation Forest and Polynomial Features
# We combine outlier removal and polynomial features

# In[ ]:


num_pipeline = Pipeline([ 
    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize = True)),
    ('polynomial', PolynomialFeatures(degree = 2, include_bias = False)),
])

categorical_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

pipe = Pipeline([
    ('day', FunctionTransformer(day, validate=False)),
    ('drop_features', FunctionTransformer(drop_features, kw_args={'cols': ['dteday','casual', 'registered'] + correlated_cols + under_rep}, validate=False)),
    ('column_transformer', ColumnTransformer([
        ('numerical_pipeline', num_pipeline, ['hum', 'temp', 'windspeed']),
        ('categorical_pipeline', categorical_pipeline, ['day', 'hr', 'mnth', 'precipitation', 'season', 'weathersit', 'weekday', 'yr']),
    ], remainder='passthrough')),
    ('outliers', FunctionSampler(func = outlier_rejection)),
])

all_scores = hp.pipeline(train_data, models, pipe, all_scores)


# ### Pipeline Performance by Model
# Here we can see the performance of each model in the different pipelines we created.

# In[ ]:


hp.plot_models(all_scores)


# ### Top Pipelines per Model
# 
# Here we show the top pipelines per model.

# In[ ]:


hp.show_scores(all_scores, top = True)


# ## Randomized Grid Search
# 
# Once we have a list of models, we perform a cross validated, randomized grid search on the best performing one to define the final models.

# In[ ]:


rf_grid = {
    'random_forest__criterion': ['mse', 'mae'],
    'random_forest__max_depth': [50, 100],
    'random_forest__min_samples_leaf': [5,10],
    'random_forest__min_samples_split': [10, 20],
    'random_forest__max_leaf_nodes': [None, 80],
}

final_scores, f_pipe = hp.cross_val(train_data, model = clone(hp.top_pipeline(all_scores)), grid = rf_grid)
final_scores


# ### Best Parameters for the models

# In[ ]:


print(f_pipe.best_params_)
final_pipe = f_pipe.best_estimator_


# From the scores obtained, we see that the performance of the model worsened, therefore we will stick to the original values

# # Results
# This is the final R^2 score of the best pipeline, tested against the holdout set.

# In[ ]:


y, predictions = hp.predict(train_data, holdout, clone(hp.top_pipeline(all_scores)))
score = metric_scorer(y, predictions)
score


# ## Plots of Predictions
# 
# Here we plot the different results obtained.
# 
# For this scatter plot, the straighter the diagonal line is, the better the predictions since they are closer to the actual values.

# In[ ]:


hp.scatter_predict(y, predictions)


# ### Hourly Three week prediction vs. reality plot

# In[ ]:


hp.plot_predict(y, predictions, subset = (3*7*24), x_label = 'Hour', y_label = 'Users')


# ### Entire daily predictions vs. reality plot

# In[ ]:


hp.plot_predict(y, predictions, group = 24, x_label = 'Day', y_label = 'Users')


# # Conclusions
# 
# We created a model that, based on certain parameters, determine bike usage on an hourly basis, with these results we can provide an estimation of usage which can be of great importance for all of the involved parties.
# 
# One of the key findings is that there is a great difference in usage from weekends to normal working days, this situation would need to be considered by the company to supply the correct amount of bicicles depending on the day of the week, since the demand changes drastically. Then, as can be guessed, temperature plays a big role in usage, although it is more significant in casual users.
# 
# Initially we had a baseline model with a very low r2 score, however, after performing multiple data preparation steps and transformations we achieved a much higher score, but not only that, we can see from the prediction plots that the model follows along many of the peaks and valleys of the real data, this proves that our predicting capabilities improved immensely.
# 
# Many different bike-sharing companies accross the world could use this model to estimate bike usage, planify better for expected demand and even help their governments transportation requirements. Measuring the impact of new bike infrastructure on cycling traffic and behavior is top of mind for many planners and advocacy groups.
