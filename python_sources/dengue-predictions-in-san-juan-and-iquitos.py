#!/usr/bin/env python
# coding: utf-8

# 1. # **DengAI**
# 
# **This code is based on previous works and it's still under construction**
# 
# 
# ## **Table of Contents:**
# * Introduction
# * Exploratory data analysis and Data Preprocessing
#     - Converting Features
#     - Creating new Features
# * Building Machine Learning Models

# ## Introduction
# 
# We begin by importing all the necessary libraries and the dataset. 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, log_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn import preprocessing
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from __future__ import division

from matplotlib import  pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.base.model import GenericLikelihoodModel
from sklearn.svm import SVR
import time as time

from warnings import filterwarnings
filterwarnings('ignore')

X_train = pd.read_csv('../input/dengue_features_train.csv')
X_test = pd.read_csv('../input/dengue_features_test.csv')
y_train = pd.read_csv('../input/dengue_labels_train.csv')


print(X_train.head(10))
print(X_train.describe())
print(X_train.info())

# Let's check the percentage of missing values in our dataset

total = X_train.isnull().sum().sort_values(ascending=False)
percent = (X_train.isnull().sum()/X_train['city'].count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# There are a few things to notice here. 
# 
# Firstly, from the description at first sight, from the description we cannot infer too much information. We have numbers that don't mean much to us right now. They seem well, but they may not be correct, so we will deal with them later on.
# 
# Secondly, good news. As we can see, there are not to many missing values. We will fill them with the mean later on aswell. Now we will check if there are outliers in our dataset.

# In[ ]:


ax = sns.countplot(X_train['city'],label="Count")       # M = 212, B = 357
SanJuan, Iquitos = X_train['city'].value_counts()
print('San Juan: ', SanJuan)
print('Iquitos : ', Iquitos)


# In[ ]:


data_dia = X_train.copy()
y = X_train['city']
del data_dia['city']
del data_dia['year']
del data_dia['week_start_date']



del data_dia['reanalysis_sat_precip_amt_mm']
del data_dia['precipitation_amt_mm']
del data_dia['reanalysis_avg_temp_k']
del data_dia['reanalysis_air_temp_k']
del data_dia['reanalysis_dew_point_temp_k']
del data_dia['reanalysis_max_air_temp_k']
del data_dia['reanalysis_min_air_temp_k']
del data_dia['reanalysis_precip_amt_kg_per_m2']
del data_dia['reanalysis_relative_humidity_percent']
del data_dia['reanalysis_specific_humidity_g_per_kg']
del data_dia['reanalysis_tdtr_k']

data_n_2 = (data_dia - data_dia.mean()) / (data_dia.std())              # standardization
data = pd.concat([y,data_n_2],axis=1)
data = pd.melt(data,id_vars="city",
                    var_name="features",
                    value_name='value')
pd.to_numeric(data['value'], downcast='float')
plt.figure(figsize=(10,10))
ax = sns.violinplot(x="features", y="value", hue="city", data=data,split=True, inner="quart")
plt.xticks(rotation=90)


# There's something strange going on there, right? We see that, after normalizing, the feature "station_precip_mm" alters the scale of our graph. Let's print the description of that feature.

# In[ ]:


print(X_train['station_precip_mm'].describe())


# **BOOM**. There you have it. It has an extremely large variance. Looking at its mean, it has an average of around 40mm, while the maximum is 543 (after normalizing). Let's drop it for a moment, just to have a clearer view of the plots.

# In[ ]:


data_dia2 = X_train.copy()
del data_dia2['city']
del data_dia2['year']
del data_dia2['week_start_date']


del data_dia2['station_precip_mm']
del data_dia2['reanalysis_sat_precip_amt_mm']
del data_dia2['precipitation_amt_mm']
del data_dia2['reanalysis_avg_temp_k']
del data_dia2['reanalysis_air_temp_k']
del data_dia2['reanalysis_dew_point_temp_k']
del data_dia2['reanalysis_max_air_temp_k']
del data_dia2['reanalysis_min_air_temp_k']
del data_dia2['reanalysis_precip_amt_kg_per_m2']
del data_dia2['reanalysis_relative_humidity_percent']
del data_dia2['reanalysis_specific_humidity_g_per_kg']
del data_dia2['reanalysis_tdtr_k']

data_n_22 = (data_dia2 - data_dia2.mean()) / (data_dia2.std())              # standardization

data2 = pd.concat([y,data_n_22],axis=1)

data2 = pd.melt(data2,id_vars="city",
                    var_name="features",
                    value_name='value')
pd.to_numeric(data2['value'], downcast='float')
plt.figure(figsize=(10,10))
ax = sns.violinplot(x="features", y="value", hue="city", data=data2,split=True, inner="quart")
plt.xticks(rotation=90)


# In[ ]:


data_dia2 = X_train.copy()
del data_dia2['city']
del data_dia2['year']
del data_dia2['week_start_date']
del data_dia2['ndvi_ne']
del data_dia2['ndvi_nw']
del data_dia2['station_diur_temp_rng_c']
del data_dia2['station_avg_temp_c']
del data_dia2['station_precip_mm']
del data_dia2['ndvi_se']
del data_dia2['ndvi_sw']
del data_dia2['station_max_temp_c']
del data_dia2['station_min_temp_c']


data_n_22 = (data_dia2 - data_dia2.mean()) / (data_dia2.std())              # standardization

data2 = pd.concat([y,data_n_22],axis=1)

data2 = pd.melt(data2,id_vars="city",
                    var_name="features",
                    value_name='value')
pd.to_numeric(data2['value'], downcast='float')
plt.figure(figsize=(10,10))
ax = sns.violinplot(x="features", y="value", hue="city", data=data2,split=True, inner="quart")
plt.xticks(rotation=90)


# Again, we see that there are different features which have extreme values. Exploring the dataset, it seems that they are not outleirs nor errors, so we cannot drop them, and we will have to take them into account. Those values are about precipitation, and since they are rain values, it is reasonable to assume that depending on the areas weather can change drastically. 
# 
# 
# But what else can we extract from these plots? For example, in **ndvi_ne** feature, median of the *San Juan* and *Iquitos* looks like separated so it can be good for classification. However, in **reanalysis_avg_temp_k** feature,  median of the *San Juan* and *Iquitos* does not looks like separated so it does not gives good information for classification. This happens with many other features.
# 
# Also, note that the features **reanalysis_avg_temp_k** and **reanalysis_specific_humidity_g_per_kg** have a very similar shape,  but how can we decide whether they are correlated with each other or not? Correlation is the answer.

# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#empt = ['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm','reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k','station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm']
#


# Checking this last graph, we see that some features have perfect correlation (value 1) and nearly perfect correlation (0.9). These are:
# 
# **reanalysis_sat_precip_amt_mm** and** precipitation_amt_mm **  with perfect correlation.
# **reanalysis_specific_humidity_g_per_kg** and** reanalysis_dew_point_temp_k ** with perfect correlation.
# **ndvi_nw** and **ndvi_ne** having correltion value 0.9.
# **reanalysis_avg_temp_k **and **reanalysis_air_temp_k ** having correltion value 0.9.
# **reanalysis_tdtr_k ** and **reanalysis_max_air_temp_k** having correltion value 0.9.
# **station_diur_temp_rng_c ** and **reanalysis_tdtr_k** having correltion value 0.9.
# 
# 
# Before making some actions, we will explore a little bit more our dataset.

# In[ ]:


sns.set(style="whitegrid", palette="muted")
data_dia2 = X_train.copy()
del data_dia2['city']
del data_dia2['year']
del data_dia2['week_start_date']


del data_dia2['station_precip_mm']
del data_dia2['reanalysis_sat_precip_amt_mm']
del data_dia2['precipitation_amt_mm']
del data_dia2['reanalysis_avg_temp_k']
del data_dia2['reanalysis_air_temp_k']
del data_dia2['reanalysis_dew_point_temp_k']
del data_dia2['reanalysis_max_air_temp_k']
del data_dia2['reanalysis_min_air_temp_k']
del data_dia2['reanalysis_precip_amt_kg_per_m2']
del data_dia2['reanalysis_relative_humidity_percent']
del data_dia2['reanalysis_specific_humidity_g_per_kg']
del data_dia2['reanalysis_tdtr_k']

data_n_22 = (data_dia2 - data_dia2.mean()) / (data_dia2.std())              # standardization

data2 = pd.concat([y,data_n_22],axis=1)

data2 = pd.melt(data2,id_vars="city",
                    var_name="features",
                    value_name='value')
pd.to_numeric(data2['value'], downcast='float')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="city", data=data2)
plt.xticks(rotation=90)


# In[ ]:


sns.set(style="whitegrid", palette="muted")
data_dia2 = X_train.copy()
del data_dia2['city']
del data_dia2['year']
del data_dia2['week_start_date']
del data_dia2['ndvi_ne']
del data_dia2['ndvi_nw']
del data_dia2['station_diur_temp_rng_c']
del data_dia2['station_avg_temp_c']
del data_dia2['station_precip_mm']
del data_dia2['ndvi_se']
del data_dia2['ndvi_sw']
del data_dia2['station_max_temp_c']
del data_dia2['station_min_temp_c']


data_n_22 = (data_dia2 - data_dia2.mean()) / (data_dia2.std())              # standardization

data2 = pd.concat([y,data_n_22],axis=1)

data2 = pd.melt(data2,id_vars="city",
                    var_name="features",
                    value_name='value')
pd.to_numeric(data2['value'], downcast='float')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="city", data=data2)
plt.xticks(rotation=90)


# What information does this give us? Well, we can see that there are some features in which the two cities are very separated, and this will be good for classification (the most clear example being **reanalysis_tdtr_k**. On the other hand, there are features in which the information is very mixed, and those will not be that good for classification.
# 

# But there's much more we can do with this dataset! For example, we can create a new dataframe which will be X_train plus the total_cases column of y_train, so that we can cretre new and interesting plots.

# In[ ]:




new_xtrain = X_train.copy()
new_xtrain['total_cases'] = y_train['total_cases']

plt.plot_date(new_xtrain['week_start_date'], new_xtrain['total_cases'], color='green', label='sj')
plt.plot_date(new_xtrain['week_start_date'], new_xtrain['total_cases'], color='blue', label='iq')
plt.xlabel('Year')
plt.ylabel('Total cases')
plt.legend(loc='upper right')
plt.show()


# # Feature Selection
# 
# Ok then, so we are now ready for selecting the features we will use for classification. For the moment, I will drop the following features: **reanalysis_sat_precip_amt_mm**, **reanalysis_specific_humidity_g_per_kg **. I won't do the same with the ones that have correlation value of 0.9 at this time.
# 

# In[ ]:


delete = ['week_start_date','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg']
for z in delete:
    del X_train[z]
    del X_test[z]


# If we tried to do machine learning up to this point, we would notice that the accuracy is really small. This is because the total cases vary a lot, from 0 to 400+. So what can we do to improve that? 
# 
# The first thing we'll try is to divide our dataset in two, one for San Juan and another one for Iquitos. Since we've modified our dataframes, we'll import them again and play with them.

# In[ ]:


X_trainc = pd.read_csv('../input/dengue_features_train.csv')
X_testc = pd.read_csv('../input/dengue_features_test.csv')
y_trainc = pd.read_csv('../input/dengue_labels_train.csv')


X_train_sj = X_trainc.loc[X_train.city=='sj']
y_train_sj = y_trainc.loc[y_train.city=='sj']

X_train_iq = X_trainc.loc[X_train.city=='iq']
y_train_iq = y_trainc.loc[y_train.city=='iq']

print('features: ', X_train_sj.shape)
print('labels  : ', y_train_sj.shape)

print('\nIquitos')
print('features: ', X_train_iq.shape)
print('labels  : ', y_train_iq.shape)


# In[ ]:


(X_train_sj.ndvi_ne.plot.line(lw=0.8))
plt.title('Vegetation Index over Time')
plt.xlabel('Time')


# Our target variable, total_cases is a non-negative integer, which means we're looking to make some count predictions. Standard regression techniques for this type of prediction include:
# 
# 1. Poisson regression
# 2. Negative binomial regression
# 
# Which technique will perform better depends on many things, but the choice between Poisson regression and negative binomial regression is pretty straightforward. Poisson regression fits according to the assumption that the mean and variance of the population distribution are equal. When they aren't, specifically when the variance is much larger than the mean, the negative binomial approach is better. Why? It isn't magic. The negative binomial regression simply lifts the assumption that the population mean and variance are equal, allowing for a larger class of possible models. In fact, from this perspective, the Poisson distribution is but a special case of the negative binomial distribution.

# In[ ]:


print('San Juan')
print('mean: ', y_train_sj.mean()[2])
print('var :', y_train_sj.var()[2])

print('\nIquitos')
print('mean: ', y_train_iq.mean()[2])
print('var :', y_train_iq.var()[2])


# variance >> mean suggests total_cases can be described by a negative binomial distribution, so we'll use a negative binomial regression below. We will now check the correlations to see if there is any difference between the cities. To gain more insight, we will add the colum of total cases to see if there is any correlation with other variables.

# In[ ]:


X_train_sj['total_cases'] = y_train_sj['total_cases']
X_train_iq['total_cases'] = y_train_iq['total_cases']


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X_train_sj.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
ax.set_title('Correlations for San Juan')
plt.show()

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X_train_iq.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
ax.set_title('Correlations for Iquitos')
plt.show()


# In[ ]:


sj_correlations = X_train_sj.corr()
iq_correlations = X_train_iq.corr()
(sj_correlations['total_cases'].drop('total_cases').sort_values(ascending=False).plot.barh())
plt.title('Correlations for San Juan')


# In[ ]:


(iq_correlations['total_cases'].drop('total_cases').sort_values(ascending=False).plot.barh())
plt.title('Correlations for Iquitos')


# So we have some valuable information here. For both cities,**reanalysis_specific_humidity_g_per_kg**, **reanalysis_dew_point_temp_k** and  **reanalysis_min_air_temp_k** are the ones that are most strongly correlated with the total cases. Does this make sense? Of course, we know that mosquitoes tend to live in places with high humidity. Also, temperature is highly related with the spread of mosquitoes, so it makes sense it is related with the total cases. Surprisingly, in the city of San Juan, the week of year is highly correlated as well, so we'll keep an eye on that. 

# If we plot the amount of cases as a function of the week of year, we see that there are outbreaks in both Iquitos and San Juan toward the ends of each year. The increases in cases and outbreaks tend to happen in weeks 35 to 45 in San Juan and weeks 45 to 50 in Iquitos.

# In[ ]:


for i in set(y_train_sj['year']):
    df = y_train_sj[y_train_sj['year'] == i]
    df.set_index('weekofyear', drop = True, inplace = True)
    plt.plot(df['total_cases'], alpha = .3)
    
y_train_sj.groupby('weekofyear')['total_cases'].mean().plot(c = 'k', figsize = (10,4))
plt.legend(set(y_train_sj['year']), loc='center left', bbox_to_anchor=(1, .5))

plt.title('Number of Cases per Week in San Juan, Puerto Rico')
plt.xlabel('Week of the Year')
plt.ylabel('Number of Cases')


# In[ ]:


for i in set(y_train_iq['year']):
    df = y_train_iq[y_train_iq['year'] == i]
    df.set_index('weekofyear', drop = True, inplace = True)
    plt.plot(df['total_cases'], alpha = .3)

y_train_iq.groupby('weekofyear')['total_cases'].mean().plot(c = 'k', figsize = (10,4))
plt.legend(set(y_train_iq['year']), loc='center left', bbox_to_anchor=(1, .5))

plt.title('Number of Cases per Week in Iquitos, Peru')
plt.xlabel('Week of the Year')
plt.ylabel('Number of Cases')


# # Building Machine Learning models
# 
# Now that we have a more clearer understanding of our dataset, we will proceed to build our ML models.
# 
# First of all, we will also do the same as we did before, splitting now the test data into San Juan and Iquitos, and do the same pre-processing for both.

# In[ ]:


X_test_sj = X_testc.loc[X_testc.city=='sj']
X_test_iq = X_testc.loc[X_testc.city=='iq']


X_train_sj=X_train_sj.join(X_train_sj.groupby(['city','weekofyear'])['total_cases'].mean(), on=['city','weekofyear'], rsuffix='_avg')
X_test_sj=X_test_sj.join(X_train_sj.groupby(['city','weekofyear'])['total_cases'].mean(), on=['city','weekofyear'], rsuffix='_avg')
X_train_iq=X_train_iq.join(X_train_iq.groupby(['city','weekofyear'])['total_cases'].mean(), on=['city','weekofyear'], rsuffix='_avg')
X_test_iq=X_test_iq.join(X_train_iq.groupby(['city','weekofyear'])['total_cases'].mean(), on=['city','weekofyear'], rsuffix='_avg')

features2=['total_cases','total_cases', 'reanalysis_specific_humidity_g_per_kg','station_avg_temp_c','reanalysis_dew_point_temp_k','station_min_temp_c','station_max_temp_c','reanalysis_min_air_temp_k','reanalysis_max_air_temp_k','reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_specific_humidity_g_per_kg','reanalysis_dew_point_temp_k','reanalysis_min_air_temp_k','station_min_temp_c']      

#TRAIN
X_sj= X_train_sj[features2]
Y_sj = X_train_sj['total_cases']

X_iq= X_train_iq[features2]
Y_iq = X_train_iq['total_cases']

#TEST
X_sj_t= X_test_sj[features2]
X_iq_t= X_test_iq[features2]

X_sj.fillna(method='bfill', inplace=True)
X_iq.fillna(method='bfill', inplace=True)

X_sj_t.fillna(method='bfill', inplace=True)
X_iq_t.fillna(method='bfill', inplace=True)


# In[ ]:


##SAN JUAN
train_size = 100
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X_sj,Y_sj)
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)
model_sj=svr.best_estimator_
print(model_sj)


# In[ ]:


##IQUITOS
train_size = 100
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X_iq,Y_iq)
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)
model_iq=svr.best_estimator_
print(model_iq)


# In[ ]:


#KNN 
k_range = list(range(1, 50))
knnsj = KNeighborsClassifier(n_neighbors = 30)
knnsj.fit(X_sj,Y_sj)


knniq = KNeighborsClassifier(n_neighbors = 30)
knniq.fit(X_iq,Y_iq)


# ----------------
# LogisticRegression
# ----------------
logregsj = LogisticRegression()
logregsj.fit(X_sj, Y_sj)

logregiq = LogisticRegression()
logregiq.fit(X_iq, Y_iq)


# In[ ]:


map_city = {'sj':0, 'iq':1}

datasets = [X_train_sj, X_train_iq, y_train_sj ,y_train_iq,X_test_sj, X_test_iq ]

for df in datasets:
    df['city'] = df['city'].map(map_city)

#sj_train_subtrain = X_train_sj.head(700)
#sj_train_subtest = X_train_sj.tail(X_train_sj.shape[0] - 700)
sj_train_subtrain = X_train_sj.sample(frac=0.7)
sj_train_subtest = X_train_sj.loc[~X_train_sj.index.isin(sj_train_subtrain.index)]


#iq_train_subtrain = X_train_iq.head(350)
#iq_train_subtest = X_train_iq.tail(X_train_iq.shape[0] - 350)
iq_train_subtrain = X_train_iq.sample(frac=0.7)
iq_train_subtest = X_train_iq.loc[~X_train_iq.index.isin(iq_train_subtrain.index)]



iq_train_subtrain.fillna(method='bfill', inplace=True)
iq_train_subtest.fillna(method='bfill', inplace=True)
sj_train_subtrain.fillna(method='bfill', inplace=True)
sj_train_subtest.fillna(method='bfill', inplace=True)


# In[ ]:


#create preds
preds_sj= model_sj.predict(sj_train_subtest[features2]).astype(int)
preds_iq=model_iq.predict(iq_train_subtest[features2]).astype(int)
#add to the dataframes
sj_train_subtest['fitted'] = preds_sj
iq_train_subtest['fitted'] = preds_iq
### reset axis
sj_train_subtest.index = sj_train_subtest['week_start_date']
iq_train_subtest.index = iq_train_subtest['week_start_date']
## make plot


# KNN
preds_sj2 = knnsj.predict(sj_train_subtest[features2]).astype(int)
preds_iq2 = knniq.predict(iq_train_subtest[features2]).astype(int)

sj_train_subtest['fitted2'] = preds_sj2
iq_train_subtest['fitted2'] = preds_iq2

sj_train_subtest.index = sj_train_subtest['week_start_date']
iq_train_subtest.index = iq_train_subtest['week_start_date']

# Logistic regression

preds_sj3 = logregsj.predict(sj_train_subtest[features2]).astype(int)
preds_iq3 = logregiq.predict(iq_train_subtest[features2]).astype(int)

sj_train_subtest['fitted3'] = preds_sj3
iq_train_subtest['fitted3'] = preds_iq3

sj_train_subtest.index = sj_train_subtest['week_start_date']
iq_train_subtest.index = iq_train_subtest['week_start_date']


"""
plt.figure(figsize=(15,10))
sj_train_subtest.total_cases.plot(label="Actual")
sj_train_subtest.fitted.plot( label="Predictions")
plt.title("Dengue Predicted Cases vs. Actual Cases in San Juan")
plt.legend()

"""


# In[ ]:



## make plot
plt.figure(figsize=(15,10))
sj_train_subtest.total_cases.plot(label="Actual")
sj_train_subtest.fitted.plot( label="Predictions SVM")
sj_train_subtest.fitted2.plot( label="Predictions KNN")
sj_train_subtest.fitted3.plot( label="Predictions logreg")
plt.title("Dengue Predicted Cases vs. Actual Cases in San Juan")
plt.legend()


# In[ ]:


plt.figure(figsize=(15,10))
iq_train_subtest.total_cases.plot(label="Actual")
iq_train_subtest.fitted.plot(label="Predictions SVM")
iq_train_subtest.fitted2.plot( label="Predictions KNN")
iq_train_subtest.fitted3.plot( label="Predictions logreg")
plt.title("Dengue Predicted Cases vs. Actual Cases in Iquitos")
plt.legend()


# In[ ]:



scores = cross_val_score(knnsj, X_sj,Y_sj, cv=5, scoring = "accuracy")
print("Scores for San Juan using SVM:", scores)
print("Mean for San Juan using SVM:", scores.mean())

scores2 = cross_val_score(knniq, X_iq,Y_iq, cv=5, scoring = "accuracy")
print("Scores for San Juan using SVM:", scores2)
print("Mean for San Juan using SVM:", scores2.mean())


# In[ ]:


knnsj.fit(X_sj,Y_sj)
knniq.fit(X_iq,Y_iq)

sj_predictions = knnsj.predict(X_sj_t).astype(int)
iq_predictions = knniq.predict(X_iq_t).astype(int)

print(sj_predictions)

submission = pd.read_csv("../input/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])

submission.to_csv("knn.csv")


# In[ ]:




model_sj.fit(X_sj,Y_sj)
model_iq.fit(X_iq,Y_iq)

sj_predictions2 = model_sj.predict(X_sj_t).astype(int)
iq_predictions2 = model_iq.predict(X_iq_t).astype(int)

print(sj_predictions2)

submission2 = pd.read_csv("../input/submission_format.csv",
                         index_col=[0, 1, 2])

submission2.total_cases = np.concatenate([sj_predictions2, iq_predictions2])

submission2.to_csv("svm.csv")

