#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION:
# 
# The Humanoid concept was first introduced in the early ages through comlplex mechanics and electricity (Tesla). This concept has its debut not in the 90's but dated back to the ancient world. They can be errorniously termed the 'The Magic of The Ages'. The first use of this mechakraft was in industries to boost production with less human efforts. Digital control with AI developed machines were introduced in 2000's based on volume of data presently available.
# 
# The three-legged walking table of the Greeks, the humanoid text reader of the Buddhist, the stolen technology by the the Indians from Rome to build the automated soldiers which protects the relics of Buddha, Albertus android which has the capacity to perform domestic task, mechanical imitations of animals and demons invented by the early chinese and so on are prooves of existence of this technical demons as some fanatics might have called it. An holistic study of Robot mophology left me with the memories of the gods. How strong is our assurance that some of this inventions are not worshiped as gods in the primitive era? If doubtful, then Robots and gods can be lyrically alike or synonymous from human perception (Let me leave this for the deep thinkers and readers of history to phantom that out).
# 
# The almighty Tesla advanced this mechakraft inventions to a point of domestic use through the ability to remotely control them in 1910's. I still remember the great unveiling of ELEKTRO 'the seven feet tall Robot that can walk by voice command in USA'. It has ability to speak 700 words. The giant toys for the rich kids. It can even smokes cigar lol....By reflections through my knowledge, this gives me a great feeling about this competition. A chance to be part of Robot creation or can I say gods creation? 
# For more deep thoughts write me or vote this kernel.
# Now lets get started with the assignment before us.
# I hope this kernel will be of a great help.
#         

# **About The Data:
# 
# The data was collected by IMU sensor data while driving a small mobile Robot over different surface in a university premises. We are asked to predict the type of floor the Robot is walking on through the test set data provided. 
# The data provided are:
# 1. X_train having 13 columns
# 2. y_train having 3 columns 
# 3. Sample_submission
# 
# **Kernel
# 
# In this kernel, I will perform data preparation:
# data visualization,
# features engineering ,
# feature selections, 
# hyper-parameter optimization,
# modelling, 
# choice of model,
# predictions and 
# submission.
# 
# 

# **Preparing Data for Analysis**
# 
# Loading Packages

# In[ ]:


import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.preprocessing import StandardScaler
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import xgboost as xgb
import lightgbm as lgb
from seaborn import countplot,lineplot, barplot
from sklearn import model_selection
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import itertools
import json
import time
from sklearn import linear_model
import eli5
from eli5.sklearn import PermutationImportance
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import shap
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
pd.set_option('max_columns', None)
import datetime
import seaborn as sns
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold


# **Loading Data**
# 
# We will check the data files that are available and also print its formats. This helps us to know the real data we are working with. The nature of the data and how the data was collated.
# This is very crucial in for data cleansing and visualisation.
# The first code below is to access the location of the file and to check for the available file.

# In[ ]:


IS_LOCAL = False
if (IS_LOCAL):
    location = "../input/careercon/"
else:
    location = "../input/"
os.listdir(location)


# In this code, we are to load the data into our working environment for further engineering and predictions.

# In[ ]:


get_ipython().run_cell_magic('time', '', "X_train = pd.read_csv(os.path.join(location, 'X_train.csv'))\nX_test = pd.read_csv(os.path.join(location, 'X_test.csv'))\ny_train = pd.read_csv(os.path.join(location, 'y_train.csv'))")


# I will like to print the shape of the files to know the topography of my data.

# In[ ]:


print("Xtrain: {}\nXtest: {}\nytrain: {}".format(X_train.shape, X_test.shape, y_train.shape))


# We noticed that the Xtrain and Xtest have the same column but ytrain (the label) is different to Xtrain. Let us dig deep

# **Data Exploration:
# 
# We are to check the datas and have a better understanding on how the data are distributed. We will check the train and the test set by printing them.

# In[ ]:


print('Size of the Xtrain')
print('Numbers of Measurements: {0}\nNumbers of columns: {1}'.format(X_train.shape[0], X_train.shape[1]))


# In[ ]:


print('Size of the Xtest')
print('Numbers of Measurements: {0}\nNumbers of columns: {1}'.format(X_test.shape[0], X_test.shape[1]))


# In[ ]:


print('Size of the Labels')
print('Numbers of Measurements: {0}\nNumbers of columns: {1}'.format(y_train.shape[0], y_train.shape[1]))


# I built a function called show_head to show us the first five elements of our data

# In[ ]:


def show_head(data):
    return(data.head())


# In[ ]:


show_head(X_train)


# In[ ]:


show_head(X_test)


# In[ ]:


show_head(y_train)


# I will like to check if there are missing values because the ytrain and the xtrain are not equal. So i will build a function for that called missing_data comprising of total, percentage of missing data and data type of each column

# In[ ]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (total/data.isnull().count()*100)
    miss_column = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    miss_column['Types'] = types
    return(np.transpose(miss_column))            


# In[ ]:


missing_data(X_train)


# In[ ]:


missing_data(X_test)


# In[ ]:


missing_data(y_train)


# This shows that there are no missing values in both train and labels.
# I will now creat a function called describe_data, this will describe our data. We will know waht exactly the problem is in this function. this function will show us the mean, std, counts and max and min of each measurement. it seems the measurement are categorised into series.

# In[ ]:


def describe_data(data):
    return(data.describe())


# In[ ]:


describe_data(X_train)


# In[ ]:


describe_data(X_test)


# In[ ]:


describe_data(y_train)


# There is the same number of series in X_train and y_train, numbered from 0 to 3809 (total 3810). Each series have 128 measurements.
# Each series in train dataset is part of a group (numbered from 0 to 72, 72 being the half of 128).
# The number of rows in X_train and X_test differs with 6 x 128, 128 being the number of measurements for each group.

# In this code we rename the surface with Labels and also list out the numbers of floors and the type of floors. Thus shows that we are dealing with 9 classification model problem. We can also see the level of imbalance in this data

# In[ ]:


Surface_count = y_train['surface'].value_counts().reset_index().rename(columns = {'index' : 'Labels'})
Surface_count


# **Data Visualization:
# 
# In this section, we will visualize the label data to know which floor is most common. 
# 
# Density plot, count_plot, Boxplot, frequency distribution, pie_chart, histogram_plot etc......

# **COUNTPLOT**

# In the plot below, we represent each counts of the surface on a barchat to clearly see the most common floor. The concrete floor is the most common this may be as a result of little funding allocated to universities lol.....

# In[ ]:


countplot(y = 'surface', data = y_train)
plt.show()


# **PieChartPlot**

# In[ ]:


trace = go.Pie(labels = y_train['surface'].value_counts().index,
              values = y_train['surface'].value_counts().values,
              domain = {'x':[0.55,1]})

data = [trace]
layout = go.Layout(title = 'PieChat Distribution of Floors')
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# **Density Plot**
# 
# I will show the density plot of the variables in train abd test set. These plots will be represented in different colors of different surface values.
# 
# This plot shows how the surface are distributed over the columns and how columns are distributed over the test and train dataset.
# 
# From the density plot below, the following are noticed:
# 
# 1. Orientation x, y are not normally distributed
# 2. Orientation z, w are normally distributed
# 3. linear_accelaration are normally distributed

# In[ ]:


def plot_columns_distribution(df1, df2, label1, label2, columns):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,5,figsize=(16,8))

    for col in columns:
        i += 1
        plt.subplot(2,5,i)
        sns.kdeplot(df1[col], bw=0.5,label=label1)
        sns.kdeplot(df2[col], bw=0.5,label=label2)
        plt.xlabel(col, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();
    
columns = X_train.columns.values[3:]
plot_columns_distribution(X_train, X_test, 'train', 'test', columns)


# In[ ]:


def plot_columns_class_distribution(classes,series_group, columns):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(5,2,figsize=(16,24))

    for col in columns:
        i += 1
        plt.subplot(5,2,i)
        for clas in classes:
            series_groups = series_group[series_group['surface']==clas]
            sns.kdeplot(series_groups[col], bw=0.5,label=clas)
        plt.xlabel(col, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();
    
classes = (y_train['surface'].value_counts()).index
series_group = X_train.merge(y_train, on='series_id', how='inner')
plot_columns_class_distribution(classes, series_group, columns)


# **Data Engineering**
# 
# This was borrowed from: https://www.kaggle.com/prashantkikani/help-humanity-by-helping-robots and https://www.kaggle.com/artgor/where-do-the-robots-drive 

# In[ ]:


def quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

def data_engineering(actual):
    new = pd.DataFrame()
    actual['total_angular_velocity'] = (actual['angular_velocity_X'] ** 2 + actual['angular_velocity_Y'] ** 2 + actual['angular_velocity_Z'] ** 2) ** 0.5
    actual['total_linear_acceleration'] = (actual['linear_acceleration_X'] ** 2 + actual['linear_acceleration_Y'] ** 2 + actual['linear_acceleration_Z'] ** 2) ** 0.5
    
    actual['acc_vs_vel'] = actual['total_linear_acceleration'] / actual['total_angular_velocity']
    
    x, y, z, w = actual['orientation_X'].tolist(), actual['orientation_Y'].tolist(), actual['orientation_Z'].tolist(), actual['orientation_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    actual['euler_x'] = nx
    actual['euler_y'] = ny
    actual['euler_z'] = nz
    
    actual['total_angle'] = (actual['euler_x'] ** 2 + actual['euler_y'] ** 2 + actual['euler_z'] ** 2) ** 5
    actual['angle_vs_acc'] = actual['total_angle'] / actual['total_linear_acceleration']
    actual['angle_vs_vel'] = actual['total_angle'] / actual['total_angular_velocity']
    
    def f1(x):
        return np.mean(np.diff(np.abs(np.diff(x))))
    
    def f2(x):
        return np.mean(np.abs(np.diff(x)))
    
    for col in actual.columns:
        if col in ['row_id', 'series_id', 'measurement_number']:
            continue
        new[col + '_mean'] = actual.groupby(['series_id'])[col].mean()
        new[col + '_min'] = actual.groupby(['series_id'])[col].min()
        new[col + '_max'] = actual.groupby(['series_id'])[col].max()
        new[col + '_std'] = actual.groupby(['series_id'])[col].std()
        new[col + '_max_to_min'] = new[col + '_max'] / new[col + '_min']
        
        # Change. 1st order.
        new[col + '_mean_abs_change'] = actual.groupby('series_id')[col].apply(f2)
        
        # Change of Change. 2nd order.
        new[col + '_mean_change_of_abs_change'] = actual.groupby('series_id')[col].apply(f1)
        
        new[col + '_abs_max'] = actual.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))
        new[col + '_abs_min'] = actual.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))

    return new


# In[ ]:


get_ipython().run_cell_magic('time', '', 'xtrain = data_engineering(X_train)\nxtest = data_engineering(X_test)')


# In[ ]:


describe_data(xtrain)


# From the data engineering we generated about 9 more columns making 22 columns in our train and test set

# In[ ]:


print("Xtrain: {}\nXtest: {}".format(X_train.shape, X_test.shape))


# In[ ]:


print("Xtrain: {}\nXtest: {}".format(xtrain.shape, xtest.shape))


# In[ ]:


show_head(xtest)


# **Correlation Matrix**

# In[ ]:


corr_xtrain = xtrain.corr()
corr_xtrain


# **Pearson Correlation**

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(24,18))
plt.title('Pearson Correlation of Features', y=1.05, size=20)
sns.heatmap(X_train.astype(float).corr(),linewidths=0.05,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# **Introducing The Model**
# 
# Firstly, RandomForestClassifier model will be used because i believe is the best random for this type od classification.
# 
# I will run trough feature selection process and four other models will be introduced 
# 
# The choice of model will later be made base on some few creterials that will be listed later.

# Encoding The  Labels using labelEncoder

# In[ ]:


le = LabelEncoder()
y_train['surface'] = le.fit_transform(y_train['surface'])


# In[ ]:


xtrain.fillna(0, inplace = True)
xtrain.replace(-np.inf, 0, inplace = True)
xtrain.replace(np.inf, 0, inplace = True)
xtest.fillna(0, inplace = True)
xtest.replace(-np.inf, 0, inplace = True)
xtest.replace(np.inf, 0, inplace = True)


# In[ ]:



folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
def feed_model(train,test,label,folds=folds,averaging='usual',clf=None,clf_type='rfc',params=None,
               plot_feature_importance=False,groups=y_train['group_id']):
    forcast = np.zeros((test.shape[0], 9))
    con_pred = np.zeros((train.shape[0]))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_, (train_index, valid_index) in enumerate(folds.split(train, label, groups)):
        print('Fold', fold_, 'started at', time.ctime())
        x_train, x_valid = train.iloc[train_index], train.iloc[valid_index]
        y_train, y_valid = label.iloc[train_index], label.iloc[valid_index]
        
        if clf_type == 'rfc':
            clf = clf
            clf.fit(x_train, y_train)
            Valid_pred = clf.predict(x_valid).reshape(-1,)
            score = accuracy_score(y_valid, Valid_pred)
            Real_pred = clf.predict_proba(test)
            
        con_pred[valid_index] = clf.predict(x_valid).reshape(-1,)
        scores.append(accuracy_score(y_valid, Valid_pred))
        if averaging == 'usual':
            forcast += Real_pred
        elif averaging == 'rank':
            forcast += pd.Series(Real_pred).rank().values
        
    forcast /= folds.n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if clf_type == 'lgb':
        feature_importance["importance"] /= folds.n_splits
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                       ascending=False)[:50].index

            best_features = feature_importance[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return Valid_pred, forcast, feature_importance
        return Valid_pred, forcast, scores
    
    else:
        return Valid_pred, forcast, scores


# In[ ]:


clf = RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=0)
con_pred_rfc1, forcast_rfc1, scores_rfc1 = feed_model(train=xtrain, test=xtest, label=y_train['surface'],folds=folds,clf_type='rfc',
                                                   plot_feature_importance=True,clf=clf)


# **Confusion Matrix**
# 
# This helps to know the effective of the model. The true positive and true negative are important. 
# 
# Heavily copied form :https://www.kaggle.com/prashantkikani/help-humanity-by-helping-robots

# **Feature Importance and Visualization**
# 
# This graph shows the level of importance of each features. It will help to determine feature selection later.

# In[ ]:


FeatureImportance = clf.feature_importances_
indices = np.argsort(FeatureImportance)
features = xtrain.columns

hm = 30
plt.figure(figsize=(16, 12))
plt.title('RFC Features Avg Over Folds')
plt.barh(range(len(indices[:hm])), FeatureImportance[indices][:hm], color='b', align='center')
plt.yticks(range(len(indices[:hm])), [features[i] for i in indices])
plt.xlabel('Importance')
plt.show()


# In[ ]:


feat_labels = xtrain.columns


# The scores shows the importance of each feature. The higher the scores the higher the importance of the feature. The score are addedup to 100%.
# 
# To identify and select the most important features we need to select a range of threshold as a cut off point for the importance and compare. this is what we will use to train the model again and compare the importance. 
# 
# I ranges the cut off from 0.004 to 0.006, i noticed that the score at 0.005 was the highers. Above 0.005, the score begins to reduce.

# In[ ]:


for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)


# In[ ]:


from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn import datasets
sfm = SelectFromModel(clf, threshold=0.001)
sfm.fit(xtrain, y_train['surface'])
for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])
    
xtrain_importance = sfm.transform(xtrain)
xtest_importance = sfm.transform(xtest)


# Wow this is a drop. If we can tune parameters and check again

# In[ ]:


folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
def feed_model_importance(train,test,label,folds=folds,averaging='usual',clf=None,clf_type='rfc',params=None,
               plot_feature_importance=False,groups=y_train['group_id']):
    forcast = np.zeros((test.shape[0], 9))
    con_pred = np.zeros((train.shape[0]))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train, label, groups)):
        print('Fold', fold_n, 'started at', time.ctime())
        x_train, x_valid = train[train_index], train[valid_index]
        y_train, y_valid = label[train_index], label[valid_index]
        
        if clf_type == 'rfc':
            clf = clf
            clf.fit(x_train, y_train)
            Valid_pred = clf.predict(x_valid).reshape(-1,)
            score = accuracy_score(y_valid, Valid_pred)
            Real_pred = clf.predict_proba(test)
            
        con_pred[valid_index] += (Valid_pred).reshape(-1,)
        scores.append(accuracy_score(y_valid, Valid_pred))
        if averaging == 'usual':
            forcast += Real_pred
        elif averaging == 'rank':
            forcast += pd.Series(Real_pred).rank().values
        
    forcast /= folds.n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if clf_type == 'lgb':
        feature_importance["importance"] /= n_folds
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return Valid_pred, forcast, feature_importance
        return Valid_pred, forcast, scores
    
    else:
        return Valid_pred, forcast, scores


# In[ ]:


clf1 = RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=0)
con_pred_rfc, forcast_rfc, scores_rfc = feed_model_importance(train=xtrain_importance, test=xtest_importance, label=y_train['surface'],folds=folds,clf_type='rfc',
                                                   plot_feature_importance=False,clf=clf1)


# I will try to use Neural Network model.
# 
# The validation as seen in the output below is low....still working on how to use cross val in a NN.
# Ideas on this will be acceted. The can also be posted in the comment section.

# In[ ]:


from keras import models
from keras import layers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
np.random.seed(0)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)

def create_network():
    network = models.Sequential()
    network.add(layers.Dense(units=16, activation='relu', input_dim= 171))
    network.add(layers.Dense(units=16, activation='relu'))
    network.add(layers.Dense(units=1, activation='sigmoid'))

    network.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    
    # Return compiled network
    return network
# Wrap Keras model so it can be used by scikit-learn
es = EarlyStopping(monitor='val_loss',mode='auto',verbose=1,patience=20)
mc = ModelCheckpoint('best_model.h5',monitor='val_acc',mode='max',verbose=1,save_best_only=True)
neural_network = KerasClassifier(build_fn=create_network,epochs=10,batch_size=100,verbose=0)
cross_val_score(neural_network, xtrain, y_train['surface'],fit_params={'callbacks': [es,mc]}, cv=folds)


# **Submission**
# 
# I will submit this values for now. 
# 
# work will continue on the hyper parameters and feature selections.......

# In[ ]:


sub = pd.read_csv(os.path.join(location,'sample_submission.csv'))
sub['surface'] = le.inverse_transform(forcast_rfc.argmax(axis=1))
sub.to_csv('Forcasting.csv', index=False)
sub.head(20)


# If you like this kernel kindly upvote 
# 
# Any help on using Kfolds validation in Neural Network will be highly appreciated.

# **Refrences**
# 
# 1. https://www.kaggle.com/prashantkikani/help-humanity-by-helping-robots
# 2. https://www.kaggle.com/artgor/where-do-the-robots-drive 

# Work Continues.............
