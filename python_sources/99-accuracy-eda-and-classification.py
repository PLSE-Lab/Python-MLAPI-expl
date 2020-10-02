#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 08:59:51 2020


# In[ ]:


@author: jaket
"""
#j2p
### Exercise pattern classification


# This data was collected from activity tracking device(s) (unspecified). There is a training data set with a known class ranging<br>
# From A-E, which should be preticted in the test data set. Many of the features are not defined, limiting the domain-specific<br>
# preprocessing and EDA we can do.

# # Set up<br>
# Import, set, read, initial exploration

# Import packages

# In[ ]:


import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings 
import seaborn as sns
import sklearn
from datetime import datetime
import calendar
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', 1000)
import warnings
warnings.filterwarnings('ignore')


# Set wd and read in data

# In[ ]:





# In[ ]:


train_df = pd.read_csv('../input/exercisepatternpredict/pml-training.csv', error_bad_lines=False, index_col=False).drop('Unnamed: 0', axis=1)
test_df = pd.read_csv('../input/exercisepatternpredict/pml-testing.csv', error_bad_lines=False, index_col=False).drop('Unnamed: 0', axis=1)


# Randomise the rows, it is currently very structured and improve training predictability

# In[ ]:


train_df = train_df.sample(frac=1).reset_index(drop=True)


# Explore

# In[ ]:


print(train_df.columns.values)
print(train_df.isna().sum()) 


# In[ ]:


sns.heatmap(train_df.isnull(), cbar=False) # Heatmap to visualise NAs


# It's clear we have a large amount of missing data in some columns. It's likely that where this data is available is specific to<br>
# a certain exercise type/class. Due to the amount, imputation will not be effective. In all other columns there is no missing data<br>
# It could be appropriate to create a seperate df in the cases which have this additional data.

# In[ ]:


train_df.describe()


# # EDA<br>
# First, it would make sense to run some pairplots using class as the Hue, so we can begin to determine which variables are related<br>
# To the target. We can only do this in the training data set. Lets start by examining the frequency of classes and participants

# In[ ]:


train_df['classe']=train_df['classe'].astype('category')


# In[ ]:


freq_plot1=train_df.filter(items=['user_name', 'classe'])
freq_plot1=freq_plot1.groupby(['user_name'])['classe'].agg(counts='value_counts').reset_index()


# In[ ]:


sns.barplot(data = freq_plot1, x = 'counts', y = 'user_name', hue = 'classe', ci = None)


# Its clear that the frequency of class A is much greater than each other class. The classes are not divided equally between users.<br>
# For example, Jeremy has around 2x as many A's as other classes. This is important as user may be an important predictive<br>
# Variable when predicting the test data, and we will later need to OH Encode user.

# Now lets plot some pairplots...

# In[ ]:



pairplot1=train_df.filter(items=['num_window', 'roll_belt', 'pitch_belt', 'yaw_belt', 'total_accel_belt', 'classe'])
sns.pairplot(pairplot1, hue='classe',  plot_kws = {'alpha': 0.6, 'edgecolor': 'k'},size = 4)


# There aren't any large differences in these relationships regarding class. However, the distribution does seem to be slightly different<br>
# Across different classes. Lets look at the X and Y axes of the gyros, accel and magnet belt.

# In[ ]:


pairplot2=train_df.filter(items=['num_window', 'gyros_belt_x', 'gyros_belt_y', 'accel_belt_x', 'accel_belt_y',  'magnet_belt_x','magnet_belt_y', 'classe'])
sns.pairplot(pairplot2, hue='classe',  plot_kws = {'alpha': 0.6,  'edgecolor': 'k'},size = 4)


# Again the data is closely grouped, though we can see that movement D has some determinable features not shared by the others on<br>
# These axes. Can the Z-axes provide any more information?

# In[ ]:


pairplot3=train_df.filter(items=['num_window', 'gyros_belt_z', 'accel_belt_z', 'magnet_belt_z', 'classe'])
sns.pairplot(pairplot3, hue='classe',  plot_kws = {'alpha': 0.6, 'edgecolor': 'k'},size = 4)


# Now we see some extremely distinctive features of class D on the Z axis. This means these variables will be important in classification.

# In[ ]:


pairplot4=train_df.filter(items=['roll_arm', 'pitch_arm', 'yaw_arm', 'total_accel_arm', 'classe'])
sns.pairplot(pairplot4, hue='classe',  plot_kws = {'alpha': 0.6, 'edgecolor': 'k'},size = 4)


# The arm data shows significantly different patterns from the belt data. However, there is not a considerable difference between classes.<br>
# Again, lets look at them on the x/y and then the Z axis.

# In[ ]:


pairplot5=train_df.filter(items=['num_window', 'gyros_arm_x', 'gyros_arm_y', 'accel_arm_x', 'accel_arm_y',  'magnet_arm_x','magnet_arm_y', 'classe'])
sns.pairplot(pairplot5, hue='classe',  plot_kws = {'alpha': 0.6,  'edgecolor': 'k'},size = 4)


# And on the Z:

# In[ ]:


pairplot6=train_df.filter(items=['num_window', 'gyros_arm_z', 'accel_arm_z', 'magnet_arm_z', 'classe'])
sns.pairplot(pairplot6, hue='classe',  plot_kws = {'alpha': 0.6, 'edgecolor': 'k'},size = 4)


# The differences here are again subtle. The distribution of A seems to take a distinctly different pattern from the others.

# We can do similar plots for the forearm variables if we wanted but i'll skip it for now. Lastly, lets take a look at some of the<br>
# Variables where nearly all data is missing. If this is uninformative, it makes sense for us to remove it, however it may give<br>
# away a certain class clearly. Since there are many of these I shall just pick a few out.

# In[ ]:


pairplot7=train_df.filter(items=['skewness_roll_belt', 'max_roll_belt', 'max_picth_belt', 
                                 'var_total_accel_belt', 'stdev_roll_belt',
                                 'avg_yaw_belt', 'classe'])
sns.pairplot(pairplot7, hue='classe',  plot_kws = {'alpha': 0.6, 'edgecolor': 'k'},size = 4)


# These dont show any clear associations with class. Given the small (<1%) fraction of the data available, it makes sense that we remove these.

# # Preprocessing

# Before we move to modelling we should consider feature removal, feature engineering and categorical encoding amongst other things.

# # Feature Removal

# First, drop cols with high % NA

# In[ ]:


print(train_df.isna().sum()) 
train_df = train_df.loc[:, train_df.isnull().mean() < .8] #remove cols with <80% completeness.
test_df = test_df.loc[:, test_df.isnull().mean() < .8] #remove cols with <80% completeness.


# The timestamps are not going to be useful for predicting on the other data sets. Also, they would not be informative in real-life<br>
# activity prediction. It make be that correctly used these timestampts could give away the entire answer if they align with the<br>
# Test data, which will be the case if the test data is a random sample of train. We'll lose these for now. to make it more realistic.<br>
# For the same reasons, it makes sense to also lose num window and new window

# In[ ]:


train_df = train_df.drop(['raw_timestamp_part_1', 'raw_timestamp_part_2' ,'cvtd_timestamp', 'new_window','num_window'], axis=1)
test_df = test_df.drop(['raw_timestamp_part_1', 'raw_timestamp_part_2' ,'cvtd_timestamp', 'new_window','num_window', 'problem_id'], axis=1)


# # Feature Engineering

# As we dont have an excessive number of features, nor a considerable amount of categorical variables to expand our X-cols, we<br>
# Can consider generating some interaction features. For example, lets combine all x y and z

# Generate a fn to turn 0s into 1s as it not ruin the interaction variables

# In[ ]:


def zeros_to_ones(x):
    x = np.where(x==0, 1, x)
    return(x)


# Np.prod will give us the product (multiple) of all columns for a given row, creating an interaction variable on the axis.    

# In[ ]:


def feat_eng (df):
    df['x_axis_feat']=df[df.columns[df.columns.to_series().str.contains('_x')]].apply(zeros_to_ones).apply(np.prod, axis=1)
    df['y_axis_feat']=df[df.columns[df.columns.to_series().str.contains('_y')]].apply(zeros_to_ones).apply(np.prod, axis=1)
    df['z_axis_feat']=df[df.columns[df.columns.to_series().str.contains('_z')]].apply(zeros_to_ones).apply(np.prod, axis=1)
    
    # Lets interact all belt, arm, dumbell and forearm variables
    
    df['belt_feat']=df[df.columns[df.columns.to_series().str.contains('_belt')]].apply(zeros_to_ones).apply(np.prod, axis=1)
    df['arm_feat']=df[df.columns[df.columns.to_series().str.contains('_arm')]].apply(zeros_to_ones).apply(np.prod, axis=1)
    df['forearm_feat']=df[df.columns[df.columns.to_series().str.contains('_forearm')]].apply(zeros_to_ones).apply(np.prod, axis=1)
    
    # Let's interact all magnet, accel and gyros variables
    
    df['accel_feat']=df[df.columns[df.columns.to_series().str.contains('accel_')]].apply(zeros_to_ones).apply(np.prod, axis=1)
    df['magnet_feat']=df[df.columns[df.columns.to_series().str.contains('magnet_')]].apply(zeros_to_ones).apply(np.prod, axis=1)
    df['gyros_feat']=df[df.columns[df.columns.to_series().str.contains('gyros_')]].apply(zeros_to_ones).apply(np.prod, axis=1)
    
    return(df)


# In[ ]:


train_df=feat_eng(train_df)
test_df=feat_eng(test_df)


# We could continue to generate more features by interacting newly engineered features, or in new combinations, and this may give<br>
# us some additional model performance however due to time and computation restraints we'll leave it here.

# # Encoding

# Only 2 encoding processes need to be done. (1) to one hot encode the user and (2) to label encode the outcome.

# In[ ]:


def Encode_fn(df):
    users=pd.get_dummies(df['user_name']) #OneHot encode username
    df=pd.concat([df, users], axis=1).reset_index(drop=True) #Join to modelling df
    df=df.drop('user_name', axis=1) #Drop original username var
    return(df)


# In[ ]:


train_df=Encode_fn(train_df)
test_df=Encode_fn(test_df)


# Label encode target

# In[ ]:


train_df['classe']=train_df['classe'].astype('category') # Ensure the target is cat
train_df['target']=train_df['classe'].cat.codes # Label encoding
train_df['target']=train_df['target'].astype('category') # Ensure the target is cat
train_df=train_df.drop('classe', axis=1)


# # Splitting

# In[ ]:


from sklearn.model_selection import train_test_split


# Define features and labels

# In[ ]:


X=train_df.drop('target', axis=1).reset_index(drop=True)
y=train_df['target']


# efine train and test

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# # Initial classification testing

# Load packages

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,  QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, precision_score, recall_score, f1_score


# Select classification algos

# In[ ]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]


# Log results for performance vis

# In[ ]:


log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)


# Run algo loop

# In[ ]:


for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    
    # calculate score
    precision = precision_score(y_test, train_predictions, average = 'macro') 
    recall = recall_score(y_test, train_predictions, average = 'macro') 
    f_score = f1_score(y_test, train_predictions, average = 'macro')
    
    
    print("Precision: {:.4%}".format(precision))
    print("Recall: {:.4%}".format(recall))
    print("F-score: {:.4%}".format(recall))
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# # Plot results of algo testing

# Accuracy

# In[ ]:


sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")


# Log Loss.

# In[ ]:


sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")


# It is clear that random forest does an extremely good job of classifying these. Usually I would opt to tune multiple algos but based<br>
# On the accuracy of the RF i'll just do some brief tuning.

# First, lets consider variable importance

# In[ ]:


rf = RandomForestClassifier(n_estimators=500, random_state = 42)
rf.fit(X_train, y_train);
feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(25).plot(kind='barh')


# No engineered features seem that important according to the RF, likely because theyre interactions (/derivatives). It would make sense<br>
# to go back and remove these to test the RF accuracy without engineering, but i'll leave that for now.

# # Parameter tuning<br>
# We'll tune the RF using a random search grid. We could use a grid search but it is computationally expensive and given that we're<br>
# On >99% accuracy and only need to predict a data set size of 20, I think we can manage without.

# Run randomized search

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# Number of trees in random forest

# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 10, stop = 20, num = 10)]


# Number of features to consider at every split

# In[ ]:


max_features = ['auto', 'sqrt']


# Maximum number of levels in tree

# In[ ]:


max_depth = [int(x) for x in np.linspace(10, 1000, num = 10)]
max_depth.append(None)


# Minimum number of samples required to split a node

# In[ ]:


min_samples_split = [2, 5, 10]


# Minimum number of samples required at each leaf node

# In[ ]:


min_samples_leaf = [2, 4, 10, 100]


# Method of selecting samples for training each tree

# In[ ]:


bootstrap = [True, False]


# Create the random grid

# In[ ]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# Use the random grid to search for best hyperparameters<br>
# Random search of parameters, using 3 fold cross validation, <br>
# Search across 100 different combinations, and use all available cores

# In[ ]:


rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               n_iter = 100, cv = 3, verbose=2, 
                               random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[ ]:


print(rf_random.best_params_)


# Fit the tuned model

# In[ ]:


best_params_rf = rf_random.best_estimator_
best_params_rf.fit(X_train,y_train)


# Predict test data

# In[ ]:


y_pred_rf = best_params_rf.predict(X_test)


# Evaluate

# In[ ]:


precision = precision_score(y_test, y_pred_rf, average = 'macro') 
recall = recall_score(y_test, y_pred_rf, average = 'macro') 
f_score = f1_score(y_test, y_pred_rf, average = 'macro')
    
    
print("Precision: {:.4%}".format(precision))
print("Recall: {:.4%}".format(recall))
print("F-score: {:.4%}".format(recall))


# # Final Predictions

# In[ ]:


final_predictions = best_params_rf.predict(test_df)


# In[ ]:


print(final_predictions)


# Convert Py to Notebook

# In[ ]:


get_ipython().system(' p2j Exercise classification.py')


# In[ ]:




