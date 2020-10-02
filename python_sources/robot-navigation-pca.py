#!/usr/bin/env python
# coding: utf-8

# In[57]:


import os
import random

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.feature_selection import RFECV

from keras import utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_Xtrain = pd.read_csv('../input/X_train.csv')
df_ytrain = pd.read_csv('../input/y_train.csv')
df_Xtest = pd.read_csv('../input/X_test.csv')


# In[3]:


df_Xtrain.head()


# In[4]:


df_Xtrain.describe()


# In[5]:


df_Xtrain.shape


# In[6]:


df_Xtrain[df_Xtrain['series_id']==1].shape


# In[7]:


df_ytrain.head()


# In[8]:


df_ytrain.shape


# In[9]:


df_Xtest.head()


# In[10]:


df_Xtest.shape


# ## Data Preprocessing

# In[12]:


#checking if there are missing data
df_Xtrain.isnull().sum()


# In[13]:


#checking duplicates
df_Xtrain.duplicated().value_counts()


# Summary: no missing values, no duplicates

# # Data Exploration

# In[14]:


#merge X_train and y_train datasets to explore data
df_explore = pd.merge(df_Xtrain, df_ytrain,  on='series_id', how='inner')
df_explore.head()


# In[15]:


robot_stats = ['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W',
              'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z',
              'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']


# In[16]:


#convert surface variable from plain object type into a categorical type
#ordered according to their proportions in the dataset
surface_types = df_explore['surface'].unique()
surfaces = pd.api.types.CategoricalDtype(ordered = True, categories = surface_types)
df_explore['surface'] = df_explore['surface'].astype(surfaces)
targets = df_explore['surface'].value_counts().index
targets


# ## Surfaces

# ### Surfaces countplot

# In[18]:


# create a list of tick positions by computing the length of the longest bar in terms of proportion
n_surface = df_ytrain['surface'].shape[0]
type_counts = df_ytrain['surface'].value_counts()
max_type_count = type_counts.max()
max_prop = max_type_count/n_surface
max_prop


# In[19]:


# create evenly spaced proportioned values between 0 and max in steps of 2%
tick_props = np.arange(0, max_prop, 0.02)
tick_props


# In[20]:


# create evenly spaced proportioned values between 0 and max in steps of 2%
tick_props = np.arange(0, max_prop, 0.02)
tick_props


# In[21]:


# format tick labels
tick_names = ['{:0.2f}'.format(v) for v in tick_props]


# In[22]:


base_color = sns.color_palette()[0]

sns.countplot(data = df_ytrain, y = 'surface', color=base_color, order=targets);
plt.xticks(tick_props*n_surface, tick_names);
plt.xlabel('proportion');
for i in range(type_counts.shape[0]):
    #X position just after the end of the bar:
    count = type_counts[i]
    #Y position which starts at 0 and increments with the loop
    pct_string = '{:0.1f}%'.format(100*count/n_surface)
    # the string to be printed, the percentage, centered vertically with VA parameter
    plt.text(count+1, i, pct_string, va = 'center');


# ## Summary: Surfaces
# So the most common surfaces that we have recordings are concrete (20%) and soft_pvc (19%). Then we have subcategory of concrete (fine_concrete - 9.5%) and different type of tiles (hard, hard/large_space, soft, tiled = ca. 30%). So the whole picture of surfaces looks like this (aggregated by all subtypes): concrete = ca. 30%, tiles = ca. 30%, soft_pvc = 19%, wood = 16%, carpet = 5%. <br>
# The most difficult part will be probably to *differentiate between different tiles types*, because they together constitute a large part of recordings with subtle differences and so may increase the number of false positives.

# ## Surfaces Summary table: by group_id and series_id

# In[24]:


#adding unique group_id list as column
surface = df_ytrain.groupby(df_ytrain['surface']).group_id.unique().reset_index()
#adding number of groups as column
surface['num_groups'] = [len(surface['group_id'].iloc[x]) for x in range(9)]
#adding unique series_id list as a column
surface['series_id'] = df_ytrain.groupby(df_ytrain['surface']).series_id.unique().reset_index().series_id
#adding number of series per surface as column
surface['num_series'] = df_ytrain.groupby(df_ytrain['surface']).series_id.count().values
surface


# # Robot stats

# ## Correlation matrix for robot stats

# In[25]:


sns.heatmap(df_explore[robot_stats].corr(), cmap = 'rocket_r', annot = True, fmt = '.2f');


# Summary:
# - Orientation_X and Orientation_W are strongly correlated (1.00)
# - Orientation_Y and Orientation_Z are strongly correlated (1.00)
# - Angular_velocity_Y and Angular_velocity_Z are highly negatively correlated (-0.76)
# - Linear_acceleration_Z and Linear_acceleration_Y are positively correlated (0.38)

# In[27]:


def surface_allseries_subset(surface_type):
    surface_series = surface[surface['surface'] == surface_type]['series_id'].values[0]
    data_surface = pd.DataFrame()
    for i in range(len(surface_series)):
        subset = df_explore[df_explore['series_id']==surface_series[i]]
        data_surface = pd.concat([data_surface, subset], axis=0)
    return data_surface


# In[28]:


def plot_stats_corr_for_surface(surface_type):
    subset = surface_allseries_subset(surface_type)
    sns.heatmap(subset[robot_stats].corr(), cmap = 'viridis', annot = True, fmt = '.2f');
    plt.title('Correlations for {} series'.format(surface_type),  fontsize=20);


# In[29]:


#Change surface type (e.g. 'carpet', 'concrete'...) to plot correlations for robot stat per surface
plot_stats_corr_for_surface('carpet');


# In[30]:


plot_stats_corr_for_surface('concrete');


# ## Mean values per surface for all series:

# In[31]:


plt.figure(figsize=(26, 13))
for i,col in enumerate(df_explore.columns[3:13]):
    ax = plt.subplot(2,5,i+1)
    ax = plt.title(col)
    df_per_surface = df_explore.groupby(df_explore['surface']).mean()
    sns.barplot(targets, df_per_surface[col], color = base_color);
    plt.xticks(rotation=90);


# ## Standard deviation per surface for all series:

# In[32]:


plt.figure(figsize=(26, 13))
for i,col in enumerate(df_explore.columns[3:13]):
    ax = plt.subplot(2,5,i+1)
    ax = plt.title(col)
    df_per_surface = df_explore.groupby(df_explore['surface']).std()
    sns.barplot(targets, df_per_surface[col], color = base_color);
    plt.xticks(rotation=90);


# ## Plotting raw data for each robot stat for ALL series per surface:

# In[33]:


def plot_robot_stat_persurface(robot_stat):
    plt.figure(figsize=(13, 13))
    print(robot_stat)
    subset = df_explore[[robot_stat, 'surface', 'series_id']]
    for i, target in enumerate(targets):
        surface_subset = subset[subset['surface']==target]
        ax = plt.subplot(3,3,i+1)
        ax = plt.title(target)
        plt.plot(surface_subset[robot_stat]);
        if robot_stat in('orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W'):
            plt.ylim(-1.25,1.25)
        elif robot_stat in('angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z'):
            plt.ylim(-2, 2);
        else:
            plt.ylim(-75, 75);
        plt.xticks([], []);


# In[34]:


#plots all series for each surface
# change robot_stats values from 0 to 9
plot_robot_stat_persurface(robot_stats[8]);


# # Modeling:

# ### Label encoding:

# In[37]:


# encodes targets as numbers from 0 to 8
# to be used in supervised ML algorithms 
le = LabelEncoder()
y_train_ML = le.fit_transform(df_ytrain['surface'])
y_train_ML


# In[36]:


# encodes targets as dummies, but needs numerical categories (that's why used on top of y_train_ML)
# to be used in deep learning
y_train_keras = utils.to_categorical(y_train_ML)
y_train_keras


# ### Preparing X_train and X_test - running PCA and aggregating (mean, std) to the series level

# In[39]:


def preparing_data_for_prediction(data):
    X = StandardScaler().fit_transform(data[robot_stats]) 
    pca = PCA(7)
    X_pca = pca.fit_transform(X)
    print("variance explained by pca components:", (pca.explained_variance_ratio_*100).astype(int))
    
    df_pca = pd.concat([pd.DataFrame(X_pca), data['series_id']], axis=1)
    X_train_mean = df_pca.groupby(data['series_id']).mean()
    X_train_std = df_pca.groupby(data['series_id']).std()
    X_train = pd.concat([X_train_mean, X_train_std], axis=1)
    X_train = X_train.drop(['series_id'], axis=1)
    
    return X_train


# Summary: <br>
# I chose 7 components because together they explain 93% of variation in the dataset. In the modeling part, I will use pca components aggregated to means and standard deviations for each serie to train and predict the floor types. <br>

# In[40]:


X_train = preparing_data_for_prediction(df_explore)


# In[41]:


print(X_train.shape)
X_train.head()


# Summary: the final X_train dataset has 14 variables - 7 aggregated pca means for each serie and 7 aggregated pca standard deviations for each serie. We are ready to run the models!

# In[42]:


X_test = preparing_data_for_prediction(df_Xtest)


# In[43]:


print(X_test.shape)
X_test.head()


# ## Supervised ML algorithms: 
# ### Bagging, RandomForest, GradientDescent and AdaBoost Classifier

# In[44]:


# Split data into training and testing sets for validation
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, 
                                                    y_train_ML, 
                                                    test_size = 0.3, 
                                                    random_state = 7)


# In[45]:


X_train_val.shape


# In[46]:


X_test_val.shape


# In[47]:


# Initialize models
clf_rf = RandomForestClassifier(n_estimators = 400, random_state = 432)
clf_Ada = AdaBoostClassifier(n_estimators = 400, random_state = 432)
clf_gradient = GradientBoostingClassifier(n_estimators = 400, random_state = 432)
clf_bagging = BaggingClassifier(n_estimators = 400, random_state = 432)


# In[48]:


# Fit models
clf_rf.fit(X_train_val, y_train_val)
clf_Ada.fit(X_train_val, y_train_val)
clf_gradient.fit(X_train_val, y_train_val)
clf_bagging.fit(X_train_val, y_train_val)


# In[49]:


# Predict using test data
pred_rf = clf_rf.predict(X_test_val)
pred_Ada = clf_Ada.predict(X_test_val)
pred_gradient = clf_gradient.predict(X_test_val)
pred_bagging = clf_bagging.predict(X_test_val)


# In[50]:


def print_metrics(y_true, preds, model_name=None):
    '''
    INPUT:
    y_true - the y values that are actually true in the dataset (numpy array or pandas series)
    preds - the predictions for those values from some model (numpy array or pandas series)
    model_name - (str - optional) a name associated with the model if you would like to add it to the print statements 
    
    OUTPUT:
    None - prints the accuracy
    '''
    if model_name == None:
        print('Accuracy score: ', format(accuracy_score(y_true, preds)))
    
    else:
        print('Accuracy score for ' + model_name + ' :' , format(accuracy_score(y_true, preds)))


# In[51]:


# Print scores
print_metrics(y_test_val, pred_rf, "Random Forest")
print_metrics(y_test_val, pred_Ada, "AdaBoost")
print_metrics(y_test_val, pred_gradient, "Gradient")
print_metrics(y_test_val, pred_bagging, "Bagging")


# In[52]:


confusion_matrix(y_test_val,pred_rf)


# In[53]:


predicted = np.zeros((df_Xtest.shape[0],9))
predicted = clf_bagging.predict(X_test)
predicted


# In[55]:


submission = pd.read_csv('../input/sample_submission.csv')
submission['surface'] = le.inverse_transform(predicted)
submission.to_csv('submission.csv', index=False)
submission.head(10)


# In[56]:


submission.shape


# # Deep Learning Model in Keras

# In[58]:


# Building the model
xor = Sequential()

# Add required layers
xor.add(Dense(units = 3810, input_dim=14))
xor.add(Dropout(0.2))
xor.add(Activation('relu'))
xor.add(Dense(units = 128))
xor.add(Dropout(.1))
xor.add(Activation('relu'))
xor.add(Dense(units = 9))
xor.add(Activation('softmax'))

# Specify loss as "binary_crossentropy", optimizer as "adam",
# and add the accuracy metric
xor.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Uncomment this line to print the model architecture
xor.summary()


# In[60]:


# Fitting the model
fitted_model = xor.fit(X_train, y_train_keras, batch_size=32, epochs=100, verbose=1)


# In[63]:


predicted = xor.predict_classes(X_test)
predicted


# In[64]:


submission2 = pd.read_csv('../input/sample_submission.csv')
submission2['surface'] = le.inverse_transform(predicted)
submission2.to_csv('submission.csv', index=False)
submission2.head(10)


# In[ ]:




