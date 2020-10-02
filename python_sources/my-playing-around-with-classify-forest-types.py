#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import statistics

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import StackingCVClassifier

from sklearn.metrics import accuracy_score
from scipy.stats import norm, skew,skewtest #for some statistics
import warnings

#import utility scripts
import classify_forest_utility_script as utils

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv("/kaggle/input/learn-together/train.csv", index_col='Id')
test_df  = pd.read_csv("/kaggle/input/learn-together/test.csv", index_col='Id')


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# In[ ]:


train_df.head()


# We want to see the distribution of our target in the training data frame

# In[ ]:


sns.countplot(x='Cover_Type', data=train_df)


# **We have a perfect distribution of the target in our training data frame**

# A quick look into the correlation between features on the training data

# In[ ]:


training_df_cols = train_df.columns.tolist()
training_df_cols = training_df_cols[-1:] + training_df_cols[:-1]
train_df = train_df[training_df_cols]
corr = train_df.corr()


# In[ ]:


# Mask (for the upper triangle)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15,15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask andcorrect aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .5})


# # We need to investigate the Soil Types

# ## Distribution of the Soil Types in the training data

# In[ ]:


soil_type_cols = train_df.columns[15:]

utils.barplot_onehot_encoded_features(train_df, soil_type_cols, ylabel="Total amount of positive Soil Types", title="Soil Types = 1 in train data frame")


# Soil types 7, 8, 15 and 25 seem to be zero?

# In[ ]:


for idx in [7, 8, 15, 25]:
    soil_type = soil_type_cols[idx - 1]
    print("Number of datapoints in Soil Type %d: %d" % (idx, train_df[soil_type].sum()))


# ## Distribution of the Soil Types in the test data

# In[ ]:


utils.barplot_onehot_encoded_features(test_df, soil_type_cols, ylabel="Total amount of positive Soil Types", title="Soil Types = 1 in test data frame")


# In[ ]:


for idx in [7, 8, 15, 25]:
    soil_type = soil_type_cols[idx - 1]
    print("Number of datapoints in Soil Type %d: %d" % (idx, test_df[soil_type].sum()))


# Relatively speaking, there is very few data points on Soil Types 7, 8, 15 and 25. Besides, in the training data set we have 0 posibilities to train an algorithm for these features. So, it would be better to remove these features from our training and test data sets.

# In[ ]:


soil_type_extract_traindf = train_df[soil_type_cols].copy()
soil_type_extract_traindf = soil_type_extract_traindf.join(train_df.Cover_Type)

utils.stackplotbar_target_over_feature(soil_type_extract_traindf, 'Cover_Type')


# Cover Types 6 and 3 are more common in Soil Type 10 than with other Soil Types. Cover Type 3 is also common in Soil Types 1-6. Cover Type 7 seems to be more common in Soil Types 38 - 40..... In general, different Cover Types seem to be common in different soil types. Therefore, I conclude that Soil Types offer usefull information to be considered by an algorithm - especially maybe when conbined with other features like Elevation or Hillshade.
# 
# **Nevertheless, we should remove from our training dataset Soil Types 7, 8, 15 and 25, since they do not bring any benefit to the algorithm - there are not enough samples of these soil types to train the algorithm on them.**

# # We need to investigate the Wilderness Areas

# ## Distribution of the Soil Types in the training data

# In[ ]:


wilderness_area_cols = train_df.columns[11:15]

utils.barplot_onehot_encoded_features(train_df, wilderness_area_cols, ylabel="Total amount of Wilderness Area Types", title="Wilderness Area = 1 in train data frame")


# Wilderness Area number 2 seems to have a very low number of entries in respect to the other Wilderness Areas. Could it be that this might present some challenges to the algorithm?
# 
# Let's see how is our Wilderness Area distribution in the test data set:

# In[ ]:


utils.barplot_onehot_encoded_features(test_df, wilderness_area_cols, ylabel="Total amount of Wilderness Area Types", title="Wilderness Area = 1 in train data frame")


# The amount of data entries that belong to the Wilderness Area number 2 feature is also low in relation to Wilderness Area number 1 and 3. Wilderness Area number 4 is also low in relation to these other wilderness areas. Nevertheless, it might be the case that the amount of wilderness areas data entries in the training dataset is enough for the algorithm to learn from these features.

# In[ ]:


wilderness_area_extract_traindf = train_df[wilderness_area_cols].copy()
wilderness_area_extract_traindf = wilderness_area_extract_traindf.join(train_df.Cover_Type)

utils.stackplotbar_target_over_feature(wilderness_area_extract_traindf, 'Cover_Type')


# This last graph explains the correlation we can see between some Soil Types and some Wilderness Areas.
# 
# Wilderness Areas seem to be very good features to help the algorithm to rule out cover types - hopefully.

# # Now we need to investigate the numerical features

# ## Let's see the correlation between the numerical features, target included.

# In[ ]:


onehot_encoded_features = np.concatenate([wilderness_area_cols,soil_type_cols])
numerical_features_train_df = train_df.drop(onehot_encoded_features, axis=1)

f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(numerical_features_train_df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()


# Correlatation between features above 0:
#  * Vertical Distance to Hydrology and Horizontal Distance to Hydrology (0.7)
#  * Elevation and Horizontal Distance to Roadway (0.6)
#  * Aspect and Hillshade 3 pm (0.6)
#  * Hillshade Noon and Hillshade 3 pm (0.6)
#  * Horizontal Distance to Roadways and Horizontal Distance to Fire Point (0.5)
#  * Elevation and Horizontal Distance to Hydrology (0.4)
#  * Elevation and Horizontal Distance to Fire Points (0.4)
#  * Aspect and Hillshade Noon (0.3)
#  * Slope and Vertical Distance to Hydrology (0.3)
#  * Elevation and Hillshade Noon (0.2)
#  * Horizontal Distance to Hydrology and Horizontal Distance to Roadway (0.2)
#  * Horizontal Distance to Hydrology and Horizontal Distance to Fire Points (0.2)
#  * Horizontal Distance to Roadway and Hillshade Noon (0.2)
#  * Horizontal Distance to Roadway and Hillshade 3 pm (0.2)
#  * Elevation and Vertical Distance to Hydrology (0.1)
#  * Elevation and Hillshade 9am (0.1)
#  * Elevation and Hillshade 3pm (0.1)
#  * Aspect and Vertical Distance to Hydrology (0.1)
#  * Aspect and Horizontal Distance to Roadways (0.1)
#  * Horizontal Distance to Hydrology and Hillshade Noon (0.1)
#  * Horizontal Distance to Hydrology and Hillshade 3pm (0.1)
#  * Hillshade 9am and Horizontal Distance to Fire Points (0.1)
#  * Hillshade Noon and Horizontal Distance to Hydrology (0.1)
#  * Hillshade Noon and Horizontal Distance to Fire Points (0.1)

# ## Let's investigate the Hillshades - and additionally the features that are highly correlated to them
# 
# Let's see how does the distribution of the different Hillshades look like (starting with some statistics):

# In[ ]:


hillshades = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

# I need to see some statistics on the Hillshades
numerical_features_train_df[hillshades].describe()


# In[ ]:


# additionally we need to know the median and mode of each Hillshade feature
for feature in hillshades:
    print("Feature %s" % feature)
    print("     Mean:   %d" % (statistics.mean(numerical_features_train_df[feature])))
    print("     Median: %d" % (statistics.median(numerical_features_train_df[feature])))
    print("     Mode:   %d" % (statistics.mode(numerical_features_train_df[feature])))


# In[ ]:


# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(numerical_features_train_df[hillshades], alpha = 0.3, figsize = (15,12), diagonal = 'kde')

f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(numerical_features_train_df[hillshades].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()


# In[ ]:


sns.distplot(a=numerical_features_train_df.Hillshade_9am, label="Hillshade 9am")
sns.distplot(a=numerical_features_train_df.Hillshade_Noon, label="Hillshade_Noon")
sns.distplot(a=numerical_features_train_df.Hillshade_3pm, label="Hillshade_3pm")
# Add title
plt.title("Histogram of Hillshades")
# Force legend to appear
plt.legend()


# The distribution graphs show that features Hillshade-9am and Hillshade_Noon are left skewed.
# 
# I'd like to see now the same information and graphs of the Hillshades of the test dataframes:

# In[ ]:


# I need to see some statistics on the Hillshades of the test data frame
test_df[hillshades].describe()


# In[ ]:


# additionally we need to know the median and mode of each Hillshade feature on the test data frame
for feature in hillshades:
    print("Feature %s" % feature)
    print("     Mean:   %d" % (statistics.mean(test_df[feature])))
    print("     Median: %d" % (statistics.median(test_df[feature])))
    print("     Mode:   %d" % (statistics.mode(test_df[feature])))


# In[ ]:


# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(test_df[hillshades], alpha = 0.3, figsize = (15,12), diagonal = 'kde')

f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(test_df[hillshades].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()


# In[ ]:


sns.distplot(a=test_df.Hillshade_9am, label="Hillshade 9am")
sns.distplot(a=test_df.Hillshade_Noon, label="Hillshade_Noon")
sns.distplot(a=test_df.Hillshade_3pm, label="Hillshade_3pm")
# Add title
plt.title("Histogram of Hillshades")
# Force legend to appear
plt.legend()


# So, we see that the same features that are skewed in the train data frame are also skewed in the test data frame.
# 
# We can see also that the median, mean and mode of the skewed features are very similiar in both the train and the test data frames.
# 
# Now, let's try to see if we can use some power-to-the-Nth transformation to try to get a better distribution on the skewed features

# In[ ]:


skewed_hillshades = ['Hillshade_9am', 'Hillshade_Noon']


# In[ ]:


features_log_transformed_train_df = numerical_features_train_df.copy()
features_log_transformed_train_df[skewed_hillshades] = features_log_transformed_train_df[skewed_hillshades].apply(lambda x: np.power(x, 5))
features_log_transformed_train_df[skewed_hillshades].describe()


# In[ ]:


# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(features_log_transformed_train_df[hillshades], alpha = 0.3, figsize = (15,12), diagonal = 'kde')

f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(features_log_transformed_train_df[hillshades].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()


# Let's replace the original hillshades with the transformed hillshades on the numerical data set to see if the correlation between these three features (i.e. the hillshades) and the rest of feature is modified / improved(?)

# In[ ]:


# let's make a copy o
numerical_veatures_train_df_transformed_hillshades = numerical_features_train_df.drop(hillshades, axis=1).copy()
numerical_veatures_train_df_transformed_hillshades = numerical_veatures_train_df_transformed_hillshades.join(features_log_transformed_train_df[hillshades],
                                                                                                             on='Horizontal_Distance_To_Roadways')


f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(numerical_features_train_df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()

f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(numerical_veatures_train_df_transformed_hillshades.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()


# In[ ]:


# let get now the ouliers we can find in the hillshades
hillshade_outliers = utils.detect_outliers(numerical_veatures_train_df_transformed_hillshades, min_num_outliers=2, features=hillshades)
hillshade_outliers


# # Data preprocessing and the Models

# ## Data preparation

# In[ ]:


# Remove Soil Types that are not needed.
def Remove_SoilTypes(df):
    return df.drop(['Soil_Type7', 'Soil_Type8', 'Soil_Type15', 'Soil_Type25'], axis=1).copy()

def Transform_Hillshades(df):
    df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']] = df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']].apply(lambda x: np.power(x, 5))
    return df.copy()

def Preprocess_data_noHllshadeTrans(df):
    return Remove_SoilTypes(df)

def Preprocess_data(df):
    return Transform_Hillshades(Remove_SoilTypes(df))


# In[ ]:


# we must remove Soil Types 7, 8, 15, 25
prep_data_noHllshadeTrans = Preprocess_data_noHllshadeTrans(train_df.drop('Cover_Type', axis=1).copy())
prep_data = Preprocess_data(train_df.drop('Cover_Type', axis=1).copy())
#outlier_indexes = utils.detect_outliers(prep_data, min_num_outliers=2, features=None)
#prep_data.drop(outlier_indexes, axis=0, inplace=True)
#len(outlier_indexes)


# ## The models

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
classifier_rf = RandomForestClassifier(random_state=81)
params_grid_rf = {'n_estimators' : [560, 561, 562],
                  #n_estimators' : [560<-, 561, 562],
                  #'n_estimators' : [560<-, 563, 565],
                  #'n_estimators' : [560<-, 565, 570],
                  #'n_estimators' : [550, 560<-, 570, 600],
                  #'n_estimators' : [550<-, 600],
                  #'n_estimators' : [500<-, 600],
                  #'n_estimators' : [450, 500<-, 719, 800],
                  'min_samples_leaf': [1,2,3,4,5],
                  'min_samples_split': [2,3,4,5],
                  'oob_score': [False, True]}

#grid =GridSearchCV(estimator = classifier_rf, cv=5, param_grid=params_grid_rf,
#                   scoring='accuracy', verbose=1, n_jobs=-1, refit=True)
#grid.fit(prep_data_noHllshadeTrans, train_df.Cover_Type)
#print("Best Score: " + str(grid.best_score_))
#print("Best Parameters: " + str(grid.best_params_))

#best_parameters = grid.best_params_
#Best Score: 0.7880952380952381
#Best Parameters: {'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 560, 'oob_score': False}


# In[ ]:


# split data into training and validation data
train_X, test_X, train_y, test_y = train_test_split(prep_data_noHllshadeTrans, train_df.Cover_Type, test_size=0.2, random_state=81)

clf_rf = RandomForestClassifier(random_state=81, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 560, oob_score = False)
clf_rf.fit(train_X, train_y)
predict = clf_rf.predict(test_X)

acc = accuracy_score(test_y, predict)
print("Accuracy on data with no transformation on Hillshades: ", acc)


# In[ ]:


# split data into training and validation data
train_X, test_X, train_y, test_y = train_test_split(prep_data, train_df.Cover_Type, test_size=0.2, random_state=81)

clf_rf = RandomForestClassifier(random_state=81, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 560, oob_score = False)
clf_rf.fit(train_X, train_y)
predict = clf_rf.predict(test_X)

acc = accuracy_score(test_y, predict)
print("Accuracy on data with transformation on Hillshades: ", acc)


# In[ ]:


outliers = utils.detect_outliers(numerical_veatures_train_df_transformed_hillshades, min_num_outliers=2)
outliers


# In[ ]:


prep_data_noOutliers = prep_data.drop(outliers, axis=0)
prep_data_noOutliers.iloc[9610:9614,:]


# In[ ]:


y_transHillshades_noOutliers = train_df.drop(outliers, axis=0)['Cover_Type']
# split data into training and validation data
train_X, test_X, train_y, test_y = train_test_split(prep_data_noOutliers, y_transHillshades_noOutliers, test_size=0.2, random_state=81)

clf_rf = RandomForestClassifier(random_state=81, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 560, oob_score = False)
clf_rf.fit(train_X, train_y)
predict = clf_rf.predict(test_X)

acc = accuracy_score(test_y, predict)
print("Accuracy on data with transformation on Hillshades and without outliers (2) : ", acc)


# In[ ]:


# StandardScaling the dataset before splitting it and fitting the algorithms
scaler = StandardScaler()
scaled_data = scaler.fit_transform(prep_data_noOutliers)


# In[ ]:


# split data into training and validation data
train_X, test_X, train_y, test_y = train_test_split(scaled_data, y_transHillshades_noOutliers, test_size=0.2, random_state=81)

clf_rf = RandomForestClassifier(random_state=81, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 560, oob_score = False)
clf_rf.fit(train_X, train_y)
predict = clf_rf.predict(test_X)

acc = accuracy_score(test_y, predict)
print("Accuracy on data with transformation on Hillshades and without outliers but with data set scaled (2) : ", acc)


# In[ ]:


classifier_rf = RandomForestClassifier(n_estimators = 560,
                                       max_features = 0.3,
                                       max_depth = 464,
                                       min_samples_split = 2,
                                       min_samples_leaf = 1,
                                       bootstrap = False,
                                       random_state=81)
classifier_rf.fit(train_X, train_y)
predict = classifier_rf.predict(test_X)
acc = accuracy_score(test_y, predict)
print("Accuracy on data with transformation on Hillshades and without outliers (2) : ", acc)


# In[ ]:


### define the classifiers
### Parameters from :https://www.kaggle.com/joshofg/pure-random-forest-hyperparameter-tuning

classifier_rf = RandomForestClassifier(n_estimators = 560,
                                       max_features = 0.3,
                                       max_depth = 464,
                                       min_samples_split = 2,
                                       min_samples_leaf = 1,
                                       bootstrap = False,
                                       random_state=81)
classifier_xgb = OneVsRestClassifier(XGBClassifier(n_estimators = 560,
                                                   max_depth = 464,
                                                   random_state=81))
classifier_et = ExtraTreesClassifier(random_state=81)

classifier_adb = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=81), random_state=81)
classifier_bg = BaggingClassifier(random_state=81)


# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sclf = StackingCVClassifier(classifiers=[classifier_rf,
                                         classifier_xgb,
                                         classifier_et,
                                         classifier_adb,
                                         classifier_bg],
                            use_probas=True,
                            meta_classifier=classifier_rf)



labels = ['Random Forest', 'XGBoost', 'ExtraTrees', 'AdaBoost', 'Bagging', 'MetaClassifier']




for clf, label in zip([classifier_rf, classifier_xgb, classifier_et, classifier_adb, classifier_bg, sclf], labels):
    scores = cross_val_score(clf, train_X, train_y,
                             cv=5,
                             scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# In[ ]:


sclf.fit(train_X, train_y)
prediction = sclf.predict(test_X)
accuracy = accuracy_score(test_y, prediction)
print("Accuracy obtained by stacking classifiers: ", accuracy)


# In[ ]:


# prepare the final test and train data
# remove the SoilTypes we didn't train
test_processed_data = Preprocess_data(test_df.copy())
train_processed_data = Preprocess_data(train_df.drop('Cover_Type', axis=1).copy())
target_training = train_df.Cover_Type

# StandardScaling the test dataset
train_scaler = StandardScaler()
test_scaler = StandardScaler()
test_scaled_data = test_scaler.fit_transform(test_processed_data)
train_scaled_data = train_scaler.fit_transform(train_processed_data)


# In[ ]:


sclf_final = StackingCVClassifier(classifiers=[classifier_rf,
                                         classifier_xgb,
                                         classifier_et,
                                         classifier_adb,
                                         classifier_bg],
                                  use_probas=True,
                                  meta_classifier=classifier_rf)

sclf_final.fit(train_scaled_data, target_training)

# prepare submission
test_ids = pd.read_csv("/kaggle/input/learn-together/test.csv")["Id"]
final_prediction = sclf_final.predict(test_scaled_data)

# save file with predictions
submission = pd.DataFrame({'Id' : test_ids,
                           'Cover_Type' : final_prediction})
submission.to_csv('submission.csv', index=False)


# Notebooks I took ideas from:
# 
# https://www.kaggle.com/phsheth/forestml-part-6-stacking-eval-selected-fets
