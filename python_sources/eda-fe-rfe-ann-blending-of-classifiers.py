#!/usr/bin/env python
# coding: utf-8

# In this notebook we are going to:
#   * load the data
#   * visualise it briefly
#   * engineer some features
#   * use RFE to select the best subset of features
#   * train some simple base classifiers
#   * blend the results using a neural network
#   * produce a submission with the trained blending ensemble classifier

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC

import keras
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical

import random


# Let's load the data using pandas and have a quick look at the first 5 rows...

# In[ ]:


# set up dataset
number_classes = 7
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# lets take a look...
train_df.head()


# Now we will:
#   * remove the ID field
#   * split the data into the features and targets/classes
#   * concatenate all the feature data together for analysis and feature engineering

# In[ ]:


# create train datasets
X_train = train_df.drop(['Id', 'Cover_Type'], axis=1)
Y_train = train_df[['Cover_Type']].values
Y_train = Y_train.reshape(len(Y_train))

# create test dataset and ID's
X_test = test_df.drop(['Id'], axis=1)
ID_test = test_df['Id'].values
ID_test = ID_test.reshape(len(ID_test))

# concatenate both together for feature engineering and normalisation
X_all = pd.concat([X_train, X_test], axis=0)


# Now its time to visualise some of the features and how they are related!

# In[ ]:


# do some EDA
cols_non_onehot = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                   'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                   'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                   'Horizontal_Distance_To_Fire_Points', 'Cover_Type']

cols_onehot = [col_name for col_name in train_df.columns if col_name not in cols_non_onehot]
cols_onehot.append('Cover_Type')


# Lets start by using seaborn to view the distributions and relations between the non onehot encoded features (view in full screen!)

# In[ ]:


sns.pairplot(train_df[cols_non_onehot], hue='Cover_Type')


# From the above chart we can see that elevation is one of the most important variables, see how each class has an almost distinguishable distribution? Cover type's 3 and 6 are the hardest to discriminate using elevation alone, however some of the other variables may help our models distinguish this!

# Lets look into the distribution of elevation a little bit further...

# In[ ]:


fig, ax = plt.subplots()

# Sort the dataframe by target
target_1 = train_df.loc[train_df['Cover_Type'] == 1]
target_2 = train_df.loc[train_df['Cover_Type'] == 2]
target_3 = train_df.loc[train_df['Cover_Type'] == 3]
target_4 = train_df.loc[train_df['Cover_Type'] == 4]
target_5 = train_df.loc[train_df['Cover_Type'] == 5]
target_6 = train_df.loc[train_df['Cover_Type'] == 6]
target_7 = train_df.loc[train_df['Cover_Type'] == 7]

sns.distplot(target_1[['Elevation']], ax=ax)
sns.distplot(target_2[['Elevation']], ax=ax)
sns.distplot(target_3[['Elevation']], ax=ax)
sns.distplot(target_4[['Elevation']], ax=ax)
sns.distplot(target_5[['Elevation']], ax=ax)
sns.distplot(target_6[['Elevation']], ax=ax)
sns.distplot(target_7[['Elevation']], ax=ax)

plt.show()


# Now lets visualise some of the other interesting relationships in a larger pairplot

# In[ ]:


interesting_cols = ['Elevation', 'Horizontal_Distance_To_Fire_Points',
                    'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
                    'Horizontal_Distance_To_Hydrology', 'Cover_Type']
sns.pairplot(train_df[interesting_cols], hue='Cover_Type', height=5)


# Now we've got an idea about how the data set is distributed and some of the relations between the data, lets put that knowlege into practice by creating some new features of the data. Since we are using RFE in the next step we can be generous in our engineering of features.

# In[ ]:


# mean hillshade
def mean_hillshade(df):
    df['mean_hillshade'] = (df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']) / 3
    return df

# calculate the distance to hydrology using pythagoras theorem
def distance_to_hydrology(df):
    df['distance_to_hydrology'] = np.sqrt(np.power(df['Horizontal_Distance_To_Hydrology'], 2) +                                           np.power(df['Vertical_Distance_To_Hydrology'], 2))
    return df

# calculate diagnial distance down to sea level?
def diag_to_sealevl(df):
    df['diag_to_sealevel'] = np.divide(df['Elevation'], np.cos(180-df['Slope']))
    return df

# calculate mean distance to features
def mean_dist_to_feature(df):
    df['mean_dist_to_feature'] = (df['Horizontal_Distance_To_Hydrology'] +                                   df['Horizontal_Distance_To_Roadways'] +                                   df['Horizontal_Distance_To_Fire_Points']) / 3
    return df

def mean_shade(df):
    df['Shadiness_morn_noon'] = df.Hillshade_9am/(df.Hillshade_Noon+1)
    df['Shadiness_noon_3pm'] = df.Hillshade_Noon/(df.Hillshade_3pm+1)
    df['Shadiness_morn_3'] = df.Hillshade_9am/(df.Hillshade_3pm+1)
    df['Shadiness_morn_avg'] = (df.Hillshade_9am+df.Hillshade_Noon)/2
    df['Shadiness_afernoon'] = (df.Hillshade_Noon+df.Hillshade_3pm)/2
    df['Shadiness_total_mean'] = (df.Hillshade_9am+df.Hillshade_Noon+df.Hillshade_3pm)/3
    return df
 
def hydro_fire_combinations(df):
    df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']
    df['HF2'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
    df['Mean_HF1'] = df.HF1/2
    df['Mean_HF2'] = df.HF2/2
    return df

def hydro_road_combinations(df):
    df['HR1'] = (df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])
    df['HR2'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
    df['Mean_HR1'] = df.HR1/2
    df['Mean_HR2'] = df.HR2/2
    return df

def fire_road_combinations(df):
    df['FR1'] = (df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])
    df['FR2'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
    df['Mean_FR1'] = df.FR1/2
    df['Mean_FR2'] = df.FR2/2
    return df

def elevation_hydro_combinations(df):
    df['EV1'] = df.Elevation+df.Vertical_Distance_To_Hydrology
    df['EV2'] = df.Elevation-df.Vertical_Distance_To_Hydrology
    df['Mean_EV1'] = df.EV1/2
    df['Mean_EV2'] = df.EV2/2
    return df

def binned_columns(df):
    bin_defs = [
        # col name, bin size, new name
#         ('Elevation', 200, 'Binned_Elevation'),
        ('Aspect', 45, 'Binned_Aspect'),
        ('Slope', 6, 'Binned_Slope'),
        ('Horizontal_Distance_To_Hydrology', 140, 'Binned_Horizontal_Distance_To_Hydrology'),
        ('Horizontal_Distance_To_Roadways', 712, 'Binned_Horizontal_Distance_To_Roadways'),
        ('Hillshade_9am', 32, 'Binned_Hillshade_9am'),
        ('Hillshade_Noon', 32, 'Binned_Hillshade_Noon'),
        ('Hillshade_3pm', 32, 'Binned_Hillshade_3pm'),
        ('Horizontal_Distance_To_Fire_Points', 717, 'Binned_Horizontal_Distance_To_Fire_Points')
    ]
    
    for col_name, bin_size, new_name in bin_defs:
        df[new_name] = np.floor(df[col_name]/bin_size)
    
    return df

X_all = mean_hillshade(X_all)
X_all = distance_to_hydrology(X_all)
# X_all = diag_to_sealevl(X_all)
X_all = mean_dist_to_feature(X_all)
X_all = mean_shade(X_all)
X_all = hydro_fire_combinations(X_all)
X_all = hydro_road_combinations(X_all)
X_all = fire_road_combinations(X_all)
# X_all = elevation_hydro_combinations(X_all)
X_all = binned_columns(X_all)

X_all.head()


# Now we have a generous set of features lets normalise them while being careful to leave the one hot encoded variables untouched.

# In[ ]:


# normalise dataset
def normalise_df(df):
    df_mean = df.mean()
    df_std = df.std()    
    df_norm = (df - df_mean) / (df_std)
    return df_norm, df_mean, df_std

# define columsn to normalise
cols_non_onehot = [#'Elevation', 
                   'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                   'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                   'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                   'Horizontal_Distance_To_Fire_Points', 'mean_hillshade',
                   'Shadiness_morn_noon', 'Shadiness_noon_3pm', 'Shadiness_morn_3',
                   'Shadiness_morn_avg', 'Shadiness_afernoon', 'Shadiness_total_mean',
                   'HF1', 'HF2', 'Mean_HF1', 'Mean_HF2',
                   'HR1', 'HR2', 'Mean_HR1', 'Mean_HR2',
                   'FR1', 'FR2', 'Mean_FR1', 'Mean_FR2',
                   #'EV1', 'EV2', 'Mean_EV1', 'Mean_EV2',
                   'distance_to_hydrology', 
                   #'diag_to_sealevel', 
                   'mean_dist_to_feature']

X_all_norm, df_mean, df_std = normalise_df(X_all[cols_non_onehot])

# replace columns with normalised versions
X_all = X_all.drop(cols_non_onehot, axis=1)
X_all = pd.concat([X_all_norm, X_all], axis=1)


# In[ ]:


# elevation was found to have very different distributions on test and training sets
# lets just drop it for now to see if we can implememnt a more robust classifier!
X_all = X_all.drop('Elevation', axis=1)


# In[ ]:


# split back into test and train sets
X_train = np.array(X_all[:len(X_train)])
X_test = np.array(X_all[len(X_train):])


# In[ ]:


from sklearn.feature_selection import RFECV

# rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, criterion='entropy',
#                             min_samples_split=3, class_weight='balanced')

lgb = LGBMClassifier(n_estimators=100, max_depth=3)

rfecv = RFECV(estimator=lgb, step=1, cv=StratifiedKFold(3),
              scoring='accuracy', verbose=1)

rfecv.fit(np.array(X_train), np.array(Y_train))

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[ ]:


# reduce features to the best subset found with rfecv
X_train = rfecv.transform(np.array(X_train))


# In[ ]:


def create_blender(X, Y):
    blender = Sequential()

    blender.add(Dense(512, activation='elu', input_shape=(len(X[0]),), 
                      kernel_regularizer='l2'))
    blender.add(BatchNormalization())
    blender.add(Dropout(0.3))

    blender.add(Dense(256, activation='elu', kernel_regularizer='l2'))
    blender.add(BatchNormalization())
    blender.add(Dropout(0.3))
    
    blender.add(Dense(128, activation='elu', kernel_regularizer='l2'))
    blender.add(BatchNormalization())
    blender.add(Dropout(0.3))
    
    blender.add(Dense(32, activation='elu', kernel_regularizer='l2'))
    blender.add(BatchNormalization())
    blender.add(Dropout(0.3))
    
    blender.add(Dense(len(Y[0]), activation='softmax'))
    sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.85, nesterov=True)
    blender.compile(loss='categorical_crossentropy', optimizer='sgd')
    return blender


# In[ ]:


class BlenderClassifier(object):
    def __init__(self, learners, create_blender_func, n_folds=5, max_epoch=50):
        self.learners = learners
        self.n_folds = n_folds
        self.max_epoch = max_epoch
        self.history = {}
        self.create_blender = create_blender_func
        
    def fit(self, X, Y, verbose=True):
        kfolds = StratifiedKFold(self.n_folds, shuffle=True)
        predictions = [[] for x in self.learners]
        targets = []
        for fold_idx, (t, v) in enumerate(kfolds.split(X, Y)):
            if verbose:
                print('  commencing fold {}/{}...'.format(fold_idx + 1, 
                                                          self.n_folds))
            targets.extend(Y[v])
            for l_idx, learner in enumerate(self.learners):
                if verbose:
                    print('    fitting {}...'.format(type(learner)))
                learner.fit(X[t], Y[t])
                if verbose:
                    print('      predicting {}...'.format(type(learner)))                
                predictions[l_idx].extend(learner.predict(X[v]))
        
        if verbose:
            print('  creating blender...')
        
        targets = to_categorical(targets)
        predictions = np.swapaxes(predictions, 0, -1)
        x_new = np.concatenate([X, predictions], axis=-1)
        self.blender = self.create_blender(x_new, targets)
        
        if verbose:
            print('  fitting blender...')
        self.blender.fit(x_new, targets, epochs=self.max_epoch, 
                         shuffle=True, validation_split=0.1)
        if verbose:
            print('done!')
        
        
    def predict(self, X, verbose=True):
        if verbose:
            print('predicting X...')
        predictions = [[] for x in self.learners]
        for l_idx, learner in enumerate(self.learners):
            if verbose:
                print('  predicting using {}...'.format(type(learner)))                
            predictions[l_idx].extend(learner.predict(X))
        
        predictions = np.swapaxes(predictions, 0, -1)
        
        if verbose:
            print('  blending predictions...')

        x_new = np.concatenate([X, predictions], axis=-1)
            
        return self.blender.predict(x_new)


# In[ ]:


logreg = LogisticRegression()

gnb = GaussianNB()

rfc1 = RandomForestClassifier(n_estimators=100, 
                              min_samples_split=2, 
                              class_weight='balanced',
                              criterion='entropy',
                              n_jobs=-1)

rfc2 = RandomForestClassifier(n_estimators=100, 
                              min_samples_split=3, 
                              class_weight='balanced',
                              criterion='entropy',
                              n_jobs=-1)

rfc3 = RandomForestClassifier(n_estimators=100, 
                              min_samples_split=5, 
                              class_weight='balanced',
                              criterion='entropy',
                              n_jobs=-1)

etc1 = ExtraTreesClassifier(n_estimators=100, 
                            min_samples_split=2,
                            class_weight='balanced',
                            criterion='entropy',
                            n_jobs=-1)

etc2 = ExtraTreesClassifier(n_estimators=100, 
                            min_samples_split=3,
                            class_weight='balanced',
                            criterion='entropy',
                            n_jobs=-1)

etc3 = ExtraTreesClassifier(n_estimators=100, 
                            min_samples_split=5,
                            class_weight='balanced',
                            criterion='entropy',
                            n_jobs=-1)

lbc1 = LGBMClassifier(n_estimators=500,
                     learning_rate=0.001)

lbc2 = LGBMClassifier(n_estimators=500,
                     learning_rate=0.01)

lbc3 = LGBMClassifier(n_estimators=500,
                     learning_rate=0.1)


# In[ ]:


default_learners = [logreg, gnb, 
                    rfc1, rfc2, rfc3, 
                    etc1, etc2, etc3, 
                    lbc1, lbc2, lbc3]


# In[ ]:


Xt, Xv, Yt, Yv = train_test_split(np.array(X_train), np.array(Y_train), shuffle=True, stratify=np.array(Y_train))

bc = BlenderClassifier(learners=default_learners, 
                       create_blender_func=create_blender)
    
bc.fit(Xt, Yt)


# In[ ]:


y_pred = bc.predict(Xv)

# for stupid sklearn metric
y_pred = np.argmax(y_pred, axis=-1)
y_pred = to_categorical(y_pred)
Yv = to_categorical(Yv)

from sklearn.metrics import accuracy_score

score = accuracy_score(Yv, y_pred)

print('final validation score: {}'.format(score))


# In[ ]:


print('producing test data predictions...')
X_test = rfecv.transform(X_test)
y_pred = bc.predict(X_test)


# In[ ]:


y_pred = np.argmax(y_pred, axis=-1)

print(max(y_pred))
print(min(y_pred))


# In[ ]:


sub = pd.DataFrame()
sub['Id'] = ID_test
sub['Cover_Type'] = y_pred
sub.to_csv('my_submission.csv', index=False)
print('good luck!')


# In[ ]:




