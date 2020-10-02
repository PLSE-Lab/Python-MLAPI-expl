#!/usr/bin/env python
# coding: utf-8

# # Stumped - can't improve accuracy so moving on to stacking.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier,                 RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
import time
SEED = 123


# In[ ]:


#reading the files
train = pd.read_csv("../input/learn-together/train.csv")
test = pd.read_csv("../input/learn-together/test.csv")

print(train.groupby('Cover_Type')['Id'].count())


# In[ ]:


y = train.Cover_Type
test_id = test['Id']

#dropping Ids
train = train.drop(['Id'], axis = 1)
test = test.drop(['Id'], axis = 1)

#prepare data for training the model
X = train.drop(['Cover_Type'], axis = 1)


# In[ ]:


print(X.columns[(X < 0).any()])


# In[ ]:



clf = RandomForestClassifier()
clf = clf.fit(X,y)

features = pd.DataFrame({'Features': X.columns, 
                         'Importances': clf.feature_importances_})
features.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)
plt.figure(figsize=(12,4))
sns.barplot(x='Features', y='Importances', data=features)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


def preprocess(df):
    #horizontal and vertical distance to hydrology can be easily combined
    cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']
    df['Distance_to_hydrology'] = df[cols].apply(np.linalg.norm, axis=1)
    
    #adding a few combinations of distance features to help enhance the classification
    cols = ['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points',
            'Horizontal_Distance_To_Hydrology']
    df['distance_mean'] = df[cols].mean(axis=1)
    df['distance_sum'] = df[cols].sum(axis=1)
    df['distance_dif_road_fire'] = df[cols[0]] - df[cols[1]]
    df['distance_dif_hydro_road'] = df[cols[2]] - df[cols[0]]
    df['distance_dif_hydro_fire'] = df[cols[2]] - df[cols[1]]
    
    #taking some factors influencing the amount of radiation
    df['Cosine_of_slope'] = np.cos(np.radians(df['Slope']) )
    #X['Diff_azimuth_aspect_9am'] = np.cos(np.radians(123.29-X['Aspect']))
    #X['Diff_azimuth_aspect_12noon'] = np.cos(np.radians(181.65-X['Aspect']))
    #X['Diff_azimuth_aspect_3pm'] = np.cos(np.radians(238.56-X['Aspect']))

    df['Elevation_VDH'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    return df

X = preprocess(X)
test = preprocess(test)
print(X.columns)


# In[ ]:


# Plotting mode frequencies as % of data size
#take from: https://www.kaggle.com/kwabenantim/forest-cover-feature-engineering
n_rows = X.shape[0]
mode_frequencies = [X[col].value_counts().iat[0] for col in X.columns]
mode_frequencies = 100.0 * np.asarray(mode_frequencies) / n_rows

mode_df = pd.DataFrame({'Features': X.columns, 
                        'Mode_Frequency': mode_frequencies})

mode_df.sort_values(by=['Mode_Frequency'], axis='index', ascending=True, inplace=True)

fig = plt.figure(figsize=(14, 4))
sns.barplot(x='Features', y='Mode_Frequency', data=mode_df)
plt.ylabel('Mode Frequency %')
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


def drop_unimportant(df):
    df_ = df.copy()
    n_rows = df_.shape[0]
    hi_freq_cols = []
    for col in X.columns:
        mode_frequency = 100.0 * df_[col].value_counts().iat[0] / n_rows 
        if mode_frequency > 99.0:
            hi_freq_cols.append(col)
    df_ = df_.drop(hi_freq_cols, axis='columns')
    return df_

X = drop_unimportant(X)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


# In[ ]:


#code from here:http://ataspinar.com/2017/05/26/classification-with-scikit-learn/
clf_dict = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),
}

def batch_classify(X_train, y_train, X_val, y_val, no_clf = 5, verbose = True):
    dict_models = {}
    for clf_name, clf in list(clf_dict.items())[:no_clf]:
        t_start = time.clock()
        clf.fit(X_train, y_train)
        t_end = time.clock()
        
        t_diff = t_end - t_start
        train_score = clf.score(X_train, y_train)
        val_score = clf.score(X_val, y_val)
        
        dict_models[clf_name] = {'model': clf, 'train_score': train_score, 
                                 'val_score': val_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=clf_name, f=t_diff))
    return dict_models

def display_dict_models(dict_models, sort_by='val_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['val_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 
                                                                     'val_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'val_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    print(df_.sort_values(by=sort_by, ascending=False))

#dict_models = batch_classify(X_train, y_train, X_val, y_val, no_clf = 8)
#display_dict_models(dict_models)


# In[ ]:


feature_names = list(X.columns)
test = test[feature_names]
print(X.shape)
print(test.shape)


# In[ ]:


rfc = RandomForestClassifier(n_estimators=181, bootstrap=False, 
                               max_features='auto', random_state=SEED)
clf1 = KNeighborsClassifier(n_neighbors=1, p=1)
clf2 = GaussianNB()
clf3 = DecisionTreeClassifier(max_features='auto', random_state=SEED)
clf5 = AdaBoostClassifier(base_estimator=clf3)
gbc = GradientBoostingClassifier(n_estimators=750, learning_rate=0.01,max_depth=6,
                                 max_features=6,subsample=0.75,random_state=SEED)
lr = LogisticRegression(multi_class='multinomial', solver='newton-cg', random_state=SEED)


# In[ ]:


from mlxtend.classifier import StackingCVClassifier
sclf = StackingCVClassifier(classifiers=[rfc, gbc],meta_classifier=lr)


# In[ ]:


for clf, label in zip([rfc, gbc, sclf],
                      ['RandomForest',
                       'GradientBoost',
                       'StackingClassifier']):

    scores = cross_val_score(clf, X.values, y.values, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))


# In[ ]:


sclf.fit(X,y)
test_pred = sclf.predict(test)
output = pd.DataFrame({'Id': test_id, 'Cover_Type': test_pred.astype(int)})
output.to_csv('submission.csv', index=False)

