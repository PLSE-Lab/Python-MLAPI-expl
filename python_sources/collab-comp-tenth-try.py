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

train = pd.read_csv("../input/learn-together/train.csv")
test = pd.read_csv("../input/learn-together/test.csv")
print(train.groupby('Cover_Type')['Id'].count())

y = train.Cover_Type
test_id = test['Id']

#dropping Ids
train = train.drop(['Id'], axis = 1)
test = test.drop(['Id'], axis = 1)

#prepare data for training the model
X = train.drop(['Cover_Type'], axis = 1)

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

    #sum of Hillshades
    shades = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    #df['Sum_of_shades'] = df[shades].sum(1)
    weights = pd.Series([0.299, 0.587, 0.114], index=cols)
    df['Hillshade'] = (df[shades]*weights).sum(1)

    df['Elevation_VDH'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    return df

X = preprocess(X)
test = preprocess(test)
#print(X.columns)

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

feature_names = list(X.columns)
test = test[feature_names]
print(X.shape)
print(test.shape)

#----------------------------------------------------------
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
import numpy as np
import warnings
warnings.simplefilter('ignore')

clf1 = KNeighborsClassifier(n_neighbors=1)
#LogisticRegression(solver='newton-cg', multi_class='multinomial',random_state=1)
clf2 = RandomForestClassifier(n_estimators=181, max_features='sqrt', random_state=1)
clf3 = ExtraTreesClassifier(n_estimators=400,max_depth=50,min_samples_split=2,random_state=1)
clf4 = GradientBoostingClassifier(n_estimators=150,max_features='sqrt',max_depth=15,
                                  min_samples_split=200,random_state=1)


print('5-fold cross validation:')
print('-'*20)
clfs = [clf1, clf2, clf3, clf4]
labels = ['KNeighbors', 'Random Forest', 'Extra Trees', 'Gradient Boost']
for clf, label in zip(clfs, labels):
    scores = model_selection.cross_val_score(clf, X, y, cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
print('-'*20)

from mlxtend.classifier import EnsembleVoteClassifier
eclf = EnsembleVoteClassifier(clfs=clfs, weights=[1,1,1,1])
clfs = [clf1, clf2, clf3, clf4, eclf]
labels = ['KNeighbors', 'Random Forest', 'Extra Trees', 'Gradient Boost', 'Ensemble']
for clf, label in zip(clfs, labels):
    scores = model_selection.cross_val_score(clf, X, y, cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
print('-'*20)

eclf.fit(X,y)
test_pred = eclf.predict(test)
output = pd.DataFrame({'Id': test_id, 'Cover_Type': test_pred.astype(int)})
output.to_csv('submission.csv', index=False)
