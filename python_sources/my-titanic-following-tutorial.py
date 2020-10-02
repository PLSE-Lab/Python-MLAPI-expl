## Largely inspired by Kaggle user sashr07's "Kaggle Titanic Supervised Learning Tutorial" -- https://www.kaggle.com/sashr07/kaggle-titanic-tutorial
## Some code is copied directly from there



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    
# testing imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#### generic tools (I originally had these in a separate module)

# python imports
import re

def process_age(df,cut_points,label_names):
    ## From tutorial
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

def discretize_feature(df, feature, bin_limits, label_names=None, nan_handling=None) :
    """discretize a continuous feature into bins
    :param DataFrame df: -- pandas DataFrame
    :param list bin_limits: -- list of bin limits -- length nbins+1
    :param list label_names: -- list of bin labels -- if none, label with zero-indexed bin numbers
    :param nan_handling: -- if not None, value to convert NaN entries
    :return: pandas DataFrame (likely modifies in place)
    """

    if nan_handling is not None :
        df[feature] = df[feature].fillna(nan_handling)

    if label_names is None :
        label_names = [ str(i) for i in range(len(bin_limits)-1) ]
    df[feature+"_discretized"] = pd.cut(df[feature], bin_limits, labels=label_names)

    return df


def create_dummies(df,column_name):
    # From tutorial
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df
    
def binarize_features(df, features) :
    """Explode a single feature into several binary-valued columns
    1 is truthy; 0 is falsy
    :param features: -- list (or tuple) of feature names, or a sigle feature name
    :return: modified dataframe
    """

    # allow non-list feature specs
    if not isinstance(features, list) :
        if isinstance(features, tuple) :
            features = list(features)
        else :
            features = [features]

    for feature in features :
        dummies = pd.get_dummies(df[feature], prefix=feature)
        df = pd.concat([df, dummies], axis=1)

    return df

def filter_feature_names(df, explicit=None, binarized=None, discretized=None, disc_bin=None) :
    """Populate a list of feature names effectively using regex patterns
    Designed to avoid having to do a lot of redundant typing for 
        discretized or binarized features
    """
    all_features = list(df)

    patterns = []
    if explicit is not None :
        patterns.append(explicit)
    if binarized is not None :
        for feature in binarized :
            patterns.append(feature + '_.*')
    if discretized is not None :
        for feature in discretized :
            patterns.append(feature + '_discretized')
    if disc_bin is not None :
        for feature in disc_bin :
            patterns.append(feature + '_discretized_.*')

    pattern = "|".join(patterns)

    matches = []
    for feature in all_features :
        if re.fullmatch(pattern, feature) :
            matches.append(feature)
            
    return matches








    
    
    
    
    
    #### Code specific to competition
    
    
    
    



train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

all_data = (train_data, test_data)

print(train_data.shape)
print(test_data.shape)
print(type(train_data))


nan_val = -0.5
cut_points = [-1,0,18,100]
label_names = ["Missing","Child","Adult"]

to_binarize = ["Age_discretized", "Sex", "Pclass"]
new_data = []
for df in all_data :
    df = discretize_feature(df, "Age", cut_points, label_names, nan_val)
    df = binarize_features(df, to_binarize)
    new_data.append(df)
all_data = tuple(new_data)
(train_data, test_data) = all_data


columns = filter_feature_names(train_data, binarized=to_binarize)
print(columns)


## cross validate
logistic = LogisticRegression()

cv_results = cross_val_score(logistic, train_data[columns], train_data['Survived'], cv=10)
print(cv_results)
print(np.mean(cv_results))


## train on all data

logistic = LogisticRegression()
logistic.fit(train_data[columns], train_data['Survived'])

final_predictions = logistic.predict(test_data[columns])

## save results

submission_dict = {
        "PassengerId" : test_data["PassengerId"], 
        "Survived" : final_predictions
    }
submission = pd.DataFrame(submission_dict)
submission.to_csv('titanic_submission.csv', index=False)