# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.

import spacy                   #Python package for NLP
import Levenshtein as lv       #To compute levenshtein distance
from sklearn.preprocessing import StandardScaler       #Used to normalize data (substract the mean and divide by std)
from sklearn.linear_model import LogisticRegression      #Logistic regression classifier

#The general idea on this script is to extract some features that somehow represent the similarity between every pair of security names and
#train a logistic regression model to make predictions on the test set. 

nlp = spacy.load('en')    #Loading english language models in spacy

df_train = pd.read_csv('../input/train.csv')  #Loading training data
df_test = pd.read_csv('../input/test.csv')   #Loading test data


def get_features(str1, str2):
    """
    This function computes some features from a pair of strings. It is an auxiliary function that will be used to generate a
    feature represenation for each example in the data. In this sample code I am computing 6 features (many more could be added):
    - difference in length in both strings
    - levenshtein ratio between the two strings
    - levenshtein distance between the two strings
    - difference in number of words
    - similarity between the two strings using spacy
    - difference in length after removing stop words from both strings
    """
    feats = {}
    feats['diff_len'] = len(str1) - len(str2)  #adding difference in len as a feature
    
    feats['lev_ratio'] = lv.ratio(str1, str2)   # adding levenshtein ratio
    feats['lv_dist'] = lv.distance(str1, str2)  # adding levenshtein distance
    
    s1 = nlp(str1)
    s2 = nlp(str2)

    feats['diff_words'] = len(s1) - len(s2)    #adding difference in number of words as a feature
    
    feats['spacy_sim'] = s1.similarity(s2)    #adding similarity between strings from spacy
    
    no_stop_s1 = [w for w in s1 if not w.is_stop]
    no_stop_s2 = [w for w in s2 if not w.is_stop]
    feats['diff_no_stop'] = len(no_stop_s1) - len(no_stop_s2)  #adding the difference in len after removing stop words
    
    #You can add many more features here...
    
    return feats


def feature_engineering(df):
    """
    Given a data frame with 'description_x' and 'description_y', it will apply 'get_features' function to all rows to compute the matrix 
    with the data representation. This will be applied to df_train and df_test before training the model. 
    """
    list_dicts = df.apply(lambda x: get_features(x['description_x'], x['description_y']), axis=1).values.tolist()
    df_feats = pd.DataFrame(list_dicts)
        
    return df_feats.values 


X_train = feature_engineering(df_train)    #training set data
y_train = df_train.same_security.values    #training set labels

X_test = feature_engineering(df_test)      #test set

#We should normalize features before applying a logistic regression algorithm 
ss = StandardScaler()  
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#Applying logistic regresion classifier
lr = LogisticRegression()  #Using default parameters.
lr.fit(X_train, y_train)    #training the model with X_train, y_train
y_pred = lr.predict_proba(X_test)[:,1]  #making predictions on the test set and getting probabilities of the prediction to be equal to 1.

#generating the submission file.
df_sub = df_test.copy()
df_sub['same_security'] = y_pred
df_sub.loc[:, ['same_security']].to_csv('logistic_regression.csv', index=True, index_label='id')
