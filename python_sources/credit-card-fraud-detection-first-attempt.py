import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA, KernelPCA, FastICA
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# I have done some preliminary EDA (see my kernel on Credit card fraud detection EDA) so the priliminary steps here are guided by that
# Plan:
# Load data and undersample it (because of huge class imbalance)
# Standardize data 
# Pass it to classifiers 

# Since I plan to use a wide variety of classifiers, I won't do any hyperparameter tuning here, just trying to get a sense of potential best methods for this problem
# This is a very simple plan



if __name__ == "__main__":
    add_embedded_data = True  # set to false if don't want to add embedded data from difussion mapping to the features
    remove_outliers = False # set to True if you want to remove some of the outlier data
    # loading data and undersampling negative samples (non-fradulent labels)
    data = pd.read_csv("../input/creditcard.csv")
    fraud = data[data['Class'] == 1]
    non_fraud = data[data['Class'] == 0].sample(len(fraud) * 5)
    non_fraud.reset_index(drop=True, inplace=True)
    fraud.reset_index(drop=True, inplace=True)
    new_data = pd.concat([non_fraud, fraud]).sample(frac=1).reset_index(drop=True)

    
    # trying mild outlier detection, one can play with the bounds below to see if outlier detection is a good thing to do! 
    if remove_outliers:
        lb = new_data.quantile(0.05)
        ub = new_data.quantile(0.95)
        rang = ub - lb
        reduced_data = new_data[~((new_data < (lb - 1 * rang)) |(new_data > (ub + 1 * rang))).any(axis=1)]
    else:
        reduced_data = new_data

    features = reduced_data.drop(["Class"], axis=1)
    labels = reduced_data["Class"]
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)
    
    if add_embedded_data:
        #spec_embedding = SpectralEmbedding(n_components=5, affinity='rbf')
        embedding = PCA(n_components=10)
        transformed_data = embedding.fit_transform(features.values)
        # new_features = pd.DataFrame()
        for i in range(transformed_data.shape[1]):
             features["new_feat_{}".format(i)] =  pd.Series(transformed_data[:, i], index=features.index)
            
        # normalizing the features again
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
    
    
    models = []
    
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('SVM', SVC()))
    models.append(('XGB', XGBClassifier()))
    models.append(('RF', RandomForestClassifier()))
    
    #testing models
    
    results = []
    names = []
    
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=0)
        cv_results = cross_val_score(model, features, labels, cv=kfold, scoring='roc_auc')
        results.append(cv_results)
        names.append(name)
        msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
        
    # using boxplots to compare the algorithms
    
    fig = plt.figure(figsize=(12,10))
    plt.title('Comparison of Classification Algorithms')
    plt.xlabel('Algorithm')
    plt.ylabel('ROC-AUC Score')
    plt.boxplot(results)
    ax = fig.add_subplot(111)
    ax.set_xticklabels(names)
    plt.show()