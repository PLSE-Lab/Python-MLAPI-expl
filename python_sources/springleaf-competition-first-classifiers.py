# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, KernelPCA, FastICA
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
import time

import os
print(os.listdir('../input'))

# This script is using data that has already been processed and cleaned
# Refer to "Springleaf competition: EDA" and "Springleaf competition: Data preperation" kernels
# to see the data cleaning and preparation processes

# This is a naieve yet helpful comparison of different models for springleaf competition, no hyperparameter tuning is done


def one_hot_encoder(df, ohe_cols):
    '''
    One-Hot Encoder function
    '''
    print('Creating OHE features..\nOld df shape:{}'.format(df.shape))
    df = pd.get_dummies(df, columns=ohe_cols)
    print('New df shape:{}'.format(df.shape))
    return df


if __name__ == "__main__":

    normalize_features = True  # wether or not the features should be normalized (mean of zero, std of one)
    embed_data = False  # wether or not the features should be reduced and embedded into a different space
    embedding_method = KernelPCA # dimensionality reduction method to use, choose between PCA, KernelPCA, FastICA, SpectralEmbedding
    frac = 1  # specifies reduced dimensions of feature space when performing embedding 

    # specifications for whether or not we need one-hot-encoding and which columns to encode
    ohe = False
    prep_info = np.load("../input/springleaf-competition-eda/prep_info.npz")   # loading info from EDA kernel
    dates, ohe_cols, potentials_to_remove = prep_info["arr_5"], prep_info["arr_8"], prep_info["arr_7"]  
    # do not want to one-hot encode date features since that just blows up feature space
    ohe_cols = [col for col in ohe_cols if col not in dates and col not in potentials_to_remove and len(col) < 9]


    # since size of data is very large we sample the training data 
    train_df = pd.read_csv('../input/springleaf-competition-data-preparation/train_clean.csv').sample(frac=0.1).reset_index(drop=True)
    train_df.drop(potentials_to_remove, axis=1, inplace=True)
    train_df.drop(dates, axis=1, inplace=True)
    print("Data is loaded!")

    if ohe:
        # performing one-hot encoding of some categorical features
        train_df = one_hot_encoder(train_df, ohe_cols)

    features = train_df.drop(["target"], axis=1)
    labels = train_df["target"]

    if embed_data:
        # perform dimensionality reduction and ditch the original features
        start_time = time.time()
        if embedding_method != KernelPCA:
            embedding = embedding_method(n_components=1500)
        elif embedding_method == SpectralEmbedding:
            embedding = embedding_method(n_components=int(train_df.shape[1] / frac), affinity='nearest_neighbors')
        else:
            embedding = embedding_method(n_components=1500, kernel='rbf')
        features = embedding.fit_transform(features.values)
        print("Dimensionality reduction took {} seconds".format(time.time() - start_time))

    if normalize_features:
        print("Normalizing features.")
        features = (features - np.mean(features)) / (np.std(features) + 1e-7)

    # models to test, note that this is just a first attempt, each model has hyperparameters that need to be tuned
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('SVM', SVC()))
    models.append(('XGB', XGBClassifier()))
    models.append(('RF', RandomForestClassifier()))

    # testing models
    results = []
    names = []
    times = []
    for name, model in models:
        start_time = time.time()
        kfold = KFold(n_splits=5, random_state=0)
        cv_results = cross_val_score(model, features, labels, cv=kfold, scoring='roc_auc')
        elapsed_time = time.time() - start_time
        results.append(cv_results)
        times.append(elapsed_time)
        names.append(name)
        msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
        print(msg)
        print("Elapsed time for {} is {}".format(name, elapsed_time))

    # using boxplots to compare the algorithms

    fig = plt.figure(figsize=(12, 10))
    plt.title('Comparison of Classification Algorithms')
    plt.xlabel('Algorithm')
    plt.ylabel('ROC-AUC Score')
    plt.boxplot(results)
    ax = fig.add_subplot(111)
    ax.set_xticklabels(names)
    plt.show()