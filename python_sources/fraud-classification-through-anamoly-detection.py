# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA, KernelPCA, FastICA
import time

# In this script I treat the creadit card fraud detection as an anamoly detection problem rather than a classification
# this is due to the large class imbalance problem with fraud detection data
# This is a naieve approach just to test anamoly detection vs. classificaiton, not much hyperparameter tuning was performed


if __name__ == "__main__":

    add_embedded_data = True
    remove_outliers = False

    # loading data and undersampling negative samples (non-fradulent labels)
    data = pd.read_csv("../input/creditcard.csv")
    fraud = data[data['Class'] == 1]
    non_fraud = data[data['Class'] == 0]
    outliers_fraction = len(fraud) / len(non_fraud)    
    features = data.drop(["Class"], axis=1)
    labels = data["Class"]
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)

    if add_embedded_data:
        embedding = PCA(n_components=5)
        transformed_data = embedding.fit_transform(features.values)
        for i in range(transformed_data.shape[1]):
            features["new_feat_{}".format(i)] = pd.Series(transformed_data[:, i], index=features.index)

        features = (features - np.mean(features)) / (np.std(features) + 1e-8)

    anomaly_algorithms = [
        ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
        ("Isolation Forest", IsolationForest(behaviour='new', contamination=outliers_fraction, random_state=0)),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=30, contamination=outliers_fraction))]
        #("Robust covariance", EllipticEnvelope(contamination=outliers_fraction, support_fraction = 0.55)),]
    
    #testing models
    
    results = []

    for name, model in anomaly_algorithms:
        start_time = time.time()
        if name == "Local Outlier Factor":
            predictions = model.fit_predict(features)
        else:
            predictions = model.fit(features).predict(features)

        # converting prediction results to classification labels
        predictions[predictions == 1] = 0
        predictions[predictions == -1] = 1
        auc = roc_auc_score(labels, predictions)
        report = classification_report(labels, predictions)

        print('%s:  auc = %f' % (name, auc))
        print(report)
        print("Elapsed time is {}".format(time.time() - start_time))
