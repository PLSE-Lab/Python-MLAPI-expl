import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsOneClassifier

from sklearn.svm import SVC


############################################################


DIR = '../input/'
train_set = pd.read_csv(DIR + 'train.csv')

X_TRAIN =  train_set.drop('label', axis=1)
TARGET = train_set['label']


pca = PCA(n_components=35)
pca.fit(X_TRAIN)
X_pca = pca.transform(X_TRAIN)

X_train, X_test, y_train, y_test = train_test_split(X_pca, TARGET, test_size = 0.1, random_state = 45)

start = dt.now()


ML = [OneVsOneClassifier(SVC(probability=True, gamma='auto', C=1)),
     OneVsOneClassifier(SVC(probability=True, gamma='auto', C=10))]#,
     #OneVsOneClassifier(SVC(probability=True, gamma='auto', C=100))]

MLA_columns =  ['PCA_n', 'MLA Name', 'MLA Parameters', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
MLA = pd.DataFrame(columns=MLA_columns)

n = 35

for row, alg in enumerate(ML):

    cv_results = cross_validate(alg, X_pca, TARGET, cv=4)
    MLA.loc[row, 'PCA_n'] = n
    MLA_name = alg.__class__.__name__
    MLA.loc[row, 'MLA Name'] = MLA_name
    MLA.loc[row, 'MLA Parameters'] = str(alg.get_params())
    MLA.loc[row, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA.loc[row, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    MLA.loc[row, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3

    print('\tMLA algorithm: {} has finished in {}'.format(MLA_name, dt.now() - start))


MLA.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)
MLA.to_csv('MLA_Results.csv', index=False)   

# params = {'C' : [0.1, 1]}

# model=

# clf =  GridSearchCV(model, params, cv=2, refit=True, verbose=2).fit(X_pca, TARGET)

# print(clf.cv_results_)
# pd.DataFrame(clf.cv_results_).to_csv('clf.csv', index=False)

# gs_results = pd.DataFrame(clf.cv_results_).loc[:, ['mean_test_score', 'rank_test_score', 'params']].sort_values(by='rank_test_score')

# gs_results.to_csv('SVM.csv', index=False)