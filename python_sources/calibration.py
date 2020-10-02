# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

def plot_calibration_curve(estimator, X, y, name, random_state, prefit = False):
    
    cv = 3
    if prefit == True:
        cv = 'prefit'
    
    #Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(estimator, cv = cv, method = 'isotonic')
    #Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(estimator, cv = cv, method = 'sigmoid')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state)
    
    fig = plt.figure(1, figsize = (10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label = "Perfectly calibrated")
    #Whatever the value of the prefit, the CalibratedClassifierCV should be fitted
    for clf, name, fit in [(estimator, name, not prefit),
                      (isotonic, name + " + Isotonic", True),
                      (sigmoid, name + " + Sigmoid", True)]:
        
        #print('X: type of %s, Y: type of %s' % (X_train.dtype, y_train.dtype))
        if fit == True:
            clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        if hasattr(clf, 'predict_proba'):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else: # use decision fuction
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        
        clf_score = brier_score_loss(y_test, prob_pos, pos_label = y.max())
        
        print('Estimator: %s' % name)
        print('\tBrier: %1.3f' % (clf_score,))
        print('\tPrecision: %1.3f' % precision_score(y_test, y_pred))
        print('\tRecall: %1.3f' % recall_score(y_test, y_pred))
        print('\tF1: %1.3f' % f1_score(y_test, y_pred))
        
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins = 20)
        
        ax1.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label = '%s (%1.3f)' % (name, clf_score))
        
        ax2.hist(prob_pos, range = (0, 1), bins = 20, label = name, histtype = 'step', lw = 2)
        
    ax1.set_ylabel('Frraction of positives')
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc = 'lower right')
    ax1.set_title('Calibration plots (reliability curve)')
    
    # Plotting histogram
    ax2.set_xlabel('Mean predicted value')
    ax2.set_ylabel('Count')
    ax2.legend(loc = 'upper center', ncol = 2)

    plt.tight_layout()
    plt.show()
        
if __name__ == '__main__':

    X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                        n_informative=2, n_redundant=10,
                                        random_state=42)
    
    # Plot calibration curve for Gaussian Naive Bayes
    plot_calibration_curve(GaussianNB(), X, y, "Naive Bayes", 1, False)
    plt.show()