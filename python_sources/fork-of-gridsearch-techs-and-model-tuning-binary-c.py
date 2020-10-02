# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import radians, sin, cos, sqrt, asin
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import csv as csv
import re
import string
import sys
import difflib
from sklearn.model_selection import GridSearchCV
np.set_printoptions(suppress=True)
from sklearn import datasets ## imports datasets from scikit-learn
from sklearn import * #linear_model
from sklearn.tree import DecisionTreeClassifier , export_graphviz
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.svm import SVC  
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


#let begin with ready datasets
datax = datasets.load_breast_cancer() ## loads Boston dataset from datasets library 

#You can see all data set list
dir(datasets)

#Should seperate target column from predictors. Otherwise model pretends to be seen as awesome:)
X = pd.DataFrame(datax.data, columns=datax.feature_names)
Y = pd.Series(datax.target)

#Since our data set is not required for data standardization, we just use dataset in its initial form.
#In our next kernel, i will try to give examples on 
#      -standardization
#      -dealing with categorical variables
#      -PCA and LDA techniques for dimension reduction



#define params of classifer. It is important that each param is classifier specisic.
param_grid_TREE = {
    #splitting criterion is just for directing model to choose correct metric.
    #General possibilities are: gini, entropy, chi square
    #All of them has tendency to split node to significant differentiate of target variable.
    #For example you have 100sammples ant target variable YES (50 samples) and NO (50 samples). If your splitting   
    #variable generates two sub nodes of which has 50% probabilty of having YES and NO meaning shit.
    #It should significantly differentiate YES ond NO, for instance 10%YES and %90 No for a subnode is better than it.
    #Useful detailed info is at:
    #https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
    #One more note: if it is regression tree, i.e target is continous, splitting critterion is:
    #lower variance in splitted sub nodes. Because if variance between actual and predicted becomes
    #decreased compared the parent, it means classifier is on the way that correctly differentiated 
    #target
   # 'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 3, 5, 8], 
    'max_depth': [3, 5, 8, 10],
    'min_samples_leaf': [3, 5, 10, 12],
    'random_state':  [0]
    #,'class_weight': ["balanced"]
}

#Note that SVM is so time consuming especially kernels different than linear
param_grid_SVM = {
 #'kernel':('linear', 'rbf'), 
 'kernel':['linear'], 
# 'C':(1,0.25,0.5,0.75), # C is just for regularization. There is a tradeoff between train error 
#                        # and the largest minimum margin of hyperplane (test set error which model does not see yet)
#                        # for detailed info, following site is so helpful: https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
'C':[1] , # C is just for regularization. There is a tradeoff between train error 
                        # and the largest minimum margin of hyperplane (test set error which model does not see yet)
                        # for detailed info, following site is so helpful: https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel                        
 #'gamma': (1,2,3,'auto'),
 'gamma': ['auto'],  #When gamma is low, the ‘curve’ of the decision boundary is very low and thus the decision region is very broad. When gamma is high, the ‘curve’ of the decision boundary is high, which creates islands of decision-boundaries around data points.
 'decision_function_shape':['ovr'], # ovo: one vs. one : Assume that you have 3 classes A,B,C. OVO assigns classifiers for a sample being A or B, A or C, B or C
                                    # ovr: one vs. rest : Assume that you have 3 classes A,B,C. OVO assigns classifiers for a sample being A or Not, B or NOT, C or NOT    
                                    # https://scikit-learn.org/dev/modules/multiclass.html
 #'shrinking':[True,False]
 'shrinking':[True]
}

#because NB has not any params
param_grid_NB = {
}


#https://machinelearningmastery.com/start-with-gradient-boosting/
#For Gradient  Boosting Algorithm
param_grid_GB = {#'learning_rate': [0.1, 0.05, 0.02, 0.01],
                  'learning_rate': [0.1, 0.2, 0.5, 1],
                  #'max_depth': [3, 5, 8, 10],
                  #'min_samples_leaf': [3, 5, 10, 12],
                  'n_estimators' : [100],
                  'random_state':  [0]
              #'max_features': [1.0, 0.3, 0.1] 
              }


#For Gradient  Boosting Algorithm
param_grid_AB = {#'learning_rate': [0.1, 0.05, 0.02, 0.01],
                  'learning_rate': [0.1, 0.5, 1],
                  #'max_depth': [3, 5, 8, 10],
                  #'min_samples_leaf': [3, 5, 10, 12],
                  'n_estimators' : [100],
                   'random_state':  [0 ]
              #'max_features': [1.0, 0.3, 0.1] 
              }

param_grid_MLP = {
                'activation' : ['tanh'],
                'solver' : ['lbfgs'],
                'alpha': [0.0001],
                'batch_size': ['auto'],
                'hidden_layer_sizes': [(3, 3)],
                'max_iter': [200],
                'tol' : [0.0001],
                'verbose': [False],
                'warm_start': [False]
                }


#Just split training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 137)

#for different type of classifiers
clf_v = DecisionTreeClassifier()
clf_v=RandomForestClassifier()
clf_v=SVC()
clf_v = GaussianNB()
clf_v = GradientBoostingClassifier()
clf_v = OneVsRestClassifier(GradientBoostingClassifier())
clf_v = AdaBoostClassifier()
clf_v = MLPClassifier()



#the function is for performing all grid search operation with given parms, and shows model performance
#with confusion matrix

    

#Classifiers calculate feature importance which can be shown basically as follows
def featureImportanceGraph( clf_v ):
    feature_imp = pd.Series(clf_v.feature_importances_, index=X.columns).sort_values(ascending=False)
    #feature_imp = pd.Series(clf_v.feature_importances_, index=data_v.feature_names).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()


#For detailed information behind the interpretability of ROC is explained in detailed from following:
#https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5    
def ROCGraph(data_v, clf_v, title_v ):
    y_pred_prob = clf_v.predict_proba(X_test)
    #if it is a binary classification, then 1 value is reference point
    #but if you have multiclass problem ROC curve becomes one vs. ALL graph. 
    #one has to change y_pred_prob[:,1] according to reference class
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob[:,1] ) 
    plt.plot(fpr, tpr)
    plt.xlim([-0.05, 1.1]) ##just for seeing line in case of perferct classifier alligns with Y axis
    plt.ylim([-0.05, 1.1])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for ' + title_v)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()


## Prepare Model
clf_final = gridsearchCLF(param_grid_TREE,clf_v)
clf_final = gridsearchCLF(param_grid_SVM,clf_v)
clf_final = gridsearchCLF(param_grid_NB,clf_v)
clf_final = gridsearchCLF(param_grid_GB,clf_v)
clf_final = gridsearchCLF(param_grid_AB,clf_v) # very useful site for algorithm behind adaboost: https://www.datacamp.com/community/tutorials/adaboost-classifier-python
clf_final = gridsearchCLF(param_grid_MLP,clf_v)


## Visualize feature importance
featureImportanceGraph(dfx, clf_final)

## Visualize ROC Curve
ROCGraph(dfx, clf_final, 'Whatever' )
