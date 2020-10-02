#!/usr/bin/env python
# coding: utf-8

# The dataset is the Video Game Sales with Ratings dataset from Kaggle. Initially
# there are 16 attributes and 16719 observations, but after data preparation and 
# EDA there are 6825 observations. I use 9 attributes in my model as features: 
# 'Global_Sales' is the target and Kmeans is imputed and the label is added as a 
# feature, with all features in the Kmeans model undergoing normalization and 
# encoding because of the different scales and data types. I use the elbow method
# to determine the ideal number of clusters for Kmeans. 

# The target is highly non-linear so I made this a classification problem: 
# can I predict if a game will be a 'blockbuster', defined as over 1 million units 
# sold globlally?

# Using domain knowledge, I choose the features, train/test split the data and created
# a pipeline to scale and encode them before going into the models. Kmeans is used 
# on my features (3 categorical, 4 numeric) to identify clustering within the data. 
# This label is added as a imputed feature to my model features. I use Logistic 
# Regression and Random Forrest Classification for my Supervised models. I print 
# the confusion matrices, ROC curves, and metrics for both models.

# Attributes
# 
# Name               6825 non-null str,
# Platform           6825 non-null str     Feature OHE Kmeans,
# Year_of_Release    6825 non-null float64,
# Genre              6825 non-null str     Feature OHE Kmeans,
# Publisher          6825 non-null str,
# NA_Sales           6825 non-null float64,
# EU_Sales           6825 non-null float64,
# JP_Sales           6825 non-null float64,
# Other_Sales        6825 non-null float64,
# Global_Sales       6825 non-null float64 Target  OHE Kmeans "Binned into 2 classes, non-linear distribution",
# Critic_Score       6825 non-null float64 Feature StandardScaled Kmeans,
# Critic_Count       6825 non-null float64 Feature StandardScaled Kmeans,
# User_Score         6825 non-null float64 Feature StandardScaled Kmeans,
# User_Count         6825 non-null float64 Feature StandardScaled Kmeans,
# Developer          6825 non-null str,
# Rating             6825 non-null str     Feature OHE Kmeans "Binned into 3 classes, non-uniform distribution",
# Kmeans_labels      6825 non-null int32   Feature n_clusers=4,
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
from sklearn.preprocessing import StandardScaler, OneHotEncoder # scaling data for pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer # apply scaling to dataset
from sklearn.pipeline import make_pipeline # create pipeline for data processing
from sklearn.model_selection import train_test_split # train/test split data
from sklearn.linear_model import LogisticRegression # classificatino model
from sklearn.ensemble import RandomForestClassifier # classification model
from sklearn.cluster import KMeans # calculate Kmeans clustering w/num_clusters determined by elbow method
from sklearn.metrics import confusion_matrix # verify model accuracy metrics
import sklearn.metrics as metrics # accuracy/precision/recall/f1
import matplotlib # plotting
from sklearn.metrics import roc_curve, auc # roc/auc curve visualization
from matplotlib import pyplot # plotting
from sklearn.model_selection import cross_val_score # cross validation
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Define dataset and check that it is well imported

# In[ ]:


dataset = pd.read_csv('/kaggle/input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv')
dataset.head()


# Bin Global_Sales into Blockbuster based on selling over 1 million units --> drop all rows with NAs (6825 remain)

# In[ ]:


# Bin NA_Sales into Blockbuster or NB based on units sold
BlockBuster = dataset.loc[:, "Global_Sales"] > 1.0
#dataset.loc[:,"Global_Sales"] = dataset.loc[:,"Global_Sales"].astype(int)
dataset.loc[BlockBuster, "Global_Sales"] = 1
dataset.loc[~BlockBuster, "Global_Sales"] = 0
# Drop nulls
nullRemoved = dataset.dropna(axis=0) #6825 observations with no nulls
# Convert to int
nullRemoved = nullRemoved.astype({'User_Score':float})


# Check data shape and dtype, especially of converted User_Score attribute

# In[ ]:


from tabulate import tabulate
tabulate(nullRemoved.info(), headers='keys', tablefmt='psql')


# Group like categories: AO -> M, RP -> T, K-A/E10+ -> E

# In[ ]:


nullRemoved.loc[nullRemoved.loc[:, "Rating"] == "AO", "Rating"] = "M"
nullRemoved.loc[nullRemoved.loc[:, "Rating"] == "K-A", "Rating"] = "E"
nullRemoved.loc[nullRemoved.loc[:, "Rating"] == "RP", "Rating"] = "T"
nullRemoved.loc[nullRemoved.loc[:, "Rating"] == "E10+", "Rating"] = "E"


# Define target and featuress for model using domain knowledge of likely significant attributes related to sales

# In[ ]:


target =  nullRemoved[['Global_Sales']].values #y
features = nullRemoved[['Platform','Genre','Critic_Score','Critic_Count','User_Score','User_Count','Rating']] #X


# Make pipeline that scales and encodes features

# In[ ]:


preprocess = make_column_transformer(
    (StandardScaler(), ['Critic_Score', 'Critic_Count','User_Score','User_Count' ]), #features scale
    (OneHotEncoder(), ['Platform', 'Genre', 'Rating']) #OHE categorical features
)


# Scale data for use in Kmeans

# In[ ]:


kmeans_features = preprocess.fit_transform(features).toarray()


# Use elbow method to determine ideal number of clusters
# 

# In[ ]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(kmeans_features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()  #4 is ideal number of clusters


# Train/test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0)


# Fitting K-Means to the dataset

# In[ ]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 69)
y_kmeans = kmeans.fit_predict(kmeans_features)


# Add Kmeans labels to features df

# In[ ]:


features = features.assign(Kmeans_labels=pd.Series(y_kmeans, index=features.index))


# Train/test split with KMEANS attribute added

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0)


# Make Logistic Regression Classification model

# In[ ]:


model = make_pipeline(
    preprocess,
    LogisticRegression(solver='lbfgs',penalty='l2'))
model.fit(X_train, y_train.ravel())
y_pred = model.predict(X_test)
print("Logistic Regression Score: %f" % model.score(X_test, y_test))


# Probabiliities for threshold

# In[ ]:


BothProbabilities = model.predict_proba(features)
probabilities = BothProbabilities[:,1]


# Make Random Forest Classification model

# In[ ]:


model2 = make_pipeline(
        preprocess,
        RandomForestClassifier(n_estimators=100))
model2.fit(X_train, y_train.ravel())
y_pred2 = model2.predict(X_test)
print("\nRandom Forest Score: %f" % model2.score(X_test, y_test)) 


# ProbaProbabiliities for thresholdbiliities for threshold

# In[ ]:


BothProbabilities2 = model2.predict_proba(features)
probabilities2 = BothProbabilities2[:,1]


# Make confusion matrices

# In[ ]:


cm = confusion_matrix(y_test, y_pred)
print ('\nLogistic Regression Confusion Matrix and Metrics')
Threshold = 0.3 # Some number between 0 and 1
print ("\nProbability Threshold is chosen to be:", Threshold)
predictions = (probabilities > Threshold).astype(int)
tn, fp, fn, tp = cm.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
print("\nAccuracy:",metrics.accuracy_score(y_test, y_pred))
print("\nPrecision:",metrics.precision_score(y_test, y_pred))
print("\nRecall:",metrics.recall_score(y_test, y_pred))
print("\nF1:",metrics.f1_score(y_test, y_pred))
print("\nAverage precision-recall score:",metrics.average_precision_score(y_test, y_pred))


# In[ ]:


cm2 = confusion_matrix(y_test, y_pred2)
print ('\nRandom Forest Confusion Matrix and Metrics')
Threshold = 0.01 # Some number between 0 and 1
print ("Probability Threshold is chosen to be:", Threshold)
predictions = (probabilities2 > Threshold).astype(int)
tn, fp, fn, tp = cm2.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
print("\nAccuracy:",metrics.accuracy_score(y_test, y_pred2))
print("\nPrecision:",metrics.precision_score(y_test, y_pred2))
print("\nRecall:",metrics.recall_score(y_test, y_pred2))
print("\nF1:",metrics.f1_score(y_test, y_pred2))
print("\nAverage precision-recall score:",metrics.average_precision_score(y_test, y_pred2))


# LR ROC curve

# In[ ]:


fpr, tpr, th = roc_curve(target, probabilities)
AUC = auc(fpr, tpr)

plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('Logistic Regression ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show() #ideal threshold is around 0.3


# RF ROC curve

# In[ ]:


fpr, tpr, th = roc_curve(target, probabilities2)
AUC = auc(fpr, tpr)

plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('Random Forest ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show() #ideal threshold is around 0.1


# Spot check algorithm performance against other models.

# In[ ]:


X_train2  = preprocess.fit_transform(X_train).toarray()
models = []
models.append(('LR', LogisticRegression(solver='lbfgs')))
models.append(('RC', RidgeClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RF', RandomForestClassifier(n_estimators=100)))
# evaluate each model in turn
results = []
names = []
out = []
# need to use preprocess to scale and encode X_train and y_train
for name, model in models:
    #model = preprocess.fit_transform(features).toarray()
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, X_train2, y_train.ravel(), cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Compare Algorithms

# In[ ]:


plt.figure() #created empty frame for plt
pyplot.boxplot(results, labels=names)
#pyplot.subplot()
pyplot.title('Algorithm Comparison')
pyplot.xlabel('Models')
pyplot.ylabel('Accuracy')
#plt.xlim(right=20)
#plt.subplot(888)
pyplot.show()


# Based upon the ROC curves for each model I tuned the Threshold value to the point
# on the curve closest to the upper-left corner. The Random Forest Classifier is more
# accurate than the Logistic Regression model when it comes to identifying blockbuster
# videogames. The ideal number of clusters for Kmeans was 4. Both models are highly
# accurate, but the recall and F1 rates are much much better for my Random Forest
# model. I conclude that I can can classify games based on whether they will sell 
# one million global units with decent accuracy. My model of models shows that I have
# used two of the highest accuracy classifiers.
