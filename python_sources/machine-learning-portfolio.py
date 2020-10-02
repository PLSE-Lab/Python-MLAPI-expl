#!/usr/bin/env python
# coding: utf-8

# # Predicting the quality of red wine
# The research questions that will be answered in the results section are:
# 1. What are the important factors for predicting the quality of red wine?
# 2. What is the best algorithm to predict the quality of red wine in terms of accuracy scores?
# 3. How to avoid the algorithm from overfitting and underfitting?
# ## Describe the data
# 
# The data set is specifically about the anonymous wine products that are measured in various attributes. These attributes are included into the data for purposes of the final result, quality of the red wine. The results are shown for each of every wine products in the dataset. However, the name of every red wine products remain anonymous to avoid conflicts of scandalized. 
# 
# 
# The reason is because publishers wish to avoid The URL to the dataset can be found below: 
# https://www.kaggle.com/sgus1318/winedata?fbclid=IwAR1PlqpiVhf3aFyHDA9MdZ5olmgLmDRr6PMVYWbO63maZLYBR-Mi-IuJd0g#winequality_red.csv
# 
# Domain of the data set is based on chemical science because of the attributes that are provided as well as the products summary.

# ##  Exploration of the data

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
from sklearn import datasets, linear_model

#%matplotlib inline

import seaborn as sns
df = pd.read_csv("../input/winedata/winequality_red.csv")


# This is to check how the data is distributed for the first 3 rows.

# In[ ]:


df.head(3)


# The following function finds the NULL values in the data. If there are NULL values in the dataset, then the data needs to be cleaned. In this case, there are no NULL values in any features. This means that the data does not need to be cleaned.

# In[ ]:


df.isnull().any()


# The data then is described into different sets of data for further preparation for splitting data. This data is observed through describe() function to check if the values for each features have high variance which can explain the outliers of the data.
# 
# Too much difference between values can explain the outliers. Below are the features that can have outliers.
# 1. residual sugar minimum value is 0.9, maximum value is 15.5
# 2. chlorides minimum value is 0.012, maximum value is 0.611.
# 3. free sulfur dioxide minimum value is 1, maximum value is 72.
# 4. total suflur dioxide minimum value is 6, maximum value is 289.

# In[ ]:


df.describe()


# The data types are checked in order to get better understanding of the values within each features of the dataset.

# In[ ]:


df.dtypes


# The figure shows the total numbers of quality that were put into the dataset. The quality values of 5 are commonly found followed by quality of 6. However, there are no quality values 1, 9, 10 that are included in this dataset. By this, the machine learning will not be able to predict those values because there are no data with the specific quality indicated.

# In[ ]:


sns.countplot(x="quality", data=df)


# From the observation above, I have decided to determine the quality between good quality wine and bad quality wine. This line is to add extra columns into the data frame. The quality score from 1-5 will be considered as bad quality and the quality score from 6-10 are good quality scores respectively.

# In[ ]:


reviews = []
for i in df['quality']:
    if i >= 1 and i <= 5:
        reviews.append('0')
    elif i >= 6 and i <= 10:
        reviews.append('1')
df['reviews'] = reviews


# ### Pair plot
# The pair plot is the first step to show the correlation of features by visualization.

# In[ ]:


sns.pairplot(df, vars=df.columns[:-1])


# ### Heat map
# 
# Heatmap assists in helping to recognize the features that have most correlation values. In this case, the features column was investigated to observe the correlation values. I will exclude the features that have less than or more than 0.099/-0.099 correlation values to the "quality" feature.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
corr = df.corr()

# plot the heatmap
sns.heatmap(corr,annot=True,
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# As you can see, the features that can correlate to quality of the red wine are 
# - fixed acidity
# - volatile acidity
# - citirc acid
# - chlorides 
# - total sulfur dioxide 
# - density
# - sulphates
# - alcohol
# 
#  The data is split into train and test data to avoid overfitting problem.

# In[ ]:


# Get from the dataframe (the independent variables)

X = df[['fixed acidity','citric acid','residual sugar','chlorides'
        ,'free sulfur dioxide','total sulfur dioxide','density', 'pH','sulphates','alcohol']] 

# Get from the dataframe the just created label variable (dependent variables)
y = df['reviews']


# ### SelectKBest
# 
# SelectKBest is another method to identify the K numbers of features that have the highest scores. For the scores, I have use f_classif because the machine learning algorithm is entirely for classification. I have choose the number 8 due to the observations on Heatmap that 8 features are correlated to the quality. In this case, SelectKBest will have the target on dependent variable which is the bad/good quality of wine.

# In[ ]:


from sklearn.feature_selection import SelectKBest,f_classif

print(X.shape)
# Find K  best features   8 is good
kbest = SelectKBest(f_classif, k=8)
kbest = kbest.fit(X,y)
kbest.transform(X).shape
print(kbest.scores_)
print(kbest.transform(X).shape)
X.columns[kbest.get_support(indices=True)]
vector_names = list(X.columns[kbest.get_support(indices=True)])
print(vector_names)


# After feature selection by observation of HeatMap and SelectKBest, these features will be put into X as follows.

# In[ ]:


X = df[['fixed acidity', 'citric acid', 'chlorides', 'free sulfur dioxide'
        , 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']]


# The dataset is then split into a training and a testing set. In this case, the test set will be 20% that are taken randomly

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)


# Standardisation technique is performed below to handle the outliers. This is because the features that have outliers are present after observation of HeatMap and SelectKBest. Outliers can cause noises when performing machine learning. These two features are
# 1. free sulfur dioxide
# 2. total suflur dioxide
# 
# By applying standard scaler, it is a consistent way to optimize the performance of training the data because this technique transform the data into the same units with the mean value of 0 instead of comparing data with different units.

# In[ ]:


from sklearn.preprocessing import StandardScaler
#Standarize the features
#create a copy of the dataset
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train[:5]


# ### Supervised Learning
# 
# #### Preparation of functions
# 
# ##### Classification report
# 
# Performance of machine learning algorithms can be determined in various scores. For classification, the scores are shown follow:
# 1. Accuracy scores is the most important performance measurement since it is the amount of correct classification.
# 
# Varying in accuracy scores, there are train accuracy scores and test accuracy scores. Train accuracy is the accuracy of the model on train examples it was used. Test accuracy is the accuracy of the model on train examples it hasn't seen (Test set).
# 2.  Precision scores determine the ratio of correctly predicted positive observations to the total predicted positive observations.
# 3. Recall, commonly known as sensitivity or true positive rates, is the ratio of how often the value is correct when the actual value is positive.
# 4. F-1 score is the average between Precision and Recall. calculated with  2 * (precision * recall) / (precision + recall)
# 5. Support is the number of samples of the true response that lie in that class.
# 
# For my case, im going to focus on recall score to get insights of the true positive rates in order to choose the best machine learning algorithm. However, observing one score is not enough to determine the best algorithm used. With the assists of other scores at hand, I am able to answer my research question.
# ##### Confusion matrix
#  For this scenario, the confusion matrix will be applied to multiple classes. Structure of the multiple class confusion matrix consists of rows and columns as follows:
# 1. Upper left = True negatives 
# 2. Bottom right = True positives
# 2. Upper right = False postivies
# 3. Buttom left = False negatives

# By examining values further in the matrix, there are:
#      1. Accuracy scores = TP + TN/total number of classifications
#      2. Precision scores = TP/(TP + FP)
#      3. Recall = TP/(TP+FN)
#      
# ##### Learning Curves
# 
# Learning curves provides the graph that can be used to observe if the results give overfitting or underfitting with the assists of cross validation scores and accuracy scores for training set. Cross-validation is a technique to evaluate predictive models by partitioning the original sample into a training set to train the model, and a test set to evaluate if the results are overfitting and underfitting.

# In[ ]:


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print('Confusion matrix')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


from sklearn import metrics
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    
    clf.fit(X_train, y_train)
    
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    print("Accuracy on testing set:")
    print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    
    # Compute confusion matrix
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[0,1],
                      title='Confusion matrix')

    plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# #### Naive Bayes 
# 
# Naive Bayes is the algorithm to use because of all input features are independent from one another. .There is no optimization because there are no parameters to optimize. One of the disadvantages of Naive Bayes algorithm is that it over-simplified assumptions which can cause underfitting.
# 

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import math
gnb = GaussianNB()
params = {}
train_and_evaluate(gnb, X_train, X_test, y_train, y_test)


# #### K-Nearest Neighbor
# 
# KNN is an algorithm that stores all the available cases and classifies the new data or case based on a similarity measure. K numbers labels the numbers of nearest neighbors to perform majority voting in the training set. The reason behind choosing KNN algorithm is that the data that is used for machine learning is not high-dimensional data. The target variable is only justify by good quality or bad quality.
# I have optimized the KNN algorithm by iterating through K numbers and apply it into the algorithm. Since this is the binary class classification, the value of K should be an odd number.

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)
train_and_evaluate(neigh, X_train, X_test, y_train, y_test)

k_accuracy = list()
for k in range(1,13):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_pred_knn = neigh.predict(X_test)
    k_accuracy.append(accuracy_score(y_test, y_pred_knn))
    
plt.plot(range(1,13),k_accuracy)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy scores')
plt.title('The optimal numbers of Neighbour in KNN')
plt.show()

neigh = KNeighborsClassifier(n_neighbors=11)
train_and_evaluate(neigh, X_train, X_test, y_train, y_test)


# #### Support Vector Machine
# 
# 
# The SVM algorithm finds a boundary that maximizes the distance between the closest members of separate classes.
# 
# Chosen kernel is Gaussian radial basis function (RBF)
# The reason that I chose the RBF kernel is by comparing to linear and polynomial kernels, they are less time consuming and provides less accuracy than the RBF or Gaussian kernels

# In[ ]:


from sklearn import svm
from sklearn.svm import SVC

clf = SVC(gamma='auto', kernel='rbf')
train_and_evaluate(clf, X_train, X_test, y_train, y_test)


# SVC Parameter Tuning in Scikit Learn using GridSearchCV to optimize and attempt to increase the score by finding the best parameters for the classifier. Two parameters that can be determined are C and gamma.

# In[ ]:


#function for GridSearchCV with import of GridSearchCV library
from sklearn.model_selection import GridSearchCV
def svc_param_selection(X_train, y_train, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    return grid_search.best_params_
    
print(svc_param_selection(X_train, y_train, 5))


# In[ ]:


#C score
#gamma
clf = SVC(C=10, gamma=0.1, kernel='rbf')
train_and_evaluate(clf, X_train, X_test, y_train, y_test)


# ### Comparing algorithms using Learning Curves
# 
# |Algorithms|Accuracy on training set|Accuracy on testing set|Average Precision|Average Recall|Average F1-score
# |---|---|---|---|---|---|---|
# |Naive Bayes|73.7%|69.4%|69%|69%|69%|
# |KNN|78.4%|70.9%|71%|71%|71%|
# |SVC|81.3%|76.9%|77%|77%|77%|
# 
# 
# 
# The observation from the scores give results in comparison between classification scores of three machine learning algorithms. The algorithm with the highest overall scores is Support Vector Classifier, followed by K-Nearest Neighbors with the slight lower acciracu scores. The performance of Naive Bayes is the least, comparing to two other scores.
# 
#  To compare the algorithms above further, I have used learning curves by having training scores and cross validation scores with 10 iterations and choose variable X as Training samples sizes, and Y as the training accuracy scores. 
#  
#  According to the learning curves, it shows underfitting when applying Naive Bayes algorithm and K-Nearest Neighbors to the data set. Several reasons can be the causes of underfitting. One of the reasons is that Naive Bayes algorithm and KNN are simple algorithm and it has been used on the dataset after reducing numbers of features, as well as performing standardization. As for Support Vector Classifier, there are some variances shown in the graph which is the gap between training scores and the validation scores. However, the lines are converging and do not overlap each others. This also means that algorithm does not causes underfitting.
#  

# In[ ]:


title = "Learning Curves (Naive Bayes)"
# Cross validation with 10 iterations to get smoother mean test and train
# score curves, each time with 25% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
plot_learning_curve(gnb, title,X, y, ylim=(0.2, 1.0), cv=cv,)
title = "Learning Curves (KNN)"
plt.show()
plot_learning_curve(neigh, title,X, y, ylim=(0.2, 1.0), cv=cv,)
plt.show()
title = "Learning Curves (SVM, RBF kernel, $\gamma=0.1$)"
plt.show()
plot_learning_curve(clf, title,X, y, ylim=(0.2, 1.0), cv=cv,)


# ## Results
# 
# ##### 1. What are the important factors for predicting the quality of red wine?
# 
# The important factors are the features that correlate to the quality indicators of red wine. The section, exploration of data explains and visualize the features that are correlated to the quality of red wine.
# 
# ##### 2. What is the best algorithm to predict the quality of red wine in terms of accuracy scores?
# 
# The best machine learning algorithm to apply to the data set is Support Vector Classifier with RBF kernel for classification methods. The algorithm is used to classify between good quality and bad quality of red wine. The reason that the algorithm is best used is because it has the best total/average scores out of other algorithms. Furthermore, the learning curves show that the algorithm performs well without overfitting or underfitting.
# 
# ##### 3. How to avoid the algorithm from overfitting and underfitting?
# 
# I have used several methods to avoid overfitting and underfitting. Lists of methods are:
# 
# - Splitting into training and testing data
# - Standardization
# - Evaluation with cross validation scores
# - Comparison between different algorithms to observe the complexity of each of machine learning algorithm to be applied to the data set
# 
# 

# ### Unsupervised Learning
# #### K-means clustering
#  For the features that will be used for clustering are the features that have highest correlation values to quality of red wine. The elbow method is used to verify the best number of clusters for the following features. Then clusering is performed by grouping in to K group.

# In[ ]:


# Importing Modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Dataset Slicing
x_axis = df["alcohol"] 
y_axis = df["sulphates"]  

# Plotting
plt.scatter(x_axis, y_axis, c=y)
plt.show()


# In[ ]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

X = df[['alcohol','sulphates','citric acid']]

x = df["alcohol"] 
y = df["sulphates"]
z = df["citric acid"]

# create new plot and data
plt.plot()
X = np.array(list(zip(x, y))).reshape(len(x), 2)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow method into the graph
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=4)
kmeans = kmeans.fit(X)
labels = kmeans.labels_

centroids = kmeans.cluster_centers_
plt.scatter(
    x, 
    y,
    c=labels,
    cmap='plasma')
plt.xlabel('alcohol', fontsize=18)
plt.ylabel('sulphates', fontsize=16)


# By adding additional feature to prove that clustering algorithm is flexible

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
dim = plt.figure(1, figsize=(8, 6))
ax = Axes3D(dim, rect=[0, 0, 1, 1], elev=48, azim=134)

ax.set_xlabel('alcohol')
ax.set_ylabel('sulphates')
ax.set_zlabel('citric acid')

ax.scatter(x, y, z, c = labels)


# ## Conclusion
#  Throughout the process of machine learning and comparing between different algorithms, the supervised learning which is Support Vector Classifier is best used for this data set. Unsupervised learning is not useful for the data set Because the data set has already labeled as the quality of red wine. Support Vector Classifier gives highest overall classification scores in determining between good quality or bad quality of red wine. By giving 800 training examples to the algorithm, it can yield good validation scores as well as reasonable training scores.
