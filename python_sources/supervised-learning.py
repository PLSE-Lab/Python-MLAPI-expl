#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
iris=datasets.load_iris()


# In[ ]:


type(iris)


# In[ ]:


print(iris.keys())


# In[ ]:


type(iris.data),type(iris.target)


# In[ ]:


iris.data.shape


# In[ ]:


print(iris.target_names)
print(iris.feature_names)


# In[ ]:


x=iris.data
y=iris.target


# In[ ]:


df= pd.DataFrame(x,columns=iris.feature_names)


# In[ ]:


df.head()


# In[ ]:


_=pd.scatter_matrix(df,c=y,figsize=[8,8],s=150,marker='D')


# In[ ]:


_=pd.plotting.scatter_matrix(df,c=y,figsize=[8,8],s=150,marker='D')


# EDA in python
# * .head()
# * .info()
# * .describe()

# In[ ]:


plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()


# Classification
# * .fit()
# * .predict()

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'],iris['target'])


# In[ ]:


# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

#using the model to predict values
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))


# Measuring model performance
# 
# calculation of accuracy
# 
# split data in train and validation set
# 
# X_train,X_test,Y_train,Y_test = train_test_split(Data, targetvariable, test_size=0.3, random_state = 21, stratify = y)
# 
#  random_state --> seed for test train random data generation
#  test_size -->
#  stratify --> list or array containing the labels
#  
#  knn.score(X_test,Y_test)

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = 
train_test_split(Data, targetvariable, test_size=0.3, random_state = 21, stratify = y)


# **Train/Test Split + Fit/Predict/Accuracy**

# In[ ]:


# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors = 7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))


# Overfitting and underfitting

# In[ ]:


# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# **Regression Analysis**
# 
# using built in datasets in sklearn package
# 
# We will import the boston dataset and will run regression on the data to predict target variable

# In[ ]:


from sklearn import datasets
import numpy as np
import pandas as pd
boston = datasets.load_boston()


# In[ ]:


print(type(boston))
print(boston.keys())


# In[ ]:


print(boston.feature_names)


# In[ ]:


boston_df = pd.DataFrame(boston.data,columns=boston.feature_names)


# In[ ]:


boston_df.head()


# In[ ]:


X=boston_df.values
y=boston.target


# In[ ]:


print(X.shape)
print(type(X))
print(y.shape)
print(type(y))


# In[ ]:


plt.scatter(X[:,5],y)
plt.ylabel('Value of house / 1000($)')
plt.xlabel('Number of rooms')
plt.margins(0.2)
plt.show()


# Fitting a regression model
# * from sklearn import linear_model
# * reg = linear_model.LinearRegression()
# * reg.fit(X_room,y)

# In[ ]:


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X[:,5].reshape(-1,1),y.reshape(-1,1))
prediction_space = np.linspace(min(X[:,5]),max(X[:,5])).reshape(-1,1)
plt.scatter(X[:,5],y,color='blue')
plt.plot(prediction_space,reg.predict(prediction_space),color='black',linewidth=3)
plt.show()

print(reg.score(X[:,5].reshape(-1,1),y.reshape(-1,1)))


# **Constructing heatmap to explore data and see correlation **

# In[ ]:


sns.heatmap(boston_df.corr(), square=True, cmap='RdYlGn')
plt.show()


# Ordinary least squares (OLS) :  Minimize the sum of squares of residuals

# In[ ]:


# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# **Cross Validation**
# 1. split dataset info multiple folds (multiple subsets)
# 2. keeping one fold as test data, run model on remaining folds.
# 3. We will repeat this process multiple times, by keeping different folds aside and running the model on the remaining folds
# 4. compute the metric of interest in this iterative process
# 
# K-fold Cross Validation
# 
# advt:  removes the dependency of calculated metric data on test and validation set.
# 
# from sklearn.model_selection import cross_val_score
# reg = linear_model.LinearRegression()
# cv_results = cross_val_score(reg, X, y, cv=5)
# 

# In[ ]:


# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


# %timeit cross_val_score(reg, X, y, cv = ____)  #run to get time

# Regularized Regression
# 
# linear regression --> minimize the loss function
# regularized regression --> 
#     ridge regression
#     lasso function
# 
# example
# 
# **Ridge**
# * from sklearn.linear_model import Ridge
# * X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=, random_state=)
# * ridge = Ridge(alpha=0.1, normalize=True)
# * ridge.fit(X_train, y_train)
# * ridge_pred = ridge.predict(X_test)
# * ridge.score(X_test,y_test)
# 
# 
# **Lasso**
# * from sklearn.linear_model import Lasso
# * X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=, random_state=)
# * lasso = Lasso(alpha=0.1, normalize=True)
# * lasso.fit(X_train, y_train)
# * lasso_pred = ridge.predict(X_test)
# * lasso.score(X_test,y_test)
# * 
# * names= boston.drop('MEDV', axis=1).columns
# * lasso = Lasso(alpha=0.1)
# * lasso_coef = lasso.fit(X,y).coef_
# * _=plt.plot(range(len(names)),lasso_coef)
# * _=plt.xticks(range(len(names)),names,rotation=60)
# * _=plt.ylabel('Coefficients')
# * plt.show()
# 
# 
# 

# ***Lasso is great for feature selection, but when building regression models, Ridge regression should be your first choice.***

# In[ ]:


# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4,normalize=True)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()


# In[ ]:


def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


# In[ ]:


# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    ridge.fit(X,y)
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge,X,y,cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


# **Fine Tuning your model**
# 
# Classification : generally ***accuracy*** is used
# 
# 
# **confusion matrix**
# 1. Accuracy = tp + tn / (tp +tn +fp +fn)
# 2. Precision =  tp / (tp + fp)
# 3. Recall = tp / (tp + fn)
# 4. F1 score =  
# 
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# 
# print( confusion_matrix( y_test, y_pred ) )
# print( classification_report( y_test, y_pred ) )
# 
# 

# In[ ]:


# Import necessary modules
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.4,random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# **Logistic Regression**
# 
# from sklearn.linear_model import LogisticRegression
# 
# ROC (Receiver operator characteristic) curve
# 
# from sklearn.metrics import roc_curve
# y_pred_prob = logreg.predict_proba(X_test)[:,1]
# fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob)  fpr - false positive rate, tpr - true positive rate
# 
# 

# In[ ]:


# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train,y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# Larger the area under ROC curve better the model.
# known as AUC (area under the curve)
# 
# from sklearn.metrics import roc_auc_score
# 
# y_pred_prob = logreg.predict(X_test)[:,1]
# roc_auc_score(y_test, y_pred_prob)
# 
# * AUC using cross validation
# 
# from sklearn.metrics import cross_val_score
# cv_scores = cv_val_score(logreg, X, y, cv=5, scoring = 'roc_auc')
# 
# 

# In[ ]:


# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


# linear regression --> chossing parameters
# 
# ridge/lasso regression --> choosing alpha
# 
# k-nearest neighbours --> choosing n_neighbours
# 
# all these parameters are called hyperparameters 
# chooing correct hyperparameters
# * try bunch of different parameters
# * fit them all separately
# * see how well each performs
# * choose the best one
# 
# #Hyperparameter tuning
# 
# 
# Use Gridsearch CV 
# 
# * from sklearn.model_selection import GridSearchCV
# * param_grid = {}
# * knn = KNeighborClassifier()
# * knn_cv = GridSearchCV(knn, param_grid, cv=5)
# * knn_cv.fit(X,y)
# * knn_cv.best_params_
# * knn_cv.best_score_
# 
# 

# In[ ]:


# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))


# GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter space and dealing with multiple hyperparameters. A solution to this is to use **RandomizedSearchCV**
# 
# Note that RandomizedSearchCV will never outperform GridSearchCV. Instead, it is valuable because it saves on computation time.
# 
# 

# In[ ]:


# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


# Holdout sample from the dataset
# 
# The idea is to tune the model's hyperparameters on the training set, and then evaluate its performance on the hold-out set which it has never seen before.
# 
# scikit-learn :  OneHotEncoder()
# 
# pandas : get_dummies()
# 
# Imputing missing data
# 
# from sklean.preprocessing import Imputer
# 
# imp = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
# imp.fit(X)
# X = imp.transform(X)
# 
# Imputing within a pipeline
# 
# from 
# 
# 
# 

# In[ ]:


# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train,y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


# Centring and scaling data
# 
# 
# df.describe()
# 
# Normalizing the data
# 
# subtract mean and divide by variance
# 
# subtract minimum and divide by range
# 
# from sklearn.preprocessing import scale
# 
# X_scaled = scale(X)
# 
# Scaling in a pipeline
# from sklearn.preprocessing import StandardScalar
# 
# 
