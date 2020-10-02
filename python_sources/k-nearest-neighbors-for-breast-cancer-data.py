#!/usr/bin/env python
# coding: utf-8

# ***We want to fit a KNN model classifier for the Breast Cancer Data.***
# 
# We preform here 3 main models: model1, model2, model3, and model4
# 
# Our purpose is to chose the model that perform the best accuracy on test data and give an acceptable estimation of AUC.
# 
# Model 1 is a K Nearest Neighbors with K=4 as number of neighbors.
# 
# Model 2 tries to fit multiple KNN models based on different values of K and then choose K_optimal that gives the best results.
# 
# Model 3 exploits the subclass "GridSearch" of sklearn, and uses a grid of value for K and explore the best model.
# 
# Model4 exploits the subclass RandomizedSearchCV of sklearn, it uses a grid of multiple parameters that can affect the results of the KNN model, we talk here about whether to affcet equal weights to observation or favor the nearest points of a query observation and give them more influence on that query point. We also try using that subclass to know which algorithm works better to define the space of nearest neighbors (KD-tree, Ball-tree), and we used the euclidean distance since all the data is numeric. 
# 
# In the last part, we compare the evolution of accuracy and AUC in function of different values of k

# In[ ]:


## Loading the packages 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


## Import and Explore the data 
df = pd.read_csv("../input/breast-cancer.csv")


# After importing the data, we drop 'id' column, and use a list comprehension to recode the 'diagnosis' variable as 0 if diagnosis == 'B' and 1 if diagnosis == 'M'.

# In[ ]:


df = df.drop(['id'], axis=1)


# In[ ]:


# A list comprehension to code the positive label as 1 and the negative label as 0
df['diagnosis'] = [1 if df['diagnosis'][i] =='M' else 0 for i in np.arange(0,len(df['diagnosis']),1)]


# Let's explore the shape of data, the type and, descriptive statistics of each variable, and the name of each one.

# In[ ]:


# Columns names list
df.columns


# In[ ]:


# DataFrame shape
df.shape


# In[ ]:


# Statistical Descriptive of df DataFrame
df.describe()


# In[ ]:


# df's variable type
df.dtypes


# In[ ]:


# Detect missing values
print(df.isnull().sum())


# We convert the 'diagnosis' variable from object type to categorical.

# In[ ]:


# Convert the 'diagnosis' target variable to categorical data 
df['diagnosis'] = df.diagnosis.astype('category')
assert df['diagnosis'].dtype=='category' ## if the 'diagnosis' variable wasn't converted to categorical, the assert function return an error


# After exploring the data, we split it to features and target variable.

# In[ ]:


# Split the data into train and test
X = df[df.columns[1:]] ## Features
Y = df[df.columns[0]]  ## target
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)


# **Let's begin modeling using KNN with 4 nearest neighbors.**
# **We begin with model1.**

# In[ ]:


#-----------------------------------------
# Creat a KNN model and predictions Model1
#-----------------------------------------
model_knn = KNeighborsClassifier(n_neighbors=4)
model_knn.fit(X_train, Y_train)

# predict using the knn model
Y_pred = model_knn.predict(X_test)

# Print the classification metrics
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# We plot the ROC Curve of model1's predictions, and calculate its accuracy and AUC.

# In[ ]:


# Plot the ROC curve 
Y_pred_prob = model_knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
plt.plot([0,1], [0,1], 'r--')
plt.plot(fpr, tpr, label='KNN ROC curve model1')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K Nearest Neighbors for knn')
plt.show()

# calculate the AUC and accuracy 
acc_model1 = model_knn.score(X_test, Y_test)
auc_model1 = roc_auc_score(Y_test, Y_pred_prob)
print('model 1: the AUC is equal to :{}'.format(auc_model1))
print('model 1: the ACCURACY is equal to : {}'.format(acc_model1))


# **Let's fit multiple models that depend to K, and choose the best one. **
# **We call it model2.**

# In[ ]:


#--------------------------------------------------------
# Overfitting and underfitting in function of k : Model 2
#--------------------------------------------------------
neighbor = np.arange(4,15) # range of value of K we test
train_accuracy = np.empty(len(neighbor)) # A list of multiple train's accuracies
test_accuracy = np.empty(len(neighbor)) # A list of multiple test's accuracies
for ind, k in enumerate(neighbor):
    mod_knn = KNeighborsClassifier(n_neighbors=k)
    mod_knn.fit(X_train, Y_train)
    train_accuracy[ind] = mod_knn.score(X_train, Y_train)
    test_accuracy[ind] = mod_knn.score(X_test, Y_test)
    
# Compare the accuracy in function of multiple value of k
plt.figure()
plt.plot(neighbor, train_accuracy, label='training accuracy')
plt.plot(neighbor, test_accuracy, label='testing accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# While we implement the value of K each time, we notice that the training accuracy trends to decrease, and the testig accuracy look to increase. So we take the K_optimal=12 that is the best number of neighbors we can have, for better Accuracy on test data. However, we need to take into consideration the AUC too.

# In[ ]:


# Get the best model and it evaluation
k0 = np.argmax(test_accuracy) + 4
b_knn = KNeighborsClassifier(n_neighbors=k0)
b_knn.fit(X_train, Y_train)
# predict using the b_knn model
bY_pred = b_knn.predict(X_test)
# Print the classification metrics
print(confusion_matrix(Y_test, bY_pred))
print(classification_report(Y_test, bY_pred))


# In[ ]:


# ROC curve for the b_knn
bY_pred_prob = b_knn.predict_proba(X_test)[:,1]
fpr, tpr, threshold = roc_curve(Y_test, bY_pred_prob, pos_label=1) 
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label = 'KNN-ROC Curve model 2')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K Nearest Neighbors for b_knn')
plt.show()

# calculate the AUC and accuracy 
acc_model2 = b_knn.score(X_test, Y_test)
auc_model2 = roc_auc_score(Y_test, bY_pred_prob)
print('model 2 : the AUC is equal to: {}'.format(auc_model2))
print('model 2: the ACCURACY is equal to: {}'.format(acc_model2))


# It seems that modeling with multiple values of K and choose the best one, give us a model with better accuracy and AUC than fitting a KNN model with 4 neighbors.

# **Let's tune our KNN model using the sub class GridSearche of sklearn.**
# 
# **We call it model3.**

# In[ ]:


# ---------------------------------------
# Tune the parameter K of the KNN Model 3
# ---------------------------------------
param_grid = {'n_neighbors': np.arange(1,30)}
knn = KNeighborsClassifier()
Grid_knn = GridSearchCV(knn, param_grid, scoring='roc_auc', cv=5)
Grid_knn.fit(X_train, Y_train)

# predict using the tunde model
tY_pred = Grid_knn.predict(X_test)

# Print the classification metrics
print(confusion_matrix(Y_test, tY_pred))
print(classification_report(Y_test, tY_pred))


# In[ ]:


# ROC curve for the b_knn
tY_pred_prob = Grid_knn.predict_proba(X_test)[:,1]
fpr, tpr, threshold = roc_curve(Y_test, tY_pred_prob)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label = 'KNN-ROC Curve model 3')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K Nearest Neighbors for Grid_knn')
plt.show()

# calculate the AUC and accuracy 
acc_model3 = Grid_knn.score(X_test, Y_test)
auc_model3 = roc_auc_score(Y_test, tY_pred_prob)
print('model 3 : the AUC is equal to: {}'.format(auc_model3))
print('model 3: the ACCURACY is equal to: {}'.format(acc_model3))

# Get the best parameters for KNN model:
print(Grid_knn.best_params_)
print(Grid_knn.best_score_)


# So the GridSearch subclass found that using 25 as number of nearest neighbors gives better accuracy and AUC.

# In[ ]:


# -----------------------------------------------------------------------------------
# Let's tune our KNN model using a grid search to find best parameters to model with: model4
# -----------------------------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV
params = {'n_neighbors': np.arange(4,30,1),
          'weights':['uniform', 'distance'],
          'algorithm':['ball_tree', 'kd_tree'],
          'metric':['euclidean']}
knn = KNeighborsClassifier()
Rando_knn = RandomizedSearchCV(knn, params, scoring='roc_auc', cv = 5)
Rando_knn.fit(X_train, Y_train)

# Predict using the tuned model
rY_pred = Rando_knn.predict(X_test)

# Print the classification metrics
print(confusion_matrix(Y_test, rY_pred))
print(classification_report(Y_test, rY_pred))


# In[ ]:


# Plot the ROC curve and accuracy
rY_pred_prob = Rando_knn.predict_proba(X_test)[:,1]
fpr, tpr, threshold = roc_curve(Y_test, rY_pred_prob)
plt.plot([0,1], [0,1], 'b--')
plt.plot(fpr, tpr, label='KNN-ROC Curve model 4')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K Nearest Neighbors for Rando_knn')
plt.show()


# In[ ]:


# calculate the AUC and accuracy 
acc_model4 = Rando_knn.score(X_test, Y_test)
auc_model4 = roc_auc_score(Y_test, rY_pred_prob)
print('model 4 : the AUC is equal to: {}'.format(auc_model4))
print('model 4: the ACCURACY is equal to: {}'.format(acc_model4))

# Get the best parameters for KNN model:
print(Rando_knn.best_params_)
print(Rando_knn.best_score_)


# it seems that the RandomizedSearchCV gives the same number of neighbors:25, the best algorithm was 'ball-tree', and it offers an acceptable accuracy of 98.37%

# In[ ]:


# --------------------------------------------------------------------------------------------
# Cross validation for Knn : Relation between variation of accuracy and AUC in Function of value of K
# --------------------------------------------------------------------------------------------
from sklearn.model_selection import cross_val_score
k_range = range(1,30)
k_scores_auc=[]
k_scores_acc=[]
for k in k_range:
    #1.Rune the KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    #2.obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores_auc = cross_val_score(knn, X, Y, cv=5, scoring='roc_auc')
    scores_acc = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
    k_scores_auc.append(scores_auc.mean())
    k_scores_acc.append(scores_acc.mean())

plt.plot(k_range, k_scores_auc, label = 'AUC')
plt.plot(k_range, k_scores_acc, label = 'ACCURACY')
plt.legend()
plt.xlabel('k for Knn')
plt.ylabel('Evaluation in %')
plt.show()


# Using a 5 Folds Cross Validation for multiple values of nearest neihgbors, we notice that in a range between 12 to 17 nearest neeighbors we got the best accuracy but not for the AUC, since it reachs its maximum for a range value of K between 25 and 30. 
# Both of these results confirm what we get in three models.
# Since AUC give best evaluation of a classifier model, we can conclude that model2 is the suitable one for our Breast Cancer Data.

# In[ ]:


#---------
## SUM UP
#---------
sumup = pd.DataFrame({'model':[1,2,3,4], 'accuracy':[acc_model1, acc_model2, acc_model3, acc_model4],
                      'AUC':[auc_model1, auc_model2, auc_model3,auc_model4]})
print(sumup)

