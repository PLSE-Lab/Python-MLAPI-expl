#!/usr/bin/env python
# coding: utf-8

# ### Run or Walk Analysis using Decision Tree & Random Forest

# In[ ]:


#Importing necessary libraries
import pandas as pd
from sklearn import tree
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Reading the files
dataset = pd.read_csv("../input/dataset.csv", parse_dates=[['date', 'time']])


# In[ ]:


# Checking the data types
dataset.info()


# In[ ]:


#Shape of dataset
dataset.shape


# In[ ]:


#checking null values
dataset.isnull().sum()


# In[ ]:


#Viewing the top 5 records
dataset.head()


# In[ ]:


#Removing the username
dataset.drop(dataset.columns[1], axis=1, inplace=True)


# In[ ]:


#Setting date and time as index
dataset = dataset.set_index('date_time')
dataset.head()


# In[ ]:


#Checking corelation of Activity with other columns
dataset.corr()['activity']


# In[ ]:


# Defining x and y variables
columns = ['wrist', 'acceleration_x','acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y','gyro_z']
x = dataset[list(columns)].values
y = dataset["activity"].values


# In[ ]:


#Splitting data into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 123)


# ### Decision Tree

# In[ ]:


#Fitting classifier to training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state= 123)
classifier = classifier.fit(x_train, y_train)


# In[ ]:


#predicting test test results
y_pred = classifier.predict(x_test)


# In[ ]:


#Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
cm


# In[ ]:


#Plotting Confusion Matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(cm, cmap=cmap) # imshow
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

plot_confusion_matrix(cm)


# In[ ]:


#Print the score of the new decison tree
print(classifier.score(x_train, y_train))


# In[ ]:


#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, x_train, y_train, cv=10)
print("Cross Validation score : " + str(accuracies))
print("Cross Validation Mean score : " + str(accuracies.mean()))


# ### Random Forest

# In[ ]:


#Fitting classifier to training set
from sklearn.ensemble import RandomForestClassifier

classifier1 = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 123)
classifier1.fit(x_train, y_train)


# In[ ]:


#predicting test test results
y_pred = classifier1.predict(x_test)


# In[ ]:


#Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
cm


# In[ ]:


#Plotting Confusion Matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(cm, cmap=cmap) # imshow
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

plot_confusion_matrix(cm)


# In[ ]:


# Print the score of the fitted random forest
print(classifier1.score(x_train, y_train))


# In[ ]:


#Applying k-fold cross validation
accuracies = cross_val_score(classifier1, x_train, y_train, cv=10)
print("Cross Validation score : " + str(accuracies))
print("Cross Validation Mean score : " + str(accuracies.mean()))


# In[ ]:


#Roc curve on predicted data
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


# ROC curve on Predicted probabilities
pred_proba = classifier1.predict_proba(x_test)
fpr, tpr, _ = metrics.roc_curve(y_test, pred_proba[:,1])
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




