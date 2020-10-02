#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# This is my first Kernel on Kaggle, I wanted to build foundation with a simple classification model.
# 
# I choose this dataset because it is clean and simple, with less number of variables and observations, an ideal dataset for me to work on.
# 
# I have structured the notebook into the following tasks:
# 1. Importing and exploring the dataset
# 2. EDA on the dataset
# 3. Defining classification labels
# 4. Modelling
# 5. Conclusion
# 6. References

# # Importing and exploring the datasets
# 
# For importing the packages, I just like to put them in alphabetical order of the package, so that it is easy to manage and review if needed

# In[ ]:


#Importing the necessary packages

import collections

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import scipy
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, plot_roc_curve, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Reading the datasets
data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
data.head()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# Good that there are no missing values in the dataset. It makes our data pre-processing very much easier.
# 
# Now let's check the correlation between the variables.

# In[ ]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[ ]:


#Removing the serial number column as it adds no correlation to any columns
data = data.drop(columns = ["Serial No."])

#The column "Chance of Admit" has a trailing space which is removed
data = data.rename(columns={"Chance of Admit ": "Chance of Admit"})

data.head()


# # Exploratory data analysis
# The main EDA that I performed on this dataset is to see how the variables are distributed, to check if the variables are distributed normally. For that the pair plot is used to check the histogram of the variables as well as for the scatter plot to see how the variables are corelated to each other.

# In[ ]:


plt.hist(data["Chance of Admit"])
plt.xlabel("Chance of Admit")
plt.ylabel("Count")
plt.show()


# In[ ]:


sns.pairplot(data)


# In[ ]:


sns.kdeplot(data["Chance of Admit"], data["GRE Score"], cmap="Blues", shade=True, shade_lowest=False)


# In[ ]:


sns.kdeplot(data["Chance of Admit"], data["University Rating"], cmap="Blues", shade=True, shade_lowest=False)


# In[ ]:


sns.kdeplot(data["GRE Score"], data["University Rating"], cmap="Blues", shade=True, shade_lowest=False)


# In[ ]:


sns.scatterplot(data["GRE Score"], data["University Rating"])


# # Defining the class labels for classification
# 
# For ease of working with the classifier, it will be nice to have a 50/50 split on the data. 
# 
# For class balance, let us assume that the bottom 50% of the observations fall in class 0 (no or less chance of admit), and the top 50% of the observations fall in class 1.
# 
# Binning the *Chance of Admit* variable and seeing where the 50% lies

# In[ ]:


collections.Counter([i-i%0.1+0.1 for i in data["Chance of Admit"]])


# In[ ]:


data['Label'] = np.where(data["Chance of Admit"] <= 0.72, 0, 1)
print(data['Label'].value_counts())
data.sample(10)


# We now have 252 observations in class 0 and 248 observations in class 1, which is good enough balance that we are expecting
# 
# # Checking variable importance
# Let us now check what variables are important for out labels. For checking variable importance, we will use a basic decision tree classifier and then check what is the variable importance within the classifier
# 
# 

# In[ ]:


#Checking feature importance with DTree classifier
# define the model
model = DecisionTreeClassifier()

x = data.drop(columns = ['Chance of Admit', 'Label'])
y = data['Label']

# fit the model
model.fit(x, y)

# get importance
importance = model.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nsmallest(7).plot(kind='barh')


# # Modeling
# Splitting the dataset into train and test and seeing the size

# In[ ]:


x_train, x_test, y_train, y_test = x[:400], x[400:], y[:400], y[400:]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


def plot_roc(false_positive_rate, true_positive_rate, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# # Model 1: Logistic regression
# 

# In[ ]:


parameters = [
    {
        'penalty' : ['l1', 'l2', 'elasticnet'],
        'C' : [0.1, 0.4, 0.5],
        'random_state' : [0]
    }
]

gscv = GridSearchCV(LogisticRegression(),parameters,scoring='accuracy')
gscv.fit(x_train, y_train)

print('Best parameters set:')
print(gscv.best_params_)
print()

print("*"*50)
print("Train classification report: ")
print("*"*50)
print(classification_report(gscv.predict(x_train), y_train))
print(confusion_matrix(gscv.predict(x_train), y_train))

print()
print("*"*50)
print("Test classification report: ")
print("*"*50)
print(classification_report(gscv.predict(x_test), y_test))
print(confusion_matrix(gscv.predict(x_test), y_test))

#Crossvalidation:
cvs = cross_val_score(estimator = LogisticRegression(), 
                      X = x_train, y = y_train, cv = 12)

print()
print("*"*50)
print(cvs.mean())
print(cvs.std())


# In[ ]:


lr = LogisticRegression(C= 0.1, penalty= 'l2', random_state= 0)
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
y_proba=lr.predict_proba(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)

plot_roc(false_positive_rate, true_positive_rate, roc_auc)

print('Accurancy Score :',accuracy_score(y_test, y_pred))

cm=confusion_matrix(y_test,y_pred)
print(cm)


# # Model 2: Decision tree

# In[ ]:


parameters = [
    {
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [3, 4, 5],
        'min_samples_split' : [10, 20, 5],
        'random_state': [0],
        
    }
]

gscv = GridSearchCV(DecisionTreeClassifier(),parameters,scoring='accuracy')
gscv.fit(x_train, y_train)

print('Best parameters set:')
print(gscv.best_params_)
print()

print("*"*50)
print("Train classification report: ")
print("*"*50)
print(classification_report(gscv.predict(x_train), y_train))
print(confusion_matrix(gscv.predict(x_train), y_train))

print()
print("*"*50)
print("Test classification report: ")
print("*"*50)
print(classification_report(gscv.predict(x_test), y_test))
print(confusion_matrix(gscv.predict(x_test), y_test))

#Crossvalidation:
cvs = cross_val_score(estimator = DecisionTreeClassifier(), 
                      X = x_train, y = y_train, cv = 12)

print()
print("*"*50)
print(cvs.mean())
print(cvs.std())


# In[ ]:


dt = DecisionTreeClassifier(criterion= 'gini', max_depth= 3, min_samples_split= 10, 
                            random_state= 0)
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
y_proba=dt.predict_proba(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)

plot_roc(false_positive_rate, true_positive_rate, roc_auc)

print('Accurancy Score :',accuracy_score(y_test, y_pred))

cm=confusion_matrix(y_test,y_pred)
print(cm)


# # Model 3: Random forest

# In[ ]:


parameters = [
    {
        'n_estimators': np.arange(10, 40, 5),
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [3, 4, 5],
        'min_samples_split' : [10, 20, 5],
        'random_state': [0],
        
    }
]

gscv = GridSearchCV(RandomForestClassifier(),parameters,scoring='accuracy')
gscv.fit(x_train, y_train)

print('Best parameters set:')
print(gscv.best_params_)
print()

print("*"*50)
print("Train classification report: ")
print("*"*50)
print(classification_report(gscv.predict(x_train), y_train))
print(confusion_matrix(gscv.predict(x_train), y_train))

print()
print("*"*50)
print("Test classification report: ")
print("*"*50)
print(classification_report(gscv.predict(x_test), y_test))
print(confusion_matrix(gscv.predict(x_test), y_test))

#Crossvalidation:
cvs = cross_val_score(estimator = RandomForestClassifier(), 
                      X = x_train, y = y_train, cv = 12)

print()
print("*"*50)
print(cvs.mean())
print(cvs.std())


# In[ ]:


rf = RandomForestClassifier(criterion= 'gini', max_depth= 5, 
                            min_samples_split= 10, n_estimators= 15, 
                            random_state= 0)
rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)
y_proba=rf.predict_proba(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)

plot_roc(false_positive_rate, true_positive_rate, roc_auc)

print('Accurancy Score :',accuracy_score(y_test, y_pred))

cm=confusion_matrix(y_test,y_pred)
print(cm)


# # Model 4: Gradient boost classifier

# In[ ]:


parameters = [
    {
        'learning_rate': [0.01, 0.02, 0.002],
        'n_estimators' : np.arange(10, 100, 5),
        'max_depth' : [3, 4, 5],
        'min_samples_split' : [10, 20, 5],
        'random_state': [0],
        
    }
]

gscv = GridSearchCV(GradientBoostingClassifier(),parameters,scoring='accuracy')
gscv.fit(x_train, y_train)

print('Best parameters set:')
print(gscv.best_params_)
print()

print("*"*50)
print("Train classification report: ")
print("*"*50)
print(classification_report(gscv.predict(x_train), y_train))
print(confusion_matrix(gscv.predict(x_train), y_train))

print()
print("*"*50)
print("Test classification report: ")
print("*"*50)
print(classification_report(gscv.predict(x_test), y_test))
print(confusion_matrix(gscv.predict(x_test), y_test))

#Crossvalidation:
cvs = cross_val_score(estimator = GradientBoostingClassifier(), 
                      X = x_train, y = y_train, cv = 12)

print()
print("*"*50)
print(cvs.mean())
print(cvs.std())


# In[ ]:


gbm = GradientBoostingClassifier(learning_rate= 0.02, max_depth= 3, 
                                 min_samples_split= 10, n_estimators= 80, 
                                 random_state= 0)
gbm.fit(x_train,y_train)

y_pred = gbm.predict(x_test)
y_proba = gbm.predict_proba(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)

plot_roc(false_positive_rate, true_positive_rate, roc_auc)

print('Accurancy Score :',accuracy_score(y_test, y_pred))

cm=confusion_matrix(y_test,y_pred)
print(cm)


# # Conclusion:
# In this kernel I have learnt and demonstrated how a simple two class binary classification is performed with this dataset.
# 
# Please upvote the kernel if you like it, and to motivate me!
# 
# Hopefully, this is first of many of my kernels on Kaggle!

# # References:
# 
# I refered to lot of other kernels and notebooks as well as lot of stack overflow for the coding doubts, here are the prominent ones that I refered to. Thanks to all the contributers!
# 
# 1. https://stackoverflow.com/questions/15697350/binning-frequency-distribution-in-python
# 2. https://machinelearningmastery.com/calculate-feature-importance-with-python/
# 3. https://www.kaggle.com/kralmachine/analyzing-the-graduate-admission-eda-ml
# 

# In[ ]:


#for submission using the random forest
y_proba=rf.predict(x_test)
np.sqrt(mean_squared_error(y_proba, y_test))

