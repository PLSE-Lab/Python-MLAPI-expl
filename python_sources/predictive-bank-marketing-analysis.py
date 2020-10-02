#!/usr/bin/env python
# coding: utf-8

# # Predictive analysis of Bank Marketing
# 
# #### Problem Statement
# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
# 
# #### What to achieve?
# The classification goal is to predict if the client will subscribe a term deposit (variable y).
# 
# #### Data Contains information in following format:
# 
# ### Categorical Variable :
# 
# * Marital - (Married , Single , Divorced)",
# * Job - (Management,BlueCollar,Technician,entrepreneur,retired,admin.,services,selfemployed,housemaid,student,unemployed,unknown)
# * Contact - (Telephone,Cellular,Unknown)
# * Education - (Primary,Secondary,Tertiary,Unknown)
# * Month - (Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec)
# * Poutcome - (Success,Failure,Other,Unknown)
# * Housing - (Yes/No)
# * Loan - (Yes/No)
# * Default - (Yes/No)
# 
# ### Numerical Variable:
# 
# * Age
# * Balance
# * Day
# * Duration
# * Campaign
# * Pdays
# * Previous
# 
# #### Class
# * deposit - (Yes/No)

# 1. **Importing required libraries**

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


#Classification Algorithms 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as m
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Importing and displaying the data**

# In[ ]:


data = pd.read_csv("../input/bank.csv", delimiter=";",header='infer')
data.head()


# **Finding correlation between features and class for selection**

# **1. Using Pairplot**

# In[ ]:


sns.pairplot(data)


# We can observe that data here is not-symmetric. So lets find out the correlation matrix to look into details.

# **2. Correlation Matrix**

# In[ ]:


data.corr()


# **3. Heatplot to visualise correlation**

# In[ ]:


sns.heatmap(data.corr())


# #### As per the pairplot, correlation matrix, and heatmap, observations as follow:
# * Data is non-linear, asymmetric
# * Hence selection of features will not depend upon correlation factor.
# * Also not a single feature is correlated completely with class, hence requires combinantion of features.

# ## Feature Selection techniques:
# 1. Univariate Selection (non-negative features)
# 2. Recursive Feature Elimination (RFE)
# 3. Principal Component Analysis (PCA) (data reduction technique)
# 4. Feature Importance (decision trees)
# 
# #### Which feature selection technique should be used for our data?
# * Contains negative values, hence Univariate Selection technique cannot be used.
# * PCA is data reduction technique. Aim is to select best possible feature and not reduction and this is classification type of data. 
# * PCA is an unsupervised method, used for dimensionality reduction.
# * Hence Decision tree technique and RFE can be used for feature selection.
# * Best possible technique will be which gives extracts columns who provide better accuracy.

# **Encoding Categorical and numerical data into digits form.**

# In[ ]:


data.dtypes


# Converting object type data into One-Hot Encoded data using get_dummies method.

# In[ ]:


data_new = pd.get_dummies(data, columns=['job','marital',
                                         'education','default',
                                         'housing','loan',
                                         'contact','month',
                                         'poutcome'])


# In[ ]:


#Class column into binary format
data_new.y.replace(('yes', 'no'), (1, 0), inplace=True)


# In[ ]:


#Successfully converted data into  integer data types
data_new.dtypes


# **Exploring features: Age as a example**

# In[ ]:


#Whole dataset's shape (ie (rows, cols))
print(data.shape)


# In[ ]:


#Unique education values
data.education.unique()


# In[ ]:


#Crosstab to display education stats with respect to y ie class variable
pd.crosstab(index=data["education"], columns=data["y"])


# In[ ]:


#Education categories and there frequency
data.education.value_counts().plot(kind="barh")


# ### Classifiers : Based on the values of different parameters we can conclude to the following classifiers for Binary Classification.
# 
#     1. Gradient Boosting
#     2. AdaBoosting
#     3. Logistics Regression
#     4. Random Forest Classifier
#     5. Linear Discriminant Analysis
#     6. K Nearest Neighbour
#     7. Decision Tree
#     8. Gaussian Naive Bayes 
#     9. Support Vector Classifier
# 
# #### And performance metric using precision and recall calculation along with roc_auc_score & accuracy_score

# In[ ]:


from xgboost import XGBClassifier
classifiers = {
               'Adaptive Boosting Classifier':AdaBoostClassifier(),
               'Linear Discriminant Analysis':LinearDiscriminantAnalysis(),
               'Logistic Regression':LogisticRegression(),
               'Random Forest Classifier': RandomForestClassifier(),
               'K Nearest Neighbour':KNeighborsClassifier(8),
               'Decision Tree Classifier':DecisionTreeClassifier(),
               'Gaussian Naive Bayes Classifier':GaussianNB(),
               'Support Vector Classifier':SVC(),
               }


# In[ ]:


#Due to one hot encoding increase in the number of columns
data_new.shape


# In[ ]:


data_y = pd.DataFrame(data_new['y'])
data_X = data_new.drop(['y'], axis=1)
print(data_X.columns)
print(data_y.columns)


# In[ ]:


log_cols = ["Classifier", "Accuracy","Precision Score","Recall Score","F1-Score","roc-auc_Score"]
log = pd.DataFrame(columns=log_cols)


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
rs = StratifiedShuffleSplit(n_splits=2, test_size=0.3,random_state=2)
rs.get_n_splits(data_X,data_y)
for Name,classify in classifiers.items():
    for train_index, test_index in rs.split(data_X,data_y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X,X_test = data_X.iloc[train_index], data_X.iloc[test_index]
        y,y_test = data_y.iloc[train_index], data_y.iloc[test_index]
        # Scaling of Features 
#         from sklearn.preprocessing import StandardScaler
#         sc_X = StandardScaler()
#         X = sc_X.fit_transform(X)
#         X_test = sc_X.transform(X_test)
        cls = classify
        cls =cls.fit(X,y)
        y_out = cls.predict(X_test)
        accuracy = m.accuracy_score(y_test,y_out)
        precision = m.precision_score(y_test,y_out,average='macro')
        recall = m.recall_score(y_test,y_out,average='macro')
        #roc_auc = roc_auc_score(y_out,y_test)
        f1_score = m.f1_score(y_test,y_out,average='macro')
        log_entry = pd.DataFrame([[Name,accuracy,precision,recall,f1_score,roc_auc]], columns=log_cols)
        #metric_entry = pd.DataFrame([[precision,recall,f1_score,roc_auc]], columns=metrics_cols)
        log = log.append(log_entry)
        #metric = metric.append(metric_entry)
        
print(log)
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="g")  
plt.show()

#Scroll complete output to view all the accuracy scores and bar graph.


# As we can see highest accuracy for Logistic Regression.
# 
# ### Why logistic regression?
# 
# The models are equivalent in terms of the functions they can express, so with infinite training data and a function where the input variables don't interact with each other in any way they will both probably asymptotically approach the underlying joint probability distribution. This would definitely not be true if your features were not all binary.
# 
# Gradient boosted stumps adds extra machinery that sounds like it is irrelevant to your task. Logistic regression will efficiently compute a maximum likelihood estimate assuming that all the inputs are independent. I would go with logistic regression.
# 

# **For independent Execution of Logistic Regression, Code as follows:**

# In[ ]:


#Divide records in training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=2, stratify=data_y)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)


# In[ ]:


#Create an Logistic classifier and train it on 70% of the data set.
from sklearn import svm
from xgboost import XGBClassifier
clf = LogisticRegression()
clf


# In[ ]:


#Fiting into model
clf.fit(X_train, y_train)


# In[ ]:


#Prediction using test data
y_pred = clf.predict(X_test)


# In[ ]:


#classification accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:


#Predictions
predictions = clf.predict(X_test)


# In[ ]:


# Imports
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
print(confusion_matrix(y_test, predictions))

# New line
print('\n')

# Classification report
print(classification_report(y_test,predictions))


# ## Conclusion:
# ### Used the following:
# * Feature Selection - RFE-LogisticRegression
# * Fiting - SVM
# * With 0.8938 Accuracy (0.3% Test data)****

# In[ ]:




