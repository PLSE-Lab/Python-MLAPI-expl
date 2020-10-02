#!/usr/bin/env python
# coding: utf-8

# #### Introduction
# 
# 1. Explore dataset: Heart Disease dataset from UCI
# 
# 
# 2. Model building in Scikit-learn
# 
#     - Decision Tree
#     - Logistic Regression
#     - Support Vector Machine
#     - K-Neighbors
#     - Naive Bayes
#     - Random Forest
#     - Neural Network
#     
# 
# 3. Model Evaluation
# 
#     - Confusion Matrix
#     - Accuracy score
#     - ROC AUC score
#     
# 4. Visualization
# 

# In[ ]:


# Import Python modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # ploting statistical graphs

import warnings
warnings.filterwarnings('ignore')


# **1. Explore dataset: Heart Disease dataset from UCI**

# In[ ]:


# read heart.csv data into a Pandas DataFrame
data = pd.read_csv('../input/heart.csv')

# Returns the dimensionality of the DataFrame (rows, columns).
print(data.shape)
#print(len(data))


# In[ ]:


# Explore data
data.head()


# In[ ]:


#print last 3 rows
data.tail(3)


# ###### Dataset features:
# 
# 1. age in years
# 2. sex(1 = male; 0 = female)
# 3. cp chest pain type
# 4. trestbpsresting blood pressure (in mm Hg on admission to the hospital)
# 5. cholserum cholestoral in mg/dl
# 6. fbs(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7. restecgresting electrocardiographic results
# 8. thalachmaximum heart rate achieved
# 9. exangexercise induced angina (1 = yes; 0 = no)
# 10. oldpeakST depression induced by exercise relative to rest
# 11. slopethe slope of the peak exercise ST segment
# 12. canumber of major vessels (0-3) colored by flourosopy
# 13. thal3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. target 1 or 0

# ###### Data summary:
# 
# Total number of obsevations (rows): 303 # note that row numbering starts from 0
# 
# Total number of features (columns): 14
# 
# Response (target column): (0,1)
# 
#     0: no disease
#     1: presence of disease

# In[ ]:


# Generate descriptive statistics of DataFrame columns (I am using display to print table).
display(data.describe())


# In[ ]:


#Print a concise summary of a DataFrame
data.info()


# In[ ]:


# Check Null values
print(data[pd.isnull(data).any(axis=1)])


# In[ ]:


# Another way to check for null values
data.isnull().sum()


# In[ ]:


# Print column names
data.columns


# In[ ]:


# print Pairwise relationships between the features
sns.pairplot(data)


# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(data.corr(), linewidths=.01, annot = True, cmap='Greens')
plt.show()


# **2. Model building in Scikit-learn**

# In[ ]:


# Create column list (used by graphviz for ptintin decision tree)
feature_cols = ['age','sex','chest_pain','rest_bp','chol','fasting_bloodsugar','rest_ecg','max_heartrate','excercise_angina','oldpeak','slope','n_major_vasel','thal']


# In[ ]:


# Split dataset in features and target variable
X = data.drop(['target'],axis=1) #features
y = data.target # response/target
print ("Feature: ", X.shape)  # metrix
print ("Response: ", y.shape) # series


# In[ ]:


import sklearn
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Split the dataset into two sets, so that the model can be trained and tested on different data
from sklearn.model_selection import train_test_split
# Split dataset into training set (80%)) and test set (20%)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)
print ("Taining data: ", X_train.shape)
print ("Test data: ", X_test.shape)


# In[ ]:


# Model: 1. Create Decision Tree classifer object
from sklearn.tree import DecisionTreeClassifier
cf = DecisionTreeClassifier()
# Train the model using the training sets
cf = cf.fit(X_train,y_train)
# make Predictions on the test dataset
cf_predicted = cf.predict(X_test)


# In[ ]:


# Classification accuracy, how often is the classifier correct (percentage of correct predictions)?
# Determine the accuracy of the model (compare actual value:y_test with predicted value:cf_predicted)
print ("DECISION TREE:")
print ("Accuracy Score:")
print (metrics.accuracy_score(y_test, cf_predicted))
# Compute confusion matrix to evaluate the accuracy of a classification
print ("Confusion metrix:")
print (metrics.confusion_matrix(y_test, cf_predicted))


# ###### Confusion matrix explained
# 	       Predicted 0		Predicted 1
#     
#     Actual 0	  TN(24)             FP(6)
# 
#     Actual 1	  FN(5)		      TP(26)
# 
# ### Terminology
# True negative  (TN): correctly predicted the patient has NO heart disease (actual value: No disease).
# 
# True positive  (TP): correctly predicted patient has heart disease(actual value: disease).
# 
# False negative (FN): incorrectly predicted patient has NO heart disease (actual value: disease).
# 
# False positive (FP): incorrectly predicted patient has heart disease (actual value: No disease).

# In[ ]:


# Null accuracy for binary classification problems
#Examine class distribution of testing set
y_test.value_counts()


# In[ ]:


# calculate percentage of NO heart disease (target: 0)
y_test.mean()


# In[ ]:


# calculate percentage of heart disease (target: 1)
1-y_test.mean()


# In[ ]:


## generate classification tree for DecisionTreeClassifier
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(cf, out_file=None, feature_names=feature_cols,  class_names=['0','1'],  filled=True, rounded=True,  special_characters=True)
graph = graphviz.Source(dot_data)
#graph.render("class") ## print pdf file
graph


# In[ ]:


# Model: 2. Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
#Train the model using the training sets
lr.fit(X_train,y_train)
#Predict the response for test dataset
lr_predicted=lr.predict(X_test)
print ("LOGISTIC REGRESSION:")
print ("Accuracy Score:")
print (metrics.accuracy_score(y_test, lr_predicted))
# Compute confusion matrix to evaluate the accuracy of a classification
print ("Confusion metrix:")
print (metrics.confusion_matrix(y_test, lr_predicted))


# In[ ]:


# Model: 3. Support Vector Machine
from sklearn.svm import SVC
sm = SVC(gamma='scale')
#Train the model using the training sets
sm.fit(X_train,y_train)
#Predict the response for test dataset
#sm.score(X_test, y_test)
sm_predicted=sm.predict(X_test)
print ("SUPPORT VECTOR MACHINE")
print ("Accuracy Score:")
print (metrics.accuracy_score(y_test, sm_predicted))
# Compute confusion matrix to evaluate the accuracy of a classification
print ("Confusion metrix:")
print (metrics.confusion_matrix(y_test, sm_predicted))


# In[ ]:


# Model: 4. K-Neighrest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
#Train the model using the training sets
knn.fit(X_train,y_train)
#Predict the response for test dataset
knn_predicted = knn.predict(X_test)
print ("K-NEIGHEST NEIGHBORS")
print ("Accuracy Score:")
print (metrics.accuracy_score(y_test, knn_predicted))
# Compute confusion matrix to evaluate the accuracy of a classification
print ("Confusion metrix:")
print (metrics.confusion_matrix(y_test, knn_predicted))


# In[ ]:


# Model: 5. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(X_train,y_train)
gb_predicted = gb.predict(X_test)
print ("NAIVE BAYES")
print ("Accuracy Score:")
print (metrics.accuracy_score(y_test, gb_predicted))
# Compute confusion matrix to evaluate the accuracy of a classification
print ("Confusion metrix:")
print (metrics.confusion_matrix(y_test, gb_predicted))


# In[ ]:


# Model: 6. Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf_predicted = rf.predict(X_test)
print ("RANDOM FOREST")
print ("Accuracy Score:")
print (metrics.accuracy_score(y_test, rf_predicted))
# Compute confusion matrix to evaluate the accuracy of a classification
print ("Confusion metrix:")
print (metrics.confusion_matrix(y_test, rf_predicted))


# In[ ]:


# Model: 7. Neural Network
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier()
nn.fit(X_train,y_train)
nn_predicted = nn.predict(X_test)
print ("NEURAL NETWORK")
print ("Accuracy Score:")
print (metrics.accuracy_score(y_test, nn_predicted))
# Compute confusion matrix to evaluate the accuracy of a classification
print ("Confusion metrix:")
print (metrics.confusion_matrix(y_test, nn_predicted))


# **3. Model Evaluation**

# In[ ]:


# We have learn how to use different ML classifier. Next we will use Python script to comapre different classifiers.
# Import classifiers (commented out as we have already imported in previous section)
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier

# Create a dictionary
classifier_collection = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "Neural Network": MLPClassifier()
}

# Create and evaluate models
# Evaluation criteria: accuracy_score (help you to choose between models and qualify model performance)
accuracy_score_dict = {}
confusion_matrix_dict = {}
roc_auc_dict = {}
count=0
for classifier_model, classifier in (classifier_collection.items()):
    #print(classifier_name)
    count +=1
    classifier.fit(X_train,y_train)
    predicted = classifier.predict(X_test)
    accuracy_score_dict[classifier_model] = {'accuracy_score' :  metrics.accuracy_score(y_test, predicted)}
    confusion_matrix_dict[classifier_model] = {'matrix' : metrics.confusion_matrix(y_test, predicted)}
    roc_auc_dict[classifier_model] = {'roc-auc' : metrics.roc_auc_score(y_test, predicted)}


# In[ ]:


accuracy_score_dict


# In[ ]:


confusion_matrix_dict


# In[ ]:


roc_auc_dict


# **4. Visualization**

# In[ ]:


# Model evaluation : Confusion matrix
plt.figure(figsize=(20,10))
plt.suptitle("Confusion matrix for different models",fontsize=24)
# initialize n with zero
n = 0
for classifier, class_score in confusion_matrix_dict.items():
    for item, score in class_score.items():
        # excluded confusion matrix for SVM (with lowest accuracy score))
        if classifier != "SVM":
            pass
            n +=1
            plt.subplot(2, 3, n) 
            plt.title(classifier, fontsize=8)
            sns.heatmap(score,annot=True,cbar=False,cmap="Greens",fmt="d")


# In[ ]:


# Model evaluation : Accuracy score
ax = pd.DataFrame(accuracy_score_dict).plot(kind='bar', figsize=(16,8), title="Accuracy score compared between the models")
x_axis = ax.axes.get_xaxis()
x_axis.set_visible(False)
ax.set_ylabel("Accuracy score")
ax.set_ylim(0,1)


# In[ ]:


# Model evaluation : 
ax = pd.DataFrame(roc_auc_dict).plot(kind='bar', figsize=(16,8), title="ROC AUC score compared between the models")
x_axis = ax.axes.get_xaxis()
x_axis.set_visible(False)
ax.set_ylabel("ROC AUC score")
ax.set_ylim(0,1)


# ###### Conclusion: Our best performing model is Logistic Regression  with ~ 90% accuray.

# Further improvements: In this study all the algorithms were tested with default parameters. These parameters influence the outcome of learning process. The classifier's parameters can be tuned to improve the accuracy of the model.

# I am new to machine learning and this is my 1st kaggle kernel. Any comments and suggestions are welcome!
