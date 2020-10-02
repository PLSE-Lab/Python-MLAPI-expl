#!/usr/bin/env python
# coding: utf-8

# OBJECTIVE : The dataset contains 14 attributes which determine whether a patient has heart disease or not. We will create several classification models to accurately predict the heart disease.

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, roc_curve


# In[2]:


# reading data                        
data=pd.read_csv("../input/heart.csv")


# In[3]:


# displaying first 5 rows
data.head()


# Attributes Description :
# 
# age : age in years 
# 
# sex : (1 = male; 0 = female)
# 
# cp : chest pain type (values- 0,1,2,3)
# 
# trestbps : resting blood pressure (in mm Hg on admission to the hospital)
# 
# chol : serum cholestoral in mg/dl
# 
# fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 
# restecg : resting electrocardiographic results (values- 0,1,2)
# 
# thalach : maximum heart rate achieved
# 
# exang : exercise induced angina (1 = yes; 0 = no)
# 
# oldpeak : ST depression induced by exercise relative to rest
# 
# slope : the slope of the peak exercise ST segment
# 
# ca : number of major vessels (0-3) colored by flourosopy
# 
# tha : l3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# target : 1 or 0

# In[4]:


# (no. of rows, no. of columns)
data.shape 


# In[5]:


data.describe()


# In[6]:


# finding any null values in data
data.isnull().any()


# In[7]:


data.info()


# In[8]:


# Finding the number of patients with heart disease.
sns.countplot(x="target",data=data,palette="pastel")
plt.show()


# In[9]:


# Finding the ration of males and females in the data (1 = male; 0 = female)
sns.countplot(x="sex",data=data,palette="colorblind")
plt.show()


# In[10]:


# Finding correaltion between all the parameters in the dataset.
fig,ax = plt.subplots(figsize=(11,8))
sns.heatmap(data.corr(),annot=True,cmap="Blues" ,ax=ax)
plt.show()


# As wee see that cp, restecg, ca, thal and slope are categorical variables so we turn them into dummy variables.

# In[11]:


# creating dummy variables
a=pd.get_dummies(data["cp"],prefix="cp")
b=pd.get_dummies(data["restecg"],prefix="restecg")
c=pd.get_dummies(data["ca"],prefix="ca")
d=pd.get_dummies(data["thal"],prefix="thal")
e=pd.get_dummies(data["slope"],prefix="slope")


# In[12]:


# joining dummy variables in the dataset.
data=pd.concat([data,a,b,c,d],axis=1)
data.head()


# In[13]:


# no. of rows and columns after addition of dummy variables
data.shape


# In[14]:


# dropping of columns whose dummy variables have been created.
data=data.drop(columns=["cp","restecg","thal","ca","slope"])
data.head()


# In[15]:


# x= independent variables
x=data.drop("target",axis=1)
x.head()


# In[16]:


# y=dependent variable (target) 
y=data["target"]
y.head()


# In[17]:


# splitting data into train and test set.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# # Logistic Regression

# In[18]:


# making object classifier of class LogisticRegression 
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()


# In[19]:


# Fitting training data set into classifier
classifier.fit(x_train,y_train)


# In[20]:


# Predicting test results
y_pred=classifier.predict(x_test)


# In[21]:


# Making confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm


# Number of correct predictions : 23 + 31 = 54
# 
# Number of incorrect predictions : 3 + 4 = 7

# In[22]:


# Heatmap of confusion matrix
sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")
plt.show()


# Performance Measures :

# In[23]:


print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")
print("Precision = ",precision_score(y_test,y_pred)*100,"%")
print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")


# In[24]:


sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])
print ("Sensitivity =",sensitivity)
specificity= cm[0,0]/(cm[0,0] + cm[0,1])
print("Specificity =",specificity)


# ROC Curve and AUC Value : Receiver Operating Characteristics (ROC) Curve is a graph between True Positive Rate (y-axis) and False Positive Rate (x-axis). AUC Value is the area under ROC curve. 
# 
# AUC = 0 -- Bad Model
# 
# AUC = 1 -- Good Model

# In[25]:


# calculating AUC
auc=roc_auc_score(y_test,y_pred)
auc


# In[26]:


# calculating ROC curve
fpr,tpr,thresholds= roc_curve(y_test,y_pred)


# In[27]:


# plotting the roc curve for the model
plt.plot([0,1],[0,1],linestyle="--")
plt.plot(fpr,tpr,marker=".")
plt.xlabel("False Positive Rate")
plt.ylabel("TruePositive Rate")
plt.title("ROC Curve")
plt.show()


# # KNN Classifier

# In[28]:


# making object classifier of class KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)


# In[29]:


# Fitting training data set into classifier
classifier.fit(x_train,y_train)


# In[30]:


# Predicting test results
y_pred=classifier.predict(x_test)


# In[31]:


# Making confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm


# Number of correct predictions : 17 + 22 = 39
# 
# Number of incorrect predictions : 10 + 12 = 22 

# In[32]:


# Heatmap of confusion matrix
sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")
plt.show()


# Performance Measures :

# In[33]:


print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")
print("Precision = ",precision_score(y_test,y_pred)*100,"%")
print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")


# In[34]:


sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])
print ("Sensitivity =",sensitivity)
specificity= cm[0,0]/(cm[0,0] + cm[0,1])
print("Specificity =",specificity)


# ROC Curve and AUC Value :

# In[35]:


# calculating AUC
auc=roc_auc_score(y_test,y_pred)
auc


# In[36]:


# calculating ROC curve
fpr,tpr,thresholds= roc_curve(y_test,y_pred)


# In[37]:


# plotting the roc curve for the model
plt.plot([0,1],[0,1],linestyle="--")
plt.plot(fpr,tpr,marker=".")
plt.xlabel("False Positive Rate")
plt.ylabel("TruePositive Rate")
plt.title("ROC Curve")
plt.show()


# In[38]:


# Cross Validation : Calculating cross validation score
from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier,x_train,y_train,cv=10,scoring="accuracy")
score


# In[39]:


score.mean()


# # SVM Classifier

# In[40]:


# making object classifier of class SVC
from sklearn.svm import SVC
classifier= SVC(kernel="linear")


# In[41]:


# Fitting training data set into classifier
classifier.fit(x_train,y_train)


# In[42]:


# Predicting test results
y_pred=classifier.predict(x_test)


# In[43]:


# Making confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm


# Number of correct predictions : 22 + 29 = 51
# 
# Number of incorrect predictions : 5 + 5 = 10

# In[44]:


# Heatmap of confusion matrix
sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")
plt.show()


# Performance Measures :

# In[45]:


print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")
print("Precision = ",precision_score(y_test,y_pred)*100,"%")
print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")


# In[46]:


sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])
print ("Sensitivity =",sensitivity)
specificity= cm[0,0]/(cm[0,0] + cm[0,1])
print("Specificity =",specificity)


# ROC Curve and AUC Value :

# In[47]:


# calculating AUC
auc=roc_auc_score(y_test,y_pred)
auc


# In[48]:


# calculating ROC curve
fpr,tpr,thresholds= roc_curve(y_test,y_pred)


# In[49]:


# plotting the roc curve for the model
plt.plot([0,1],[0,1],linestyle="--")
plt.plot(fpr,tpr,marker=".")
plt.xlabel("False Positive Rate")
plt.ylabel("TruePositive Rate")
plt.title("ROC Curve")
plt.show()


# # Naive Bayes Classifier

# In[50]:


# making object classifier of class GaussianNB
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()


# In[51]:


# Fitting training data set into classifier
classifier.fit(x_train,y_train)


# In[52]:


# Predicting test results
y_pred=classifier.predict(x_test)


# In[53]:


# Making confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm


# Number of correct predictions : 20 + 31 = 51
# 
# Number of incorrect predictions : 7 + 3 = 10

# In[54]:


# Heatmap of confusion matrix
sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")
plt.show()


# Performance Measures :

# In[55]:


print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")
print("Precision = ",precision_score(y_test,y_pred)*100,"%")
print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")


# In[56]:


sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])
print ("Sensitivity =",sensitivity)
specificity= cm[0,0]/(cm[0,0] + cm[0,1])
print("Specificity =",specificity)


# ROC Curve and AUC Value :

# In[57]:


# calculating AUC
auc=roc_auc_score(y_test,y_pred)
auc


# In[58]:


# calculating ROC curve
fpr,tpr,thresholds= roc_curve(y_test,y_pred)


# In[59]:


# plotting the roc curve for the model
plt.plot([0,1],[0,1],linestyle="--")
plt.plot(fpr,tpr,marker=".")
plt.xlabel("False Positive Rate")
plt.ylabel("TruePositive Rate")
plt.title("ROC Curve")
plt.show()


# # Decision Tree Classifier

# In[60]:


# making object classifier of class DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="gini",random_state=0)


# In[61]:


# Fitting training data set into classifier
classifier.fit(x_train,y_train)


# In[62]:


# Predicting test results
y_pred=classifier.predict(x_test)


# In[63]:


# Making confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm


# Number of correct predictions : 21 + 26 = 47
# 
# Number of incorrect predictions : 8 + 6 = 14

# In[64]:


# Heatmap of confusion matrix
sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")
plt.show()


# Performance Measures :

# In[65]:


print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")
print("Precision = ",precision_score(y_test,y_pred)*100,"%")
print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")


# In[66]:


sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])
print ("Sensitivity =",sensitivity)
specificity= cm[0,0]/(cm[0,0] + cm[0,1])
print("Specificity =",specificity)


# ROC Curve and AUC Value :

# In[67]:


# calculating AUC
auc=roc_auc_score(y_test,y_pred)
auc


# In[68]:


# calculating ROC curve
fpr,tpr,thresholds= roc_curve(y_test,y_pred)


# In[69]:


# plotting the roc curve for the model
plt.plot([0,1],[0,1],linestyle="--")
plt.plot(fpr,tpr,marker=".")
plt.xlabel("False Positive Rate")
plt.ylabel("TruePositive Rate")
plt.title("ROC Curve")
plt.show()


# # Random Forest Classifier

# In[70]:


# making object classifier of class RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion="gini")


# In[71]:


# Fitting training data set into classifier
classifier.fit(x_train,y_train)


# In[72]:


# Predicting test results
y_pred=classifier.predict(x_test)


# In[73]:


# Making confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm


# Number of correct predictions : 23 + 28 = 52
# 
# Number of incorrect predictions : 4 + 6 = 10

# In[74]:


# Heatmap of confusion matrix
sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")
plt.show()


# Performance Measures :

# In[75]:


print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")
print("Precision = ",precision_score(y_test,y_pred)*100,"%")
print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")


# In[76]:


sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])
print ("Sensitivity =",sensitivity)
specificity= cm[0,0]/(cm[0,0] + cm[0,1])
print("Specificity =",specificity)


# ROC Curve and AUC Value :

# In[77]:


# calculating AUC
auc=roc_auc_score(y_test,y_pred)
auc


# In[78]:


# calculating roc curve
fpr,tpr,thresholds= roc_curve(y_test,y_pred)


# In[79]:


# plotting the roc curve for the model
plt.plot([0,1],[0,1],linestyle="--")
plt.plot(fpr,tpr,marker=".")
plt.xlabel("False Positive Rate")
plt.ylabel("TruePositive Rate")
plt.title("ROC Curve")
plt.show()


# # Comparison Between Classifiers

# In[80]:


methods = ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Decision Tree", "Random Forest"]
accuracy = [88.5,63.93,83.6,83.6,77.04,85.24]
plt.subplots(figsize=(11,8))
sns.barplot(x=methods,y=accuracy)
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.title("Comparison between Classifiers")
plt.show()


# From above graph we can say that Logistic Regression classification model is best suited for our dataset with accuraacy of 88.5% and KNN classification model is least suite with accuracy of 63.93%.
