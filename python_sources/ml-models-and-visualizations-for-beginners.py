#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#reading the file
df=pd.read_csv('../input/mushrooms.csv')
df.head()


# In[ ]:


df.info() #this method will show if there are any null character in our data 


# In[ ]:


#to check how many unique values are there in a column 
print('unique_elements: ',df['class'].unique())
#to check how many are there of each kind in a column
df['class'].value_counts()


# In[ ]:


df.describe()


# **Applying Countplot**

# In[ ]:


sns.countplot(x='gill-color',hue='class',data=df)


# In[ ]:


sns.countplot(x='cap-surface',hue='class',data=df)


# In[ ]:


sns.countplot(x='odor',hue='class',data=df)


# LabelEncoder
# 
# since our data are all  in character/strings if we directly feed the raw data in our model it will end up throwing errors so before feeding the data into our models we will have to pass it through Label Encoder which will assign all the characters of each column a particular number.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in df.columns:
    df[i]=le.fit_transform(df[i])
df.head()


# In[ ]:


X=df.iloc[:,1:23]
y=df['class']
X.head()
y.head()


# **Boxplot to see the distribution of data**

# In[ ]:


sns.set_style('whitegrid')
sns.boxplot( x=df['class'],y=df['cap-color'])


# In[ ]:


sns.boxplot( x='class',y='stalk-color-above-ring',data=df,palette="Set3")


# In[ ]:


sns.boxplot( x='class',y='stalk-color-below-ring',data=df,palette="Set2")
sns.stripplot(x='class',y='stalk-color-below-ring',data=df,jitter=True,color=".6")


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)
X_train.head()
y_train.head()


# ** Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
pred1=logreg.predict(X_test)
print('lr.score:',round(logreg.score(X_train,y_train)*100,3))
print('metrics.accuracy_score:',round(metrics.accuracy_score(y_test,pred1)*100,3))


# **Logistic Regression with Cross_validation**
# 

# In[ ]:


from sklearn.model_selection import cross_val_score
cv_result=cross_val_score(logreg,X,y,cv=10)
print("cv_score:",cv_result)
print('average_score: ',np.sum(cv_result)/10)


# Confusion matrix using decision_tree_classifier
# 
# Now lets discuss accuracy. Is it enough for measurement of model selection. For example, there is a data that includes 95% normal and 5% abnormal samples and our model uses accuracy for measurement metric. Then our model predict 100% normal for all samples and accuracy is 95% but it classify all abnormal samples wrong. Therefore we need to use confusion matrix as a model measurement matris in imbalance data. While using confusion matrix lets use Random forest classifier to diversify classification methods.
# 
# tp = true positive, fp = false positive, fn = false negative, tn = true negative tp = Prediction is positive(normal) and actual is positive(normal).
# 
# fp = Prediction is positive(normal) and actual is negative(abnormal).
# 
# fn = Prediction is negative(abnormal) and actual is positive(normal).
# 
# tn = Prediction is negative(abnormal) and actual is negative(abnormal)
# 
# precision = tp / (tp+fp)
# 
# recall = tp / (tp+fn)
# 
# f1 = 2 precision recall / ( precision + recall)
# 

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,pred1)
print('confusion_matrix: \n',cm)
print('classification_report:\n',classification_report(y_test,pred1))


#Heatmap visualization
sns.heatmap(cm,annot=True,fmt='d')
plt.show()


# In[ ]:


#plotting ROC
from sklearn.metrics import roc_curve,auc
y_pred_logreg=logreg.predict_proba(X_test)[:,1]
fpr,tpr,threshold=roc_curve(y_test,y_pred_logreg)
roc_auc=auc(fpr,tpr)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC')
plt.show()
print(roc_auc)


# **Decision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dr=DecisionTreeClassifier()
dr.fit(X_train,y_train)
pred=dr.predict(X_test)
#pred = np.where(prob > 0.5, 1, 0)[:,1]
dr.score(X_test,y_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,pred)
print('confusion_matrix: \n',cm)
print('classification_report:\n',classification_report(y_test,pred))


#Heatmap visualization
sns.heatmap(cm,annot=True,fmt='d')
plt.show()


# In[ ]:


#plotting ROC

from sklearn.metrics import roc_curve,auc
y_pred=dr.predict_proba(X_test)[:,1]
print(y_pred)
fpr,tpr,threshold=roc_curve(y_test,y_pred)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC')
plt.show()


# **CORRELATION  MATRIX**
# 
# What is a Correlation matrix??
# 
# Correation matrix gives the correlation between the features.Correlation coeff can be negative or it can be positiveor it can also be zero. So which feature is how much correlated with  each other can  be undersatand ussing this matrix.
# 
# The feature of a dataset is directly proportional to another feature is the correlation coeff is positve, the more +ve the coeff becomes the more is the correlation.
# The features are inversely proportional if the correlation coeff is -ve ,the more -ve it becomes the more it get correlated.
# and if the coeff is zero the features don't have any correlation between them.

# In[ ]:


import seaborn as sns
corr=df.corr()
f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(corr,xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()


# In[ ]:


cols = corr.nlargest(15, 'class')['class'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
f,ax=plt.subplots(figsize=(10,9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# **Support Vector Machine(SVM)**

# In[ ]:


from sklearn import svm
#as it is clear from the matrix that the feature veil-type hast no importace so lets drop the feature
x=X.drop(['veil-type'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)
clf=svm.SVC()
clf.fit(x_train,y_train)
clf.predict(x_test)
clf.score(x_test,y_test)

