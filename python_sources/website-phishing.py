#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df=pd.read_csv(r"../input/phishing-website-detector/phishing.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score


# In[ ]:


X= df.drop(columns='class')
X.head()


# In[ ]:


Y=df['class']
Y=pd.DataFrame(Y)
Y.head()


# In[ ]:


train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.3,random_state=2)


# In[ ]:


print(train_X.shape)
print(test_X.shape)
print(train_Y.shape)
print(test_Y.shape)


# **Lets apply Logistic Regression and check its accuracy**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


logreg=LogisticRegression()
model_1=logreg.fit(train_X,train_Y)


# In[ ]:


logreg_predict= model_1.predict(test_X)


# In[ ]:


accuracy_score(logreg_predict,test_Y)


# In[ ]:


print(classification_report(logreg_predict,test_Y))


# In[ ]:


def plot_confusion_matrix(test_Y, predict_y):
 C = confusion_matrix(test_Y, predict_y)
 A =(((C.T)/(C.sum(axis=1))).T)
 B =(C/C.sum(axis=0))
 plt.figure(figsize=(20,4))
 labels = [1,2]
 cmap=sns.light_palette("blue")
 plt.subplot(1, 3, 1)
 sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Confusion matrix")
 plt.subplot(1, 3, 2)
 sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Precision matrix")
 plt.subplot(1, 3, 3)
 sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Recall matrix")
 plt.show()


# In[ ]:


plot_confusion_matrix(test_Y, logreg_predict)


# **Lets apply K-Nearest Neighbors Classifier and check its accuracy**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=3)
model_2= knn.fit(train_X,train_Y)


# In[ ]:


knn_predict=model_2.predict(test_X)


# In[ ]:


accuracy_score(knn_predict,test_Y)


# In[ ]:


print(classification_report(test_Y,knn_predict))


# In[ ]:


plot_confusion_matrix(test_Y, knn_predict)


# **Lets apply Decision Tree Classifier and check its classifier **

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtree=DecisionTreeClassifier()
model_3=dtree.fit(train_X,train_Y)


# In[ ]:


dtree_predict=model_3.predict(test_X)


# In[ ]:


accuracy_score(dtree_predict,test_Y)


# In[ ]:


print(classification_report(dtree_predict,test_Y))


# In[ ]:


plot_confusion_matrix(test_Y, dtree_predict)


# **Lets apply Random Forest Classifier and check its accuracy**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier()
model_4=rfc.fit(train_X,train_Y)


# In[ ]:


rfc_predict=model_4.predict(test_X)


# In[ ]:


accuracy_score(rfc_predict,test_Y)


# In[ ]:


print(classification_report(rfc_predict,test_Y))


# In[ ]:


plot_confusion_matrix(test_Y, rfc_predict)


# **Lets Apply SVM and check its accuracy**

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc=SVC()
model_5=svc.fit(train_X,train_Y)


# In[ ]:


svm_predict=model_5.predict(test_X)


# In[ ]:


accuracy_score(svm_predict,test_Y)


# In[ ]:


print(classification_report(svm_predict,test_Y))


# In[ ]:


plot_confusion_matrix(test_Y, svm_predict)


# **Lets apply AdaBoost Classifier and check its accuracy **

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


adc=AdaBoostClassifier(n_estimators=5,learning_rate=1)
model_6=adc.fit(train_X,train_Y)


# In[ ]:


adc_predict=model_6.predict(test_X)


# In[ ]:


accuracy_score(adc_predict,test_Y)


# In[ ]:


print(classification_report(adc_predict,test_Y))


# In[ ]:


plot_confusion_matrix(test_Y, adc_predict)


# Lets apply XGBoost Classifier and check its accuracy

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb=XGBClassifier()
model_7=xgb.fit(train_X,train_Y)


# In[ ]:


xgb_predict=model_7.predict(test_X)


# In[ ]:


accuracy_score(xgb_predict,test_Y)


# In[ ]:


plot_confusion_matrix(test_Y, xgb_predict)


# In[ ]:


print('Logistic Regression Accuracy:',accuracy_score(logreg_predict,test_Y))
print('K-Nearest Neighbour Accuracy:',accuracy_score(knn_predict,test_Y))
print('Decision Tree Classifier Accuracy:',accuracy_score(dtree_predict,test_Y))
print('Random Forest Classifier Accuracy:',accuracy_score(rfc_predict,test_Y))
print('support Vector Machine Accuracy:',accuracy_score(svm_predict,test_Y))
print('Adaboost Classifier Accuracy:',accuracy_score(adc_predict,test_Y))
print('XGBoost Accuracy:',accuracy_score(xgb_predict,test_Y))


# From all the models we developed , Random forest accuracy has highest accuracy and followed by decision tree and XGBoost. Lowest accuracy model is SVM. 

# Now lets consider only two imporatant features Prefix_Suffix and URL_of_Anchor.

# In[ ]:


df.columns


# In[ ]:


X=df[['PrefixSuffix-','AnchorURL']]
X.head()


# In[ ]:


train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.3,random_state=2)


# In[ ]:


print(train_X.shape)
print(test_X.shape)
print(train_Y.shape)
print(test_Y.shape)


# Now lets apply logistic Regression for this new model which is having only two features

# In[ ]:


model_8=logreg.fit(train_X,train_Y)


# In[ ]:


logreg_predict=model_8.predict(test_X)


# In[ ]:


accuracy_score(test_Y,logreg_predict)


# In[ ]:


logreg.classes_


# In[ ]:


x = np.array(X)
x


# In[ ]:


X = X.to_numpy()
y = df['class']
y= y.to_numpy()


# Now lets plot the decision boundary 

# In[ ]:


from mlxtend.plotting import plot_decision_regions


# In[ ]:


plot_decision_regions(x, y, clf=model_1, legend=2)

# Adding axes annotations
plt.xlabel('features')
plt.ylabel('class')
plt.title('Logistic regression')
plt.show()


# Plot for only Logistic regression is made 
# Plots for remaining model will be made soon 

#  **TO BE CONTINUED**
