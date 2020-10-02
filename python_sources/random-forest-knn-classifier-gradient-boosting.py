#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df= pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")


# In[ ]:


df.head(5)


# In[ ]:


df.info()


# In[ ]:


# Lets drop the unneccasary Varaibles- ID, Unnamed


# In[ ]:


df.drop(["id","Unnamed: 32"],axis=1,inplace=True)


# In[ ]:


# Lets we map the target varaible and have a look of those values


# In[ ]:


df['diagnosis']=df['diagnosis'].map({'B':0,'M':1})


# In[ ]:


df["diagnosis"].value_counts(normalize=True).plot(kind='bar')
plt.show()


# In[ ]:


# Before we proceed we have a visualisation about the feature and its influence


# In[ ]:


# Correlation plot(Heat Map)--only values the relation
# Lets use pair plot to get an overall idea about the data for comparision
#sns.pairplot(df,hue="diagnosis")

plt.figure(figsize=(10,10))
sns.boxplot(x= df.columns[-2:], y=df["diagnosis"])
plt.xticks(rotation=90)
# In[ ]:


# Make a X and y ready for the classification


# In[ ]:


X= df.drop("diagnosis",axis=1)
y= df["diagnosis"]

# Taken from kaggle as a reference, where we use standardization of the variable and have a look of the distribution
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)
# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Random Forest for prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
rfc= RandomForestClassifier(n_estimators=10)
rfc.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score


# In[ ]:


# Validating the train on the model
y_train_pred =rfc.predict(X_train)
y_train_prob =rfc.predict_proba(X_train)[:,1]

print("Accuracy Score of train", accuracy_score(y_train,y_train_pred))
print("AUC of the train ", roc_auc_score(y_train,y_train_prob))
print(" confusion matrix \n" , confusion_matrix(y_train,y_train_pred))


# In[ ]:


# Model on Test data 
y_test_pred =rfc.predict(X_test)
y_test_prob =rfc.predict_proba(X_test)[:,1]

print("Accuracy Score of test", accuracy_score(y_test,y_test_pred))
print("AUC od the test ", roc_auc_score(y_test,y_test_prob))
print(" confusion matrix \n" , confusion_matrix(y_test,y_test_pred))


# ## KNN Classifier
# 

# In[ ]:


# Since KNN is a distance based Algorithm- we need to do standardization of values


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3) 
clf.fit(X_train, y_train)  
print(clf.score(X_test, y_test))


# In[ ]:


# Validating the train on the model
y_train_pred =clf.predict(X_train)
y_train_prob =clf.predict_proba(X_train)[:,1]

print("Accuracy Score of train", accuracy_score(y_train,y_train_pred))
print("AUC of the train ", roc_auc_score(y_train,y_train_prob))
print(" confusion matrix \n" , confusion_matrix(y_train,y_train_pred))


# In[ ]:


# Model on Test data 
y_test_pred =clf.predict(X_test)
y_test_prob =clf.predict_proba(X_test)[:,1]

print("Accuracy Score of test", accuracy_score(y_test,y_test_pred))
print("AUC od the test ", roc_auc_score(y_test,y_test_prob))
print(" confusion matrix \n" , confusion_matrix(y_test,y_test_pred))


# ## GradientBoostingClassifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

lr_list= [0.05,0.075,0.1,0.25,0.5,0.75,1] #Learning Rate

for learning_rate in lr_list:
    clf= GradientBoostingClassifier(n_estimators=20, 
                                    learning_rate=learning_rate,max_features=2,max_depth=2,random_state=0)
    clf.fit(X_train,y_train)

    print("Learning Rate :", learning_rate)
    print(" Accuracy rate of training ", clf.score(X_train,y_train))
    print("Accuracy score of the test :", clf.score(X_test,y_test))


# In[ ]:


# Best Accuracy rate is been observed in the Learning Rate : 0.5
# Accuracy rate of training  1.0
# Accuracy score of the test : 0.9415204678362573


# In[ ]:


clf_new= GradientBoostingClassifier(n_estimators=10,max_features=2,learning_rate=5,random_state=1)
clf_new.fit(X_train,y_train)


# In[ ]:


# Validating the train on the model
y_train_pred =clf_new.predict(X_train)
y_train_prob =clf_new.predict_proba(X_train)[:,1]

print("Accuracy Score of train", accuracy_score(y_train,y_train_pred))
print("AUC of the train ", roc_auc_score(y_train,y_train_prob))
print(" confusion matrix \n" , confusion_matrix(y_train,y_train_pred))


# In[ ]:


# Model on Test data 
y_test_pred =clf_new.predict(X_test)
y_test_prob =clf_new.predict_proba(X_test)[:,1]

print("Accuracy Score of test", accuracy_score(y_test,y_test_pred))
print("AUC od the test ", roc_auc_score(y_test,y_test_prob))
print(" confusion matrix \n" , confusion_matrix(y_test,y_test_pred))


# ### Overall Inference:
#       - KNN Classifier using the concept of the distance algorithm performed the best in determining the AUC(Area Under the curve as it been more valid in the classifier compared to the accuracy in classifiers)
#       - Random Forest Classfier(with 10 estimators) provided with the comparable auc score of 97%
#       - Gradient Descent (Using the leraning rate to minimise the error rate) provided with the accuracy of 88%. 

# In[ ]:




