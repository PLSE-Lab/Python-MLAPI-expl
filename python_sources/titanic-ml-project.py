#!/usr/bin/env python
# coding: utf-8

# In[84]:


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


# In[85]:


#Data Analysis
import numpy as np
import pandas as pd

#Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Machine Learning
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
sns.set_style('whitegrid')


# In[86]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[87]:


train.head()


# In[88]:


train.info()


# In[89]:


train.describe()


# In[90]:


train.drop(['PassengerId','Name','Ticket','Fare'],axis =1 , inplace = True)


# In[91]:


train.head()


# In[92]:


corr = train.corr()
sns.heatmap(corr)


# In[93]:


sns.heatmap(train.isnull())


# In[94]:


train.drop('Cabin', axis=1, inplace = True)


# In[95]:


train.head()


# In[96]:


train['Age'].mean()


# In[97]:


sns.boxplot(x ='Pclass', y ='Age', data =train)


# In[98]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[99]:


train['Age']= train[['Age','Pclass']].apply(impute_age,axis = 1)


# In[100]:


sns.heatmap(train.isnull())


# In[101]:


train.head()


# In[102]:


sex = pd.get_dummies(train['Sex'], drop_first = True)


# In[103]:


sex.head()


# In[104]:


embark = pd.get_dummies(train['Embarked'], drop_first = True)


# In[105]:


embark.head()


# In[106]:


train = pd.concat([train,sex,embark], axis = 1)


# In[107]:


train.head()


# In[108]:


train.drop(['Sex', 'Embarked'],axis=1,inplace = True)


# In[109]:


train.head()


# In[110]:


from sklearn.model_selection import train_test_split


# In[111]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], 
                                                    test_size = 0.30, random_state = 101) 


# Logistic Regression

# In[112]:


from sklearn.linear_model import LogisticRegression


# In[113]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[114]:


predictions = logmodel.predict(X_test)


# In[115]:


acc_Logistic_Regression = round(logmodel.score(X_train, y_train) * 100, 2)
acc_Logistic_Regression


# In[116]:


from sklearn.metrics import confusion_matrix, classification_report


# In[117]:


print(confusion_matrix(y_test,predictions))


# In[118]:


print(classification_report(y_test,predictions))


# In[119]:


models = []
acc = []
precision = []
recall = []
f1 = []


# In[120]:


models.append('Logistic Regression')
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score ,make_scorer)


# In[121]:


print('Confusion Matrix for LR: \n',confusion_matrix(y_test, logmodel.predict(X_test)))
print('Accuracy for LR: \n',accuracy_score(y_test, logmodel.predict(X_test)))
acc.append(accuracy_score(y_test, logmodel.predict(X_test)))
print('Precision for LR: \n',precision_score(y_test, logmodel.predict(X_test)))
precision.append(precision_score(y_test, logmodel.predict(X_test)))
print('Recall for LR: \n',recall_score(y_test, logmodel.predict(X_test)))
recall.append(recall_score(y_test, logmodel.predict(X_test)))
print('f1_score for LR: \n',f1_score(y_test, logmodel.predict(X_test)))
f1.append(f1_score(y_test, logmodel.predict(X_test)))


# Naive Bayes

# In[122]:


from sklearn.naive_bayes import GaussianNB


# In[123]:


classifier = GaussianNB()
classifier.fit(X_train,y_train)


# In[124]:


predictions = classifier.predict(X_test)


# In[125]:


acc_Naive_Bayes = round(classifier.score(X_train, y_train) * 100, 2)
acc_Naive_Bayes


# In[126]:


from sklearn.metrics import confusion_matrix, classification_report


# In[127]:


print(confusion_matrix(y_test,predictions))


# In[128]:


print(classification_report(y_test,predictions))


# In[176]:


models.append('Naive Bayes')


# In[177]:


print('Confusion Matrix for RF: \n',confusion_matrix(y_test, classifier.predict(X_test)))
print('Accuracy for RF: \n',accuracy_score(y_test, classifier.predict(X_test)))
acc.append(accuracy_score(y_test, classifier.predict(X_test)))
print('Precision for RF: \n',precision_score(y_test, classifier.predict(X_test)))
precision.append(precision_score(y_test, classifier.predict(X_test)))
print('Recall for RF: \n',recall_score(y_test, classifier.predict(X_test)))
recall.append(recall_score(y_test, classifier.predict(X_test)))
print('f1_score for RF: \n',f1_score(y_test, classifier.predict(X_test)))
f1.append(f1_score(y_test, classifier.predict(X_test)))


# K Nearest Neighbour

# In[129]:


from sklearn.neighbors import KNeighborsClassifier


# In[130]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


# In[131]:


predictions = knn.predict(X_test)


# In[132]:


acc_KNN = round(knn.score(X_train, y_train) * 100, 2)
acc_KNN


# In[133]:


from sklearn.metrics import confusion_matrix, classification_report


# In[134]:


print(confusion_matrix(y_test,predictions))


# In[135]:


print(classification_report(y_test,predictions))


# In[174]:


models.append('K Nearest Neighbour')


# In[175]:


print('Confusion Matrix for RF: \n',confusion_matrix(y_test, knn.predict(X_test)))
print('Accuracy for RF: \n',accuracy_score(y_test, knn.predict(X_test)))
acc.append(accuracy_score(y_test, knn.predict(X_test)))
print('Precision for RF: \n',precision_score(y_test, knn.predict(X_test)))
precision.append(precision_score(y_test, knn.predict(X_test)))
print('Recall for RF: \n',recall_score(y_test, knn.predict(X_test)))
recall.append(recall_score(y_test, knn.predict(X_test)))
print('f1_score for RF: \n',f1_score(y_test, knn.predict(X_test)))
f1.append(f1_score(y_test, knn.predict(X_test)))


# Support Vector Machines

# In[136]:


from sklearn.svm import SVC


# In[137]:


model = SVC()
model.fit(X_train,y_train)


# In[138]:


predictions = model.predict(X_test)


# In[139]:


acc_SVM = round(model.score(X_train, y_train) * 100, 2)
acc_SVM


# In[140]:


from sklearn.metrics import confusion_matrix,classification_report


# In[141]:


print(confusion_matrix(y_test,predictions))


# In[142]:


print(classification_report(y_test,predictions))


# In[172]:


models.append('Support Vector Machine')


# In[173]:


print('Confusion Matrix for RF: \n',confusion_matrix(y_test, model.predict(X_test)))
print('Accuracy for RF: \n',accuracy_score(y_test, model.predict(X_test)))
acc.append(accuracy_score(y_test, model.predict(X_test)))
print('Precision for RF: \n',precision_score(y_test, model.predict(X_test)))
precision.append(precision_score(y_test, model.predict(X_test)))
print('Recall for RF: \n',recall_score(y_test, model.predict(X_test)))
recall.append(recall_score(y_test, model.predict(X_test)))
print('f1_score for RF: \n',f1_score(y_test, model.predict(X_test)))
f1.append(f1_score(y_test, model.predict(X_test)))


# Decision Tree Classifier

# In[143]:


from sklearn.tree import DecisionTreeClassifier


# In[144]:


clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)


# In[145]:


y_pred = clf.predict(X_test)


# In[146]:


acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
acc_decision_tree


# In[147]:


print(confusion_matrix(y_test, y_pred))


# In[148]:


print(classification_report(y_test,y_pred))


# In[149]:


#Importing the accuracy metric from sklearn.metrics library

from sklearn.metrics import accuracy_score
print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))


# In[150]:


clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=50)
clf.fit(X_train, y_train)
print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
print('Accuracy Score on the test data: ', accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))


# In[166]:


models.append('Decision Tree')


# In[167]:


print('Confusion Matrix for RF: \n',confusion_matrix(y_test, clf.predict(X_test)))
print('Accuracy for RF: \n',accuracy_score(y_test, clf.predict(X_test)))
acc.append(accuracy_score(y_test, clf.predict(X_test)))
print('Precision for RF: \n',precision_score(y_test, clf.predict(X_test)))
precision.append(precision_score(y_test, clf.predict(X_test)))
print('Recall for RF: \n',recall_score(y_test, clf.predict(X_test)))
recall.append(recall_score(y_test, clf.predict(X_test)))
print('f1_score for RF: \n',f1_score(y_test, clf.predict(X_test)))
f1.append(f1_score(y_test, clf.predict(X_test)))


# Random Forest Classifier

# In[151]:


from sklearn.ensemble import RandomForestClassifier


# In[152]:


rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)


# In[153]:


y_pred = rfc.predict(X_test)


# In[154]:


acc_random_forest = round(rfc.score(X_train, y_train) * 100, 2)
acc_random_forest


# In[155]:


print(confusion_matrix(y_test,y_pred))


# In[156]:


print(classification_report(y_test,y_pred))


# In[157]:


models.append('Random Forest')


# In[158]:


print('Confusion Matrix for RF: \n',confusion_matrix(y_test, rfc.predict(X_test)))
print('Accuracy for RF: \n',accuracy_score(y_test, rfc.predict(X_test)))
acc.append(accuracy_score(y_test, rfc.predict(X_test)))
print('Precision for RF: \n',precision_score(y_test, rfc.predict(X_test)))
precision.append(precision_score(y_test, rfc.predict(X_test)))
print('Recall for RF: \n',recall_score(y_test, rfc.predict(X_test)))
recall.append(recall_score(y_test, rfc.predict(X_test)))
print('f1_score for RF: \n',f1_score(y_test, rfc.predict(X_test)))
f1.append(f1_score(y_test, rfc.predict(X_test)))


# In[159]:


model_dict = {'Models': models,
             'Accuracies': acc,
             'Precision': precision,
             'Recall': recall,
             'f1-score': f1}


# In[178]:


model_df = pd.DataFrame(model_dict)


# In[179]:


model_df = model_df.sort_values(['Accuracies', 'f1-score', 'Recall', 'Precision'],
                               ascending=False)


# In[180]:


model_df


# In[ ]:




