#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score as score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


# In[ ]:


train_data =pd.read_csv("../input/sf-crime/train.csv", parse_dates =['Dates'])
test_data =pd.read_csv("../input/sf-crime/test.csv", parse_dates =['Dates'])

print("The size of the train data is:", train_data.shape)
print("The size of the test data is:", test_data.shape)


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data.dtypes.value_counts()


# In[ ]:


test_data.dtypes.value_counts()


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


train_data.columns


# In[ ]:


train_data.Category.value_counts()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_data['Category'] = le.fit_transform(train_data.Category)
train_data.Category.head()


# In[ ]:


train_data.PdDistrict.value_counts()


# In[ ]:


feature_cols =['DayOfWeek', 'PdDistrict']
train_data = pd.get_dummies(train_data, columns=feature_cols)
test_data = pd.get_dummies(test_data, columns=feature_cols)

train_data


# In[ ]:


test_data


# In[ ]:


for x in [train_data, test_data]:
    x['years'] = x['Dates'].dt.year
    x['months'] = x['Dates'].dt.month
    x['days'] = x['Dates'].dt.day
    x['hours'] = x['Dates'].dt.hour
    x['minutes'] = x['Dates'].dt.minute
    x['seconds'] = x['Dates'].dt.second


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data = train_data.drop(['Dates', 'Address','Resolution'], axis = 1)


# In[ ]:


train_data = train_data.drop(['Descript'], axis = 1)
train_data.head()


# In[ ]:


test_data = test_data.drop(['Dates', 'Address'], axis = 1)
test_data.head()


# In[ ]:


feature_cols = [x for x in train_data if x!='Category']
X = train_data[feature_cols]
y = train_data['Category']
X_train, x_test,y_train, y_test = train_test_split(X, y)


# In[ ]:


DTC = DecisionTreeClassifier(criterion = 'gini', max_features = 25, max_depth = 13)
DTC = DTC.fit(X_train,y_train)
y_pred_DTC = DTC.predict(x_test)
y_pred_test_DTC = DTC.predict(X_train)
print("score is {:.3f}".format (score(y_test, y_pred_DTC, average = 'micro')*100))
print("Accuracy for the test data is {:.3f} ".format (accuracy_score(y_test, y_pred_DTC)*100))
print("Accuracy for the train data is {:.3f} ".format (accuracy_score(y_train, y_pred_test_DTC)*100))
print("acc voting classifier:", accuracy_score(y_test, y_pred_DTC))


# In[ ]:


from sklearn.metrics import f1_score
from sklearn.kernel_approximation import Nystroem
nystroemSVC = Nystroem(kernel = 'rbf')
sgd = SGDClassifier()

X_train_svc = nystroemSVC.fit_transform(X_train)
X_test_svc = nystroemSVC.transform(x_test)

linSVC = sgd.fit(X_train_svc, y_train)
y_pred_svc = linSVC.predict(X_test_svc)
y_pred_test_svc = linSVC.predict(X_train_svc)

print("score is {:.3f}".format (score(y_test, y_pred_svc, average = 'micro')*100))
print("Accuracy for the test data is {:.3f}".format (accuracy_score(y_test, y_pred_svc)*100))
print("Accuracy for the train data is {:.3f}".format (accuracy_score(y_train, y_pred_test_svc)*100))


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(dual=False, tol=0.1)
lr.fit(X_train, y_train)
y_pred_test_lr = lr.predict(X_train)
y_pred_lr = lr.predict(x_test)


print("score is {:.3f}".format (score(y_test, y_pred_lr, average = 'micro')*100))
print("Accuracy for the test data is {:.3f}".format (accuracy_score(y_test, y_pred_lr)*100))
print("Accuracy for the train data is {:.3f} ".format (accuracy_score(y_train, y_pred_test_lr)*100))
print("acc voting classifier:", accuracy_score(y_test, y_pred_lr))


# In[ ]:


from sklearn.linear_model import SGDClassifier
svm = SGDClassifier( max_iter=13 ,loss="hinge")
svm.fit(X_train, y_train)
y_pred_test_svm = svm.predict(X_train)
y_pred_svm = svm.predict(x_test)

print("score is {:.3f}".format (score(y_test, y_pred_svm, average = 'micro')*100))
print("Accuracy for the test data is {:.3f}".format (accuracy_score(y_test, y_pred_svm)*100))
print("Accuracy for the train data is {:.3f} ".format (accuracy_score(y_train, y_pred_test_svm)*100))
print("acc voting classifier:", accuracy_score(y_test, y_pred_svm))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=39)
knn.fit(X_train, y_train)
y_pred_test_knn = knn.predict(X_train)
y_pred_knn = knn.predict(x_test)
print("acc voting classifier:", accuracy_score(y_test,y_pred_knn))


# In[ ]:


from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators=[('lr', lr) ,('knn', knn),('DTC', DTC),('svm', svm)], n_jobs=-1,voting='hard')
y_pred_vc=vc.fit(X_train, y_train)
y_pred_test_vc = vc.predict(x_test)
print("acc voting classifier:", accuracy_score(y_test, y_pred_test_vc))


# In[ ]:


print("acc voting classifier:", accuracy_score(y_test, y_pred_test_vc)*100)


# In[ ]:


from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(y_test, y_pred_test_vc)
cmat


# In[ ]:


X_test =test_data.drop(['Id'], axis = 1)

my_prediction = DTC.predict(X_test)

SFCC_submission_final = pd.DataFrame({'Id': test_data.Id, 'Category': my_prediction})
print(SFCC_submission_final.shape)
SFCC_submission_final.to_csv("../input/sf-crime/sampleSubmission.csv", index = False)

