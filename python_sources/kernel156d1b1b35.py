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


dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
dataset.sample(5)


# In[ ]:


dataset.info()


# In[ ]:


dataset.isna().sum()


# In[ ]:


X = dataset.iloc[:, 2:].values
Y = dataset.iloc[:, 1].values


# In[ ]:


X


# In[ ]:


Y


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.99, random_state = 1,stratify =Y)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(21)


# In[ ]:


X_test[1:2]


# In[ ]:


knn.fit(X_train, y_train)


# In[ ]:


y_preds = knn.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix


# In[ ]:


accuracy_score(y_test, y_preds)


# In[ ]:


target_names = ['good rate', 'Not bad rate']


# In[ ]:


print(classification_report(y_test, y_preds, target_names=target_names))


# In[ ]:


test_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename.csv')


# In[ ]:





# ### KNN FINISHED

# In[ ]:


cm = confusion_matrix(y_test, y_preds)


# In[ ]:


cm


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:


test_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename1.csv')


# ### SVM

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svclassifier = SVC()


# In[ ]:


plt.scatter(X_train[:, 0], X_train[:, 4], c=y_train, cmap = 'spring')


# In[ ]:


svclassifier.fit(X_train, y_train)


# In[ ]:


y_pred = svclassifier.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


svclassifier1 = SVC(kernel='sigmoid')


# In[ ]:


svclassifier1.fit(X_train, y_train)


# In[ ]:


y_pred = svclassifier1.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


test_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename2.csv')


# ### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


gnb = GaussianNB()


# In[ ]:


gnb.fit(X_train, y_train)


# In[ ]:


y_pred = gnb.predict(X_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[ ]:


print("Accuracy:",metrics.classification_report(y_test, y_pred))


# In[ ]:


test_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename3.csv')


# In[ ]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split


# In[ ]:


bnb = BernoulliNB(binarize=0.0)


# In[ ]:


bnb.fit(X_train, y_train)


# In[ ]:


bnb.score(X_test, y_test)


# In[ ]:


y_pred = bnb.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[ ]:


print("Accuracy:",metrics.classification_report(y_test, y_pred))


# In[ ]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename4.csv')


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


mnb = MultinomialNB(alpha=0.01)


# In[ ]:


from sklearn.preprocessing import Normalizer


# In[ ]:


normalizer = Normalizer(norm='l2', copy=True)


# In[ ]:


X_train = Normalizer(copy=False).fit_transform(X_train)


# In[ ]:


X_train


# ### DTress and RF

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


from sklearn import metrics 


# In[ ]:


from sklearn.tree import export_graphviz 


# In[ ]:


clf = DecisionTreeClassifier()


# In[ ]:


clf = clf.fit(X_train,y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


pip install pydotplus


# In[ ]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  


# In[ ]:


clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)


# In[ ]:


clf = clf.fit(X_train,y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[ ]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf=RandomForestClassifier(n_estimators=100)


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


y_pred=clf.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename4.csv')


# ### XGboost

# In[ ]:


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[ ]:


import xgboost as xgb
import pandas as pd


# In[ ]:


xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed=123 )


# In[ ]:


xg_cl.fit(X_train, y_train)


# In[ ]:


y_pred = xg_cl.predict(X_test)


# In[ ]:


import numpy as np
accuracy = float(np.sum(y_pred==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


# In[ ]:


dataset_dmatrix = xgb.DMatrix(data = X,label = Y)
dataset_dmatrix


# In[ ]:


params = {"objective":"reg:logistic", "max_depth":3}
params


# In[ ]:


# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "rmse", as_pandas = True, seed = 123)


# In[ ]:


print(cv_results)


# In[ ]:


print(1-cv_results["test-rmse-mean"].tail(1))


# In[ ]:


# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)


# In[ ]:


print(cv_results)


# In[ ]:


print(cv_results["test-auc-mean"].tail(1))


# In[ ]:


result = pd.DataFrame({"target" : np.array(y_pred).T})
pd.concat([test_ds['ID_code'],result], axis=1).to_csv('filename5.csv')

