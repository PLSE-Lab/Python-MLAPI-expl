#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/US_Heart_Patients.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# We can see that there are few null values. Hence we can fill it by using forward fill as the null values are small.

# In[ ]:


df = df.fillna(method='ffill')


# In[ ]:


df.info()


# In[ ]:


from sklearn.model_selection import train_test_split

X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10 )
dt.fit(X, y)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred_train = dt.predict(X_train)
y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))
print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))


# ## Hyperparameter Tuning
# ### Grid Search

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[ ]:


dt = DecisionTreeClassifier()

params = {'max_depth' : [2,3,4,5,6,7,8],
        'min_samples_split': [2,3,4,5,6,7,8,9,10],
        'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10]}

gsearch = GridSearchCV(dt, param_grid=params, cv=3)

gsearch.fit(X,y)

gsearch.best_params_


# For Grid search we considered max_depth between 2 to 8, min_samples_split between 2 to 10 and min_samples_leaf between 1 to 10. The best parameter for the decision was found to be 2, 5 and 2.
# 
# Now we will build decision tree using these hyperparameters.
# Also we will Select the best model using the AUC curve.
# The area covered by the curve is the area between the orange line (ROC) and the axis. This area covered is AUC. The bigger the area covered, the better the machine learning models is at distinguishing the given classes. Ideal value for AUC is 1.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.3, random_state=1)


# In[ ]:


dt = DecisionTreeClassifier(**gsearch.best_params_)

dt.fit(X_train, y_train)

y_pred_train = dt.predict(X_train)
y_prob_train = dt.predict_proba(X_train)[:,1]

y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:,1]

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))
print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))


# In[ ]:


print('AUC of Decision Tree-Train: ', roc_auc_score(y_train, y_prob_train))
print('AUC of Decision Tree-Test: ', roc_auc_score(y_test, y_prob))


# ### Hyperparameter Tuning - Random Search

# In[ ]:


from scipy.stats import randint as sp_randint

dt = DecisionTreeClassifier(random_state=1)

params = {'max_depth' : sp_randint(2,10),
        'min_samples_split': sp_randint(2,50),
        'min_samples_leaf': sp_randint(1,20),
         'criterion':['gini', 'entropy']}

rand_search = RandomizedSearchCV(dt, param_distributions=params, cv=3, 
                                 random_state=1)

rand_search.fit(X, y)
print(rand_search.best_params_)


# In[ ]:


dt = DecisionTreeClassifier(**rand_search.best_params_)

dt.fit(X_train, y_train)

y_pred_train = dt.predict(X_train)
y_prob_train = dt.predict_proba(X_train)[:,1]

y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:,1]

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))
print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))


# In[ ]:


print('AUC of Decision Tree-Train: ', roc_auc_score(y_train, y_prob_train))
print('AUC of Decision Tree-Test: ', roc_auc_score(y_test, y_prob))


# ## Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, random_state=1)

rfc.fit(X_train, y_train)

y_pred_train = rfc.predict(X_train)
y_prob_train = rfc.predict_proba(X_train)[:,1]

y_pred = rfc.predict(X_test)
y_prob = rfc.predict_proba(X_test)[:,1]

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

print('Accuracy of Random Forest-Train: ', accuracy_score(y_pred_train, y_train))
print('Accuracy of Random Forest-Test: ', accuracy_score(y_pred, y_test))


# In[ ]:


print('AUC of Random Forest-Train: ', roc_auc_score(y_train, y_prob_train))
print('AUC of Random Forest-Test: ', roc_auc_score(y_test, y_prob))


# We noticed that there is large difference between the model performance Train and Test. Hence, we will tune the Hyperparamter and check the results.

# ### Hyperparameter Tuning of Random Forest

# In[ ]:


from scipy.stats import randint as sp_randint

rfc = RandomForestClassifier(random_state=1)

params = {'n_estimators': sp_randint(5,25),
    'criterion': ['gini', 'entropy'],
    'max_depth': sp_randint(2, 10),
    'min_samples_split': sp_randint(2,20),
    'min_samples_leaf': sp_randint(1, 20),
    'max_features': sp_randint(2,15)}

rand_search_rfc = RandomizedSearchCV(rfc, param_distributions=params,
                                 cv=3, random_state=1)

rand_search_rfc.fit(X, y)
print(rand_search_rfc.best_params_)


# In[ ]:


rfc = RandomForestClassifier(**rand_search_rfc.best_params_)

rfc.fit(X_train, y_train)

y_pred_train = rfc.predict(X_train)
y_prob_train = rfc.predict_proba(X_train)[:,1]

y_pred = rfc.predict(X_test)
y_prob = rfc.predict_proba(X_test)[:,1]

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

print('Accuracy of Random Forest-Train: ', accuracy_score(y_pred_train, y_train))
print('Accuracy of Random Forest-Test: ', accuracy_score(y_pred, y_test))


# In[ ]:


print('AUC of Random Forest-Train: ', roc_auc_score(y_train, y_prob_train))
print('AUC of Random Forest-Test: ', roc_auc_score(y_test, y_prob))


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.plot(fpr, fpr, 'r-')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


# ## k-NN Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

Xs = ss.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_trains = ss.fit_transform(X_train)
X_tests = ss.transform(X_test)


# In[ ]:


knn.fit(X_trains, y_train)

y_pred_train = knn.predict(X_trains)
y_prob_train = knn.predict_proba(X_trains)[:,1]

y_pred = knn.predict(X_tests)
y_prob = knn.predict_proba(X_tests)[:,1]

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

print('Accuracy of kNN-Train: ', accuracy_score(y_pred_train, y_train))
print('Accuracy of kNN-Test: ', accuracy_score(y_pred, y_test))


# In[ ]:


print('AUC of kNN-Train: ', roc_auc_score(y_train, y_prob_train))
print('AUC of kNN-Test: ', roc_auc_score(y_test, y_prob))


# In[ ]:


knn = KNeighborsClassifier()

params = {'n_neighbors': sp_randint(1,20),
        'p': sp_randint(1,5)}

rand_search_knn = RandomizedSearchCV(knn, param_distributions=params,
                                 cv=3, random_state=1)
rand_search_knn.fit(Xs, y)
print(rand_search_knn.best_params_)


# In[ ]:


knn = KNeighborsClassifier(**rand_search_knn.best_params_)

knn.fit(X_trains, y_train)

y_pred_train = knn.predict(X_trains)
y_prob_train = knn.predict_proba(X_trains)[:,1]

y_pred = knn.predict(X_tests)
y_prob = knn.predict_proba(X_tests)[:,1]

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

print('Accuracy of Random Forest-Train: ', accuracy_score(y_pred_train, y_train))
print('Accuracy of Random Forest-Test: ', accuracy_score(y_pred, y_test))


# In[ ]:


print('AUC of kNN-Train: ', roc_auc_score(y_train, y_prob_train))
print('AUC of kNN-Test: ', roc_auc_score(y_test, y_prob))


# ## Stacking Algorithms

# In[ ]:


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


lr = LogisticRegression(solver='liblinear')
rfc = RandomForestClassifier(**rand_search_rfc.best_params_)
knn = KNeighborsClassifier(**rand_search_knn.best_params_)


# In[ ]:


clf = VotingClassifier(estimators=[('lr',lr), ('rfc',rfc), ('knn',knn)], 
                       voting='soft')
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_prob_train = clf.predict_proba(X_train)[:,1]

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]


# In[ ]:


print('Accuracy of Stacked Algos-Train: ', accuracy_score(y_pred_train, y_train))
print('Accuracy of Stacked Algos-Test: ', accuracy_score(y_pred, y_test))


# In[ ]:


print('AUC of Stacked Algos: ', roc_auc_score(y_train, y_prob_train))
print('AUC of Stacked Algos: ', roc_auc_score(y_test, y_prob))


# In[ ]:


clf = VotingClassifier(estimators=[('lr',lr), ('rfc',rfc), ('knn',knn)], 
                       voting='soft', weights=[2,3,1])
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_prob_train = clf.predict_proba(X_train)[:,1]

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]


# In[ ]:


print('Accuracy of Stacked Algos-Train: ', accuracy_score(y_pred_train, y_train))
print('Accuracy of Stacked Algos-Test: ', accuracy_score(y_pred, y_test))


# In[ ]:


print('AUC of Stacked Algos: ', roc_auc_score(y_train, y_prob_train))
print('AUC of Stacked Algos: ', roc_auc_score(y_test, y_prob))


# In[ ]:




