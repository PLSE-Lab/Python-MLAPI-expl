#!/usr/bin/env python
# coding: utf-8

# # Load the iris dataset and the ensemble library for sklearn

# In[ ]:


from sklearn.datasets import load_iris
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# # Load Iris dataset and split the train and test data

# In[ ]:


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # Create a simple random forest with basic attributes and fit the data

# In[ ]:


clf = ensemble.RandomForestClassifier(n_estimators=50,criterion = "gini", verbose = 1, random_state=0, max_depth=2)
clf = clf.fit(X_train, y_train)


# # Predict the test data

# In[ ]:


y_test_predicted = clf.predict(X_test)


# # Print confusion matrix and accuracy

# In[ ]:


print(confusion_matrix(y_test, y_test_predicted))
accuracy_score(y_test, y_test_predicted)


# In[ ]:




