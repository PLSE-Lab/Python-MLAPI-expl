#!/usr/bin/env python
# coding: utf-8

# # SGDClassifier is a linear classifier that uses SGD for training (that is, looking for the minima of the loss using SGD)
# >SGD loss = 'log' implements Logistic regression
# 
# >SGD loss = 'hinge' implements Linear SVM.
# 
# # RandomForest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting

#  read train and test csv 

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# # Separate Features and Labels

# X = Features
# Y = Labels

# In[ ]:


X, y = train.drop(labels = ["label"],axis = 1).as_matrix(), train["label"]
X.shape


# In[ ]:


y.shape


# 1. feature X[20] contains '8' (image_pixel data) pixels 784 = 28*28 
# 2. y[20] contain 8 value

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X[20]
some_digit_show = plt.imshow(X[20].reshape(28,28), cmap=mpl.cm.binary)
y[20]


# In[ ]:


y = y.astype(np.uint8)


# Spliting Train and Test sets

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# # Binary Classifier's
# We were just training our model to predict 8 

# In[ ]:


y_train_8 = (y_train == 8)
y_test_8 = (y_test == 8)


# In[ ]:


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_8)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train_8)


# In[ ]:


sgd_clf.predict([some_digit])


# In[ ]:


forest_clf.predict([some_digit])


# # some_digit X[20] == 8, True

# # Lets Try to Predict Other
# example -> X[0] contain 1
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X[33]
some_digit_show = plt.imshow(X[33].reshape(28,28), cmap=mpl.cm.binary)
y[33]
sgd_clf.predict([X[33]])


# In[ ]:


forest_clf.predict([X[33]])


# # False its not 8 Our Model Working Fine 
# 

# # Lets Do CrossValidation

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score


# In[ ]:


cv_score_sgd = cross_val_score(sgd_clf, X_train, y_train_8, cv=3, scoring='accuracy')
cv_score_sgd = np.mean(cv_score_sgd)
cv_score_sgd


# In[ ]:


SGDpred = cross_val_predict(sgd_clf, X_train, y_train_8, cv=3)
confusion_matrix(y_train_8, SGDpred)


# In[ ]:


precision_score(y_train_8, SGDpred)


# In[ ]:


recall_score(y_train_8, SGDpred)


# In[ ]:


f1_score(y_train_8, SGDpred)


# In[ ]:


cv_score_forest = cross_val_score(forest_clf, X_train, y_train_8, cv=3, scoring='accuracy')
cv_score_forest = np.mean(cv_score_forest)
cv_score_forest


# In[ ]:


ForestPred = cross_val_predict(forest_clf, X_train, y_train_8, cv=3)
confusion_matrix(y_train_8, ForestPred)


# In[ ]:


precision_score(y_train_8, ForestPred)


# In[ ]:


recall_score(y_train_8, ForestPred)


# In[ ]:


f1_score(y_train_8, ForestPred)


# In[ ]:


y_scores_sgd = cross_val_predict(sgd_clf, X_train, y_train_8, cv=3,
                             method="decision_function")


# In[ ]:


y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_8, cv=3,
                                    method="predict_proba")
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


roc_auc_score(y_train_8, y_scores_sgd)


# In[ ]:


roc_auc_score(y_train_8, y_scores_forest)


# # Forest Win (;

# test

# In[ ]:


predictions_forest = forest_clf.predict(X_test)
predictions_forest


# In[ ]:


predictions_sgd = sgd_clf.predict(X_test)
predictions_sgd


# In[ ]:


predictions_sgd = sgd_clf.predict(X_test).astype(int)

submissions_sgd=pd.DataFrame({"ImageId": list(range(1,len(predictions_sgd)+1)),
                         "Label": predictions_sgd})
submissions_sgd.to_csv("binary_sgd_clf.csv", index=False, header=True)


# In[ ]:


predictions_forest = forest_clf.predict(X_test).astype(int)

submissions_forest=pd.DataFrame({"ImageId": list(range(1,len(predictions_forest)+1)),
                         "Label": predictions_forest})
submissions_forest.to_csv("binary_forest_clf.csv", index=False, header=True)


# In[ ]:


final = pd.concat([submissions_sgd, submissions_forest], axis =1)
final.to_csv('final.csv', index=False, header=True)


# In[ ]:




