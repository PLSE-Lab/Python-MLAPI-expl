#!/usr/bin/env python
# coding: utf-8

# > Notebook Updated
# 
# # Author: Kazi Amit Hasan
# 
# Department of Computer Science & Engineering, <br>
# Rajshahi University of Engineering & Technology (RUET) <br>
# Website: https://amithasanshuvo.github.io/ <br>
# Linkedin: https://www.linkedin.com/in/kazi-amit-hasan-514443140/ <br>
# Email: kaziamithasan89@gmail.com <br>
# 
# 
# ### Comment: 
# This notebook represents the chapter three(3) of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems book. I tried to implemented each examples while I was reading and practing this book. <br>
# <b>Please give your feedback how I can improve the notebook. Happy Learning!
# 
# 
# ##### Reference 
# Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems book (The best book I ever got on Data Sciene <3) <br>
# 
# Language: Python
# 
# # Plese upvote if you like it.

# ### Loading the dataset

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:



train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.iloc[:,1:].values.astype('float32')
y_test = test.iloc[:,0:].values.astype('int32')


# In[ ]:


print ("X_train: ", X_train)
print ("y_train: ", y_train)
print("X_test: ", X_test)
print ("y_test: ", y_test)


# In[ ]:


#X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
#X_train.shape


# In[ ]:


#X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
#X_test.shape


# ## Making it a Binary classification problem by making it "5-detector" problem
# ### It will distinguish between two classes. 5 or not! Sounds cool!

# In[ ]:


y_train_5 = (y_train ==5)
y_test_5 = (y_test ==5)

# This means true for all 5's and false for others


# In[ ]:


# Now we have to pick a classifier and train it.

# Stochastic Gradient Descent (it can handle big datasets every efficiently)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier( random_state=42)
sgd_clf.fit(X_train,y_train_5)


# In[ ]:


# Performance Measure with k fold cross validation with three folds
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# ### Accuracy isn't always the base of performance measure

# In[ ]:


# A classifier that only images that are in 'not 5' class.

from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# In[ ]:


never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# Good accuracy but only 10% images are 5's. So if that image is not a 5, then we are right about 90% time.


# In[ ]:


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv =3)


# Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix (y_train_5,y_train_pred)


# In[ ]:



y_train_perfect_predictions = y_train_5  
# pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)


# In[ ]:


from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)


# In[ ]:


recall_score(y_train_5, y_train_pred)


# In[ ]:



from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)

