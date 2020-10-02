#!/usr/bin/env python
# coding: utf-8

# # My work on car evaluation

# In[ ]:


# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[ ]:


# data doesn't have headers, so let's create headers
_headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'car']


# In[ ]:


# read in cars dataset
df = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter06/Dataset/car.data', names=_headers, index_col=None)


# In[ ]:


df.head()


# In[ ]:


training, evaluation = train_test_split(df, test_size=0.3, random_state=0)


# In[ ]:


validation, test = train_test_split(evaluation, test_size=0.5, random_state=0)


# # Creating classification model

# In[ ]:


# encode categorical variables
_df = pd.get_dummies(df, columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
_df.head()


# In[ ]:


# target column is 'car'

features = _df.drop(['car'], axis=1).values
labels = _df[['car']].values

# split 80% for training and 20% into an evaluation set
X_train, X_eval, y_train, y_eval = train_test_split(features, labels, test_size=0.3, random_state=0)

# further split the evaluation set into validation and test sets of 10% each
X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=0.5, random_state=0)


# In[ ]:


# train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[ ]:


# make predictions for the validation dataset
y_pred = model.predict(X_val)


# # Creating confusion matrix

# In[ ]:


#import libraries
from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_val, y_pred)


# # Creating precision score

# In[ ]:


#import libraries
from sklearn.metrics import precision_score


# In[ ]:


precision_score(y_val, y_pred, average='macro')


# # Creating recall score

# In[ ]:


# import libraries
from sklearn.metrics import recall_score


# In[ ]:


recall_score = recall_score(y_val, y_pred, average='macro')
print(recall_score)


# # creating f1 score

# In[ ]:


#import libraries
from sklearn.metrics import f1_score


# In[ ]:


f1_score = f1_score(y_val, y_pred, average='macro')
print(f1_score)


# # Creating accuracy score

# In[ ]:


# import necessary library
from sklearn.metrics import accuracy_score


# In[ ]:


_accuracy = accuracy_score(y_val, y_pred)
print(_accuracy)


# # Computing Log Loss

# In[ ]:


# import libraries
from sklearn.metrics import log_loss


# In[ ]:


_loss = log_loss(y_val, model.predict_proba(X_val))
print(_loss)


# In[ ]:





# In[ ]:




