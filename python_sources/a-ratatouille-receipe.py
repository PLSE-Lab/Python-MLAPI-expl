#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
Image("../input/ratatouille/ratatouille.jpg")


# In[ ]:


import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (15,10)


# ## Import Data 

# In[ ]:


data = pd.read_json("../input/whats-cooking-kernels-only/train.json")
test = pd.read_json("../input/whats-cooking-kernels-only/test.json")
data.head()


# ### Removing NULL/NAN values

# In[ ]:


data.isnull().values.any()
data.dropna(axis=0, how='any',inplace = True)
data.isnull().values.any()
data.isnull().sum()


# In[ ]:


data.shape


# ### List of cuisines

# In[ ]:


data.cuisine.unique()


# In[ ]:


pd.value_counts(data['cuisine']).plot.bar()


# ### Input to the model

# In[ ]:


#Convert into proper format
data.ingredients =data.ingredients.str.join(' ')
test.ingredients =test.ingredients.str.join(' ')


# In[ ]:


# convert text to unique integers with HashingVectorizer
vect = HashingVectorizer()
features = vect.fit_transform(data.ingredients)
testfeatures = vect.transform(test.ingredients)


# In[ ]:


#Split the dataset into train and test sets
labels = data.cuisine


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


# In[ ]:


from sklearn.linear_model import LogisticRegression

start = time.time()

log_reg = LogisticRegression(C=12)
log_reg.fit(X_train,y_train)

print("Accuracy: ",log_reg.score(X_test, y_test))
print("Time: " , time.time() - start )


# In[ ]:


from sklearn.svm import LinearSVC

start = time.time()

linear_svm = LinearSVC(random_state=0, max_iter = 1500)
linear_svm.fit(X_train, y_train)

print("Accuracy: ",linear_svm.score(X_test, y_test))
print("Time: " , time.time() - start )


# In[ ]:


from sklearn.svm import SVC

start = time.time()

rbf_svm = SVC(kernel='rbf', gamma=0.8, C=12)
rbf_svm.fit(X_train, y_train)

print("Accuracy: ",rbf_svm.score(X_test, y_test))
print("Time: " , time.time() - start )


# In[ ]:


prediction = rbf_svm.predict(testfeatures)
sub = pd.DataFrame({'id':test.id,'cuisine':prediction})
output = sub[['id','cuisine']]
output.to_csv("sample_submission.csv",index = False)


# In[ ]:




