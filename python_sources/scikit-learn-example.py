#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import svm
import pandas as pd
import math


# ## Make a simple sklearn classifier
# First, read the data in using `pandas.read_csv()`.
# Note that the final column contains the `class_type` field that we are interested in.

# In[ ]:


data = pd.read_csv("../input/zoo.csv")
data.head(6)


# ## Preprocess the data
# Split the data up for training and evaluation.

# In[ ]:


def preprocess(data):
    X = data.iloc[:, 1:17]  # all rows, all the features and no labels
    y = data.iloc[:, 17]  # all rows, label only

    return X, y


# In[ ]:


# Shuffle and split the dataset
# We don't need to use this any more, thanks to scikit-learn!

data = data.sample(frac=1).reset_index(drop=True)
data_total_len = data[data.columns[0]].size

data_train_frac = 0.9
split_index = math.floor(data_total_len*data_train_frac)

train_data = data.iloc[:split_index]
eval_data = data.iloc[split_index:]


# Split the data using scikit-learn instead, using fewer lines!

# In[ ]:


from sklearn.model_selection import train_test_split

all_X, all_y = preprocess(data)
X_train, X_test, y_train, y_test = train_test_split(all_X, all_y)


# In[ ]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ## Train and Evaluate the model
# It's easy to swap in a different model of your choice.

# In[ ]:


clf = svm.SVC()
clf.fit(X_train, y_train)  


# In[ ]:


clf.score(X_test, y_test)


# ## Predict on some new data
# We can predict new values with a one line call.

# In[ ]:


clf.predict(X_test[15:25])


# In[ ]:


# Show what the correct answer is
y_test[10:15]


# In[ ]:





# In[ ]:




