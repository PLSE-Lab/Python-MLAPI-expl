#!/usr/bin/env python
# coding: utf-8

# # Decision Trees 
# 
# The Don't overfit competition is a classification problem, so you must predict 1 or 0 based on the information of the input variables such as "0", "1", and so on. 
# 
# I suggest you to read the following web pages in order to understand decision trees:
# 
# https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/
# 
# http://scott.fortmann-roe.com/docs/BiasVariance.html
# 
# https://medium.com/@chiragsehra42/decision-trees-explained-easily-28f23241248
# 
# Next, we are going to fit our first decision tree.

# In[ ]:


# Loading the packages
import numpy as np
import pandas as pd 
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# Loading the training dataset
df_train = pd.read_csv("../input/train.csv")


# In[ ]:


y = df_train["target"]
# We exclude the target and id columns from the training dataset
df_train.pop("target");
df_train.pop("id")
X = df_train 
del df_train


# We are going to use the default parameters that provide the class DecisionTreeClassifier and because of that we do not need to use a validation dataset.

# In[ ]:


# Split data into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# The class DecisionTreeClassifier has many parameters that controls the complexity of the model, if you want to know more about it, please read the following link: 
# 
# [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
# 
# In order to prevent the problem of  [imbalanced dataset ](https://www.quora.com/What-is-an-imbalanced-dataset) we are going to create the decision tree as follows: 

# In[ ]:


model = DecisionTreeClassifier(class_weight='balanced')
# Good reference: 
# https://stackoverflow.com/questions/37522191/sklearn-how-to-balance-classification-using-decisiontreeclassifier


# Next, we fit the model with the training dataset. 

# In[ ]:


model.fit(X_train, y_train)


# We are going to visualize the decision tree: 

# In[ ]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# The decision tree above is too big and because of that this model is prone to overfitting. Next, we are going to calculate the AUC score in the training and testing dataset.

# In[ ]:


y_train_predict = model.predict_proba(X_train)
y_train_predict = y_train_predict[:,1]
y_test_predict = model.predict_proba(X_test, )
y_test_predict = y_test_predict[:,1]
auc_train = roc_auc_score(y_train, y_train_predict)
auc_test = roc_auc_score(y_test, y_test_predict)


# In[ ]:


print("The AUC in the training dataset is {}".format(auc_train))
print("The AUC in the test dataset is {}".format(auc_test))


# The auc in the training dataset is greater than the auc in the validation dataset, the model exhibits [overfitting](https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76). We are going to generate predictions over the test.csv file with this model. 

# In[ ]:


df_test = pd.read_csv("../input/test.csv")
df_test.pop("id");
X = df_test 
del df_test
y_pred = model.predict_proba(X)
y_pred = y_pred[:,1]


# In[ ]:


# submit prediction
smpsb_df = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


smpsb_df.head()


# In[ ]:


smpsb_df["target"] = y_pred
smpsb_df.to_csv("decision_tree.csv", index=None)


# In[ ]:




