#!/usr/bin/env python
# coding: utf-8

# # Table of contents
# 1. [Librarys and Dataset](#libdatasets)
# 2. [Data Analysis](#dataanalysis)
# 3. [Pre-Processing Step](#preprocessing)
# 4. [Decision Tree Model](#dtmodel)
# 5. [Model Evaluation](#evaluation)

# <div id='libdatasets'/>
# # Librarys and Dataset

# In[ ]:


import numpy as np, matplotlib.pyplot as plt, pandas as pd

from collections import Counter
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


dataset = pd.read_csv('../input/kag_risk_factors_cervical_cancer.csv', header=0, na_values='?')


# <div id='dataanalysis'/>
# # Data Analysis
# Let's have a look in some rows of the dataset...

# In[ ]:


dataset.head(10)


# How we can see, in the first 10 elements he can note that the data have some NaN values. Therefore, let's have a look at the amount of NaN for every variable:

# In[ ]:


dataset.isnull().sum()


# The dataset have a large number of NaN values, being that for the variabes *STDs: Time since first diagnosis* and *STDs: Time since last diagnosis* the values arrive to be missing in almost all rows.
# 
# **The NaN values will be treated in the pre-processing step .**
# 
# Second, we'll have a look in the amount of each class:
# 

# In[ ]:


dataset['Biopsy'].value_counts()


# As can be seen, the database has an imbalance between the quantity of each class. 
# 
# **Techniques will be applied to try to deal with imbalance in the pre-processing step.**

# <div id='preprocessing'/>
# # Pre-processing Step
# First, the *STDs: Time since first diagnosis* and *STDs: Time since last diagnosis* variables will be removed due to the high amount of incomplete values.
# 
# The remainder of the incomplete values will be replaced by the median of each column.

# In[ ]:


dataset.drop(dataset.columns[[26, 27]], axis=1, inplace=True)


# In[ ]:


values = dataset.values
X = values[:, 0:33]
y = values[:, 33]


# In[ ]:


imputer = Imputer(strategy='median')
X = imputer.fit_transform(X)


# Next, lets apply the InstanceHardnessTreshold algorithm to reduce the number of instances of the majority class, thus, the decision tree model will be less affected by the imbalance of the classes.

# In[ ]:


iht = InstanceHardnessThreshold(random_state=12)
X, y = iht.fit_sample(X, y)
print('Amount of each class after under-sampling: {0}'.format(Counter(y)))


# Finally, the base is separated into training and testing for creation and evaluation of the model.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 12)


# <div id='dtmodel'/>
# # Decision Tree Model

# The parameters were chosen according to the optimization of the parameters using the *GridSearch* algorithm (not shown here). 

# In[ ]:


classifier = DecisionTreeClassifier(criterion='gini', max_leaf_nodes= None,
                                    min_samples_leaf=14, min_samples_split=2 ,
                                    random_state = 12)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# <div id='evaluation'/>
# # Model Evaluation

# The classification report and the confusion matrix using the created model are shown below.

# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Predicted No', 'Predicted Yes'],
    index=['Actual No', 'Actual Yes']
)


# Finally, the result of the macro-f1 average is shown using the model created and using the 10-fold cross validation.

# In[ ]:


scores = cross_val_score(classifier, X, y, scoring='f1_macro', cv=10)
print('Macro-F1 average: {0}'.format(scores.mean()))

