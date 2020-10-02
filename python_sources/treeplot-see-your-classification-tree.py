#!/usr/bin/env python
# coding: utf-8

# ## Vizualize your classification tree using [treeplot](https://github.com/erdogant/treeplot)
# Here I have taken a [very simple classifier script](https://www.kaggle.com/carlmcbrideellis/random-forest-classifier-minimalist-script) which uses the [random forest classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) from scikit-learn, and used the [treeplot](https://github.com/erdogant/treeplot) package, written by Erdogan Taskesen, to produce a wonderful visualization:

# In[ ]:


get_ipython().system('pip install treeplot')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import treeplot package:
import treeplot
# and the random forest classifier
from sklearn.ensemble import RandomForestClassifier

# read in the data
train_data = pd.read_csv('../input/titanic/train.csv')

# select some features
features = ["Pclass", "Sex", "SibSp", "Parch"]

X_train       = pd.get_dummies(train_data[features])
y_train       = train_data["Survived"]

# perform the classification and the fit
classifier = RandomForestClassifier(criterion='gini', n_estimators=100, 
        min_samples_split=2, min_samples_leaf=10, max_features='auto')
classifier.fit(X_train, y_train)


# In[ ]:


# now make the plot
ax = treeplot.plot(classifier)


# The above image is a `.png` image and you can right-click on it to see it in more detail.<br>
# The blue boxes represent a classification of 1 (survived) and the orange boxes are a zero.
# ## Links:
# * [treeplot on PyPI](https://pypi.org/project/treeplot/)
# * [treeplot on GitHub](https://github.com/erdogant/treeplot)
