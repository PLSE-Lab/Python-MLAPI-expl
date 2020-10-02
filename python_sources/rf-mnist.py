#!/usr/bin/env python
# coding: utf-8
Load the MNIST data (introduced in Chapter 3), and 
split it into a training set, a validation set, and a test set 
(e.g., use 50,000 instances for training, 10,000 for validation, and 10,000 for testing). 
Then train various classifiers, such as a Random Forest classifier, an Extra-Trees classifier, and an SVM. 
Next, try to combine them into an ensemble that outperforms them all on the validation set, using a soft or hard voting classifier. 
Once you have found one, try it on the test set. How much better does it perform compared to the individual classifiers?MNIST dataset, a set of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau
# In[ ]:


import pandas as pd
import numpy as np

import gzip, pickle


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.linear_model import SGDClassifier 

from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# ### 0. Functions

# In[ ]:


def plotConfusionMatrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# ### 1. Loading MNIST data

# [MNIST Dataset](http://deeplearning.net/data/mnist/mnist.pkl.gz)

# In[ ]:


get_ipython().system('ls ../input/mnistpklgz')
with gzip.open("../input/mnistpklgz/mnist.pkl.gz","rb") as ff :
    u = pickle._Unpickler( ff )
    u.encoding = "latin1"
    train, val, test = u.load()


# In[ ]:


print( train[0].shape, train[1].shape )


# In[ ]:


print( val[0].shape, val[1].shape )


# In[ ]:


print( test[0].shape, test[1].shape )


# In[ ]:


some_digit = train[0][0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="lanczos")
plt.axis("off")
plt.show()


# In[ ]:


train[1][0]


# In[ ]:


X_train = train[0]
X_val = val[0]
X_test = test[0]


# In[ ]:


y_train = train[1].astype(np.uint8)
y_val = val[1].astype(np.uint8)
y_test = test[1].astype(np.uint8)


# ### 2. Train a Ensemble of Decision Trees

# #### 2.1. Single Tree Feature Importances

# In[ ]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
iris_pd = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
clf = clf.fit(iris_pd, iris.target)


# In[ ]:


print(dict(zip(iris_pd.columns, clf.feature_importances_)))


# #### 2.2. Example

# In[ ]:


bagClf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=100,
    max_samples=100, bootstrap=True
)
bagClf.fit(X_train, y_train)
y_pred = bagClf.predict(X_test)


# In[ ]:


set(y_test)


# In[ ]:


unique_labels(y_test, y_pred)


# In[ ]:


classesName = np.array(range(10))


# In[ ]:


featureImportances = np.mean([
    tree.feature_importances_ for tree in bagClf.estimators_
], axis=0)


# In[ ]:


baggingPixelImportances = featureImportances.reshape(28, 28)
fig, ax = plt.subplots()
im = ax.imshow(baggingPixelImportances, interpolation="lanczos", cmap=mpl.cm.afmhot)
ax.figure.colorbar(im, ax=ax)
plt.show()


# In[ ]:


## Confusion matrix
plotConfusionMatrix(y_test, y_pred, classesName)


# ### 3. Train a Random Forest with the same Setup

# In[ ]:


rndClf = RandomForestClassifier(
    n_estimators=100, max_leaf_nodes=16, n_jobs=-1
)
rndClf.fit(X_train, y_train)
y_pred_rf = rndClf.predict(X_test)


# In[ ]:


rndClf.feature_importances_


# In[ ]:


pixelImportance = rndClf.feature_importances_.reshape(28, 28)


# In[ ]:


fig, ax = plt.subplots()
im = ax.imshow(pixelImportance, interpolation="lanczos", cmap=mpl.cm.afmhot)
ax.figure.colorbar(im, ax=ax)
plt.show()


# In[ ]:


## Confusion matrix
plotConfusionMatrix(y_pred_rf, y_pred, classesName)

