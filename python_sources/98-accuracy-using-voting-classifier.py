#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# Really simple introduction into Voting Classifiers. The final model achieves a 98% accuracy. 

# In[ ]:


import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import unique_labels
import os
print(os.listdir("../input"))

# Model selection and optimization was done outside of this kernel!


# In[ ]:


data = pd.read_csv("../input/data.csv")


# In[ ]:


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
data.drop(['Unnamed: 32',"id"], axis=1, inplace=True)
Y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(x_data)
data.head()


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X, Y, random_state=1)


# In[ ]:


# Instansiate models
tree = DecisionTreeClassifier(random_state=1)
svm = SVC(probability=True, kernel='linear', C=1, gamma = 0.001)
log = LogisticRegression(random_state=0, solver='liblinear', 
                         multi_class='ovr')


# In[ ]:


eclf = VotingClassifier(estimators=[
     ('tree', tree), ('svm', svm), ('log', log)
], voting='soft')


# In[ ]:


eclf.fit(train_X, train_y)
pred_y = eclf.predict(test_X)
accuracy_score(test_y, pred_y)


# Results of the voting classifier model are **97.9%**. 
# 
# For a more complete view of model performance, I have included both the regular and normalized matrices below. 

# In[ ]:


def plot_confusion_matrix(y_true, y_pred, classes,
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
    classes = ['Malignant', 'Benign']
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

np.set_printoptions(precision=2)


# In[ ]:


# Plot non-normalized confusion matrix
plot_confusion_matrix(test_y, pred_y, classes=['Malignant', 'Benign'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(test_y, pred_y, classes=['Malignant', 'Benign'], normalize=True,
                      title='Normalized confusion matrix')


# In[ ]:





# In[ ]:





# In[ ]:




