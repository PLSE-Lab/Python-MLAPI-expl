#!/usr/bin/env python
# coding: utf-8

# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)
# 
# Also code from: https://github.com/amueller/scipy-2018-sklearn

# 

# Fork this notebook!

# ## Exciting times! Data sources, tools, compute resources readily available to get started!
# 
# ## Free Data Sources:
# * Too many to list!
# ### Caution: Data sources vs. Machine learning data - structured/unstrctured data vs labeled data
# ### Caution: Check out everyone's licensing before using it for your enterprise needs.
# 
# 
# ## Compute resources to use:
# Free (or free trial):
# * https://data.world/community/open-community/
# * https://colab.research.google.com/notebooks/welcome.ipynb
# * https://aws.amazon.com/sagemaker/pricing/
# * https://cloud.google.com/products/ai/
# * https://datastudio.google.com/navigation/reporting
# * https://azure.microsoft.com/en-us/pricing/details/virtual-machines/windows/
# * https://www.dominodatalab.com/domino-for-good/
# * https://www.dominodatalab.com/domino-for-good/for-students/
# * https://www.kaggle.com/sigma23/women-in-analytics-2019-workshop/edit
# * https://medium.com/@jamsawamsa/running-a-google-cloud-gpu-for-fast-ai-for-free-5f89c707bae6
# 
# ### Caution: Make sure you know how to shut them down to not rack up a huge bill!
# 
# ## Generally free tools 
# * RStudio
# * Anaconda (Jupyter, Spyder, Orange)
# * Weka
# * KNIME
# * https://www.h2o.ai/products/h2o/#how-it-works
# * https://public.tableau.com/en-us/s/
# * https://plot.ly/create/#/
# 
# 
# ## Great Resources:
# Free Code!
# * https://github.com/amueller/scipy-2018-sklearn
# * https://jakevdp.github.io/PythonDataScienceHandbook
# * https://github.com/rasbt/python-machine-learning-book-2nd-edition
# * https://github.com/josephmisiti/awesome-machine-learning
# * https://github.com/lazyprogrammer/machine_learning_examples
# * https://github.com/scikit-learn/scikit-learn
# 
# ## Free Courses:
# Coursera, edx, classcentral.com
# 
# ## Soapbox: Free means people have dedicated time and resources to creating and maintaining these things.  Be a part of the open source community by contributing!
# 

# # Python Machine Learning - Code Examples
# # Chapter 3 - A Tour of Machine Learning using Scikit-Learn
# 
# ## Agenda
# - Machine Learning basics
# - K-Nearest Neighbor
# - Decision Trees
# - Logistic Regression
# 
# Sorry no equations - read the book for theoretical basis for the algorithms!

# In[ ]:


from IPython.display import Image


# In[ ]:


Image("../input/images/images/what_is_ml.png")


# In[ ]:


Image("../input/images/images/types_oh_ml.png")


# In[ ]:


Image("../input/images/images/types_of_ml.png")


# ## Quickest intro to numpy/scipy:
# * NumPy (1995 as numeric, 2006 as NumPy)
#     * Array data types and basic operations (with some overlap with scipy)
# * SciPy (Scientific Python) - created 2001
#     * Fully featured versions of the linear algebra and numerical algorithms
#     
#     
# * Fortran/C/C++ under the hood - fast! Don't rewrite these methods!
# * Incredible SciPy conference held yearly in Austin, TX! Meet many of the scikit-learn and other python open source contributers! https://conference.scipy.org/ Watch all talks for free: https://www.youtube.com/user/EnthoughtMedia

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import sklearn

# Any results you write to the current directory are saved as output.


# Caution: Keep track of python packages using docker (kaggle does this), conda environments, virtualenv, or with watermark:  https://github.com/rasbt/watermark#installation-and-updating
# 
# Caution: Important to remember reproducible code and research!
# 
# Caution: Check internet connection for pip install
# 

# In[ ]:


get_ipython().system('pip install watermark')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'watermark')


# #### Caution: VERY important to keep track of the versions used - open source packages change frequently and the code may not work with new package versions.  Use Docker or virtual environments to keep track and test all your code anytime you want to update a package in your environment.
# 
# I like to print it out in the notebook if I'm just sharing my notebook.

# In[ ]:


get_ipython().run_line_magic('watermark', '--iversions')


# ### No free lunch! 
# * "All models are wrong but some are useful"
# * Bias-Variance tradeoff
# 
# Resource: https://www.researchgate.net/figure/Examples-of-real-life-problems-in-the-context-of-supervised-and-unsupervised-learning_fig8_319093376

# In[ ]:


Image("../input/images/images/bias_variance_tradeoff_reg.png")


# In[ ]:


Image("../input/images/images/overfitting_under_classification.png")


# ## Scikit-Learn
# - Initially the largest python machine learning package
# - Gold standard in the way we can transform data, fit models, and predict using a model
# 
# https://scikit-learn.org/stable/
# 

# ## Supervised Learning:
# * Learning with labels provided
# * Many classification methods built into scikit-learn (sklearn)
# * Because we know the 'answer' (labels), we can evaluate
# * Human element to machine learning - how to evaluate (more details in future workshop)
# 

# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png)

# 

# ## Download Iris Dataset
# 
# Loading the Iris dataset from scikit-learn. 

# In[ ]:


from sklearn import datasets
import numpy as np

iris = datasets.load_iris() # loading built in data set
X = iris.data[:, [2, 3]] # Here, the third column represents the petal length, 
                        # and the fourth column the petal width of the flower samples.
y = iris.target # The classes are already converted to integer labels 
                # .where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.

print('Class labels:', np.unique(y))


# ![](https://cdn-images-1.medium.com/max/1200/1*2uGt_aWJoBjqF2qTzRc2JQ.jpeg)

# In[49]:


X.shape # shape of the matrix, # of samples by # of features


# In[ ]:


plt.scatter(X[:,0], X[:,1])
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()


# To evaluate how well our supervised models generalize, we can split our data into a training and a test set - split the data into 70% training and 30% test data:

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)


# #### Caution: Stratify splits based on class imbalance instead of randomly.  Make sure to look at the class imbalance and deal with it appropriately based on the algorithm!
# 
# Resource: https://github.com/scikit-learn-contrib/imbalanced-learn 

# ![](https://cdn-images-1.medium.com/max/1600/1*2yd6LH2QjQ1vahVg9rx2gw.png)

# In[ ]:


print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


# In[ ]:


plt.scatter(X_train[:,0], X_train[:,1])
plt.xlabel('petal length')
plt.ylabel('petal width')


# In[ ]:





# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[ ]:


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# In[ ]:


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')


# ## Lazy learning
# - KNN - K nearest neighbors
# 
# ![](https://pbs.twimg.com/media/DmVRIqrXcAAOvtH.jpg)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=20, 
                           p=2, 
                           metric='minkowski') #euclidean distance
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_24.png', dpi=300)
plt.show()


# In[ ]:


y_pred = knn.predict(X_test_std)


# In[ ]:


knn.score(X_train_std, y_train)


# In[ ]:


knn.score(X_test_std, y_test) # 98% accurate


# Because we have "labels" or ground-truth, we can evaluate.  Extensive evaluation deepdive in a future session.

# Another evaluation method - a confusion matrix. Below we have accuracy = (50 + 100)/165 = .91

# ![](https://www.dataschool.io/content/images/2015/01/confusion_matrix_simple2.png)

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
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


# In[50]:


np.set_printoptions(precision=2)
class_names = iris.target_names
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# EXERCISE:
# - Modify above KNN code with different values of the ``n_neighbors`` and observe how the prediction boundary changes.
# - Modify above KNN code with different values of the ``n_neighbors`` and observe how training and test score changes.
# 

# ## Decision Trees:
# 

# CLICK ME: http://www.r2d3.us/visual-intro-to-machine-learning-part-1/

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_20.png', dpi=300)
plt.show()


# In[ ]:


get_ipython().system('pip install pydotplus')


# In[ ]:


from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree,
                           filled=True, 
                           rounded=True,
                           class_names=['Setosa', 
                                        'Versicolor',
                                        'Virginica'],
                           feature_names=['petal length', 
                                          'petal width'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data) 
#graph.write_png('tree.png') 


# In[ ]:


Image(graph.create_png())


# EXERCISE:
# - Modify above Decision Tree with different values of the ``max_depth`` and observe how the prediction boundary changes.
# - Modify above Decision Tree with different values of the ``max_depth`` and observe how ``Image(graph.create_png())`` changes
# 
# Time Permitted:
# - Modify above Decision Tree with different values of the ``max_depth`` and observe how training and test score changes.  What could be prolematic with decision trees?
# 

# Logistic Regression!
