#!/usr/bin/env python
# coding: utf-8

# ## Data Set Information:
# 
# This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
# 
# ![florecita iris](https://archive.ics.uci.edu/ml/assets/MLimages/Large53.jpg)
# 
# [UCI MACHINE LEARNING](https://archive.ics.uci.edu/ml/datasets/iris)

# In[ ]:


# Check the versions of libraries

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# In[ ]:


# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ## Load dataset
# 
# **Data Set Information:**
# 
# This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
# 
# Predicted attribute: class of iris plant.
# 
# This is an exceedingly simple domain.
# 
# This data differs from the data presented in Fishers article (identified by Steve Chadwick, spchadwick '@' espeedaz.net ). The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa" where the error is in the fourth feature. The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa" where the errors are in the second and third features.
# 
# 
# Attribute Information:
# 
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class:
# -- Iris Setosa
# -- Iris Versicolour
# -- Iris Virginica
# 
# 

# In[ ]:


# Check Kaggle filename
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Load dataset
url = "/kaggle/input/iris/Iris.csv"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url)


# # EDA
# 
# With the Exploratory Data Analyisis we explore the dataset, summary and stat datas.

# In[ ]:


# shape
print(dataset.shape)


# In[ ]:


# head
print(dataset.head(20))


# In[ ]:


# descriptions
print(dataset.iloc[:,1:].describe())


# In[ ]:


# Create a dataset backup
dataset_bak = dataset


# In[ ]:


dataset.iloc[:,0]


# In[ ]:


# Remove first column - Id
dataset = dataset.drop(['Id'], axis=1)
# head
print(dataset.head(20))


# In[ ]:


# box and whisker plots
fig=plt.figure(figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[ ]:


# histograms
dataset.hist()
plt.show()


# In[ ]:


# scatter plot matrix
scatter_matrix(dataset)
plt.show()


# ## Machine Learning Part
# 
# Often the hardest part of solving a machine learning problem can be finding the right estimator for the job.
# Different estimators are better suited for different types of data and different problems.
# The flowchart below is designed to give users a bit of a rough guide on how to approach problems with regard to which estimators to try on your data.
# 
# Please check official chart here > [flowchart by SciKit-Learn](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
# 
# ![](https://scikit-learn.org/stable/_static/ml_map.png)

# In[ ]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]

# Split 80-20 train-test data
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# In[ ]:


# Visualize first ten results of train data
X_train[1:10,]


# In[ ]:


# Visualize first ten results of validation data
X_validation[1:10,]


# In[ ]:


# Visualize first ten results of train predictor
Y_train[1:10,]


# In[ ]:


# Visualize first ten results of validation predictor
Y_validation[1:10,]


# ## ML models
# 
# We choose some machine learning model as:
# 
# * Logistic Regression
# * Linear Discriminant Analysis
# * K-Neighbors Classifier
# * Decision Tree Classifier
# * Gaussian NB
# * SVC
# 
# Also we can manipulate them with other parameter, or in simple words **tuning ML model**

# In[ ]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


# In[ ]:


# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# ## Final thoughts
# 
# For a simple dataset like Iris, we have only 150 observations for to calculate a machine learning model. Anyway we have a good results >95%. So, our conclusions are good and we validate ml models above.
# 
# Thank you for to read my first job and thanks for your time and upvote!
# 
# Thanks
# [marcusRB](https://www.marcusrb.com)
# 
