#!/usr/bin/env python
# coding: utf-8

# #My first project in data analysis and machine learning
# 
# **1 - Data analysis**
# 
# 1.1 - Load the data
# 
# 1.2 - Manipulating the data
# 
# 1.3 - Visualizing the data
# 
# **2 - Machine Learning**
# 
# 2.1 -  Test predictions in data input
# 
# 2.2 Test result prediction

# In[ ]:


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
import seaborn
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# **1.1 Load the data**

# In[ ]:


# Load dataset
dataset = pandas.read_csv("../input/Iris.csv")


# **1.2 Manipulating the data**

# In[ ]:


print(dataset.head(5))


# In[ ]:


print(dataset.shape)


# In[ ]:


print(dataset.describe())


# In[ ]:


#Remove a column from the data
dataset = dataset.drop('Id',axis=1)


# In[ ]:


print(dataset.head(5))


# In[ ]:


print(dataset.describe())


# **1.3 Visualizing the data**

# In[ ]:


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[ ]:


seaborn.pairplot(dataset, hue="Species", size=3, diag_kind="kde")
plt.show()


# In[ ]:


seaborn.pairplot(dataset, hue="Species", size = 3)
seaborn.set()


# In[ ]:


dataset.hist()
plt.show()


# **2 Machine Learning**

# In[ ]:


#I made an adaptation of this reference online 
#----> http://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, 
random_state=seed)


# In[ ]:


# Test options and evaluation metric
num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'


# In[ ]:


#I made an adaptation of this reference online 
#----> http://machinelearningmastery.com/machine-learning-in-python-step-by-step/
#Here we are testing various predictive algorithms from scikit-learn
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[ ]:


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[ ]:



# Make predictions on validation dataset
svn = SVC()
svn.fit(X_train, Y_train)
predictions = svn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# **2.1 Test predictions in data input**

# In[ ]:


#Input Vector 
X_new = numpy.array([[3, 2, 4, 0.2], [  4.7, 3, 1.3, 0.2 ]])
print("X_new.shape: {}".format(X_new.shape))


# In[ ]:


prediction = svn.predict(X_new)


# **2.1 Test result prediction**

# In[ ]:


#Prediction of the species from the input vector
print("Prediction of Species: {}".format(prediction))


# **Thanks!!**
