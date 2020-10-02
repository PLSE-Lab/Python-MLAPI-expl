#!/usr/bin/env python
# coding: utf-8

# # 1. Problem Definition
# 
# We are going to work on the Iris Species dataset provided by UCI Machine Learning Repository.
# 
# We are going to cover the following points:
# 2. Loading the dataset.
# 3. Summarizing the dataset.
# 4. Visualizing the dataset.
# 5. Evaluating some algorithms.
# 6. Making some predictions.
# 7. References and Credits
# 
# <u>Goal</u>: Classification of Iris flowers

# # 2. Load The Data
# 
# In this step we are going to load the libraries and the input file provided by Kaggle.

# ## 2.1 Import libraries
# 
# First, let's import all of the modules, functions and objects we are going to use in this project.

# In[ ]:


# Load libraries
from pandas import read_csv
# from pandas.tools.plotting import scatter_matrix (https://github.com/pandas-dev/pandas/issues/15893)
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')


# ## 2.2 Load Dataset

# In[ ]:


# Load dataset
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# filename = '/kaggle/input/iris/Iris.csv' 
# https://www.kaggle.com/sohier/tutorial-accessing-data-with-pandas

dataset = read_csv('/kaggle/input/iris/Iris.csv')

# df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
dataset.rename(columns={'SepalLengthCm': 'sepal-length', 'SepalWidthCm': 'sepal-width', 'PetalLengthCm':'petal-length', 'PetalWidthCm':'petal-width', 'Species':'class'}, inplace=True)

# drop the Id column
dataset = dataset.drop('Id', 1)


# # 3. Summarize the Dataset
# 
# In this step we are going to take a look at the data in a few different ways:
# * Dimensions of the dataset.
# * Peek at the data itself.
# * Statistical summary of all attributes.
# * Breakdown of the data by the class variable.

# ## 3.1 Dimensions of Dataset
# 
# We can get a quick idea of the number of instances (rows) and number of attributes (columns).

# In[ ]:


# shape
print(dataset.shape)


# ## 3.2 Peek at the Data
# 
# Let's eyeball our data.

# In[ ]:


# head
print(dataset.head(5))


# ## 3.3 Statistical Summary
# 
# Let's take a look at a summary of each attribute. This includes the count, mean, the min and max values as well as some percentiles.

# In[ ]:


# descriptions
print(dataset.describe())


# <u>Inference</u>: 
# * We can see that all of the numerical values have the same scale (centimeters) and similar ranges between 0 and 8 centimeters.
# * Hence, rescaling/normalization is not required in this case
# 
# <u>Why do we need to rescale data?</u>
# 
# * When our data is comprised of attributes with varying scales, many machine learning algorithms can benefit from rescaling the attributes to all have the same scale. 
# * This is referred to as normalization and attributes are often rescaled into the range between 0 and 1. 
# * This is useful for optimization algorithms used in the core of machine learning algorithms like gradient descent. 
# * It is also useful for algorithms that weight inputs like regression and neural networks and algorithms that use distance measures like k-Nearest Neighbors. 
# * We can rescale our data using scikit-learn using the MinMaxScaler class.

# ## 3.4 Class Distribution
# 
# Let's now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count.
# 
# * On classification problems we need to know how balanced the class values are. 
# * Highly imbalanced problems (a lot more observations for one class than another) are common and may need special handling in the data preparation stage of our project.

# In[ ]:


# class distribution
print(dataset.groupby('class').size())


# <u>Inference</u>: 
# * We can see that each class has the same number of instances (50 or 33% of the dataset).
# * Since this is not a case of imbalanced dataset, there is no need for SMOTE (Synthetic Minority Oversampling Technique)
# * In case of unbalanced dataset, we would need to refer to the techniques mentioned in https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/ 

# # 4. Data Visualization
# 
# We now have a basic idea about the data. We need to extend this with some visualizations. We are going to look at two types of plots:
# * Univariate plots to better understand each attribute.
# * Multivariate plots to better understand the relationships between attributes.

# ## 4.1 Univariate Plots
# 
# We will start with some univariate plots, that is, plots of each individual variable. Given that the input variables are numeric, we can create box and whisker plots of each.

# In[ ]:


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# We can also create a histogram of each input variable to get an idea of the distribution.

# In[ ]:


# histograms
dataset.hist()
pyplot.show()


# <u>Inference</u>: 
# * It looks like perhaps two of the input variables (sepal-length and sepal-width) have a Gaussian distribution. 
# * This is useful to note as we can use algorithms that can exploit this assumption.

# ## 4.2 Multivariate Plots
# 
# Now we can look at the interactions between the variables. Let's look at scatter plots of all pairs of attributes. This can be helpful to spot structured relationships between input variables.

# In[ ]:


# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


# <u>Inference</u>: 
# * Note the diagonal grouping of some pairs of attributes such as between
#     * sepal-length and petal-length
#     * petal-lenth and petal-width
# * This suggests a high correlation and a predictable relationship.

# # 5. Evaluate Some Algorithms
# 
# Now it is time to create some models of the data and estimate their accuracy on unseen data.
# Here is what we are going to cover in this step:
# 1. Separate out a validation dataset.
# 2. Setup the test harness to use 10-fold cross validation.
# 3. Build 6 different models to predict species from flower measurements
# 4. Select the best model.

# ## 5.1 Create a Validation Dataset
# 
# We need to know whether or not the model that we created is any good. Later, we will use statistical methods to estimate the accuracy of the models that we create on unseen data. We also want a more concrete estimate of the accuracy of the best model on unseen data by evaluating it on actual unseen data. That is, we are going to hold back some data that the algorithms will not get to see and we will use this data to get a second and independent idea of how accurate the best model might actually be. We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.

# In[ ]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# ## 5.2 Test Harness
# 
# We will use 10-fold cross validation to estimate accuracy. This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits. We are using the metric of accuracy to evaluate models. This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.

# ## 5.3 Build Models
# 
# We don't know which algorithms would be good on this problem or what configurations to use. We get an idea from the plots that some of the classes are partially linearly separable in some
# dimensions, so we are expecting generally good results. Let's evaluate six different algorithms:
# * Logistic Regression (LR)
# * Linear Discriminant Analysis (LDA)
# * k-Nearest Neighbors (KNN)
# * Classifcation and Regression Trees (CART)
# * Gaussian Naive Bayes (NB)
# * Support Vector Machines (SVM)
# 
# This list is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable. Let's build and evaluate our six models:

# We now have training data in the X_train and Y_train for preparing models and a X_validation and Y_validation sets that we can use later.

# In[ ]:


# Spot-Check Algorithms
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
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# ## 5.4 Select The Best Model
# 
# We now have 6 models and accuracy estimations for each. We need to compare the models to each other and select the most accurate. Running the example above, we get the following raw results:
# 
# * We can see that it looks like LR, KNN and SVM have the largest estimated accuracy score.
# * All 3 models (LR, KNN, SVM) have an accuracy = 0.983333 and standard deviation = 0.033333
# 
# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. 
# There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).

# In[ ]:


# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# <u>Inference</u>:
# * We can see that the box and whisker plots are squashed at the top of the range, with many samples achieving 100% accuracy.

# # 6. Make Predictions
# 
# Out of LR, KNN and SVM, we don't know which will be most accurate on our validation dataset. Let's find out. We will run these models directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report.
# 
# This will give us an independent final check on the accuracy of the best model. It is important to keep a validation set just in case we made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.

# ## 6.1 Make predictions on validation dataset based on LogisticRegression

# In[ ]:


# Make predictions on validation dataset based on LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# ## 6.2 Make predictions on validation dataset based on KNeighborsClassifier

# In[ ]:


# Make predictions on validation dataset based on KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# ## 6.3 Make predictions on validation dataset based on SVC

# In[ ]:


# Make predictions on validation dataset based on SVC
svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# <u>Inference</u>:
# * The KNN algorithm was the most accurate model that we tested on the validation dataset.
# * The accuracy of KNN is 0.9 or 90% whereas the accuracy of LR and SVM is 0.87 or 87%. 
# * The confusion matrix provides an indication of the three errors made. 
# * Finally the classification report provides a breakdown of each class by precision, recall, f1-score and support showing the results.

# # 7. References and Credits
# 
# * Thank you to Jason Brownlee for his post https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# * For formatting, this link is useful https://www.kaggle.com/chrisbow/formatting-notebooks-with-markdown-tutorial
# * How to replace dataset header with custom header https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
