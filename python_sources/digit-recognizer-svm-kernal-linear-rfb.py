#!/usr/bin/env python
# coding: utf-8

# # Objective
# You are required to develop a model using Support Vector Machine which should correctly classify the handwritten digits from 0-9 based on the pixel values given as features. Thus, this is a 10-class classification problem. 

# # Data Description
# For this problem, we use the MNIST data which is a large database of handwritten digits. The 'pixel values' of each digit (image) comprise the features, and the actual number between 0-9 is the label. 
# 
# Since each image is of 28 x 28 pixels, and each pixel forms a feature, there are 784 features. MNIST digit recognition is a well-studied problem in the ML community, and people have trained numerous models (Neural Networks, SVMs, boosted trees etc.) achieving error rates as low as 0.23% (i.e. accuracy = 99.77%, with a convolutional neural network).
# 
# Before the popularity of neural networks, though, models such as SVMs and boosted trees were the state-of-the-art in such problems. In this assigment we will build model with **SVM** only.

# # Approach
# 
# We'll divide the analysis into the following parts:
# - Data understanding and cleaning
# - Data preparation for model building
# - Building an SVM model - hyperparameter tuning, model evaluation etc.
# 

# ## Data Understanding and Cleaning
#  
#  Let's understand the dataset and see if it needs some cleaning etc.

# ### Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import gc
import warnings
from IPython.display import Markdown, display ,HTML
from sklearn.model_selection import GridSearchCV


# ### User Prefrences  

# In[ ]:


pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')

def log(string):
    display(Markdown("> <span style='color:blue'>"+str(string)+"</span>"))


# ### Reading data 

# In[ ]:


# read the dataset
digits = pd.read_csv("../input/train.csv")
digits.info()


# In[ ]:


# head
digits.head()


# In[ ]:


one = digits.iloc[2, 1:]
one.shape
one = one.values.reshape(28, 28)
plt.imshow(one, cmap='gray')


# In[ ]:


four = digits.iloc[3, 1:]
four.shape
four = four.values.reshape(28, 28)
plt.imshow(four, cmap='gray')


# #### Side note: Indexing Recall ####
# `list =    [0, 4, 2, 10, 22, 101, 10]` <br>
# `indices = [0, 1, 2, 3, ...,        ]` <br>
# `reverse = [-n           -3  -2   -1]` <br>

# In[ ]:


# visualise the array
print(four[5:-5, 5:-5])


# In[ ]:


# Summarise the counts of 'label' to see how many labels of each digit are present
count = pd.DataFrame(digits.label.astype('category').value_counts()).sort_index()
count = count.rename(columns={'label': 'Count'})


# In[ ]:


# Summarise count in terms of percentage 
percetage = pd.DataFrame(100*(round(digits.label.astype('category').value_counts()/len(digits.index), 4))).sort_index()
percetage = percetage.rename(columns={'label': 'Percetage'})


# In[ ]:


pd.concat([count, percetage], axis=1, join_axes=[count.index])


# Thus, each digit/label has an approximately 9%-11% fraction in the dataset and the **dataset is balanced**. This is an important factor in considering the choices of models to be used, especially SVM, since **SVMs rarely perform well on imbalanced data**.
# Let's quickly look at missing values, if any.

# In[ ]:


# missing values - there are none
#digits.isnull().sum()

## no null vales in dataset


# Also, let's look at the average values of each column, since we'll need to do some rescaling in case the ranges vary too much.

# In[ ]:


# average values/distributions of features
description = digits.describe()
description


# You can see that the max value of the mean and maximum values of some features (pixels) is 139, 255 etc., whereas most features lie in much lower ranges  (look at description of pixel 0, pixel 1 etc. above).
# 
# Thus, it seems like a good idea to rescale the features.

# ## Data Preparation for Model Building
# 
# Let's now prepare the dataset for building the model. We'll only use a fraction of the data else training will take a long time.
# 

# In[ ]:


# Creating training and test sets
# Splitting the data into train and test
X = digits.iloc[:, 1:]
Y = digits.iloc[:, 0]

# Rescaling the features
from sklearn.preprocessing import scale
X = scale(X)

# train test split with train_size=10% and test size=90%
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.90, random_state=101)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


# delete test set from memory, to avoid a memory error
# we'll anyway use CV to evaluate the model, and can use the separate test.csv file as well
# to evaluate the model finally

# del x_test
# del y_test


# ## Model Building
# 
# Let's now build the model and tune the hyperparameters. Let's start with a **linear model** first.
# 
# ### Linear SVM
# 
# Let's first try building a linear SVM model (i.e. a linear kernel). 

# In[ ]:


from sklearn import svm
from sklearn import metrics

# an initial SVM model with linear kernel   
svm_linear = svm.SVC(kernel='linear')

# fit
svm_linear.fit(x_train, y_train)


# In[ ]:


# predict
predictions = svm_linear.predict(x_test)
predictions[:10]


# In[ ]:


# evaluation: accuracy
# C(i, j) represents the number of points known to be in class i 
# but predicted to be in class j
confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)
confusion


# In[ ]:


# measure accuracy
log(metrics.accuracy_score(y_true=y_test, y_pred=predictions))


# In[ ]:


# class-wise accuracy
class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)
print(class_wise)


# In[ ]:


# run gc.collect() (garbage collect) to free up memory
# else, since the dataset is large and SVM is computationally heavy,
# it'll throw a memory error while training
log("Memory Claimed : "+str(gc.collect()))


# ### Non-Linear SVM
# 
# Let's now try a non-linear model with the RBF kernel.

# In[ ]:


# rbf kernel with other hyperparameters kept to default 
svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(x_train, y_train)


# In[ ]:


# predict
predictions = svm_rbf.predict(x_test)

# accuracy 
log(metrics.accuracy_score(y_true=y_test, y_pred=predictions))


# The accuracy achieved with a non-linear kernel is slightly higher than a linear one. Let's now do a grid search CV to tune the hyperparameters C and gamma.
# 
# ### Grid Search Cross-Validation

# In[ ]:


# conduct (grid search) cross-validation to find the optimal values 
# of cost C and the choice of kernel



parameters = {'C':[1, 10, 100], 
             'gamma': [1e-2, 1e-3, 1e-4]}

# instantiate a model 
svc_grid_search = svm.SVC(kernel="rbf")

# create a classifier to perform grid search
clf = GridSearchCV(svc_grid_search, param_grid=parameters, scoring='accuracy',return_train_score=True)

# fit
clf.fit(x_train, y_train)


# In[ ]:


# results
cv_results = pd.DataFrame(clf.cv_results_)
cv_results


# In[ ]:


def plot_accuracy_graph(location,gamma_value) :
    plt.subplot(location)
    gamma = cv_results[cv_results['param_gamma']==gamma_value]
    plt.plot(gamma["param_C"], gamma["mean_test_score"])
    plt.plot(gamma["param_C"], gamma["mean_train_score"])
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title("Gamma="+str(gamma_value))
    plt.ylim([0.60, 1])
    plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
    plt.xscale('log')


# In[ ]:


# converting C to numeric type for plotting on x-axis
cv_results['param_C'] = cv_results['param_C'].astype('int')

# # plotting
plt.figure(figsize=(16,6))

# subplot 1/3
plot_accuracy_graph(131,0.01)
plot_accuracy_graph(132,0.001)
plot_accuracy_graph(133,0.0001)

plt.show()


# From the plot above, we can observe that (from higher to lower gamma / left to right):
# - At very high gamma (0.01), the model is achieving 100% accuracy on the training data, though the test score is quite low (<75%). Thus, the model is overfitting.
# 
# - At gamma=0.001, the training and test scores are comparable at around C=1, though the model starts to overfit at higher values of C
# 
# - At gamma=0.0001, the model does not overfit till C=10 but starts showing signs at C=100. Also, the training and test scores are slightly lower than at gamma=0.001.
# 
# Thus, it seems that the best combination is gamma=0.001 and C=1 (the plot in the middle), which gives the highest test accuracy (~92%) while avoiding overfitting.
# 
# Let's now build the final model and see the performance on test data.
# 
# 

# ### Final Model
# 
# Let's now build the final model with chosen hyperparameters.

# In[ ]:


print(clf.best_score_)
print(clf.best_params_)


# In[ ]:


# optimal hyperparameters
best_C = 10
best_gamma = 0.001


# model
svm_final = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)

# fit
svm_final.fit(x_train, y_train)


# In[ ]:


# predict
predictions = svm_final.predict(x_test)


# In[ ]:


# evaluation: CM 
confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)

# measure accuracy
test_accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)

log(test_accuracy)
print(confusion)


# ### Conclusion
# 
# The final accuracy on test data is approx. 96.90%. 
# 
# 
