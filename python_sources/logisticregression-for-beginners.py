#!/usr/bin/env python
# coding: utf-8

# ** Hello Kaggler!**
# 
# Here is a detailed case study from scratch to recognise the Breast Cancer based on few input features using Support Vector Machine.

# **If** you are new to Data Science I am sure you will get something reading this.
# 
# **Else** your feedback will definitely help me improving this kernel... :)

# # Contents 
# 1. [Problem Statement ](#PS) 
# 2. [Importing Data](#ID)
# 3. [Data Visualization](#DV)
# 4. [Model Training](#MT)
#     1. [Support Vector Machine](#SVM)        
# 5. [Evaluating the Model](#EM)
# 6. [Improving the Model](#IM)
#     1. [Grid Search](#GS)

# # STEP 1: Problem Statement <a id='PS'></a>

# - Predicting if the cancer diagnosis is benign or malignant based on several observations/features 
# - 30 features are used, few of the examples are:
#         - radius (mean of distances from center to points on the perimeter)
#         - texture (standard deviation of gray-scale values)
#         - perimeter
#         - area
#         - smoothness (local variation in radius lengths)
#         - compactness (perimeter^2 / area - 1.0)
#         - concavity (severity of concave portions of the contour)
#         - concave points (number of concave portions of the contour)
#         - symmetry 
#         - fractal dimension ("coastline approximation" - 1)
# 
# - Datasets are linearly separable using all 30 input features
# - Number of Instances: 569
# - Class Distribution: 212 Malignant, 357 Benign
# - Target class:
#          - Malignant
#          - Benign
# 
# 
# [Data can be downloaded from here.](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

# # STEP 2: Importing Data <a id='ID'></a>
# 

# In[ ]:


# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# Let's explore the data..

# In[ ]:


cancer


# In[ ]:


cancer.keys()


# Let's explore the data stored in the cancer keys.

# In[ ]:


print(cancer['DESCR'])
print(cancer['target_names'])


# In[ ]:


print(cancer['feature_names'])
print(cancer['data'])


# In[ ]:


cancer['data'].shape


# As initially told, there are 30 features and 569 instances.

# Let's create a Data Frame of all cancer features.

# In[ ]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


# In[ ]:


df_cancer.head()


# In[ ]:


df_cancer.tail()


# # STEP 3: Visualising the Data <a id='DV'></a>

# Let's create scatter plots among few features.

# In[ ]:


sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )


# Above plots showing the relationship between different feature but not showing whether cancer is Malignant or Benign ie target variable.
# 
# Let's create another set of scatter plots where Malignant or Benign Cancer is classified.

# In[ ]:


sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )


# Let's create a plot showing the number of incidences of target variable (already given but still lets create!)

# In[ ]:


sns.countplot(df_cancer['target'], label = "Count") 


# Create scatter plot with different features.

# In[ ]:


sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# In[ ]:


# Let's check the correlation between the variables 

plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 


# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter

# # STEP 4: Model Training (Finding a Training Solution) <a id='MT'></a>

# In[ ]:



# Let's drop the target label coloumns
X = df_cancer.drop(['target'],axis=1)


# In[ ]:


y = df_cancer['target']


# Let's split the dataset into train and test dataset.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)


# In[ ]:


print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)


# Let's import the SVC class and fit the model.

# ## Support Vector Machine <a id='SVM'></a>
#  
# SVM is a supervised machine learning algorithm which can be used for classification or regression problems. It uses a technique called the kernel trick to transform the data and then based on these transformations it finds an optimal boundary between the possible outputs. Simply put, it does some extremely complex data transformations, then figures out how to seperate data based on the labels or outputs that are defined.
# 
# **So what makes it so great?**
#  
# Well SVM, it is capable of doing both classification and regression. In this post we'll focus on using SVM for classification. In particular I'll be focusing on non-linear SVM, or SVM using a non-linear kernel. Non-linear SVM means that the boundary that the algorithm calculates doesn't have to be a straight line. The benefit is that we can capture much more complex relationships between datapoints without having to perform difficult transformations on our own. The downside is that the training time is much longer as it's much more computationally intensive.

# In[ ]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)


# # STEP 5: Evaluating the Model <a id='EM'></a>

# In[ ]:


y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)


# In[ ]:


print(classification_report(y_test, y_predict))

sns.heatmap(cm, annot=True)


# From above confusion matrix, out of 114 entries 48 pridictions (almmost 42%) are wrong.
# 
# So, need to improve the model.

# # STEP 6: Improving the Model <a id='IM'></a>

# In[ ]:


sns.scatterplot(x= 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# One of the reason for such error is the range of data sets. From above we can say that the range for *mean smoothness* is from *0.04 to 0.18* while for the *mean area* is *0 to 2500*.
# 
# So, the solution is to set the range from 0 to 1 or we can say to normalise the data(normalization)
# 
# ![image.png](attachment:image.png)

# In[ ]:


min_train = X_train.min()

range_train = (X_train - min_train).max()


# In[ ]:


X_train_scaled = (X_train - min_train)/range_train

X_train_scaled


# In[ ]:


sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)


# Similarily, Normalize the test data.

# In[ ]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# Again create a model using Support Vector Machine.

# In[ ]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


# In[ ]:


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

print(classification_report(y_test,y_predict))
sns.heatmap(cm,annot=True,fmt="d")


# By observing the above *Confusion Matrix* we can say, the error reduced from *42% to 4%* only.

# # Improving the Model - Part 2

# # Grid Search <a id='GS'></a>
# 
# This technique is used to find the optimal parameters to use with an algorithm. This is NOT the weights or the model, those are learned using the data. This is obviously quite confusing so I will distinguish between these parameters, by calling one hyper-parameters.
# 
# Hyper-parameters are like the *c* or *gamma* or *kernel* in *SVM*. SVM requires the user to select which neighbor to consider when calculating the distance for *hyperplane*. The algorithm then tunes a parameter, a threshold, to see if a novel example falls within the learned distribution, this is done with the data.
# 
# 
# **How does it work?**
# 
# First we need to build a grid. This is essentially a set of possible values your hyper-parameter can take. For our case we can use for example *{'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}[1,2,3,...,10]*. Then we will train our model for each value in the grid. First it would take '*C*' and do for the remaining *gamma* and *kernel* and so on. For each iteration, we will get a performance score which will tell us how well the algorithm performed using that value for the hyper-parameter. After we have gone through the entire grid we will select the value that gave the best performance.
# 
# 

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)

grid.fit(X_train_scaled,y_train)


# In[ ]:


grid.best_params_


# So, these are the best parameters used for the particular model and dataset.
# 
# Let's make prediction based on these hyperparameters. 

# In[ ]:


grid.best_estimator_


# In[ ]:


grid_predictions = grid.predict(X_test_scaled)


# In[ ]:


cm = confusion_matrix(y_test, grid_predictions)

print(classification_report(y_test,grid_predictions))
sns.heatmap(cm, annot=True)


# From above *confusion matrix*, we can say the error is further reduced ie 4% to 2.6% and these are the best hyperparameters that are selected for this particular model.

# Here, We have implemented one of the common Machine Learning classification algorithms.
# 
# I hope this kernal is useful to you to learn machine learning from the scratch with Breast_Cancer dataset.
# 
# If you find this notebook helpful to you to learn, **Please Upvote**.
# 
# *Thank You!!*
