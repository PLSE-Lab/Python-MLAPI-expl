#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbors Classifier to Predict Wood Slabs

# ## Goal for this Notebook
# 
# This notebook aims to develop a model that would accurately predict the kind of the wood, that slab belongs to. This model would be trained using the provided data set of various **Wood Slabs**, using the **K Nearest Neighbors Algorithm** and the optimized parameters.

# #### Data Handling
# 
# - Importing Data with Pandas
# - Cleaning Data
# - Exploring Data through Visualizations with Matplotlib
# 
# #### Supervised Machine Learning Technique
# - K Nearest Neighbors Classification 
#     
# #### Hyperparameter Tuning
# - Grid Search Cross Validation
#     
# #### Model Performance
# - Score 
# - Classification Report
# - Confusion Matrix
# 
# #### Required Packages
# - NumPy
# - Pandas
# - Scikit Learn
# - Matplotlib

# To begin with, let's make all the necessary imports to start working.

# In[ ]:


# for performing mathematical operations
import numpy as np 

# for data processing, CSV file I/O 
import pandas as pd 

# for plotting and visualozing data
import matplotlib.pyplot as plt 


# ## Data Handling
# 
# In this section, required dataset is imported, explored and cleaned to make it available in the right format for the implementation of K Nearest Neighbors Classification. 
# 
# Data Analysis techniques used in this section includes: 
# - Importing data set using Pandas 
# - Exploring data to find features and target 
# - Handling missing or corrupted values in the data
# - Visualizing data using Matplotlib to explore relationships
# 
# *You can skip this section if you want to play with data yourself.*
#  

# #### Dataset Details
# 
# We will be using the CSV format file called **Raw_Materials_Wood_Slabs** to train our model. The CSV file contains rows (observations) describing each wood slab using some factors defined with the help of columns (features). The dataset contains more than 700 observations. 
# 
# Data set used in this notebook is available at below link. <br />
# https://github.com/mahnoor-shahid/Machine-Learning-Models/blob/master/Raw_Materials_Wood_Slabs.csv

# #### Importing Data using Pandas Library

# In[ ]:


# read the data from the csv file into a dataframe
labels = ['Wood Slab', 'Slab Height', 'Slab Width', 'Slab Shade']
dataset = pd.read_csv('../input/raw-materials-wood-slabs/Raw_Materials_Wood_Slabs.csv', names=labels, header=None)


# #### Exploration of Data
# We have loaded our required dataset, now we will see it's first five rows to check how our data looks.

# In[ ]:


# checking first five rows of our dataset
dataset.head()


# In[ ]:


# extracting information from the dataset for the predictor and target variables
dataset.info()


# #### Analysis
# In the dataset provided, we have extracted the information regarding the columns and rows. Rows are refered as *observations*. Each column in this data set tells us something about each of our observations, like their width, height, or shade. These columns are called *Features* or *Predictor Variables* of our dataset. Columns like wood-type and wood-slab classifies our dataset and are considered as *Target Variables*. 
# 
# From the above information, this can be figured out that we have **3 Predictor Variables** of data type float-64 and **1 Target Variable** of data type object, represented by **4** data columns.
# 
# **Features**
# - Slab Height
# - Slab Width
# - Slab Shade
# 
# **Target** 
# - Wood Slab
# 
# Also, we have been provided with **791 observations**, ranged from 0 to 790.

# In[ ]:


# finding out all types of wood slabs that exist as our target and their respective count
print("\nDifferent Types of Wood")
dataset['Wood Slab'].value_counts()


# By this we know that, we have **3** different kinds of woods: Oak, Beech and Ebony.

# #### Cleaning Data - Resolving Missing Values
# 
# The data may have missing or null values which maybe useless and would contribute nothing to our analysis. To handle this, we will drop those values from the dataset to preserve the integrity of our dataset.

# In[ ]:


# finding out the null values 
dataset.isnull().sum()


# Unfortunately, we have **38** missing or corrupted data fields in our dataset, possibly represented by NaNs.

# In[ ]:


# eliminating all the null values 
dataset = dataset.dropna()


# This line of code removes the NaN values from every column.
# 
# Next, we will verify our dataset of having null-free values using the same line of code that we used to check null or missing values.

# In[ ]:


# looking for null values
dataset.isnull().sum()


# Now, we have a clean and tidy dataset that is ready for analysis. Because .dropna() removes an observation from our data even if it only has 1 NaN in one of the features.

# #### Visualize the relationship between the features and the target variable
# 
# We will visualize this data on plots using target variable as x-axis and predictor variables as y-axis for further investigation and to explore the relationship between them.

# In[ ]:


# visualize the relationship between the features and the target using scatterplots
fig, axs = plt.subplots(1, 3, figsize=(20, 8))
dataset.plot(kind='scatter',  x='Wood Slab', y='Slab Height', ax=axs[0], title="Height of Wood Slabs")
dataset.plot(kind='scatter',  x='Wood Slab', y='Slab Width', ax=axs[1], title="Width of Wood Slabs")
dataset.plot(kind='scatter',  x='Wood Slab', y='Slab Shade', ax=axs[2], title="Shade of Wood")


# In[ ]:


# visualize the flow of the target using line-plot
plt.figure(figsize=(16, 5))
plt.plot(dataset['Wood Slab'], color='Red', label="Wood Type", linewidth=3)
plt.grid()
plt.legend()


# ## Implementing K-Nearest Neighbors Algorithm
# 

# **What is K-Nearest Neighbor Algorithm?**
# 
# As explained by Wikipedia:
# 
# > In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression: 
# - In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. 
# - In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.
# 
# k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation.

# #### Importing KNeighborsClassifier using Sklearn Library
# Now we will use K-Nearest Neighbors Classification to predict a new record on the basis of this data. 

# In[ ]:


# import the KNeighborsClassifier module
from sklearn.neighbors import KNeighborsClassifier


# #### Instantiate the model
# Now we will create a knn-model classifier for making predictions.

# In[ ]:


# instantiating KNeighborsClassifier
knn = KNeighborsClassifier()


# ## Splitting Training and Testing Data
# In order to evaluate our model later for perfomance metrics or factors, we will split our data set into two groups.
# - Training Data, consisting of 70 percent of data on which we will train our model
# - Testing Data, consisting of remaing 30 percent of data on which we will perform evaluation

# #### Importing train_test_split using Sklearn Library
# We will use train_test_split to split our data set into training and testing data.

# In[ ]:


# for splitting data into training and testing data
from sklearn.model_selection import train_test_split


# #### Assignment of X_train, X_test, y_train, y_test

# In[ ]:


# defining target variables 
target = dataset['Wood Slab']

# defining predictor variables 
features = dataset.drop(['Wood Slab'], axis=1)

# assigning the splitting of data into respective variables
X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=0.4, random_state=42, stratify = target)


# #### Exploring X_train, X_test, y_train, y_test
# 

# In[ ]:


print("X_train shape: %s" % repr(X_train.shape))
print("y_train shape: %s" % repr(y_train.shape))
print("X_test shape: %s" % repr(X_test.shape))
print("y_test shape: %s" % repr(y_test.shape))


# In[ ]:


# to display the HTML representation of an object.
from IPython.display import display_html

X_train_data = X_train.describe().style.set_table_attributes("style='display:inline'").set_caption('Summary of Training Data')
X_test_data = X_test.describe().style.set_table_attributes("style='display:inline'").set_caption('Summary of Testing Data')

# to display the summary of both training and testing data, side by side for comparison 
display_html(X_train_data._repr_html_()+"\t" +X_test_data._repr_html_(), raw=True)


# ## Hyperparameter Tuning - Adjust important parameters using Grid Search CV
# Parameters that needs to be specified before fitting (training) a model are called Hyperparameters, like *n_neighbors* in our case. So, the process of choosing the optimal hyperparameters for the learning algorithm is called Hyperparameter Tuning.
# 
# We will be using the *Grid Search Cross Validation* technique to choose the hyperparameter (n_neighbors) that would perform the best, for our model.
# 
# **What is Grid Search Cross Validation technique?**
# 
# As explained by Wikipedia:
# 
# > The traditional way of performing hyperparameter optimization has been grid search, or a parameter sweep, which is simply an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm. A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set or evaluation on a held-out validation set.
# Since the parameter space of a machine learner may include real-valued or unbounded value spaces for certain parameters, manually set bounds and discretization may be necessary before applying grid search.

# #### Importing GridSearchCV using Sklearn Library
# Since, we are using the grid search cross validation technique for hyperparameter tuning, we will first import it from the Sklearn Library.

# In[ ]:


# for exhaustive search over specified parameter values for an estimator
from sklearn.model_selection import GridSearchCV


# #### Defining the Parameter Grid

# In[ ]:


# assigning the dictionary of variables whose optimium value is to be retrieved
param_grid = {'n_neighbors' : np.arange(1,50)}


# At this point, we have a classifier and paramerter_grid, so we can perform the Grid Search Cross Validation.

# In[ ]:


# performing Grid Search CV on knn-model, using 5-cross folds for validation of each criteria
knn_cv = GridSearchCV(knn, param_grid, cv=5)


# #### Training the Classifier
# The idea is to tune the model's hyperparameter on the training set and then evaluate later it's performance on the hold-out set. So, using the training set that we have obtained before, that is, X_train and y_train, we will now fit the model.

# In[ ]:


# training the model with the training data and best parameter
knn_cv.fit(X_train,y_train)


# #### Best Parameter and Best Score
# Let's look for the value of the required parameter, chosen to be the most efficient for the model. Along with the score that it achieved.

# In[ ]:


# finding out the best parameter chosen to train the model
print("The best paramter we have is: {}" .format(knn_cv.best_params_))

# finding out the best score the chosen parameter achieved
print("The best score we have achieved is: {}" .format(knn_cv.best_score_))


# The n_neighbor is set as **1** and the accuracy level is **100%**

# ## Testing the Model
# Apply the model. For supervised algorithms, this is predict.

# In[ ]:


# predicting the values using the testing data set
y_pred = knn_cv.predict(X_test)


# We can make predictions with new data as following:

# In[ ]:


# example of ebony slab , given height, width and shade
prediction1=knn_cv.predict([[5.3, 3.6, 0.1]])
prediction1


# In[ ]:


# example of oak slab , given height, width and shade
prediction2=knn_cv.predict([[3.7, 8.2, 0.9]])
prediction2


# ## Model Performance

# In classification problems, *accuracy* is the commonly used metric to evaluate the performance of the model, which means the fraction of the correct predictions.

# In[ ]:


# the score() method allows us to calculate the mean accuracy for the test data
knn_cv.score(X_test,y_test)


# This metric quantifies the fraction of the correct predictions. Thus, the accuracy of the model to predict the values is **100%**.

# #### Importing classification_report and confusion_matrix using Sklearn Library
# We will use classification_report and confusion_matrix to view the performance factors of our model.

# In[ ]:


# for performance metrics
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


# call the classification_report and print the report
print(classification_report(y_test, y_pred))


# In[ ]:


# call the confusion_matrix and print the matrix
print(confusion_matrix(y_test, y_pred))


# Although the array is printed without headings, but you can see that the majority of the predictions fall on the diagonal line of the matrix (which are correct predictions).
# 
# Finally, we have trained our model and it's successfully predicting correct values with maximum accuracy. 

# ## Summary

# In this notebook, first, we examined the data to understand it and extracted information required to know the *features* and *target*. Then, data cleaning is performed to transform the data in the right and useful format. After that, we have visualized the data to explore the relationship between the data points. We have splitted our data into testing and training data sets, as well. Later, we have intantiated the KNeighbors classifier and performed hyperparameter tuning to figure out the best and optimum parameter for training our model. Subsequently, we have evaluated our model predictions using different performance metrics and achieved maximum accuracy.
