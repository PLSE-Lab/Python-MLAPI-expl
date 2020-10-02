#!/usr/bin/env python
# coding: utf-8

# # Breast cancer Prediction
# # -[Rishit Dagli](rishitdagli.ml)

# ![](https://drive.google.com/uc?id=16c6UtqGFDrJNordq9lIursSR0Ks5W8k6)

# I have used the Wisconsin Breast cancer dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml) availaible [here](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)).
# <br>
# I have used Artificial Neural Networks for this problem and found out the best hyper parameters using cross validation.<br>For more details read my research paper [here](https://iarjset.com/papers/machine-learning-as-a-decision-aid-for-breast-cancer-diagnosis/).
# <br><br>
# ![](https://drive.google.com/uc?id=1ETVCulfECkSBOcZXtXLnaDUIjnpoZMu5)
# <br>
# <br>
# ![](https://drive.google.com/uc?id=1mIKCJ6wyvSMrx-oa4IFRGG5FUR8pPOKN)
# 
# ---
# 
# 
# <font color="red">**Data Set Information:**</font>
# 
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. A few of the images can be found [here](www.cs.wisc.edu/~street/images/)
# 
# 
# 
# ---
# 
# 
# 
# <font color="red">**Attribute Information:**</font>
# 
# 1) ID number<br>
# 2) Diagnosis (M = malignant, B = benign)<br>
# 3-32)<br>
# <br>
# <font color="light green">*Ten real-valued features are computed for each cell nucleus:*</font><br>
# 
# a) radius (mean of distances from center to points on the perimeter)<br>
# b) texture (standard deviation of gray-scale values)<br>
# c) perimeter<br>
# d) area<br>
# e) smoothness (local variation in radius lengths)<br>
# f) compactness ($\frac{perimeter^2}{area} - 1.0$)<br>
# g) concavity (severity of concave portions of the contour)<br>
# h) concave points (number of concave portions of the contour)<br>
# i) symmetry<br>
# j) fractal dimension ("coastline approximation" - 1)<br>
# 

# ### Import required libraries

# We use `matplotlib` and `seaborn` to create some wonderful visualizations of our data.<br>
# `pandas` to read our data and know some insights about the data, efficiently. With `pandas` by our side we can do many things with just simple functions which we will take a look at in the later part.<br>
# `sklearn` to<br>
# - Select the model with best hyper parameters
# - Encode the labels i.e. M and B
# - Print a confusion matrix with test data results
# - Make a train / test split easily
# - Scale values
# <br><br>
# `tensorflow` and `keras` to create our model (ANN) and make some plots of it

# In[ ]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from IPython.display import SVG
from keras.utils import model_to_dot
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

dataset = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')


# ### Analyze the dataset

# We will now analyze our data :
# 
# 
# 1.Print the features of dataset which are also mentioned above<br>
# 2.View how many samples and missing values there are for each feature and display them accordingly<br>
# We here see the missing samples and values:
# ```
# RangeIndex: 568 entries, 0 to 567
# Data columns (total 32 columns):
# 842302      568 non-null int64
# M           568 non-null object
# 17.99       568 non-null float64
# 10.38       568 non-null float64
# 122.8       568 non-null float64
# 1001        568 non-null float64
# 0.1184      568 non-null float64
# 0.2776      568 non-null float64
# ...
# ...
# 0.4601      568 non-null float64
# 0.1189      568 non-null float64
# dtypes: float64(30), int64(1), object(1)
# memory usage: 142.1+ KB
# None
# ```
# 
# 3.View numerical features of data set we will majorly focus on `mean`, `count`, `std`, `min value`, `max value`, `upper quartile`, `inter quartile` and `lower quartile`<br>
# For this all we need to do is `dataset.describe()` So this justifies our use of `pandas` library.<br>
# 4.We will now take a look at the label features specially `count`, `unique`, `top` and `frequency`. The `count` parameter just tells us the number of entries, the `unique` parameter is important.
# Here we receive -
# ```
# unique: 2
# ```
# Which tells us to perform2 class classification.<br>
# The `top` parameter is often used to check biases in the data set itself.
# 
# 
# 

# In[ ]:


def analyze(data):
    
  # View features in data set
  print("Dataset Features")
  print(data.columns.values)
  print("=" * 30)
    
  # View How many samples and how many missing values for each feature
  print("Dataset Features Details")
  print(data.info())
  print("=" * 30)
    
  # view distribution of numerical features across the data set
  print("Dataset Numerical Features")
  print(data.describe())
  print("=" * 30)
    
  # view distribution of categorical features across the data set
  print("Dataset Categorical Features")
  print(data.describe(include=['O']))
  print("=" * 30)


# In[ ]:


analyze(dataset)


# ### Make a feature pairplot

# We will now make a feature wise pairplot meaning we will plot labels $x_1$, $x_2$, $...$ and or label $y$ with each other. Where $x$ and $y$ have their usual meaning. We will use `seaborn` to help us with this. A pair plot allows us to see both distribution of single variables and relationships between two variables . Pair plots are a great method to identify trends for follow-up analysis. So this again becomes an important step for us.

# In[ ]:


sns.pairplot(dataset, hue="diagnosis", size= 2.5)


# ### Seperate the the features and laabels

# This is just a simple code which stores $X$ or the features and $y$ or the labels in different variables

# In[ ]:


X = dataset.iloc[:,2:32] 
y = dataset.iloc[:,1] 


# ### Encode the **labels** to 1, 0

# We had our labels as `M` and `B` depicting malignant and benign respectively. As these are `strings` or `char` as you might say and we are only concerned with numbers so we encode such that <br>
# 
# 
# *   All `M = 1`
# *   All `B = 0`
# 
# For this we use the `LabelEncoder` class.
# 

# In[ ]:


print("Earlier: ")
print(y[100:110])

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

print()
print("After: ")
print(y[100:110])


# ### Make a 80/ 20 train, test split

# Making a train test split is important for any AI problem without which we do not know how our model would perform to unseen values and also not overfit the data

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Scaling the values is not actually compulsory but I would recommend one to do it for a faster convergence, so we use `sklearn` to help us in this

# In[ ]:


# Scale values from faster convergence
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### Build a classifier

# We finally build a `tensorflow`, `keras` classifier

# In[ ]:


def build_classifier(optimizer):
  classifier = Sequential()
  classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
  classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
  classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
  classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
  classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
  return classifier


# In[ ]:


classifier = KerasClassifier(build_fn = build_classifier)


# ### Experiment with various hyper parameters

# Before tuning our hyper parameters we surely want to test the data with various of them and choose the best one so define few options for the model

# In[ ]:


parameters = {'batch_size': [1, 5],
               'epochs': [100, 120],
               'optimizer': ['adam', 'rmsprop']}


# ### Use Cross Vaildation to obtain best model

# Cross Vaalidation is a wonderful technique which often comes to our rescue while selecting the best model so we use a 10 fold CV here.<br>
# We could also have used `AIC`, `BIC` or even Mallows $C_p$ if the CV does not give us a good result but that's not the case here

# In[ ]:


# Cross validation
grid_search = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 10)


# Search for the best model in the complete matrix

# In[ ]:


# Get best model
# Note: this may take some time
grid_search = grid_search.fit(X_train, y_train)


# ### Finally build model according to above obtained results

# In the above step we already obtained the best model for this problem so now we are almost done all we need to do is build a classifier according to obtained results

# In[ ]:


classifier = Sequential()


# So we make an ANN now with
# 
# * Input Layer - **16** neurons and `ReLu` activator
# * Hidden Layer 1 - **8** neurons and `ReLu` activator
# * Hidden Layer 2 - **6** neurons and `ReLu` activator
# * Output Layer - **1** neuron (ok that was obvious !) and `sigmoid` activator
# 
# 

# In[ ]:


# Make the best classifier as we received earlier
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Complete the classifier

# We use `BCE` or binary cross entropy which is suited best to sigmoid.
# <br> $BCE=CE_1+CE_2$
# <br> $BCE=-y log \hat y - (1-y) log (1 - \hat y)$

# In[ ]:


classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Fit the classifier to the data

# Finally fit the classifier to training data

# In[ ]:


classifier.fit(X_train, y_train, batch_size = 1, epochs = 100, verbose=1)


# ### Evaluate model

# We cannot complete any AI algorithm or model without assesing it so we now measure the accuracy of our model.<br>
# This happens to be a classification model so we can simply use the accuracy formula:<br>
# $Accuracy=\frac{True Positive + True Negative}{Total}$

# First make a y predictions list for all entries in x test list as we have the probabilities and not 1 or 0 corresponding to `M` and `B`
# <br>
# We now receive the probabilities, This is the pseudo code used:
# * if the `prob >= 0.5`
# * we classify it as `1`
# * else `0`
# 

# In[ ]:


y_pred = classifier.predict(X_test)
# If probab is >= 0.5 classify as 1 or 0
y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]


# Now we build the confuson matrix for easy interpretation of model accuracy. A confusion matrix is very helpful in interpreting our model results in this manner.<br><br>
# ![](https://drive.google.com/uc?id=1SaflBpLkDz753uijkjzGK70KpMCmp520)

# In[ ]:


# Finally use scikit-learn to build a confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Create a heat map of the confusion matrix

# In[ ]:


sns.heatmap(cm,annot=True)


# Now print out the accuracy<br> $Accuracy=\frac{True Positive + True Negative}{Total}$

# In[ ]:


# (True positive + True Negative)/Total
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: "+ str(accuracy*100)+"%")


# Create a visualization of our ANN to see its layers at a glance

# In[ ]:


plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# So, now you can see a png file of the model architecture created as `model_plot.png`
