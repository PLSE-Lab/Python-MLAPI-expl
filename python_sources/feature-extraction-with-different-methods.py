#!/usr/bin/env python
# coding: utf-8

# # Feature Extraction with Different Methods

# The idea of this work is to show the use of different techniques for extraction of characteristics in a given database. The methods used are shown below:
# 
# <ul>
#     <li><a href='#univariate_'>Univariate Feature Selection</a></li>
#     <li><a href='#rfe_'>Recursive Feature Elimination</a></li>
#     <li><a href='#rfecv_'>Recursive Feature Elimination with Cross-Validation</a></li>
#     <li><a href='#tree_'>Tree based feature selection</a></li>
#     <li><a href='#pca_'>Feature Extraction through PCA</a></li>
# </ul>
# 
# Applying techniques shown above, we will test the effectiveness through a Random Forest classifier.

# # 1. Let's Start

# First we need to load all libraries we will use in this work.

# In[ ]:


import pandas as pd # To handle the data set.
import seaborn as sb # To display visualizations.
import matplotlib.pyplot as plt # To plot
import numpy as np

from sklearn.model_selection import train_test_split # To split data
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.metrics import confusion_matrix # To calculate the confusion matrix
from sklearn.metrics import accuracy_score # To calculate the score
from sklearn.feature_selection import SelectKBest # Univariate Feature Selection
from sklearn.feature_selection import chi2 # To apply Univariate Feature Selection
from sklearn.feature_selection import RFE # Recursive Feature Selection
from sklearn.feature_selection import RFECV # Recursive Feature Selection with Cross Validation
from sklearn.decomposition import PCA # To apply PCA
from sklearn import preprocessing # To get MinMax Scaler function

# To plot inline
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1.1 Loading and Preparing DataSet
# 
# We need to load the dataset. In this case we are going to use the dataset provided for Kaggle Contest in: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package.

# In[ ]:


# Loading file and dropping some columns (the justification is shown in my latest kernel 
# https://www.kaggle.com/ferneutron/classification-and-data-visualization

australia = pd.read_csv('../input/weatherAUS.csv') 
australia = australia.drop(['Location','Date','Evaporation','Sunshine', 'Cloud9am','Cloud3pm',
                           'WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am',
                           'WindSpeed3pm'], axis=1)


# In[ ]:


# Splitting between X and Y vector wich means the corpus and target vector respectively
Y = australia.RainTomorrow
X = australia.drop(['RainTomorrow'], axis=1)


# In[ ]:


# Switching 'Yes' and 'No' with a boolen value and handling NaN values, in this case replacing it with a zero
X = X.replace({'No':0, 'Yes':1})
X = X.fillna(0)
Y = Y.replace({'No':0, 'Yes':1})
Y = Y.fillna(0)


# ## 1.2 Scaling Data
# 
# Working with values in a wide range is not convenient, we need to scale it. 
# In this case we are going to normalize it and scaling it in a 0-1 range.

# In[ ]:


# Initializing the MinMaxScaler function
min_max_scaler = preprocessing.MinMaxScaler()


# In[ ]:


# Scaling dataset keeping the columns name
X_scaled = pd.DataFrame(min_max_scaler.fit_transform(X), columns = X.columns)
X_scaled.head()


# ## 1.3 Splitting up Data
# 
# We have scaling the values in the corpus <i>X</i>, now we need to separate it in train and test set.

# In[ ]:


# Splitting  up data, seting 75% for train and 25% for test.
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=43)


# <a id='univariate_'></a>
# # 2. Univariate Feature Selection
# 
# This method works by selection the <i>K</i> beast features acording to a score. The <i>K</i> number of features
# is setting explicity.

# In[ ]:


# Initialize SelectKBest function
UnivariateFeatureSelection = SelectKBest(chi2, k=5).fit(x_train, y_train)


# In[ ]:


# Creating a dict to visualize which features were selected with the highest score
diccionario = {key:value for (key, value) in zip(UnivariateFeatureSelection.scores_, x_train.columns)}
sorted(diccionario.items())


# As we can see, the last five elements have the highest score. So the best features are:
# 
# <ul>
#     <li>1. RainToday</li>
#     <li>2. RISK_MM</li>
#     <li>3. Humidity3pm</li>
#     <li>4. Rainfall</li>
#     <li>5. Humidity9am</li>
# </ul>
# 
# Now that we have the best features, let's extract them from the original data set and let's measure the performance 
# with the random forest algorithm.

# ## 2.1 Extracting the best <i>K</i> values

# In[ ]:


# Using the 'UnivariateFeatureSelection' based on 'SelectKBest' function,
# let's extract the best features from the original dataset

x_train_k_best = UnivariateFeatureSelection.transform(x_train)
x_test_k_best = UnivariateFeatureSelection.transform(x_test)


# In[ ]:


print("Shape of original data: ", x_train.shape)
print("Shape of corpus with best features: ", x_train_k_best.shape)


# ## 2.2 Testing with Random Forest Algorithm

# In[ ]:


# Initializing and fitting data to the random forest classifier
RandForest_K_best = RandomForestClassifier()      
RandForest_K_best = RandForest_K_best.fit(x_train_k_best, y_train)


# In[ ]:


# Making a prediction and calculting the accuracy
y_pred = RandForest_K_best.predict(x_test_k_best)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ',accuracy)


# In[ ]:


# Showing performance with a confusion matrix
confMatrix = confusion_matrix(y_test, y_pred)
sb.heatmap(confMatrix, annot=True, fmt="d")


# <a id='rfe'></a>
# # 3. Recursive Feature Elimination
# 
# The idea of this method is to make use of an estimator (in this case we are using random forest), and test with
# different sizes of features until find the best set of features.

# In[ ]:


# Initializing Random Forest Classifier
RandForest_RFE = RandomForestClassifier() 
# Initializing the RFE object, one of the most important arguments is the estimator, in this case is RandomForest
rfe = RFE(estimator=RandForest_RFE, n_features_to_select=5, step=1)
# Fit
rfe = rfe.fit(x_train, y_train)


# In[ ]:


print("Best features chosen by RFE: \n")
for i in x_train.columns[rfe.support_]:
    print(i)


# ## 3.1 Testing with Random Forest Algorithm

# In[ ]:


# Generating x_train and x_test based on the best features given by RFE
x_train_RFE = rfe.transform(x_train)
x_test_RFE = rfe.transform(x_test)


# In[ ]:


# Fitting the Random Forest
RandForest_RFE = RandForest_RFE.fit(x_train_RFE, y_train)


# In[ ]:


# Making a prediction and calculting the accuracy
y_pred = RandForest_RFE.predict(x_test_RFE)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ',accuracy)


# In[ ]:


# Showing performance with a confusion matrix
confMatrix = confusion_matrix(y_test, y_pred)
sb.heatmap(confMatrix, annot=True, fmt="d")


# <a id='rfecv'></a>
# # 4. Recursive Feature Elimination with Cross-Validation
# 
# This method is an extention of Recursive Feature Elimination showed above. In this method we have to 
# set the number of k-fold cross validation, basically takes the subset of the traing set and measure the
# performance recurively with respect to the number of features.

# In[ ]:


# Initialize the Random Forest Classifier
RandForest_RFECV = RandomForestClassifier() 
# Initialize the RFECV function setting 3-fold cross validation
rfecv = RFECV(estimator=RandForest_RFECV, step=1, cv=3, scoring='accuracy')
# Fit data
rfecv = rfecv.fit(x_train, y_train)

print('Best number of features :', rfecv.n_features_)
print('Features :\n')
for i in x_train.columns[rfecv.support_]:
    print(i)


# In[ ]:


# Plotting the best features with respect to the Cross Validation Score
plt.figure()
plt.xlabel("Number of Features")
plt.ylabel("Score of Selected Features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In this case we can visualize that the best number of features is 2.

# <a id='tree'></a>
# # 5. Tree based Feature Selection
# 
# This method is to compute the relevance of each feature in the dataset.

# In[ ]:


# Initialize the Random Forest Classifier
RandForest_Tree = RandomForestClassifier()  
# Fit the random forest with the original data
RandForest_Tree = RandForest_Tree.fit(x_train, y_train)
# Getting the relevance between features
relevants = RandForest_Tree.feature_importances_


# In[ ]:


# Apply the tree based on importance for the random forest classifier and indexing it
std = np.std([tree.feature_importances_ for tree in RandForest_Tree.estimators_], axis=0)
indices = np.argsort(relevants)[::-1]


# In[ ]:


# Printting the ranking of importance
print("Feature Rank:")

for i in range(x_train.shape[1]):
    print("%d. Feature %d (%f)" 
          % (i + 1, indices[i], relevants[indices[i]]))


# In[ ]:


# Plotting the feature importances
plt.figure(1, figsize=(9, 8))
plt.title("Feature Importances")
plt.bar(range(x_train.shape[1]), relevants[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()


# <a id='pca'></a>
# # 6. Feature Extraction through PCA
# 
# In some cases it is convenient to apply dimensionality reduction to visualize the number of components
# or elements which could be the best for our model. In this case we apply PCA to discover which ones are
# the features to obtain a acceptable performance in the model.

# In[ ]:


# Initializing PCA and fitting
pca = PCA()
pca.fit(x_train)


# In[ ]:


# Plotting to visualize the best number of elements
plt.figure(1, figsize=(9, 8))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('Number of Feautres')
plt.ylabel('Variance Ratio')


# As we can see, making use of dimensionality reduction we find that the best number of features are in a range of 2 - 4 
# features.

# In[ ]:




