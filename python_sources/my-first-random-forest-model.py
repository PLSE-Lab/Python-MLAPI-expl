#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# 
# The goal of the 'Learn with other Kaggle Users' competition is to predict what types of trees (Cover_Type) there are in an area based on various geographic features. All the areas are located in the Roosevelt National Forest and each observation is a 30m x 30m patch. The goal is to predict an integer classification for the forest cover type. The seven types are:
# 
# 1. Spruce/Fir
# 2. Lodgepole Pine
# 3. Ponderosa Pine
# 4. Cottonwood/Willow
# 5. Aspen
# 6. Douglas-fir
# 7. Krummholz
# 
# This is a multiclassification problem. To make predictions I use a random forest classifier. In this kernel I do not perform feature selection/engineering to improve the model. However I do perform a grid search to find the optimal parameters for the random classifier.
# 
# Here are the steps I did to make this model:
# 
# * Import data
# * Split the trainset in a train and test set (the testset is only used to make predictions for the competition)
# * Prepare the data using a pipeline
# * Fit random forest model using default options
# * Evaluate results using accuracy score and confusion matrices
# * Search for optimal parameters using grid search
# * Fit the optimized random forest model and evaluate results
# * Predict the Cover_Types in the test set using the final random forest model
# 
# The categorization accuracy score obtained on the test data with the random forest model was 0.76026. For me this is a benchmark which I will hopefully improve using feature selection/engineering. 
# 
# The following book was very usefull for making this notebook: Hands-on Machine Learning with Skikit-Learn and TensorFlow, Aurelie Geron, 2017.

# # 2. Import data and have quick look

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

train = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col = 'Id')
test = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col = 'Id')


# In[ ]:


train.head()


# Each row represents one patch. There are 55 attributes. The last attribute is called Cover_Type, and this is the attribute we have to predict. 

# In[ ]:


train.describe().T


# The first 10 attributes are continuous. The attributes from Wilderness_area1 to Soil_Type40 are binary. Lets have a look at the distribution for each attribute

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

train.loc[:,'Elevation':'Horizontal_Distance_To_Fire_Points'].hist(bins = 50, figsize=(20, 15))
plt.show()


# From the histograms it can be observed that the attributes have different scales. For example the attribute Aspect ranges from 0 to 350 degrees, and the attribute elevation ranges from 1800 to 3800 meters. This problem can be addressed with feature scaling. Many histograms are also tail heavy, they extend much farther to to right of the median than to the left. Later these attributes will be transformed to achieve a more bell-shaped distribution.  

# In[ ]:


train.loc[:,'Wilderness_Area1':'Wilderness_Area4'].hist(bins = 50, figsize=(20, 15))
plt.show()


# The wilderness attributes are binary features, they are coded 0 and 1. A 1 means that the patch is present in that particular wilderness area.  

# In[ ]:


train.loc[:,'Soil_Type1':'Soil_Type40'].hist(bins = 50, figsize=(20, 15))
plt.show()


# Soil_Type attributes are also binary features. In some cases there is no patch with a particular Soil type. This is the case for Soil_Type7 and Soil_Type15. These attributes need to be removed from the data before analysis. 

# # 3. Prepare data
# 
# When preparing the data we start with dividing the dataset in a train and validation set. The train set is used to train the model and the test set is used to evaluate the model. When we have a good model it will be used on the final test set provided by Kaggle and hopefully get us a good score. 

# In[ ]:


from sklearn.model_selection import train_test_split

# Split into validation and training data, set to random_state 1
train_set, test_set = train_test_split(train, test_size = 0.20, random_state = 1)


# First we need to make a target object (which is Cover_Type the attribute we have to predict) and a dataframe with all predictor attributes. 

# In[ ]:


## make training set
# Create target object and call it y
y_train = train_set.Cover_Type
X_train = train_set.drop('Cover_Type', axis = 1)

# make test set
y_test = test_set.Cover_Type
X_test = test_set.drop('Cover_Type', axis = 1)

# make final test set
X_final_test = test.copy()


# For data preprocessing I made a pipeline:

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
    
# make a function to extract columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In the code cell below you can select your favorite columns:

# In[ ]:


# here you can select your favorite numerical features
num_attribs = list(train_set.iloc[:,0:10].columns)

# features that should be removed, they were not present in train set
my_list = ['Soil_Type7', 'Soil_Type15']

# here you can select your favorite binary features
cat_attribs = list(train_set.iloc[:,10:54].columns)
cat_attribs = [e for e in cat_attribs if e not in (my_list)]


# In[ ]:


# make pipeline for numerical features
num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                         ('std_scaler', StandardScaler(),
                        )])

cat_pipeline = Pipeline([('Selector', DataFrameSelector(cat_attribs))])

# combine both pipelines
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list = [('num_pipeline', num_pipeline),
                                                 ('cat_pipeline', cat_pipeline)])


# In[ ]:


# run the pipeline to prepare the train data
X_train_prepared = full_pipeline.fit_transform(X_train)

# run the pipeline to prepare the test data
X_test_prepared = full_pipeline.transform(X_test)

# run the pipeline to prepare the final test data
X_final_test_prepared = full_pipeline.transform(X_final_test)


# In[ ]:


X_train_prepared.shape


# In[ ]:


X_test_prepared.shape


# In[ ]:


X_test_prepared.shape


# # 4 Running the model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import warnings

# to remove warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# fit the model with default options
rf = RandomForestClassifier(random_state = 0)
rf.fit(X_train_prepared, y_train)


# # 5. evaluating the random forest model

# ## 5.1 Accuracy score

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

scores_rf = cross_val_score(rf, X_train_prepared, y_train, cv=10, scoring='accuracy')

# Get the mean accuracy score
print("Average accuracy score random forest model (across experiments):")
print(scores_rf.mean())


# According to the accuracy score, 83 percent is correctly predicted.

# ## 5.2 Confusion matrix

# In[ ]:


# confusion matrix
from sklearn.metrics import confusion_matrix

y_test_predict = rf.predict(X_test_prepared)

# make a confusion matrix
conf_mx_test = confusion_matrix(y_test, y_test_predict)

# make a normalized confusion matrix
row_sums = conf_mx_test.sum(axis = 1, keepdims = True)
norm_conf_mx = conf_mx_test / row_sums


# In[ ]:


import seaborn as sns

f, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

# names for labeling
alpha = ['Spruce/Fir', 'Lodgehole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas/Fir', 'Krummholz']

sns.heatmap(conf_mx_test, annot=True, xticklabels=alpha, yticklabels = alpha, cbar=False, ax=axes[0])

sns.heatmap(norm_conf_mx, annot=True, xticklabels=alpha, yticklabels = alpha, cbar=False, ax=axes[1])


# The first confusion matrix represents the actual numbers and the second confusion matrix represents the error rates. Each row in a confusion matrix represents an actual Cover_Type, while each column represents a predicted Cover_Type. 
# 
# From the plot it can be observed that the model has difficulties distinguising between Spruce/Fir and Lodgehole Pine. It can also be observed that sometimes Douglas/Fir gets confused with Ponderosa Pine.  The model also has some difficulties distinguising between Spruce/Fir and Krummholz.
# 
# Conclusion: for improving the model it is important to find features that distinguish between Spruce/Fir and Lodgehole Pine.

# ## 5.3 Features importance

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf, random_state=1).fit(X_train_prepared, y_train)

eli5.show_weights(perm, feature_names = num_attribs + cat_attribs, top = 60)


# Elevation and horizontal distance to roadways, fire points and hydroloy are the most important features. Features with an permutation score of zero or zero have no impact. 

# # 6. Grid search for optimizing the parameters
# 
# To optimize the random forest classifier model it is needed to find the optimal set of parameters. This process is called [hyperparameter optimization](http://https://en.wikipedia.org/wiki/Hyperparameter_optimization). Here I used grid search to find the set of optimal parameters. 
# 
# The following parameters can be tuned in a random forest classifier:
# 
# * n_estimators = number of trees in the forest
# * max_features = max number of features considered for splitting a node
# * max_depth = max number of levels in each decision tree
# * min_samples_split = minimum number of data points placed in a node before the node is split
# * min_samples_leaf = minimum number of data points allowed in a leaf node
# * bootstrap = method for sampling data points (with or without replacement)
# 
# The param_grid below will tells SciLearn to first evaluate all 4 x 4 = 16 combinations of n_estimators and max_features specified in the first dict, then try all  4 x 4 = 16 combinations in the second dict, but this time bootstrap is set on False instead of true. Thus in total GridSearch will test 16 + 16 = 32 combinations, and it will try each model 5 times (because we are using 5 times cross-validation). Thus in total 32 x 5 = 160 rounds of training. Here I commented the lines of code with #, otherwise it will take a long time to run.

# In[ ]:


#from sklearn.model_selection import GridSearchCV

#param_grid = [
#    {'n_estimators': [150, 200, 250, 300], 'max_features': [4, 8, 16, 32]},
#    {'bootstrap': [False], 'n_estimators':[150, 200, 250, 300], 'max_features':[4, 8, 16, 32]}
#]

#rf_final = RandomForestClassifier()

#grid_search = GridSearchCV(rf_final, param_grid, cv = 5, scoring = 'accuracy')

#grid_search.fit(X_train_prepared, y_train)


# In[ ]:


#grid_search.best_params_

#{'bootstrap': False, 'max_features': 16, 'n_estimators': 300}


# The results are: {'bootstrap': False, 'max_features': 16, 'n_estimators': 300} Use these parameters to build a model with optimized parameters.

# # 7. Make final model and evaluate it in the test set

# In[ ]:


# fit the model with optimized parameters
rf_final = RandomForestClassifier(bootstrap=False, n_estimators = 300, 
                                      max_features = 16, random_state = 0)

#making the model using cross validation
scores_rf = cross_val_score(rf_final, X_train_prepared, y_train, cv=10, scoring='accuracy')

# and get scores
print("Average accuracy score (across experiments):")
print(scores_rf.mean())

# 0.783
# 0.816


# The obtained accuracy score was 0.86, which is an improvement compared to the random forest model with default options (0.83). Lets fit our optimized model and test how accurate it predicts in the Cover_Types in the test set.

# In[ ]:


rf_final.fit(X_train_prepared, y_train)

# make predictions using our model
y_test_predict = rf_final.predict(X_test_prepared)


# In[ ]:


# evaluate the results
from sklearn.metrics import accuracy_score

accuracy_score(y_test_predict, y_test) # 0.850


# In[ ]:


# make a confusion matrix
conf_mx_test = confusion_matrix(y_test, y_test_predict)

# make a normalized confusion matrix
row_sums = conf_mx_test.sum(axis = 1, keepdims = True)
norm_conf_mx = conf_mx_test / row_sums

# visualize confusion matrices
f, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

# names for labeling
alpha = ['Spruce/Fir', 'Lodgehole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas/Fir', 'Krummholz']

sns.heatmap(conf_mx_test, annot=True, xticklabels=alpha, yticklabels = alpha, cbar=False, ax=axes[0])

sns.heatmap(norm_conf_mx, annot=True, xticklabels=alpha, yticklabels = alpha, cbar=False, ax=axes[1])


# Compared to the default model, you see that the optimized model shows some improvement in classification.

# # 8. Make final predictions

# In[ ]:


# make predictions using the model
predictions_test_final = rf_final.predict(X_final_test_prepared)

# Save test predictions to file
output = pd.DataFrame({'ID': test.index,
                       'Cover_Type': predictions_test_final})

output.to_csv('submission.csv', index=False)

