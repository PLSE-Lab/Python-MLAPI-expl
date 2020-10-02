#!/usr/bin/env python
# coding: utf-8

# **Objectives of this kernal
# 
# Learning Feature Engineering for classification . 
# Learning use of Decision Tree. 
# Learning use of Random Forests
# **
# 

# 

# 

# # 1. Import

# In[ ]:


#Importing Libraries for data analysis

# Call data manipulation libraries
import pandas as pd
import numpy as np

# Feature creation libraries
from sklearn.random_projection import SparseRandomProjection as sr  # Projection features
from sklearn.cluster import KMeans                    # Cluster features
from sklearn.preprocessing import PolynomialFeatures  # Interaction features

# For feature selection
# Ref: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif  # Selection criteria

# Data processing
#  Scaling data in various manner
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
# Transform categorical (integer) to dummy
from sklearn.preprocessing import OneHotEncoder

# Splitting data
from sklearn.model_selection import train_test_split

# Decision tree modeling
from sklearn.tree import  DecisionTreeClassifier as dt

# RandomForest modeling
from sklearn.ensemble import RandomForestClassifier as rf

# Plotting libraries to plot feature importance
import matplotlib.pyplot as plt
import seaborn as sns

# Misc
import os, time, gc


# # 2.0 Set working directory and read file

# In[ ]:




# Read train/test files
heart = pd.read_csv("../input/heart.csv")

heart.head(5)
heart.shape         ## 303 x 14


# In[ ]:


#  Split into Test and Training Data
X_train, X_test, y_train, y_test = train_test_split(
        heart.drop('target', 1), 
        heart['target'], 
        test_size = 0.3, 
        random_state=10
        ) 


# In[ ]:


# Look at data
X_train.head(2)
X_train.shape                        # 212 x 13
X_test.shape                         # 91 x 13

y_test.shape                        # 91 x 
y_train.shape                       # 212 x

# Data types
X_train.dtypes.value_counts()   # All afeatures re integers 


# In[ ]:


# Target classes are almost balanced
heart.target.value_counts()


# In[ ]:


# Check if there are Missing values? None
X_train.isnull().sum().sum()  # 0
X_test.isnull().sum().sum()   # 0


# # 3. Feature Engineering #

# In[ ]:





# #  Using Statistical Numbers 
# 

# In[ ]:


#  Feature 1: Row sums of features  More successful
#                when data is binary.

X_train['sum'] = X_train.sum(numeric_only = True, axis=1)  # numeric_only= None is default
X_test['sum'] = X_test.sum(numeric_only = True,axis=1)


# In[ ]:


# Assume that value of '0' in a cell implies missing feature
#     Transform train and test dataframes
#     replacing '0' with NaN
#     Use pd.replace()
tmp_train = X_train.replace(0, np.nan)
tmp_test = X_test.replace(0,np.nan)


# In[ ]:


#  Check if tmp_train is same as train or is a view
#     of train? That is check if tmp_train is a deep-copy

tmp_train is X_train                # False
tmp_train._is_view                # False


# In[ ]:


# Check if 0 has been replaced by NaN
tmp_train.head(1)
tmp_test.head(1)


# In[ ]:


# Feature 2 : For every row, how many features exist
#                that is are non-zero/not NaN.
#                Use pd.notna()
tmp_train.notna().head(1)
X_train["count_not0"] = tmp_train.notna().sum(axis = 1)
X_test['count_not0'] = tmp_test.notna().sum(axis = 1)


# In[ ]:


# Similary create other statistical features
#    Feature 3

feat = [ "var", "median", "mean", "std", "max", "min"]
for i in feat:
    X_train[i] = tmp_train.aggregate(i,  axis =1)
    X_test[i]  = tmp_test.aggregate(i,axis = 1)


# In[ ]:


# Delete not needed variables and release memory
del(tmp_train)
del(tmp_test)
gc.collect()


# In[ ]:


# So what do we have finally
X_train.shape                
X_train.head(1)
X_test.shape                 
X_test.head(2)


# In[ ]:


# Before we proceed further, keep target feature separately
target = y_train
target.tail(2)


# In[ ]:


# Store column names of our data somewhere
#     We will need these later (at the end of this code)
colNames = X_train.columns.values
colNames


# # Feature creation Using Random Projections 

# In[ ]:



# Random projection is a fast dimensionality reduction feature
#     Also used to look at the structure of data


# In[ ]:


#  Generate features using random projections
#     First stack train and test data, one upon another
tmp = pd.concat([X_train,X_test],
                axis = 0,            # Stack one upon another (rbind)
                ignore_index = True
                )


# In[ ]:



tmp.shape     # 303 X 21


# In[ ]:


# Transform tmp t0 numpy array

tmp = tmp.values
tmp.shape       # 303 X 21


# In[ ]:


#  Let us create 8 random projections/columns
NUM_OF_COM = 8


# In[ ]:


#  Create an instance of class
rp_instance = sr(n_components = NUM_OF_COM)


# In[ ]:


# fit and transform the (original) dataset
#      Random Projections with desired number
#      of components are returned
rp = rp_instance.fit_transform(tmp[:, :13])


# In[ ]:


#  Look at some features
rp[: 5, :  3]


# In[ ]:


#  Create some column names for these columns
#      We will use them at the end of this code
rp_col_names = ["r" + str(i) for i in range(8)]
rp_col_names


# # Feature creation using kmeans 
# 
# 

# In[ ]:


# Before clustering, scale data
#  Create a StandardScaler instance
se = StandardScaler()
# fit() and transform() in one step
tmp = se.fit_transform(tmp)
# 
tmp.shape               


# In[ ]:


#  Perform kmeans using 13 features.
#     No of centroids is no of classes in the 'target'
centers = target.nunique()    
centers               


# In[ ]:


# Begin clustering
start = time.time()

# First create object to perform clustering
kmeans = KMeans(n_clusters=centers,
                n_jobs = 4)         

# Next train the model on the original data only
kmeans.fit(tmp[:, : 13])

end = time.time()
(end-start)/60.0      


# In[ ]:


# Get clusterlabel for each row (data-point)
kmeans.labels_
kmeans.labels_.size 


# In[ ]:


# Cluster labels are categorical. So convert them to dummy


# In[ ]:


#  Create an instance of OneHotEncoder class
ohe = OneHotEncoder(sparse = False)


# In[ ]:


# Use ohe to learn data
#      ohe.fit(kmeans.labels_)
ohe.fit(kmeans.labels_.reshape(-1,1))     
                                          


# In[ ]:


# Transform data now
dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))
dummy_clusterlabels
dummy_clusterlabels.shape    


# In[ ]:


# We will use the following as names of new two columns
#      We need them at the end of this code

k_means_names = ["k" + str(i) for i in range(2)]
k_means_names


# # Creating Interaction features 
# # Using Polynomials

# In[ ]:



#  Will require lots of memory if we take large number of features
#     Best strategy is to consider only impt features

degree = 2
poly = PolynomialFeatures(degree,                 # Degree 2
                          interaction_only=True,  # Avoid e.g. square(a)
                          include_bias = False    # No constant term
                          )


# In[ ]:


# Consider only first 8 features
#      fit and transform
df =  poly.fit_transform(tmp[:, : 8])


df.shape     # 303 X 36


# In[ ]:


#  Generate some names for these 36 columns
poly_names = [ "poly" + str(i)  for i in range(36)]
poly_names


# # concatenate all features  

# In[ ]:




# Append now all generated features together
# Append random projections, kmeans and polynomial features to tmp array

tmp.shape       


# In[ ]:


#   If variable, 'dummy_clusterlabels', exists, stack kmeans generated
#       columns also else not. 'vars()'' is an inbuilt function in python.
#       All python variables are contained in vars().

if ('dummy_clusterlabels' in vars()):               #
    tmp = np.hstack([tmp,rp,dummy_clusterlabels, df])
else:
    tmp = np.hstack([tmp,rp, df])       


tmp.shape          


# In[ ]:


# Combine train and test into X and y to split compatible datasets
X = tmp
X.shape     


# In[ ]:


# Combine y_train and y_test into y to split into compatible datasets later
y = pd.concat([y_train,y_test],
                axis = 0,            
                ignore_index = True
                )
y.shape        


# In[ ]:


# Delete tmp - as a good programming practice
del tmp
gc.collect()


# # Model building 
# 

# In[ ]:


# Split the feature engineered data into new training and test dataset
X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.3)


# ## Final data for model creation and testing
# ### X_train: Training Data with new features
# ### y_train: expected output for training data
# 
# ### X_test: test data with new features
# ### y_test: expected output for test data

# In[ ]:


# 
X_train.shape   


# In[ ]:


X_test.shape    


# In[ ]:


# Decision tree classification
# Create an instance of class
clf1_dt = dt(min_samples_split = 5,
         min_samples_leaf= 3
        )


# In[ ]:


start = time.time()
# Fit/train the object on training data
#      Build model
clf1_dt = clf1_dt.fit(X_train, y_train)
end = time.time()
(end-start)/60                     


# In[ ]:


#  Use model to make predictions
classes1_dt = clf1_dt.predict(X_test)


# In[ ]:


#  Check accuracy
(classes1_dt == y_test).sum()/y_test.size      


# 

# In[ ]:


#  Instantiate RandomForest classifier
clf1_rf = rf(n_estimators=50)


# In[ ]:


# Fit/train the object on training data
#      Build model

start = time.time()
clf1_rf = clf1_rf.fit(X_train, y_train)
end = time.time()
(end-start)/60                    


# In[ ]:


# Use model to make predictions
classes1_rf = clf1_rf.predict(X_test)


# In[ ]:


#  Check accuracy
(classes1_rf == y_test).sum()/y_test.size      


# 

# # Feature selection 

#  ****************************************
# ## Using feature importance given by model
# ****************************************
# 

# In[ ]:


#  Get feature importance
clf1_rf.feature_importances_        
clf1_rf.feature_importances_.size   


# In[ ]:


# To our list of column names, append all other col names
#      generated by random projection, kmeans (onehotencoding)
#      and polynomial features
#      But first check if kmeans was used to generate features

if ('dummy_clusterlabels' in vars()):       
    colNames = list(colNames) + rp_col_names+ k_means_names + poly_names
else:
    colNames = colNames = list(colNames) + rp_col_names +  poly_names      


# In[ ]:


# So how many columns?
len(colNames)           


# In[ ]:


#  Create a dataframe of feature importance and corresponding
#      column names. Sort dataframe by importance of feature
feat_imp = pd.DataFrame({
                   "importance": clf1_rf.feature_importances_ ,
                   "featureNames" : colNames
                  }
                 ).sort_values(by = "importance", ascending=False)


# In[ ]:


feat_imp.shape                  
feat_imp.head(13)


# ## Notice the number of derived parameters making into the top important features
# *******************

# In[ ]:


# Plot feature importance for first 20 features
g = sns.barplot(x = feat_imp.iloc[  : 20 ,  1] , y = feat_imp.iloc[ : 20, 0])
g.set_xticklabels(g.get_xticklabels(),rotation=90)


# ## Notice the K-means parameters... they're found to be most relevant!!
# ***********

# In[ ]:


# Select top 13 columns and get their indexes
#      Note that in the selected list few kmeans
#      columns also exist
newindex = feat_imp.index.values[:13]
newindex


# In[ ]:


# Use these top 13 columns for classification
#   Create DTree classifier object
clf2_dt = dt(min_samples_split = 5, min_samples_leaf= 3)


# In[ ]:


# Train the object on data
start = time.time()
clf2_dt = clf2_dt.fit(X_train[: , newindex], y_train)
end = time.time()
(end-start)/60                     


# In[ ]:


#  Make prediction
classes2_dt = clf2_dt.predict(X_test[: , newindex])


# In[ ]:


#  Accuracy?
(classes2_dt == y_test).sum()/y_test.size 


# 

# In[ ]:


# Create RForest classifier object
clf2_rf = rf(n_estimators=500)


# In[ ]:


# Traion the object on data
start = time.time()
clf2_rf = clf2_rf.fit(X_train[: , newindex], y_train)
end = time.time()
(end-start)/60                     


# In[ ]:


# Make prediction
classes2_rf = clf2_rf.predict(X_test[: , newindex])


# In[ ]:


# Accuracy?
(classes2_rf == y_test).sum()/y_test.size  


# 

# 

# ## Using top 16 features for classification as per newindex2

# In[ ]:


# Select top  columns and get their indexes
#      Note that in the selected list few kmeans
#      columns also exist
newindex2 = feat_imp.index.values[:10]
newindex2


# In[ ]:


#   Create DTree classifier object
clf3_dt = dt(min_samples_split = 5, min_samples_leaf= 3)


# In[ ]:


# Train the object on data
start = time.time()
clf3_dt = clf3_dt.fit(X_train[: , newindex2], y_train)
end = time.time()
(end-start)/60                     


# In[ ]:


#  Make prediction
classes3_dt = clf3_dt.predict(X_test[: , newindex2])


# In[ ]:


# Accuracy?
(classes3_dt == y_test).sum()/y_test.size


# 

# In[ ]:


#  Create RForest classifier object
# increasing the number of estimators to 300 from 50...
clf3_rf = rf(n_estimators=500)


# In[ ]:


# Train the object on data
start = time.time()
clf3_rf = clf3_rf.fit(X_train[: , newindex2], y_train)
end = time.time()
(end-start)/60                     


# In[ ]:


#  Make prediction
classes3_rf = clf3_rf.predict(X_test[: , newindex2])


# In[ ]:


#  Accuracy?
(classes3_rf == y_test).sum()/y_test.size 


# 

# 

# 

# In[ ]:




