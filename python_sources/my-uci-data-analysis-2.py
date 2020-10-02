#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reset', '-f')
# 1.1 Call data manipulation libraries
import pandas as pd
import numpy as np

# 1.2 Feature creation libraries
from sklearn.random_projection import SparseRandomProjection as sr  # Projection features
from sklearn.cluster import KMeans                    # Cluster features
from sklearn.preprocessing import PolynomialFeatures  # Interaction features

# 1.3 For feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif  # Selection criteria

# 1.4 Data processing
# 1.4.1 Scaling data in various manner
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
# 1.4.2 Transform categorical (integer) to dummy
from sklearn.preprocessing import OneHotEncoder

# 1.5 Splitting data
from sklearn.model_selection import train_test_split

# 1.6 Decision tree modeling
from sklearn.tree import  DecisionTreeClassifier as dt

# 1.7 RandomForest modeling
from sklearn.ensemble import RandomForestClassifier as rf

# 1.8 Plotting libraries to plot feature importance
import matplotlib.pyplot as plt
import seaborn as sns

# 1.9 Misc
import os, time, gc


# In[ ]:



data = pd.read_csv("../input/heart.csv")
data.head()
data.shape  # . (303, 14)


# In[ ]:


# Split the CSV in training and test data
X_train, X_test, y_train, y_test = train_test_split(
        data.drop('target', 1), 
        data['target'], 
        test_size = 0.3, 
        random_state=10
        ) 


# In[ ]:


# 2.2 Look at data
X_train.head(5)
X_train.shape  # (212, 13)                  
X_test.shape   # (91,)         
# 2.3 Data types
X_train.dtypes.value_counts() 
# int64      12
# float64     1
# dtype: int64


# In[ ]:


# 2.4 Target classes are almost balanc
data.target.value_counts()

# check if any row has all zeroes or nulls
x = np.sum(data, axis = 1)
x


# In[ ]:


# 3 Check if there are Missing values? None
data.isnull().sum().sum()  # 0
X_test.isnull().sum().sum()   # 0


# In[ ]:


############################ BB. Feature Engineering #########################

## i)   Shooting in dark. These features may help or may not help
## ii)  There is no theory as to which features will help
## iii) Fastknn is another method not discussed here

############################################################################
############################ Using Statistical Numbers #####################


#  4. Feature 1: Row sums of features 1:93. More successful
#                when data is binary.

X_train['sum'] = X_train.sum(numeric_only = True, axis=1)  # numeric_only= None is default
X_test['sum'] = X_test.sum(numeric_only = True,axis=1)


# In[ ]:



# 4.1 Assume that value of '0' in a cell implies missing feature
#     Transform train and test dataframes
#     replacing '0' with NaN
#     Use pd.replace()
tmp_train = X_train.replace(0, np.nan)
tmp_test = X_test.replace(0,np.nan)


# In[ ]:


# 4.2 Check if tmp_train is same as train or is a view
#     of train? That is check if tmp_train is a deep-copy

tmp_train is X_train                # False
#tmp_train is train.values.base    # False
tmp_train._is_view                # False


# In[ ]:


# 4.3 Check if 0 has been replaced by NaN
tmp_train.head(1)
tmp_test.head(1)


# In[ ]:


# 5. Feature 2 : For every row, how many features exist
#                that is are non-zero/not NaN.
#                Use pd.notna()
tmp_train.notna().head(1)
X_train["count_not0"] = tmp_train.notna().sum(axis = 1)
X_test['count_not0'] = tmp_test.notna().sum(axis = 1)


# In[ ]:



# 6. Similary create other statistical features
#    Feature 3
#    Pandas has a number of statistical functions
#    Ref: https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#computations-descriptive-stats

feat = [ "var", "median", "mean", "std", "max", "min"]
for i in feat:
    X_train[i] = tmp_train.aggregate(i,  axis =1)
    X_test[i]  = tmp_test.aggregate(i,axis = 1)


# In[ ]:




# 7 Delete not needed variables and release memory
del(tmp_train)
del(tmp_test)
gc.collect()


# 7.1 So what do we have finally
X_train.shape                # 61878 X (1+ 93 + 8) ; 93rd Index is target
X_train.head(1)
X_test.shape                 # 144367 X (93 + 8)
X_test.head(2)


# In[ ]:



# 8. Before we proceed further, keep target feature separately
target = y_train
target.tail(2)


# In[ ]:


# 9.1 And then drop 'target' column from train
#      'test' dataset does not have 'target' col
#X_train.drop(columns = ['target'], inplace = True)
X_train.shape                # 61878 X 101


# 9.2. Store column names of our data somewhere
#     We will need these later (at the end of this code)
colNames = X_train.columns.values
colNames


# In[ ]:


###########################################################################
################ Feature creation Using Random Projections ##################
# 10. Random projection is a fast dimensionality reduction feature
#     Also used to look at the structure of data

# 11. Generate features using random projections
#     First stack train and test data, one upon another
tmp = pd.concat([X_train,X_test],
                axis = 0,            # Stack one upon another (rbind)
                ignore_index = True
                )

tmp.shape


# In[ ]:




# 12.2 Transform tmp t0 numpy array
#      Henceforth we will work with array only
tmp = tmp.values
tmp.shape       # (303, 21)


# In[ ]:


# 13. Let us create 10 random projections/columns
#     This decision, at present, is arbitrary
NUM_OF_COM = 8

# 13.1 Create an instance of class
rp_instance = sr(n_components = NUM_OF_COM)

# 13.2 fit and transform the (original) dataset
#      Random Projections with desired number
#      of components are returned
rp = rp_instance.fit_transform(tmp[:, :13])

# 13.3 Look at some features
rp[: 5, :  3]
np.shape

# 13.4 Create some column names for these columns
#      We will use them at the end of this code
rp_col_names = ["r" + str(i) for i in range(8)]
rp_col_names


# In[ ]:


rp = rp_instance.fit_transform(tmp[:, :93])
rp


# In[ ]:


###############################################################################
############################ Feature creation using kmeans ####################
######################Can be skipped without loss of continuity################


# 14. Before clustering, scale data
# 15.1 Create a StandardScaler instance
se = StandardScaler()
# 15.2 fit() and transform() in one step
tmp = se.fit_transform(tmp)
# 15.3
tmp.shape               # 303 X 21 (an ndarray)


# In[ ]:


tmp[: 5, :  3]

# 16. Perform kmeans using 93 features.
#     No of centroids is no of classes in the 'target'
centers = target.nunique()    # 2 unique classes
centers               # 2


# In[ ]:



# 17.1 Begin clustering
start = time.time()

# 17.2 First create object to perform clustering
kmeans = KMeans(n_clusters=centers, # How many
                n_jobs = 4)         # Parallel jobs for n_init



# 17.3 Next train the model on the original data only
#fit the cluster 
kmeans.fit(tmp[:, : 13])

end = time.time()
(end-start)/60.0      # .39


# In[ ]:


# 18 Get clusterlabel for each row (data-point)
kmeans.labels_
kmeans.labels_.size   # 303



# In[ ]:


# 19. Cluster labels are categorical. So convert them to dummy

# 19.1 Create an instance of OneHotEncoder class
ohe = OneHotEncoder(sparse = False)

# 19.2 Use ohe to learn data
#      ohe.fit(kmeans.labels_)
ohe.fit(kmeans.labels_.reshape(-1,1))     # reshape(-1,1) recommended by fit()
                                          # '-1' is a placeholder for actual
# 19.3 Transform data now
dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))
dummy_clusterlabels
dummy_clusterlabels.shape    # 303, 2


# In[ ]:


# 19.4 We will use the following as names of new nine columns
#      We need them at the end of this code

k_means_names = ["k" + str(i) for i in range(2)]
k_means_names


# In[ ]:


############################ Interaction features #######################
# 21. Will require lots of memory if we take large number of features
#     Best strategy is to consider only impt features

degree = 2
poly = PolynomialFeatures(degree,                 # Degree 2
                          interaction_only=True,  # Avoid e.g. square(a)
                          include_bias = False    # No constant term
                          )


# 21.1 Consider only first 8 features
#      fit and transform
df =  poly.fit_transform(tmp[:, : 8])


df.shape     # 303, 36

poly_names = [ "poly" + str(i)  for i in range(36)]
poly_names


# In[ ]:


################ concatenate all features now ##############################

# 22 Append now all generated features together
# 22 Append random projections, kmeans and polynomial features to tmp array

tmp.shape          # 303 X 21


# In[ ]:


#  22.1 If variable, 'dummy_clusterlabels', exists, stack kmeans generated
#       columns also else not. 'vars()'' is an inbuilt function in python.
#       All python variables are contained in vars().

if ('dummy_clusterlabels' in vars()):               #
    tmp = np.hstack([tmp,rp,dummy_clusterlabels, df])
else:
    tmp = np.hstack([tmp,rp, df])       # No kmeans      <==

    
tmp.shape          # 303X 67  


# In[ ]:


###Data modelling done*********************************************
# 22.1 Separate train and test
X = tmp
X.shape                             # 61878 X 135 if no kmeans: (61878, 126)

# 22.2
y = pd.concat([y_train,y_test],
                axis = 0,            # Stack one upon another (rbind)
                ignore_index = True
                )
y.shape        # 303,

# 22.3 Delete tmp
del tmp
gc.collect()


# In[ ]:


################## Model building #####################


# 23. Split train into training and validation dataset
X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.3)


# In[ ]:



# 23.1
X_train.shape    # 43314 X 135  if no kmeans: (212, 67)
X_test.shape     # 18564 X 135; if no kmeans: (91, 67)


# In[ ]:


# 24 Decision tree classification
# 24.1 Create an instance of class
clf1_dt = dt(min_samples_split = 5,
         min_samples_leaf= 3
        )

start = time.time()
# 24.2 Fit/train the object on training data
#      Build model
clf1_dt = clf1_dt.fit(X_train, y_train)
end = time.time()
(end-start)/60                     
###0.0001556873321533203


# In[ ]:


# 25. Instantiate RandomForest classifier
clf1_rf = rf(n_estimators=50)

# 25.1 Fit/train the object on training data
#      Build model

start = time.time()
clf1_rf = clf1_rf.fit(X_train, y_train)
end = time.time()
(end-start)/60      


# In[ ]:


# 25.2 Use model to make predictions
classes1_rf = clf1_rf.predict(X_test)
# 25.3 Check accuracy
(classes1_rf == y_test).sum()/y_test.size 


# In[ ]:


##****************************************
## Using feature importance given by model
##****************************************

# 26. Get feature importance
# 26. Get feature importance
clf1_rf.feature_importances_        # Column-wise feature importance
clf1_rf.feature_importances_.size   # 67



# In[ ]:


# 26.1 To our list of column names, append all other col names
#      generated by random projection, kmeans (onehotencoding)
#      and polynomial features
#      But first check if kmeans was used to generate features

if ('dummy_clusterlabels' in vars()):       # If dummy_clusterlabels labels are defined
    colNames = list(colNames) + rp_col_names+ k_means_names + poly_names
else:
    colNames = colNames = list(colNames) + rp_col_names +  poly_names      # No kmeans      <==

# 26.1.1 So how many columns?
len(colNames)           # 67 


# In[ ]:


# 26.2 Create a dataframe of feature importance and corresponding
#      column names. Sort dataframe by importance of feature
feat_imp = pd.DataFrame({
                   "importance": clf1_rf.feature_importances_ ,
                   "featureNames" : colNames
                  }
                 ).sort_values(by = "importance", ascending=False)



# In[ ]:


feat_imp.shape              
feat_imp.head(20)



# In[ ]:


# 26.3 Plot feature importance for first 20 features
g = sns.barplot(x = feat_imp.iloc[  : 20 ,  1] , y = feat_imp.iloc[ : 20, 0])
g.set_xticklabels(g.get_xticklabels(),rotation=90)


# In[ ]:


# 27 Select top 13 columns and get their indexes
#      Note that in the selected list few kmeans
#      columns also exist
newindex = feat_imp.index.values[:13]
newindex


# In[ ]:


# 27.1 Use these top 13 columns for classification
# 28.1  Create DTree classifier object
clf2_dt = dt(min_samples_split = 5, min_samples_leaf= 3)

# 27.2 Train the object on data
start = time.time()
clf2_dt = clf2_dt.fit(X_train[: , newindex], y_train)
end = time.time()
(end-start)/60                     


# In[ ]:


# 27.3  Make prediction
classes2_dt = clf2_dt.predict(X_test[: , newindex])

# 27.4 Accuracy?
(classes2_dt == y_test).sum()/y_test.size 


# In[ ]:


# 27x Select top 20 columns and get their indexes
#      Note that in the selected list few kmeans    columns also exist
newindex2 = feat_imp.index.values[:20]
newindex2

# 27.1x Use these top 13 columns for classification
# 28.1x  Create DTree classifier object
clf2_dt = dt(min_samples_split = 8, min_samples_leaf= 2)

# 27.2 Train the object on data
start = time.time()
clf2_dt = clf2_dt.fit(X_train[: , newindex2], y_train)
end = time.time()
(end-start)/60  

# 27.3x  Make prediction
classes2_dt = clf2_dt.predict(X_test[: , newindex2])

# 27.4x Accuracy?
(classes2_dt == y_test).sum()/y_test.size 
##########################################################
#   *Accuracy ~ 78.02 -- increased from previous of 71%  #
##########################################################


# In[ ]:





# In[ ]:





# 

# 

# In[ ]:


data.head()


# In[ ]:


# 25x. Instantiate RandomForest classifier
clf2_rf = rf(n_estimators=100)
## changed estimators from 50 to 100

# 25.1x Fit/train the object on training data
#      Build model

start = time.time()
clf2_rf = clf2_rf.fit(X_train[:,newindex2], y_train)
# newindex2 has top 20 features

end = time.time()
(end-start)/60      

# 25.2x Use model to make predictions
classes2_rf = clf2_rf.predict(X_test[: , newindex2])
# 25.3x Check accuracy
(classes2_rf == y_test).sum()/y_test.size 
##################################################################################
# Accuracy remained almost the same around 84 to 87 percent. No major variation  #
##################################################################################

