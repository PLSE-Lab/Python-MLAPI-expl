#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Breast Cancer Diagnosis Prediction using H2O</font></center></h1>
# 
# 
# <img src="https://kaggle2.blob.core.windows.net/datasets-images/180/384/3da2510581f9d3b902307ff8d06fe327/dataset-card.jpg" width="400"></img>
# 
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Load packages</a>  
# - <a href='#3'>Read the data</a>  
# - <a href='#4'>Check the data</a>  
# - <a href='#5'>Data exploration</a>
# - <a href='#6'>Predictive model</a>  
#     - <a href='#61'>Split the data</a> 
#     - <a href='#62'>Train  GBM</a>   
#     - <a href='#63'>Model evaluation</a>  
#     - <a href='#64'>Prediction</a>     
# - <a href='#8'>References</a>
# 

# # <a id="1">Introduction</a>  
# 
# ## The dataset
# 
# The **Breast Cancer (Wisconsin) Diagnosis dataset** <a href='#8'>[1]</a> contains the diagnosis and a set of 30  features describing the characteristics of the cell nuclei present in the digitized image of a of a fine needle aspirate (FNA) of a breast mass.
# Ten real-valued features are computed for each cell nucleus:  
# + **radius** (mean of distances from center to points on the perimeter);  
# + **texture** (standard deviation of gray-scale values);  
# + **perimeter**;  
# + **area**;  
# + **smoothness** (local variation in radius lengths);  
# + **compactness** (perimeter^2 / area - 1.0);  
# + **concavity** (severity of concave portions of the contour);  
# + **concave points** (number of concave portions of the contour);  
# + **symmetry**;  
# + **fractal dimension** ("coastline approximation" - 1).
# 
# The **mean**, standard error (**SE**) and "**worst**" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features.
# 
# 
# ## H2O  
# 
# H2O is a Java-based software for data modeling and general computing. Primary purpose of H2O is as a distributed (many machines), parallel (many CPUs), in memory (several hundred GBs Xmx) processing engine. It has both Python and R interfaces <a href='#8'>[2]</a>.  
# 
# ## Analysis
# 
# We will analyze the features to understand the predictive value for diagnosis. We will then create models using two different algorithms and use the models to predict the diagnosis.
# 
# 

# # <a id="2">Load packages</a>  
# 
# We load the packages we will use in the analysis.
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import time
import itertools
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
get_ipython().run_line_magic('matplotlib', 'inline')


# # <a id="3">Read the data</a>  
# 
# For reading the data, we will use also H2O. First, we will initialize H2O.
# 
# ## Initialize H2O
# 
# H2O will first try to connect to an existing instance. If none available, will start one. Then informations about this engine are printed.  At the end connection to the H2O server is attempted and reported.

# In[ ]:


h2o.init()


# More information are presented: the H2O cluster uptime, timezone, version, version age, cluster name, hardware resources allocated ( number of nodes, memory, cores), the connection url, H2O API extensions exposed and the Python version used.
# 
# ## Import the data
# 
# We already initialized the H2O engine, now we will use H2O to import the data.

# In[ ]:


data_df = h2o.import_file("../input/data.csv", destination_frame="data_df")


# # <a id="4">Check the data</a>  
# 
# 
# We use also H2O function **describe** to check the data. 

# In[ ]:


data_df.describe()


# There are 569 rows and 33 columns in the data. For each column, the following informations are shown:  
# 
# + type;  
# + min;  
# + mean;  
# + max;  
# + standard deviation (sigma);  
# + number of zeros (zero);  
# + number of missing values (missing);  
# + a certain number of selected values (first 10);  
# 
# Notes: Calling **describe()** function this way is equivalent with calling **summary()**.   
# We can call describe with a parameter different from 0. In this case, more information about the type of chunk compression data and frame distribution, besides the data description, is given.
# 

# In[ ]:


data_df.describe(1)


# # <a id="5">Explore the data</a>  
# 
# We will use another functions from H2O to explore the data.
# 
# Let's start by showing the distribution of features, grouped by **diagnosis**, which is the **target** value.
# 
# We start by looking how many cases are with **diagnosis** of each type (malignant (M) or benign (B)).

# In[ ]:


df_group=data_df.group_by("diagnosis").count()
df_group.get_frame()


# In[ ]:


features = [f for f in data_df.columns if f not in ['id', 'diagnosis', 'C33']]

i = 0
t0 = data_df[data_df['diagnosis'] == 'M'].as_data_frame()
t1 = data_df[data_df['diagnosis'] == 'B'].as_data_frame()

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(6,5,figsize=(16,24))

for feature in features:
    i += 1
    plt.subplot(6,5,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Malignant")
    sns.kdeplot(t1[feature], bw=0.5,label="Benign")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
    


# Some of the features show good separation in terms of density plots for the subset with **malignant (M)** diagnosis and the subset with **benign (B)** diagnosis, for example:   
# 
# * radius_mean;  
# * texture_mean;  
# * perimeter_mean;  
# * area_mean;  
# * radius_worst;  
# * texture_worst;  
# * perimeter_worst;  
# * area_worst;  
# Some features show perfect identity of the density plots grouped by diagnosis, as following:  
# * compactness_se;  
# * concavity_se;  
# * concave_points_se;  
# * simmetry_se;  
# * smoothness_se;  
# 
# Let's represent the correlation between the features, excluding id, C33 and diagnosis:  
# 

# In[ ]:


plt.figure(figsize=(16,16))
corr = data_df[features].cor().as_data_frame()
corr.index = features
sns.heatmap(corr, annot = True, cmap='YlGnBu', linecolor="white", vmin=-1, vmax=1, cbar_kws={"orientation": "horizontal"})
plt.title("Correlation Heatmap for the features (excluding id, C33 & diagnosis)", fontsize=14)
plt.show()


# Some of the features are strongly correlated , as following:  
# 
# * radius_mean with perimeter_mean;  
# * radius_mean with texture_mean;  
# * perimeter_worst with radius_worst;  
# * perimeter_worst with area_worst;  
# * area_se with perimeter_se;  
# 

# # <a id="6">Predictive model</a>   
# 
# 

# # <a id="61">Split the data</a> 
# 
# Let's start by spliting the data in train, validation and test sets. We will use 60%, 20% and 20% splits.

# In[ ]:


train_df, valid_df, test_df = data_df.split_frame(ratios=[0.6,0.2], seed=2018)
target = "diagnosis"
train_df[target] = train_df[target].asfactor()
valid_df[target] = valid_df[target].asfactor()
test_df[target] = test_df[target].asfactor()
print("Number of rows in train, valid and test set : ", train_df.shape[0], valid_df.shape[0], test_df.shape[0])


# ## <a id="62">Train  GBM</a> 
# 
# We will use a GBM model.

# In[ ]:


# define the predictor list - it will be the same as the features analyzed previously
predictors = features
# initialize the H2O GBM 
gbm = H2OGradientBoostingEstimator()
# train with the initialized model
gbm.train(x=predictors, y=target, training_frame=train_df)


# 
# ## <a id="63">Model evaluation</a> 
# 
# 
# Let's inspect the model already trained. We can print the summary:

# In[ ]:


gbm.summary()


# This shows that we used 50 trees, 50 internal trees. It is also showing the min and max tree depth (4,5), the min and max number of leaves (7,14) and the mean values for tree depth and number of leaves.
# 
# We can also inspect the model further, looking to other informations.
# 
# Let's see the model performance for the train set.

# In[ ]:


print(gbm.model_performance(train_df))


# We can see that the AUC is 1 for the train set and Gini coeff is 1 as well. LogLoss is 0.01.
# 
# Let's see the model performance for the validation set.

# In[ ]:


print(gbm.model_performance(valid_df))


# We can see that the AUC is 0.9987 for validation set and Gini coeff is 0.997. LogLoss is 0.05.
# 
# Confusion matrix show that only one value in the validation set was wrongly predicted.  
# 
# With such good results in the validation set, we will not need to further tune the model.  We can now try and predict the test set values.
# 
# Let's also show the variable importance plot for the model.

# In[ ]:


gbm.varimp_plot()


# The most important features are perimeter_worst, concave_points_mean, radius_worst, concave_points_worst.
# 
# Let's now use the model for prediction.
# 
# ## <a id="64">Predict</a>   
# 
# The prediction 

# In[ ]:


pred_val = list(gbm.predict(test_df[predictors])[0])
true_val = list(test_df[target])


# 

# # <a id="8">References</a>
# 
# [1] Breast Cancer Wisconsin (Diagnostic) Data Set, https://www.kaggle.com/uciml/breast-cancer-wisconsin-data  
# [2] SRK, Getting started with H2O,  https://www.kaggle.com/sudalairajkumar/getting-started-with-h2o
