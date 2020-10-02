#!/usr/bin/env python
# coding: utf-8

# # Allstate Claims Severity
# 
# ## Task
# 
# ## Evaluation
# 
# > Submissions are evaluated on the [mean absolute error (MAE)](https://www.kaggle.com/wiki/MeanAbsoluteError) between the predicted loss and the actual loss. ([Source: Kaggle.com](https://www.kaggle.com/c/allstate-claims-severity/details/evaluation)).
# 
# # General Notes
# 
# **Visualize everything you can!** We are much better in interpreting and analyzing visual representations of data than looking at the bare numbers. However, make sure to pick visualizations, scales and colors that are suited for your particular question.
# 
# **Look at the Kernels and Forums on Kaggle!** Kaggle is a platform for learning and sharing ideas in the domain of data mining, data processing and machine learning. Many users share their work and ideas, so please use these resources. Also don't be shy to post questions and share your approaches and ideas in the Forum if you are not in the Top 10%. It's all about learning!
# 
# **Always cite (link) your sources!** Many of the ideas that you implement are not yours, and others would love to know where you took them from.
# 
# You can also use our [gitter channel](https://gitter.im/ViennaKaggle/allstate-claims-severity) to discuss your approach. 

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 


# # Competition Datasets
# 
# Each competition provides at least 2 datasets for a competition - a training set and a testing set.
# 
# ## Kaggle Training Set
# 
# The training set contains `n` input variables (number of columns - 2) for `p` observation (number of rows) including an index column and a ground truth column. We will refer to this set as *Kaggle Training Set* to avoid confusion during Cross Validation. It can be loaded and inspected using the `pandas` library.
# 
#     train_data = pd.read_csv('../input/train.csv')
# 
# ## Kaggle Testing Set
# 
# The testing set contains `n` input variables (number of columns - 1) for `q` observation (number of rows) including an index column (also called *ID*) and **without** a ground truth column. The task of the Kaggle participant is to predict this missing column. We will refer to this set as *Kaggle Testing Set* to avoid confusion during Cross Validation.
# 
#     test_data = pd.read_csv('../input/test.csv')
# 
# ## Additional Sets
# 
# Some competitions offer additional data that can/should be joined with the training data. However, this competition doesn't provide additional training sets.
# 
# # Submission
# 
# In the end, a Kaggle participant should upload a CSV file containing `q` predictions (number of rows - one for each observation in the testing set) with an index column and one (or multiple) prediction column(s). The prediction dimensions depends on the competition and task (e.g. classification vs. regression).

# In[ ]:


# Let's load both sets
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Let's load the sample submission
submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


# Let's take a look at the first 5 entries of the Kaggle training set
train_data.head(5)


# In[ ]:


print("Number of observations: %i" % len(train_data))
print("List of columns: %s" % ", ".join(train_data.columns))


# We observe that the Kaggle training set contains 188,318 observations with 132 columns
# 
#  - 1 index column **id**
#  - 130 feature columns **cat{0-99}**, **cont{0-99}**, etc.
#  - 1 ground truth column **loss**

# In[ ]:


# Let's take a look at the first 5 entries of the Kaggle testing set
test_data.head(5)


# In[ ]:


print("Number of observations: %i" % len(test_data))
print("List of columns: %s" % ", ".join(test_data.columns))


# We observe that the Kaggle training set contains 125,546 observations with 131 columns
# 
#  - 1 index column **id**
#  - 130 feature columns **cat{0-99}**, **cont{0-99}**, etc.

# In[ ]:


submission.head(5)


# In[ ]:


print("Number of observations: %i" % len(submission))
print("List of columns: %s" % ", ".join(submission.columns))


# The sample submission should contain 125,546 rows with 2 columns
# 
#  - 1 index column **id**
#  - 1 prediction column **loss**
# 
# where **id** should correspond with the index of the observation in the Kaggle testing set and **loss** should be predicted by your model.

# # Analyzing the Dataset
# 
# 

# **Task A >** What do the columns mean (age, hometown, packages, price, etc.)? -> you can find this by looking at the values, distributions, or feature importance (you can use this as an order for analyzing the columns)
# 
# **Task B >** Can we use external data? If yes, which datasets can help us (demographics, BIP, number of hospitals, money spent in health)?
# 
# **Task C >** What does the value distributions tell us? What could be useful bins for the features? Are there log dependencies? how many unique values? how often do they occur? Are there outliers? Correlations between columns? Columns with unique values? Spikes?
# 
# **Task D >** Are there missing values? how many are there, and in which columns do they occur? What could be a reason that the values are missing -> how can we replace them? 0, Min, Max, Mean, Median, etc.

# In[ ]:


train_data.describe()


# # Preparing the Dataset
# 
# As we have seen in the previous part, the dataset contains numerical and categorical values. Most Machine Learning algorithms can only work with numeric values as they are computing distances. Keep in mind that whenever you compute distances over multiple features you should normalize the features. The goal of this part is to clean the dataset and convert it to something that the Machine Learning algorithms can use.
# 
# ## Notes
# 
# Be careful when mixing pandas, numpy, and XGB matrices. We usually add a **_df** suffix when dealing with pandas dataframes and a **xgdmat_** prefix when dealing with XGB matrices.
# 
# When you apply a transformation (e.g. Log-Transformation) to the ground truth, you need to apply as well an inverse transform to the result of your prediction (we will do this in the end).

# In[ ]:


from sklearn import preprocessing

label_encoders = {}
category_labels = {}

def transform_x(data_df, phase="train"):
    """Transforms the input dataframe to a dataframe containing
    the input variables (= features)"""
    X = data_df.drop(['id'], axis=1)
    
    if 'loss' in X.columns:
        X = X.drop(['loss'], axis=1)
    
    # List of categorical features
    cat_features = X.select_dtypes(include=['object']).columns

    # List of numerical features
    num_features = X.select_dtypes(exclude=['object']).columns
    
    # Replace each categorical feature with encoded labels
    for cat in cat_features:
        if phase == "train":
            # Let's store the used labels
            category_labels[cat] = list(set(X[cat]))     
  
            # We need to fit the Label Encoder in the training phase
            label_encoders[cat] = preprocessing.LabelEncoder()
            label_encoders[cat].fit(X[cat])
        
        # We replace unseen labels by the first label
        mask = X[cat].apply(lambda x: x not in category_labels[cat])
        X.loc[mask, cat] = category_labels[cat][0]
        
        X[cat] = label_encoders[cat].transform(X[cat])
    
    return X

def transform_y(data_df):
    """Transforms the input dataframe to a dataframe containing
    the ground truth data"""
    y = data_df['loss']
    
    # You can do some crazy stuff here
    # y = np.log(y)
    
    return y

def inverse_transform_y(data):
    """Inverse transforms the y values to match the original
    Kaggle testing set"""
    y = data
    
    # You should invert all the crazy stuff
    # y = np.exp(y)
    
    return y


# In[ ]:


X_train_df = transform_x(train_data)
y_train_df = transform_y(train_data)


# # Cross-Validation
# 
# Cross-Validation is a technique used in parameter optimization to avoid overfitting (having a low training error and a high tesing error). Make sure you use the right evaluation metric for your problem.

# **Task E >** Split the Kaggle training set into a training and validation set. The training set should be used for paramter tuning using Cross Validation and the results should be verified on the validation set. The goal is to generate a validation set that gives similar results as the public/private Kaggle testing set.

# In[ ]:





# # Regression using XGB

# In[ ]:


import xgboost as xgb

# Create our DMatrix to make XGBoost more efficient
xgdmat_train = xgb.DMatrix(X_train_df.values, y_train_df.values)

params = {'eta': 0.01, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5, 
             'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3} 

num_rounds = 100
mdl = xgb.train(params, xgdmat_train, num_boost_round=num_rounds)


# In[ ]:


X_test_df = transform_x(test_data, phase="test")


# In[ ]:


xgdmat_test = xgb.DMatrix(X_test_df.values)
y_pred = mdl.predict(xgdmat_test)


# In[ ]:


submission.iloc[:, 1] = inverse_transform_y(y_pred)
submission.to_csv('vienna_kaggle_submission.csv', index=None)


# # References
# 
#  - https://www.kaggle.com/guyko81/allstate-claims-severity/just-an-easy-solution
