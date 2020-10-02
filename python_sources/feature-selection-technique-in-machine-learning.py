#!/usr/bin/env python
# coding: utf-8

# # Feature Selection
# Feature Selection is one of the most import technique for a great predictive model. It help us to know the most important features of the data set.

# **I will cover the below points :**
# 1. What is Feature Selection?
# 2. Why it is one the most important techinque to learn for a Data Scientitst?
# 3. What are the different type of Feature Selection?

# **1.  Feature Selection: **The process of selecting subset of relevant features for use in model construction which will help to increase the model prediction and decrease the error rate. 
# In other word you can say its a  process of identifying and removing as much of  irrelevant and redundent information as possible.
# 

# **2. Importance of Feature Selection: **
# * Improve the accuracy of model.
# * Reduce overfitting.
# * Shoter traning time.
# * Reduce complexity of model.
# 

# **3. Type of Feature Selection** <br/>
# *         ***Wrapper Method***
# *         ***Filter Method***
# *         ***Embedded Method***
# 

# # Wrapper Method
# In this method a subset of features are selected and train a model using them. Based on the inference that we draw from the previous model, we decide to add or remove features from subset.
# [For indepth details](https://en.wikipedia.org/wiki/Feature_selection)

# **Image from wiki**
# <p><img src="https://upload.wikimedia.org/wikipedia/commons/0/04/Feature_selection_Wrapper_Method.png" alt="Feature selection Wrapper Method.png" height="179" width="640"><br></p>

# **Type of Wrapper Method**
# * Forward Selection
# * Backward Elimination
# * Exhaustive Feature Selection 

# **Forward Selection : ** It is a iterative method in which we keep adding feature which  best improves our model till an addition of a new feature does not improve the model performance.<br/><br/>
# **Backward Elimination : ** In this we start with all features and removes the least significant feature at each iteration which improves the model performance. We repeat this until no improvemnt is observed on removal of feature.<br><br>
# **Exhaustive Feature Selection : ** In this the best subset of feature is selected, over all possible feature subsets. For example, if a dataset contains 4 features, the algorithm will evaluate all the feature combinations as follows:
# * All possible combinations of 1  feature
# * All possible combinations of 2 features
# * All possible combinations of 3 features
# * All possible combinations of 4 features
#             

# **Pros**<br>
# * Aim to find the best possible feature combintaion.
# * Better result then filter method.
# * Can we used for small dataset having less features.

# **Cons**
# * Computationally expensive
# * Often impracticable for large dataset having more features.

# **For detail please checkout [Wrapper Feature Selection Example](https://www.kaggle.com/raviprakash438/wrapper-method-feature-selection)**

# # Filter Method

# Filter methods are generally used as a preprocessing step. The selection of features is independent of any machine learning algorithms. Instead, features are selected on the basis of their scores in various statistical tests for their correlation with the outcome variable.

# **Image from wiki**
# <p><img src="https://upload.wikimedia.org/wikipedia/commons/2/2c/Filter_Methode.png" alt="Filter Methode.png" height="63" width="640"></a></p>

# **Basic Methods**<br>
# We should consider the below filter methods as a data pre processing steps.
# * Constant features - Constant features are those that show the same value for all the observations of the dataset. Remove constant features from dataset.
# * Quasi-constant features  - The column which contain 99% of same data is called Quasi constant column. Remove Quasi constant features from dataset.
# * Duplicated features - Remove duplicated features from dataset.

# **Correlation**
# * Correlation is measure of the linear relationship of 2 or more variables.
# * Through correlation we can predict one variable from other.
#     * Good variables are highly correlated with the target but uncorrelated among themselves.
# * If two variables are highly correlated with each other, then we should remove one of them.   
#   

# **Fisher Score**
# * Measures the dependence of 2 variables
# * Suited for categorical variables.
# * Target should be binary.
# * Variable values should be non negative, typically Boolean or counts.
# 

# **ANOVA (Analysis Of Variance)**
# * Measures the dependency of two variables.
# * Suited for continuous variables.
# * Requires a binary target.
# * Assumes linear relationship between variable and target.
# * Assumes variables are normally distributed.
# * Sensitive to sample size
# 

# **ROC-AUC / RMSE**
# * Measures the dependency of two variables.
# * Suited for all type of variables.
# * Makes no assumption on the distribution of the variables.

# **Steps to  select features**
# * Rank features according to a certain criteria (like correlation).
#     * Each feature is ranked independently of the feature space.
# * Select highest ranking features.    

# **Pros**
# * Fast computation.
# * Simple yet powerful to quickly remove irrelevant and redundant feature.
# * Better choice for large dataset over wrapper methods.

# **Cons**
# * It may select redundant variables because they do not consider the relationships between features.
# * The prediction accuracy is lesser than wrapper methods.

# **For detail please checkout [Filter Method Feature Selection Examples](https://www.kaggle.com/raviprakash438/filter-method-feature-selection)**

# # Embedded Method
# Embedded method combine the features of Filter and Wrapper methods. A learning algorithm takes advantage of its own variable selection process and performs feature selection and classification simultaneously.

# **Image from wiki**
# <p><img src="https://upload.wikimedia.org/wikipedia/commons/b/bf/Feature_selection_Embedded_Method.png" alt="Feature selection Embedded Method.png" height="190" width="640"></p>

# ### REGULARISATION
# Regularization consists in adding a penalty on the different parameters of the model to reduce the freedom of the model. Hence, the model will be less likely to fit the noise of the training data and will improve the generalization abilities of the model. For linear models there are in general 3 types of regularisation:
# * The L1 regularization (also called Lasso)
# * The L2 regularization (also called Ridge)
# * The L1/L2 regularization (also called Elastic net)

# **Image from Scikit learn**
# <p><img src="http://scikit-learn.org/stable/_images/sphx_glr_plot_sgd_penalties_001.png"></p>

# **For detail please checkout [Lasso and Ridge Regularisation](https://www.kaggle.com/raviprakash438/lasso-and-ridge-regularisation)**

# ***Please share your comments,likes or dislikes so that I can improve the post.***

# In[ ]:




