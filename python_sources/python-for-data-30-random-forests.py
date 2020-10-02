#!/usr/bin/env python
# coding: utf-8

# # Python for Data 30: Random Forests
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# For the final lesson in this guide, we'll learn about random forest models. As we saw last time, decision trees are a conceptually simple predictive modeling technique, but when you start building deep trees, they become complicated and likely to overfit your training data. In addition, decision trees are constructed in a way such that branch splits are always made on variables that appear to be the most significant first, even if those splits do not lead to optimal outcomes as the tree grows. Random forests are an extension of decision trees that address these shortcomings.

# # Random Forest Basics

# A random forest model is a collection of decision tree models that are combined together to make predictions. When you make a random forest, you have to specify the number of decision trees you want to use to make the model. The random forest algorithm then takes random samples of observations from your training data and builds a decision tree model for each sample. The random samples are typically drawn with replacement, meaning the same observation can be drawn multiple times. The end result is a bunch of decision trees that are created with different groups of data records drawn from the original training data.
# 
# The decision trees in a random forest model are a little different than the standard decision trees we made last time. Instead of growing trees where every single explanatory variable can potentially be used to make a branch at any level in the tree, random forests limit the variables that can be used to make a split in the decision tree to some random subset of the explanatory variables. Limiting the splits in this fashion helps avoid the pitfall of always splitting on the same variables and helps random forests create a wider variety of trees to reduce overfitting.
# 
# Random forests are an example of an ensemble model: a model composed of some combination of several different underlying models. Ensemble models often yields better results than single models because different models may detect different patterns in the data and combining models tends to dull the tendency that complex single models have to overfit the data.

# # Random Forests on the Titanic

# Python's sklearn package offers a random forest model that works much like the decision tree model we used last time. Let's use it to train a random forest model on the Titanic training set:

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


# Load and prepare Titanic data
titanic_train = pd.read_csv("../input/train.csv")    # Read the data

# Impute median Age for NA Age values
new_age_var = np.where(titanic_train["Age"].isnull(), # Logical check
                       28,                       # Value if check is true
                       titanic_train["Age"])     # Value if check is false

titanic_train["Age"] = new_age_var 


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


# In[ ]:


# Set the seed
np.random.seed(12)

# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()

# Convert some variables to numeric
titanic_train["Sex"] = label_encoder.fit_transform(titanic_train["Sex"])

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=1000, # Number of trees
                                  max_features=2,    # Num features considered
                                  oob_score=True)    # Use OOB scoring*

features = ["Sex","Pclass","SibSp","Age","Fare"]

# Train the model
rf_model.fit(X=titanic_train[features],
             y=titanic_train["Survived"])

print("OOB accuracy: ")
print(rf_model.oob_score_)


# Since random forest models involve building trees from random subsets or "bags" of data, model performance can be estimated by making predictions on the out-of-bag (OOB) samples instead of using cross validation. You can use cross validation on random forests, but OOB validation already provides a good estimate of performance and building several random forest models to conduct K-fold cross validation with random forest models can be computationally expensive.
# 
# The random forest classifier assigns an importance value to each feature used in training. Features with higher importance were more influential in creating the model, indicating a stronger association with the response variable. Let's check the feature importance for our random forest model:

# In[ ]:


for feature, imp in zip(features, rf_model.feature_importances_):
    print(feature, imp)


# Feature importance can help identify useful features and eliminate features that don't contribute much to the model.
# 
# As a final exercise, let's use the random forest model to make predictions on the titanic test set and submit them to Kaggle to see how our actual generalization performance compares to the OOB estimate:

# In[ ]:


# Read and prepare test data
titanic_test = pd.read_csv("../input/test.csv")    # Read the data

# Impute median Age for NA Age values
new_age_var = np.where(titanic_test["Age"].isnull(),
                       28,                      
                       titanic_test["Age"])      

titanic_test["Age"] = new_age_var 

# Fill missing Fare with 50
new_fare_var = np.where(titanic_test["Fare"].isnull(),
                       50,                      
                       titanic_test["Fare"])      

titanic_test["Fare"] = new_fare_var 

# Convert some variables to numeric
titanic_test["Sex"] = label_encoder.fit_transform(titanic_test["Sex"])


# In[ ]:


# Make test set predictions
test_preds = rf_model.predict(X = titanic_test[features])

# Create a submission for Kaggle
submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],
                           "Survived":test_preds})

# Save submission to CSV
submission.to_csv("tutorial_randomForest_submission.csv", 
                  index=False)        # Do not save index values


# Upon submission, the random forest model achieves an accuracy score of 0.74641, which is actually worse than the decision tree model and even the simple gender-based model. What gives? Is the model overfitting the training data? Did we choose bad variables and model parameters? Or perhaps our simplistic imputation of filling in missing age data using median ages is hurting our accuracy. Data analyses and predictive models often don't turn out how you expect, but even a "bad" result can give you insight into your problem and help you improve your analysis or model in a future iteration.

# # Python for Data Analysis Conclusion

# In this introduction to Python for data analysis series, we built up slowly from the most basic rudiments of the Python language to building predictive models that you can apply to real-world data. Although Python is a beginner-friendly programming language, it was not built specifically for data analysis, so we relied heavily upon libraries to extend base Python's functionality when doing data analysis. As a series focused on practical tools and geared toward beginners, we didn't always take the time to dig deep into the details of the language or the statistical and predictive models we covered. My hope is that some of the lessons in this guide piqued your interest and equipped you with the tools you need to dig deeper on your own.
# 
# If you're interested in learning more about Python for data science, your best bet is to use what you've learned to tackle a project that interests you. An easy way to get started is to use the Kaggle kernel platform we've used in this guide to tackle a Kaggle competition or analyze an existing Kaggle dataset. Reading guides, books and taking online courses is a good way to build your foundational knowledge, but jumping in and doing projects--and learning what you need to know along the way--is probably the best way to learn.
# 
# One of the hardest parts of learning a new skill is getting started. If any part of this guide helped you get started, it has served its purpose.
# 
# *Final Note: If you are interested in learning R, I have a [30-part introduction to R](https://www.kaggle.com/hamelg/intro-to-r-index) guide that covers most of the same topics as this Python guide and recreates many of the same examples in R.*

# ## The End!
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
