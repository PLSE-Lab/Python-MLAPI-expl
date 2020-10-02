#!/usr/bin/env python
# coding: utf-8

# This is Part Two of my analysis of the Zomato Bangalore dataset. In Part One we attempted to predict restaurant ratings with six selected features using regression. In this kernel we will keep the same features, but transform the ratings into a categorical target with four levels and build classification models to predict them. 
# 
# Exploratory Data Analysis was done in Part One. This kernel consists of:
# 
# - Data cleaning (identifying and dropping duplicates, reformatting features)
# - Preprocessing and prediction with Decision Tree, Random Forest and XGBoost 
# - Model evaluation (Accuracy, Cohen Kappa, F1 score, Precision, Recall)
# - Feature Importance visualization
# - Results summary
# 
# We will go through the data cleaning and preprocessing quickly - see Part One for explanations.

# In[ ]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

zomato = pd.read_csv("../input/zomato.csv", na_values = ["-", ""])
# Making a copy of the data to work on
data = zomato.copy()


# ## Data Cleaning and Preprocessing

# In[ ]:


# Renaming and removing commas in the cost column 
data = data.rename({"approx_cost(for two people)": "cost"}, axis=1)
data["cost"] = data["cost"].replace(",", "", regex = True)


# In[ ]:


# Converting numeric columns to their appropriate dtypes
data[["votes", "cost"]] = data[["votes", "cost"]].apply(pd.to_numeric)


# In[ ]:


# Group and aggregate duplicate restaurants that are listed under multiple types in listed_in(type)
grouped = data.groupby(["name", "address"]).agg({"listed_in(type)" : list})
newdata = pd.merge(grouped, data, on = (["name", "address"]))


# In[ ]:


# Drop rows which have duplicate information in "name", "address" and "listed_in(type)_x"
newdata["listed_in(type)_x"] = newdata["listed_in(type)_x"].astype(str) # converting unhashable list to a hashable type
newdata.drop_duplicates(subset = ["name", "address", "listed_in(type)_x"], inplace = True)


# In[ ]:


# Converting the restaurant names to rownames 
newdata.index = newdata["name"]


# In[ ]:


# Dropping unnecessary columns
newdata.drop(["name", "url", "phone", "listed_in(city)", "listed_in(type)_x", "address", "dish_liked",  "listed_in(type)_y", "menu_item", "cuisines", "reviews_list"], axis = 1, inplace = True)


# In[ ]:


# Transforming the target (restaurant ratings)

# Extracting the first three characters of each string in "rate"
newdata["rating"] = newdata["rate"].str[:3] 
# Removing rows with "NEW" in ratings as it is not a predictable level
newdata = newdata[newdata.rating != "NEW"] 
# Dropping rows that have missing values in ratings 
newdata = newdata.dropna(subset = ["rating"])
# Converting ratings to a numeric column so we can discretize it
newdata["rating"] = pd.to_numeric(newdata["rating"])


# Our EDA in Part One showed that 3.7 is the most common rating and the frequency of ratings below 2.5 and above 4 is very low. To prevent a class imbalance problem, we will create custom-sized bins that take frequency counts into account while still making sense.
# 
# Our four rating bins (classes) will be 0 to 3 < 3 to 3.5 < 3.5 to 4 < 4 to 5. To make label encoding easier later, we'll label these classes 0, 1, 2, 3. **We can think of these as Very Low, Low, Medium and High.**

# In[ ]:


# Discretizing the ratings into a categorical feature with 4 classes
newdata["rating"] = pd.cut(newdata["rating"], bins = [0, 3.0, 3.5, 4.0, 5.0], labels = ["0", "1", "2", "3"])


# In[ ]:


# Checking the number of restaurants in each rating class
np.unique(newdata["rating"], return_counts = True)


# We have 884 restaurants in rating class 0 (Very Low), 2929 in class 1 (Low), 4037 in class 2 (Medium) and 1466 in class 4 (High).

# In[ ]:


# Dropping the original rating column
newdata.drop("rate", axis = 1, inplace = True)


# In[ ]:


newdata.describe(include = "all")


# In[ ]:


# Separating the predictors and target
predictors = newdata.drop("rating", axis = 1)
target = newdata["rating"]


# In[ ]:


# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(predictors, target, random_state = 0, test_size = 0.3)


# In[ ]:


# Preprocessing the predictors
num_cols = ["votes", "cost"]
cat_cols = ["location", "rest_type", "online_order", "book_table"]

num_imputer = SimpleImputer(strategy = "median")
# Imputing numeric columns with the median (not mean because of the high variance)
num_imputed = num_imputer.fit_transform(X_train[num_cols])
scaler = StandardScaler()
# Scaling the numeric columns to have a mean of 0 and standard deviation of 1
num_preprocessed = pd.DataFrame(scaler.fit_transform(num_imputed), columns = num_cols)

cat_imputer = SimpleImputer( strategy = "most_frequent")
# Imputing categorical columns with the mode
cat_imputed = pd.DataFrame(cat_imputer.fit_transform(X_train[cat_cols]), columns = cat_cols)
# Dummifying the categorical columns
cat_preprocessed = pd.DataFrame(pd.get_dummies(cat_imputed, prefix = cat_cols, drop_first = True))

train_predictors = pd.concat([num_preprocessed, cat_preprocessed], axis=1)


# In[ ]:


test_num_imputed = num_imputer.transform(X_test[num_cols])
test_num_preprocessed = pd.DataFrame(scaler.transform(test_num_imputed), columns = num_cols)

test_cat_imputed = pd.DataFrame(cat_imputer.transform(X_test[cat_cols]), columns = cat_cols)
test_cat_preprocessed = pd.DataFrame(pd.get_dummies(test_cat_imputed, prefix = cat_cols, drop_first = True))
                                    
test_predictors = pd.concat([test_num_preprocessed, test_cat_preprocessed], axis=1)

# Accounting for missing columns in the test set caused by dummification
missing_cols = set(train_predictors) - set(test_predictors)
# Add missing columns to test set with default value equal to 0
for c in missing_cols:
    test_predictors[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test_predictors = test_predictors[train_predictors.columns]


# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(train_predictors, y_train)
pred_train = dt.predict(train_predictors)
pred_test = dt.predict(test_predictors)


# In[ ]:


accuracy_score(y_train, pred_train)


# In[ ]:


accuracy_score(y_test, pred_test)


# ### Observation
# 
# A basic decision tree is overfitting on the train data. Let's see if a Random Forest ensemble model can do better.
# 
# Below are the results of Random Forest after manual hyperparameter tuning.

# In[ ]:


rf = RandomForestClassifier(criterion = "gini", n_estimators = 250, max_depth = 10, 
                            max_features = 50, min_samples_split = 4, random_state = 0)
rf.fit(train_predictors, y_train)
pred_train = rf.predict(train_predictors)
pred_test = rf.predict(test_predictors)


# In[ ]:


accuracy_score(y_train, pred_train)


# In[ ]:


accuracy_score(y_test, pred_test)


# In[ ]:


cohen_kappa_score(y_train, pred_train)


# In[ ]:


cohen_kappa_score(y_test, pred_test)


# In[ ]:


print(classification_report(y_train, pred_train))


# In[ ]:


print(classification_report(y_test, pred_test))


# In[ ]:


# Inspecting class counts in the train predictions
np.unique(pred_train, return_counts = True)


# In[ ]:


# Doing the same for the test predictions
np.unique(pred_test, return_counts = True)


# The RF classifier has not predicted any samples for the minority class (0) in the test data, which means it has not learnt that class. 
# 
# Let's rebuild the Random Forest with a class weight parameter to handle class imbalance.

# In[ ]:


rf = RandomForestClassifier(criterion = "gini", n_estimators = 250, max_depth = 10, 
                            max_features = 50, min_samples_split = 4, random_state = 0,
                           class_weight = "balanced")
rf.fit(train_predictors, y_train)
pred_train = rf.predict(train_predictors)
pred_test = rf.predict(test_predictors)


# In[ ]:


# Inspecting class counts in the train predictions
np.unique(pred_train, return_counts = True)


# In[ ]:


# Doing the same for the test predictions
np.unique(pred_test, return_counts = True)


# The new RF is assigning a **huge** number of samples to the minority class!
# 
# Here are the actual class counts for train and test:

# In[ ]:


np.unique(y_train, return_counts = True)


# In[ ]:


np.unique(y_test, return_counts = True)


# Bagging may not be enough for this classification task. Let's try boosting with XGBoost.

# In[ ]:


# Building an XGBoost classifier
xgb = XGBClassifier(n_estimators = 250, max_depth = 20, gamma = 2, learning_rate = 0.001, random_state = 0)

xgb.fit(train_predictors, y_train)
pred_train = xgb.predict(train_predictors)
pred_test = xgb.predict(test_predictors)


# ## Model evaluation

# In[ ]:


accuracy_score(y_train, pred_train)


# In[ ]:


accuracy_score(y_test, pred_test)


# In[ ]:


cohen_kappa_score(y_train, pred_train)


# In[ ]:


cohen_kappa_score(y_test, pred_test)


# In[ ]:


print(classification_report(y_train, pred_train))


# In[ ]:


print(classification_report(y_test, pred_test))


# In[ ]:


# Inspecting class counts in the train predictions
np.unique(pred_train, return_counts = True)


# In[ ]:


# Doing the same for the test predictions
np.unique(pred_test, return_counts = True)


# Let's see which features our classifier found most important for rating class prediction.

# In[ ]:


# Visualizing a feature importances plot

plt.figure(figsize = (20, 10))
feat_importances = pd.Series(xgb.feature_importances_, index=train_predictors.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# ## Results summary
# 
# In this kernel we split the restaurant ratings into four ranges (classes) and built tree-based classifiers to predict them. **The best performer was a manually tuned XGBoost classifier with 72% accuracy on train and 67% accuracy on test.** But since the rating classes are somewhat imbalanced, we also evaluated our predictions with Cohen Kappa and F1 scores.
# 
# The Cohen Kappa score was 0.56 on train and 0.47 on test. This can be interpreted as the model **performing moderately well** (according to Landis & Koch). Average weighted F1 score was 0.69 and 0.64 on train and test respectively. 
# 
# Random Forest also gave similar scores but the class counts in RF predictions showed that minority classes were being misclassified, i.e. almost all restaurants with "rare" ratings were being misclassified as having a more "common" rating. To counter this, we added the parameter (class_weight = "balanced") to the RF. This made the classifier overcompensate for the minority classes (incorrectly classified too many restaurants into the minority classes) and brought down the scores. 
# 
# XGBoost with manually tuned depth, learning rate and gamma (regularization) hyperparameters predicted more minority samples correctly than RF. **The most important feature for prediction was Votes, followed by rest_type_Dessert_Parlor. **
# 
# In Part Three of my analysis we will apply text mining techniques to extract insights from textual features, like customer reviews, and attempt rating classification with a neural network.
