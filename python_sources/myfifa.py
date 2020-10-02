#!/usr/bin/env python
# coding: utf-8

# # **Goal: Classify weather or not a team player wins MotM Award.**

# ![https://imgur.com/a/MJAaE7W](http://)

# **Index:** <br>
# <br>
# **1. Exploratory Data Analysis <br>**
#         a. Feature Engineering <br>
#         b. Graphs <br>
# 
# **2. Data Organising** <br>
#         a. Dealing with Null Values <br>
#         b. Dealing with Categorical Variables <br>
#     
# **3. Feature Importance Calculation** <br>
#         a. XGBoost <br>
#         b. LGBM <br>
#         c. Extra Trees Classifier <br>
#     
# ** 4. Training and Classification** <br>
#         a. Support Vector Machine (RBF Kernel) <br>
#         b. Linear Regression <br>
#         c. Random Forests Classifier <br>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style = "whitegrid")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# # Exploratory Data Analysis

# In[ ]:


data = pd.read_csv("../input/FIFA 2018 Statistics.csv")
data.shape


# In[ ]:


data.head(2)


# > # Feature Engineering

# Creating a feature of goals conceded by a team in a match using goals scored by the opposition

# In[ ]:


gc = np.array([], dtype = 'int')
for i in range(0, 128, 2):
    gc = np.append(gc, data.loc[i+1, "Goal Scored"])
    gc = np.append(gc, data.loc[i, "Goal Scored"])
    
data.insert(4, "Goal Conceded", pd.Series(gc))


# In[ ]:


data.head(2)


# Creating a feature of who won the match using goals scored and goals conceded. <br> 
# Attributes: Won, Lost, Draw (0, 1, 2) to keep no. of categories low 

# In[ ]:


conditions = [(data["Goal Scored"] > data["Goal Conceded"]), (data["Goal Scored"] == data["Goal Conceded"]), (data["Goal Scored"] < data["Goal Conceded"])]
result = np.array([0, 1, 2], dtype = 'int')
data.insert(5, "Result", pd.Series(np.select(conditions, result, default = -1)))


# In[ ]:


data.head(2)


# Creating a feature of total set pieces (free kicks + corners) <br>
# (penalties ignored as information is not provided) 

# In[ ]:


data.insert(15, "Total Set Pieces", pd.Series(data["Corners"] + data["Free Kicks"], dtype = 'int'))


# In[ ]:


data.head(2)


# > # Graphs

# Plot of Ball Possession vs Attempts. 

# In[ ]:


sns.lmplot(x = 'Ball Possession %', y = 'Attempts', data = data)


# In[ ]:


data["Ball Possession %"].corr(data["Attempts"], method = 'pearson')


# It can be observed that there is a linear relation between attempts made and % ball possession. <br>
# Moreover, number of attempts tend to rise with increase in % ball possession. <br>
# Correlation between % Ball Possession and Attempts: 0.54 (good correlation) <br>

# Plot of Distance Covered vs Passes. 

# In[ ]:


sns.lmplot(x = 'Passes', y = 'Distance Covered (Kms)', data = data)


# In[ ]:


data["Passes"].corr(data["Distance Covered (Kms)"], method = 'pearson')


# This is an interesting result. <br>
# I thought there should be a good (inverse) correlation between passes and distance covered but it is very less.  <br>
# Correlation between Passes and Distance Covered: 0.18 <br>

# Comparison between number of set pieces for wins, losses and draws.

# In[ ]:


sns.set_context("paper")
sns.swarmplot( x = 'Result', y = 'Total Set Pieces', data = data)


# This again is an corroborating conclusion. <br>
# It was believed that this was the world cup of set pieces. <br> 
# We can observe that more number of teams tend to win given equal number of set pieces. (0 and 1 on x-axis) <br>

# # Data Organising

# > # Dealing with Null Values

# In[ ]:


data.isna().sum()


# In[ ]:


data[['Own goals', 'Own goal Time']].head()


# As it can be seen, there were 116/2 = 58 matches in which either of the team did not score an own goal. <br>
# Therefore, I replace NaN with 0 in both the columns, as it seems to be the most logical idea.

# In[ ]:


data[['Own goals', 'Own goal Time']] = data[['Own goals', 'Own goal Time']].fillna(0)
data.isna().sum()


# In[ ]:


data["1st Goal"].head()


# These are the matches in which one of the teams were not able to score a single score. <br>
# Here, I replace NaN with full-time minutes as replacing it with 0 will give a different inference. <br>

# In[ ]:


data["1st Goal"] = data["1st Goal"].fillna(90)
data.isna().sum()


# Now there are no Null or NaN values in the dataset.

# > # Dealing with Categorical Variables

# In[ ]:


data.dtypes.unique()


# There are features with Object, Integer and Float datatypes. <br>
# We need to deal with features having Object datatype. <br>

# In[ ]:


cat = data.columns.values[data.dtypes == object]
cat


# Out of these features: <br>
# 1. Date can be dropped. <br>
# 2. Team and Opponent contain names of countries. <br>
# 3. Man of the Match and PSO have values in Yes or No. <br>
# 4. Round has 6 different values. <br>
# 
# I'll deal with Team and Opponent separately.

# In[ ]:


data.drop(["Date"], axis = 1, inplace = True)

temp = np.array(['Man of the Match', 'Round', 'PSO'])
for i in range(0,len(temp)):
    x = temp[i]
    data[x] = data[x].astype('category').cat.codes
    
data.head(2)


# Therefore, Date is dropped. Man of the Match, Round and PSO is converted to categorical variables.

# For Team and Opponent, I'll first get categorical codes for all the countries and then directly assign that codes to Team and Opponent.

# In[ ]:


# I am currently working on the above mentioned approach.
# This is a temporary solution.

data["Team"] = data["Team"].astype('category').cat.codes
data["Opponent"] = data["Opponent"].astype('category').cat.codes


# In[ ]:


data.head()


# # Now our data is clean and ready to train.

# # Feature Importance Calculation

# **I'll calculate feature importance using gradient boosting method.** <br>
# 1. XGBoost Classifier
# 2. Gradient Boosting Machince
# 3. Extra Trees Classifier <br>
# After calculating feature importance using various algorithms, I'll select top 5 features consistent in each algortihm and finally train the actual model.

# In[ ]:


features = data.drop(["Man of the Match"], axis = 1)
target = data["Man of the Match"]


# XGBoost Classifier

# In[ ]:


modelxgb = XGBClassifier()
modelxgb.fit(features, target)

print(modelxgb.feature_importances_)


# Plot of F Scores (Feature Importance Scores) vs Features

# In[ ]:


from xgboost import plot_importance
plot_importance(modelxgb)


# In[ ]:


f_xgb = pd.DataFrame(data = {'feature' : features.columns, 'value' : modelxgb.feature_importances_})
f_xgb = f_xgb.sort_values(['value'], ascending = False)
top10xgb = f_xgb.head(10)
plt.figure(figsize=(15,8))
sns.barplot(x = top10xgb["feature"], y = top10xgb["value"])


# In[ ]:


modellgbm = LGBMClassifier()
modellgbm.fit(features, target)

print(modellgbm.feature_importances_)


# In[ ]:


f_lgbm = pd.DataFrame(data = {'feature' : features.columns, 'value' : modellgbm.feature_importances_})
f_lgbm = f_lgbm.sort_values(['value'], ascending = False)
top10lgbm = f_lgbm.head(10)
plt.figure(figsize=(15,8))
sns.barplot(x = top10lgbm["feature"], y = top10lgbm["value"])


# *Therefore, it can be observed that different algorithms give different Feature Importance Scores.*

# **The 2 new features that were created in the begining are prominent for both, XGBoost and LGBM** <br>
# **Infact, feature *"Result"*(as expected) is the most prominent and feature *"Total Set Pieces"* is also dominating**

# *As I have used 2 boosting algorithms, I'll further use Extra Trees Classifier provided by sci-kit learn to obtain feature importance.*

# In[ ]:


modeletc = ExtraTreesClassifier()
modeletc.fit(features, target)

print(modeletc.feature_importances_)


# In[ ]:


f_etc = pd.DataFrame(data = {'feature' : features.columns, 'value' : modeletc.feature_importances_})
f_etc = f_etc.sort_values(['value'], ascending = False)
top10etc = f_etc.head(10)
plt.figure(figsize=(15,8))
sns.barplot(x = top10etc["feature"], y = top10etc["value"])


# **Again, feature *Result* is the most dominant feature.**

# > # Now, based on these 3 algorithms, I will select top 5 features for training and prediction

# Using intersection, I'll select all the features that are prevalent in all the three models.

# In[ ]:


ft = pd.merge(f_xgb, f_lgbm, how = 'inner', on = ["feature"])
ft = pd.merge(ft, f_etc, how = 'inner', on = ["feature"])


# > Merging all the 3 feature importance dataframes, we get a comprehensive account of important features

# In[ ]:


ft.head(5)


# Therefore, top 5 features are:
# 1. Result
# 2. Off-Target
# 3. 1st Goal
# 4. Corners
# 5. Team
# 
# Now, I'll use these features to train data and predict the target.

# In[ ]:


features = ft["feature"].head(5).values
X = data[features]
Y = target.values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 7)


# **I am experimenting with Simple SVM (RBF Kernel), Linear Regression and Random Forests.**

# # Training and Classification

# > # Support Vector Machine (Radial Basis Function)

# In[ ]:


modelsvc = svm.SVC(kernel = 'rbf', gamma='auto')
modelsvc.fit(X_train, y_train)
y_svc = modelsvc.predict(X_test)
accuracy_score(y_test, y_svc)


# We obtain 66% accuracy using SVM.

# > # Logistic Regression

# In[ ]:


modelreg = linear_model.LogisticRegression()
modelreg.fit(X_train, y_train)
y_reg = modelreg.predict(X_test)
accuracy_score(y_test, y_reg.round(), normalize = False)


# We obtain 34% accuracy using Logistic Regression Model. 

# > # Random Forests Classifier

# In[ ]:


modelrf = RandomForestClassifier(max_depth=2, random_state=0)
modelrf.fit(X_train, y_train)
y_rf = modelrf.predict(X_test)
accuracy_score(y_test, y_rf)


# We obtain 84% accuracy using Random Forests Classifier. 

# ***Therefore, for this case, Random Forests Classifier prove to be the best classifier algorithm.***

# # This is my first attempt to work on a data science and machine learning problem so please share your views and comments regarding this kernel so that I can improve my work. <br>
# # A big thank you for taking out time and efforts to read through the entire kernel.

# In[ ]:




