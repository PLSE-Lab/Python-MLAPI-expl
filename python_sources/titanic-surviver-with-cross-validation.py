#!/usr/bin/env python
# coding: utf-8

# # 1. Load Dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## os.getwcd() --> '/kaggle/working'

# In[ ]:


train = pd.read_csv("../input/titanic/train.csv", index_col = "PassengerId")
print(train.shape)
train.head


# In[ ]:


test = pd.read_csv("../input/titanic/test.csv", index_col = "PassengerId")
print(test.shape)
test.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt


# ## 1st column : Sex

# In[ ]:


# 1. by using plot
sns.countplot(data = train, x = "Sex", hue = "Survived")


# In[ ]:


# 2. by using pivot
pd.pivot_table(train, index = "Sex", values = "Survived")


# * The data visualization is not very specific (ex: what percentage of male passengers are likely to survive?) But is very intuitive because it shows a picture.
# 
# 
# * On the other hand, the pivot table shows a specific figure (ex: 18.9% chance of a male passenger surviving), but it's hard to intuitively understand what that figure means. (ex: So how many times different is the survival rate of female passengers compared to males?)

# ## 2nd column : Pclass
# 
# * The room class is divided into 1 class (= first class), 2 class (= business) and 3 class (= economy).

# In[ ]:


sns.countplot(data = train, x = "Pclass", hue = "Survived")


# * The higher the Pclass, the higher the chance of survival.

# In[ ]:


pd.pivot_table(train, index = "Pclass", values = "Survived")


# ## 3rd column : Embarked
# * There are three types of docks: 1) Cherbourg (C) 2) Queenstown (Q) and 3) Southampton (S).

# In[ ]:


sns.countplot(data = train, x = "Embarked", hue = "Survived")


# * The analysis shows that the more likely to survive in Sherbourg, C, the more likely to die in Southampton, S. 
# 
# 
# 
# 1. To be more specific,The largest number of passengers are in Southampton, but many are killed. Nearly twice as many survivors appear to die.
# 
# 
# 2. The number of passengers in Cherbourg is relatively small compared to Southampton, but the number of survivors is higher than that of the dead.
# 
# 
# 3. People boarding in Queenstown have a slightly higher chance of dying, but initially have fewer passengers.

# In[ ]:


pd.pivot_table(train, index = "Embarked", values = "Survived")


# ## 4th column : Age & Fare
# 

# In[ ]:


sns.lmplot(data = train, x = "Age", y = "Fare", hue = "Survived", fit_reg = False)


# * Outlier Detection -> Delete Data

# In[ ]:


# delete three outlier data
low_fare = train[train["Fare"] < 500]
train.shape, low_fare.shape


# In[ ]:


sns.lmplot(data = low_fare, x = "Age", y = "Fare", hue = "Survived", fit_reg = False)


# In[ ]:


low_fare = train[train["Fare"] < 100]
sns.lmplot(data = low_fare, x = "Age", y = "Fare", hue = "Survived", fit_reg = False)


# * A closer look at the results shows that 1) the age is under 15 years old, and 2) passengers paying less than $20 for the fare have a relatively high survival rate.

# # 2. Preprocessing

# * The basic conditions for putting data in the machine learning algorithm provided by scikit-learn are:
#         1. All data should consist of numbers (integer, decimal, etc.).
#         
#         2. There should be no empty values in the data.

# ## 1) Encode Sex

# In[ ]:


train.loc[train["Sex"] == "male", "Sex_encode"] = 0
train.loc[train["Sex"] == "female", "Sex_encode"] = 1
train.head()


# In[ ]:


test.loc[test["Sex"] == "male", "Sex_encode"] = 0
test.loc[test["Sex"] == "female", "Sex_encode"] = 1
test.head()


#  * ### Fill in missing fare
# 

# In[ ]:


train[train["Fare"].isnull()]


# In[ ]:


test[test["Fare"].isnull()]


# In[ ]:


train["Fare_fillin"] = train["Fare"]
train[["Fare", "Fare_fillin"]].head()


# In[ ]:


test["Fare_fillin"] = test["Fare"]
test[["Fare", "Fare_fillin"]].head()


# In[ ]:


test.loc[test["Fare"].isnull(), "Fare_fillin"] = 0

test.loc[test["Fare"].isnull(), ["Fare", "Fare_fillin"]]


# ## 2) Encode Embarked
# 
# * C == 0
# * S == 1
# * Q == 2       ---> One Hot Encoding

# In[ ]:


train["Embarked_C"] = train["Embarked"] == "C"
train["Embarked_S"] = train["Embarked"] == "S"
train["Embarked_Q"] = train["Embarked"] == "Q"
train[["Embarked_C", "Embarked_S", "Embarked_Q"]].head()


# In[ ]:


test["Embarked_C"] = test["Embarked"] == "C"
test["Embarked_S"] = test["Embarked"] == "S"
test["Embarked_Q"] = test["Embarked"] == "Q"
test[["Embarked_C", "Embarked_S", "Embarked_Q"]].head()


# # 3. Train (by using Decision Tree)
# 
# * Feature : Values that help you to get label
#         1) Ticket Class (Pclass), 2) Gender (Sex_encode), 3) Fare_fillin, and 4) Embarked.
# 
# * Label : Survived or not

# In[ ]:


feature_names = ["Pclass", "Sex_encode", "Fare_fillin", "Embarked_C", "Embarked_S", "Embarked_Q"]
feature_names


# In[ ]:


label_name = "Survived"


# In[ ]:


X_train = train[feature_names]
print(X_train.shape)
X_train.head()


# In[ ]:


X_test = test[feature_names]
print(X_test.shape)
X_test.head()


# In[ ]:


y_train = train[label_name]
print(y_train.shape)
y_train.head()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth = 5)
model


# ### Training by using DecisionTreeClassifier

# In[ ]:


model.fit(X_train, y_train)


# ### Visualize
# #### When the learning finished, we can visualize it to see how well it learned.

# In[ ]:


import graphviz
from sklearn.tree import export_graphviz

dot_tree = export_graphviz(model,
                           feature_names = feature_names,
                           class_names = ["Perish", "Survived"],
                           out_file = None)


graphviz.Source(dot_tree)


# # 4. Predict
# * If the Decision Tree has been trained successfully, all that remains is to use this Decision Tree to predict the survival / death of the passengers in the test data.

# In[ ]:


predictions = model.predict(X_test)
print(predictions.shape)
predictions[0:10]


# # 5. Submit

# In[ ]:


submission = pd.read_csv("../input/titanic/gender_submission.csv", index_col = "PassengerId")
print(submission.shape)
submission.head()


# In[ ]:


submission["Survived"] = predictions
submission.head()


# In[ ]:


# submission.to_csv("./decision_tree.csv")


# # Cross validation

# In[ ]:


from sklearn.utils.testing import all_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
warnings.warn("FutureWarning")


# In[ ]:


allAlgorithms = all_estimators(type_filter = "classifier")
allAlgorithms
kfold_cv = KFold(n_splits = 5, shuffle = True)
result = []
for (name, algorithm) in allAlgorithms:
    if(name == 'CheckingClassifier' or name == 'ClassifierChain' or 
       name == 'MultiOutputClassifier' or name == 'OneVsOneClassifier' or 
       name =='OneVsRestClassifier' or name == 'OutputCodeClassifier' or
       name =='VotingClassifier' or name == 'RadiusNeighborsClassifier'): continue
        
    model = algorithm()
    if hasattr(model, "score"):
        scores = cross_val_score(model, X_train, y_train, cv = kfold_cv)
        result.append({"name": name, "mean": np.mean(scores)})

result


# In[ ]:


import operator
sorted(result, key = operator.itemgetter("mean", "name"))[-5:]


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
bestModel = GradientBoostingClassifier(max_depth = 5)
bestModel.fit(X_train, y_train)


# In[ ]:


bestPrediction = bestModel.predict(X_test)
print(bestPrediction.shape)

bestSubmission = pd.read_csv("../input/titanic/gender_submission.csv", index_col = "PassengerId")
bestSubmission["Survived"] = bestPrediction
bestSubmission.to_csv("./GradientBoostingClassifier.csv")

