#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Importing libraries](#1)
# 1. [Review and Preparation of Data](#2)
# 1. [Handle Missing Values and Columns Types](#3)
# 1. [Analysis of Numerical Variables with Descriptive Statistics](#4)
# 1. [Analysis of Categorical Variables with Descriptive Statistics](#5)
# 1. [Explanation of the Relationship Between Target (Survived) and Other Variables](#6)
# 1. [Model Preparation](#7)
# 1. [Modelling and Feature Importances](#8)
# 1. [Model Evaluation](#9)
# 1. [Parameter Tuning for Best Modelling](#10)

# ## 1. Importing libraries <a class="anchor" id="1"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


import numpy as np 
import pandas as pd

# Graphs
import seaborn as sns
import matplotlib.pyplot as plt

# Modelling Libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Model Evaluation
from sklearn.model_selection import cross_val_score,GridSearchCV

# Other
import warnings
warnings.filterwarnings('ignore')


# ## 2. Review and Preparation of Data <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# All data importing from csv convert to pandas DataFrame object

# In[ ]:


df_train = pd.read_csv("../input/titanic/train.csv")
df_test = pd.read_csv("../input/titanic/test.csv")
df_gender = pd.read_csv("../input/titanic/gender_submission.csv")
print(f"Train data consist of {df_train.shape[0]} rows and {df_train.shape[1]} columns      \nTest data consist of {df_test.shape[0]} rows and {df_test.shape[1]} columns      \nGender submission data consist of {df_gender.shape[0]} rows and {df_gender.shape[1]} columns")


# Sample train data review

# In[ ]:


df_train.sample(10)


# ## 3. Handle Missing Values and Columns Types <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


def missingValuesAndTaypes(data):
    df = pd.DataFrame(data.isnull().sum(),columns=["Count of Missing Values"])
    df["Rate of Missing Values"] = df["Count of Missing Values"] / data.shape[0]
    df["Types"] = [data[i].dtypes for i in data.columns]
    return df.sort_values(by="Count of Missing Values",ascending=False)

missingValuesAndTaypes(df_train)


# Values of Cabin that 77 percent of them are missing. Therefore, this column is removed from the data.

# In[ ]:


if "Cabin" in df_train.columns:
    df_train = df_train.drop(columns="Cabin")
    df_c = pd.DataFrame({"Columns":df_train.columns})
else:
    df_c = pd.DataFrame({"Columns":df_train.columns})
df_c


# Since only 2 values are missing in the variable named X, we can delete the lines belonging to this value.

# In[ ]:


df_train = df_train[df_train.Embarked.isnull() != True]
df_train['Age'] = df_train.groupby(["Sex", "Pclass"])["Age"].apply(lambda x: x.fillna(x.median()))
missingValuesAndTaypes(df_train)


# ## 4. Analysis of Numerical Variables with Descriptive Statistics <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


pd.DataFrame(df_train.describe())


# * **PassengerId:** This column cannot be evaluated numerically. Because it is an ID
# * **Survived:** That is target variable in dataset. It has the feature of being a flag.
# * **Pclass:** It is categorical variable. So it cannot be evaluated numerically
# * **Age:** Since the average age of the people on this ship and the median values are almost close, a normal distribution can be predicted.
# * **SibSp:** As it is understood from the average, one in every two people is seen as a close siblings or spouses on the ship.
# * **Parch:** As it is understood from the average, one in every three people is seen as a close parents / children on the ship.
# * **Fare:** It is seen that there is a difference between average and median value in terms of wages and standrt deviation is high.

# ## 5. Analysis of Categorical Variables with Descriptive Statistics <a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


categoricalVariables = ["Survived","Pclass","Sex","Embarked"]
for i in categoricalVariables:
    print(pd.DataFrame(df_train[i].value_counts()))


# The scale distribution of categorical data in the data is given as above.

# ## 6. Explanation of the Relationship Between Target (Survived) and Other Variables <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# Targetin variable is survivors and other variables are predictors. Let's look at the relationship between the target variable and other variables according to the type of the variable.

# **Survived and Passenger Class**

# In[ ]:


sns.violinplot( x=df_train.Survived, y=df_train.Pclass)
plt.show()


# It seems that there is an almost even distribution between the classes for the survivors. On the other hand, it is seen that those who do not survive have more people traveling in low class.

# **Survived and Gender (Sex)**

# In[ ]:


sns.violinplot( x=df_train.Survived, 
               y=df_train.Sex,
               linewidth=0)
plt.show()


# The distribution shows that women survive more than men.

# **Survived and Age**

# In[ ]:


sns.violinplot( x=df_train.Survived, 
               y=df_train.Age,
               linewidth=0)
plt.show()


# Age of the survival probability of survival decreases with age in relation to said to have increased relatively.

# **Survived and siblings / spouses**

# In[ ]:


g = sns.factorplot(x="SibSp",y="Survived",data=df_train,
                   kind="bar", height = 5, aspect= 1.6, 
                   palette = "hls")
g.set_ylabels("Probability(Survive)", fontsize=15)
g.set_xlabels("SibSp Number", fontsize=15)
plt.show()


# One or two, siblings/spouses are more likely to survive.

# **Survived and Parch**

# In[ ]:


g = sns.factorplot(x="Parch",y="Survived",data=df_train,
                   kind="bar", height = 5, aspect= 1.6, 
                   palette = "hls")
g.set_ylabels("Probability(Survive)", fontsize=15)
g.set_xlabels("Parch Number", fontsize=15)
plt.show()


# We can see a high standard deviation in the survival with 3 parents/children person's
# Also that small families (1~2) have more chance to survival than single or big families

# **Survived and Embarked**

# In[ ]:


sns.violinplot( x=df_train.Survived, 
               y=df_train.Embarked,
               linewidth=0)
plt.show()


# Passengers boarding from Southampton seem to be less likely to survive.

# **Survived and Fare**

# In[ ]:


sns.violinplot( x=df_train.Survived, 
               y=df_train.Fare,
               linewidth=0)
plt.show()


# It turns out that passengers who have paid higher tickets are more likely to survive.

# **New Variable**
# We derive a family variable by adding the x and y variables and the passenger itself.

# In[ ]:


df_train["FSize"] = df_train["Parch"] + df_train["SibSp"] + 1
pd.DataFrame(pd.crosstab(df_train.FSize, df_train.Survived))


# ## 7. Model Preparation <a class="anchor" id="7"></a>
# 
# [Back to Table of Contents](#0.1)

# Convert to dummy variable

# In[ ]:


if "Sex_male" not in df_train.columns:
    df_train = pd.get_dummies(df_train, columns=["Sex","Embarked"],drop_first=True)
df_train.head()


# **Correlation Anaylsis**

# In[ ]:


plt.figure(figsize=(15,12))
plt.title('Correlation of Features for Train Set')
sns.heatmap(df_train[["Pclass", "Age","Fare","FSize","Sex_male","Embarked_Q","Embarked_S","Survived"]].astype(float).corr(),vmax=1.0,  annot=True)
plt.show()


# Selecting target and predictors

# In[ ]:


target = df_train.Survived
predictors = df_train[["Pclass", "Age","Fare","FSize","Sex_male","Embarked_Q","Embarked_S"]]


# **Normalization for data modelling**

# In[ ]:


predictors = (predictors-predictors.min())/(predictors.max()-predictors.min())
predictors.head()


# In[ ]:


# Train and test splitting
features = predictors.columns
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size=0.25, random_state=0)
for i in [x_train,x_test,y_train,y_test ]:
    print(i.shape)


# ## 8. Modelling and Feature Importances <a class="anchor" id="8"></a>
# [Back to Table of Contents](#0.1)

# **Decision Tree**

# In[ ]:


dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred_dtc = dtc.predict(x_test)

dtc_cm = confusion_matrix(y_test,y_pred_dtc)
print("Decision Tree Confusion Matrix",dtc_cm)
dtc_acc = accuracy_score(y_test,y_pred_dtc)
print("Decision Tree Accuracy",dtc_acc)

importances_dtc = dtc.feature_importances_
indices_dtc = np.argsort(importances_dtc)

plt.title('Decision Tree Feature Importances')
plt.barh(range(len(indices_dtc)), importances_dtc[indices_dtc], color='b', align='center')
plt.yticks(range(len(indices_dtc)), [features[i] for i in indices_dtc])
plt.xlabel('Relative Importance')
plt.show()


# **Random Forrest**

# In[ ]:


rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)

rfc_cm = confusion_matrix(y_test,y_pred_rfc)
print("Random Forrest Confusion Matrix",rfc_cm)
rfc_acc = accuracy_score(y_test,y_pred_rfc)
print("Random Forrest Accuracy",rfc_acc)

importances_rfc = rfc.feature_importances_
indices_rfc = np.argsort(importances_rfc)

plt.title('Random Forrest Feature Importances')
plt.barh(range(len(indices_rfc)), importances_rfc[indices_rfc], color='b', align='center')
plt.yticks(range(len(indices_rfc)), [features[i] for i in indices_rfc])
plt.xlabel('Relative Importance')
plt.show()


# **Xgboost Classifier**

# In[ ]:


xgboast = XGBClassifier()
xgboast.fit(x_train, y_train)
print("Xgboast Classifier Accuracy",xgboast.score(x_test,y_test))

importances_xgboast = xgboast.feature_importances_
indices_xgboast = np.argsort(importances_xgboast)

plt.title('Xgboost Classifier Feature Importances')
plt.barh(range(len(indices_xgboast)), importances_xgboast[indices_xgboast], color='b', align='center')
plt.yticks(range(len(indices_xgboast)), [features[i] for i in indices_xgboast])
plt.xlabel('Relative Importance')
plt.show()


# ## 9. Model Evaluation <a class="anchor" id="9"></a>
# [Back to Table of Contents](#0.1)

# Prepare configuration for cross validation test harnessall models including a list

# In[ ]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('xgboast', XGBClassifier()))

# evaluate each model in turning kfold results
results_boxplot = []
names = []
results_mean = []
results_std = []
p,t = predictors.values, target.values
for name, model in models:
    cv_results = cross_val_score(model, p,t, cv=10)
    results_boxplot.append(cv_results)
    results_mean.append(cv_results.mean())
    results_std.append(cv_results.std())
    names.append(name)
algorithm_table = pd.DataFrame({"Algorithm":names,
                                "Accuracy Mean":results_mean,
                                "Accuracy":results_std})
algorithm_table.sort_values(by="Accuracy Mean",ascending=False)


# **boxplot algorithm comparison**

# In[ ]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_boxplot)
ax.set_xticklabels(names)
plt.show()


# ## 10. Parameter Tuning for Best Modelling <a class="anchor" id="10"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


#Grid Seach for XGboast
params = {
        'min_child_weight': [1, 2, 3],
        'gamma': [1.9, 2, 2.1, 2.2],
        'subsample': [0.4,0.5,0.6],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3,4,5]
        }
gd_sr = GridSearchCV(estimator=XGBClassifier(),
                     param_grid=params,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=1
                     )
gd_sr.fit(predictors, target)
best_parameters = gd_sr.best_params_
best_result = gd_sr.best_score_

print("Best result:", best_result)
pd.DataFrame({"Parameter":[i for i in best_parameters.keys()],
              "Best Values":[i for i in best_parameters.values()]})

