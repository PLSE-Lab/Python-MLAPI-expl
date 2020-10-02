#!/usr/bin/env python
# coding: utf-8

# # Hands-on ML using Titanic Kernel
# ### Uttara Sawant and Thalanayar Muthukumar
# #### April 2019
# 
# * **1 Introduction**
# * **2 Load and check data**
#     * 2.1 load data
#     * 2.2 massage/clean the data
#         * 2.2.1 Check missing values and fill them
#     * 2.3 Feature analysis
#         * 2.3.1 numerical values
#         * 2.3.2 categorical values
#     * 2.4 feature engineering
# * **3 Split into Training/Test Set**
# 
# * **4 Train model on data**
#     * 4.1 Two models
#     * 4.2 Feature importance from model
# * **5 Score & Evaluate**
#     * 5.1 Evaluation using classification report
#     * 5.2 Evaluation using confusion matrix
#     

# ## 1. Introduction
# 
# We will use the Titanic dataset to walk through the machine learning steps. Before we walk through the ML steps, we need to import the packages (pandas, numpy, matplotlib and seaborn). We have disabled printing of warning messages using the warnings package.
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ## 2. Machine Learning Steps
# We will walk through the following steps i. Load data, ii. Clean data, iii. Feature analysis, iv. Split into train / test set, v. Train model on data, vi. Score and evaluate
# 
# ### 2.1 Load and check data

# In[ ]:


##### Load dataset
titanic_data = pd.read_csv("../input/titanic/titanic.csv")
titanic_data.head()                                 #head prints only the first 5 rows

# The columns below such as PassengerId, Sex, Age are referred to as features


# ### 2.2 Massage/clean the data

# In[ ]:


# Check for missing values
titanic_data.isnull().sum()


# Age and Cabin features have an important part of missing values.

# ### 2.2.1. Check missing values and fill them

# In[ ]:


# 2.2.1 Filling missing value of Age with mean value
trainFareSum=titanic_data["Age"].isnull().sum()
print('Total age null values in the dataset before fill = ', trainFareSum)
titanic_data["Age"] = titanic_data["Age"].fillna(titanic_data["Age"].mean())
trainFareSum=titanic_data["Age"].isnull().sum()
print('Total age null values in the dataset after fill = ', trainFareSum)


# Since we have one missing value , we decided to fill it with the mean value which will not have an important effect on the prediction.
# You will notice we checked total number of null values for age before and after filling the null values.
# Another option to fill null values is using median value of the feature.

# ### 2.3 Feature analysis
# You understand the correlation of features (numerical and categorical) with the value being predicted
# 
# #### 2.3.1 Numerical values

# In[ ]:


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(titanic_data[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# Only Fare feature seems to have a significative correlation with the survival probability.
# 
# It doesn't mean that the other features are not useful. Subpopulations in these features can be correlated with the survival. To determine this, we need to explore in detail these features

# ##### 2.3.1.1 SibSp feature

# In[ ]:


# Explore SibSp feature vs Survived
g = sns.catplot(x="SibSp",y="Survived",data=titanic_data,kind="bar", height = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# It seems that passengers having a lot of siblings/spouses have less chance to survive
# 
# Single passengers (0 SibSP) or with two other persons (SibSP 1 or 2) have more chance to survive.

# ##### 2.3.1.2 Parch feature

# In[ ]:


# Explore Parch feature vs Survived
g  = sns.catplot(x="Parch",y="Survived",data=titanic_data,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Small families have more chance to survive, more than single (Parch 0), medium (Parch 3,4) and large families (Parch 5,6).

# ##### 2.3.1.3 Age feature

# In[ ]:


# Explore Age vs Survived
g = sns.FacetGrid(titanic_data, col='Survived')
g = g.map(sns.distplot, "Age")


# It seems that very young passengers have more chance to survive.

# ### 2.3.2 Categorical values feature analysis
# #### 2.3.2.1 Sex

# In[ ]:


g = sns.barplot(x="Sex",y="Survived",data=titanic_data)
g = g.set_ylabel("Survival Probability")


# In[ ]:


# convert Sex into categorical value 0 for male and 1 for female
titanic_data["Sex"] = titanic_data["Sex"].map({"male": 0, "female":1})


# It is clearly obvious that Male have less chance to survive than Female.
# 
# So Sex, might play an important role in the prediction of the survival.
# 
# For those who have seen the Titanic movie (1997), I am sure, we all remember this sentence during the evacuation : "Women and children first". 

# ### 2.4 Feature engineering

# In[ ]:


# Drop extraneous features, since we would like to demonstrate machine learning with minimal dataset
# For advanced training, we would include more features based on further analysis.
titanic_data.drop(["Name",  "SibSp", "Parch", "Cabin", "PassengerId", "Pclass", "Embarked", "Ticket"], axis = 1, inplace = True)


# In[ ]:


# Check the data for total number of features
titanic_data.head()


# In[ ]:


# Check the correlation between features after data cleaning
g = sns.heatmap(titanic_data[["Age","Sex","Fare"]].corr(),cmap="BrBG",annot=True)


# At this stage, we have 3 features.

# ## 3. Split into Training/Test Set

# In[ ]:


## Separate train dataset for model to see features and outcomes
from sklearn.model_selection import train_test_split

X=titanic_data.drop("Survived",axis=1)   # list of features selected for training predicton model
y=titanic_data["Survived"]               # list of features to be predicted after training the model
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.20, random_state=10)
                                    # test size indicates percent of dataset used as test set, here it is set to 20%                                    


# ### 4 Train models
# #### 4.1 Two Models
# 
# We compared 2 popular models and evaluate them. 
# 
# * Decision Tree: Simplest predictive learning model to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves)
# * Random Forest: Collective  predictive learning model which combines multitude of decision trees

# We decided to choose the DecisionTree and RandomForest models as they are the stepping stones to quickly understand predictive modeling in machine learning.

# In[ ]:


# DecisionTree Parameters tunning 
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state=10)
DTC.fit(X_train,Y_train)


# In[ ]:


# RFC Parameters tunning 
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=10,random_state=10)
RFC.fit(X_train,Y_train)


# ### 4.2 Feature importance of models
# 
# In order to see the most informative features for the prediction of passengers survival, we displayed the feature importance for the 2 models.

# In[ ]:


# Decision Tree feature importance
indices = np.argsort(DTC.feature_importances_)[::-1][:40]        
g = sns.barplot(y=X_train.columns[indices][:40],x = DTC.feature_importances_[indices][:40] , orient='h')
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("Decision Tree feature importance")


# In[ ]:


# Random Forest feature importance
indices = np.argsort(RFC.feature_importances_)[::-1][:40]        
g = sns.barplot(y=X_train.columns[indices][:40],x = RFC.feature_importances_[indices][:40] , orient='h')
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("Random Forest feature importance")


# We note that the two models have top features according to the relative importance. When several features are considered, the top features that are important may differ.

# In[ ]:


# Predict the results for both models
test_Survived_RFC = pd.Series(RFC.predict(X_test), name="RFC")
test_Survived_DTC = pd.Series(DTC.predict(X_test), name="DTC")


# At this time, prediction results for both models are ready for evaluation.

# ## 5 Score & Evaluate
# ### 5.1 Evaluation using classification report
# 
# We choose classification report to determine the model precision.

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,test_Survived_DTC))


# The precision column in the report indicates the Decision Tree model predicted 'Survived' (1) to 67%, and 'Not Survived' (0) to 85%.

# In[ ]:


print(classification_report(Y_test,test_Survived_RFC))


# The precision column in the report indicates the Random Forest model predicted 'Survived' (1) to 74%, and 'Not Survived' (0) to 84%.

# ### 5.2 Evaluation using confusion matrix

# In[ ]:


print("Training samples: {}, testing samples: {}".format(len(X_train), len(X_test)))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,test_Survived_DTC)
g = sns.heatmap(cm,annot=True)
g.set_xlabel("Predicted label - Survived",fontsize=12)
g.set_ylabel("True label - Survived",fontsize=12)
g.tick_params(labelsize=9)
g.set_title('Confusion Matrix - Decision Tree Model')


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,test_Survived_RFC)
g = sns.heatmap(cm,annot=True)
g.set_xlabel("Predicted label - Survived",fontsize=12)
g.set_ylabel("True label - Survived",fontsize=12)
g.tick_params(labelsize=9)
g.set_title('Confusion Matrix - Random Forest Model')


# At this time, you'll be familiar with how to load dataset, clean dataset, visualize features using plots, check and fill null values, train models, and predict the results.
