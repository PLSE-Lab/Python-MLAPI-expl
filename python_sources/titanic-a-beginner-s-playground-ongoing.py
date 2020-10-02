#!/usr/bin/env python
# coding: utf-8

# # **Titanic Survival Prediction**
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we are goint to complete the analysis of **what sorts of people were likely to survive**. In particular, **machine learning tools** will be applied to predict which passengers survived the tragedy.

# ### Load some libraries

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams['figure.figsize'] = (12, 10)
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn import metrics

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance


# ### Load the datasets

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/gender_submission.csv")


# ### Take a look at the datasets.

# In[ ]:


train.head()


# In[ ]:


train.describe(include="all")


# According to the table above:
# - Approximately 38.4% passengers survived;
# - There are missing values in the 'Age' column, the 'Cabin' column, and the 'Embarked' column;
# - Someone went onboard for free, it knid of makes sense if they were invited.

# In[ ]:


test.head()


# In[ ]:


test.describe(include="all")


# According to the table above:
# - There are missing values in the 'Age' column, the 'Fare' column, and the 'Cabin' column;
# - Someone went onboard for free, it knid of makes sense if they were invited.

# In[ ]:


sub.head()


# In[ ]:


sub.describe()


# The prediction above was made based on gender, where approximately 36.4% passengers survived.

# # Initial EDA

# ### Correlations among the continuous variables

# In[ ]:


corr = train.drop('PassengerId',axis=1).corr()
sns.heatmap(corr, annot=True, cmap='YlOrBr')


# According to the plot above:
#  - `Survival` is correlated with `Pclass` (-0.34) and `Fare` (0.26), while `PassengerId` (-0.005), `Age` (-0.077), `SibSp` (-0.035), and `Parch` (0.082) don't  seem to be correlated with `Survival`;
#  - `Age` and `Fare` seem to be correlated with `Pclass`;
#  - `SibSp` seems to be correlated with `Age`;
#  - `Parch` seems to be correlated with `SibSp`.

# ### Survival vs Pclass
# More people in class 1 survived, while more people in class 2 didn't and much more people in class** 3 didn't. And there are much more people in class 3 than in class 1 and 2.

# In[ ]:


sns.countplot(train['Pclass'], hue=train['Survived'])


# ### Survival vs Fare
# Passengers paid more tended to be more likely to **survive.

# In[ ]:


plt.hist(train[train["Survived"]==1]['Fare'], label="Yes", alpha=0.7)
plt.hist(train[train["Survived"]==0]['Fare'], label="No", alpha=0.7)
plt.legend(title='Survived')
plt.xlabel("Fare")


# ### Combine train and test for preprocessing

# In[ ]:


Survival = train.Survived # Label for training set
full = pd.concat([train.drop('Survived', axis=1), test])


# In[ ]:


full.describe(include="all")


# # Feature engineering
# - Remove `PassengerId` because it's not informative;
# - Remove `Cabin` because it has too many missing values;
# - Remove `Ticket`;
# - Represent categorical variables with dummy variables;
# - Extract information from `Name`;
# - Create a new variable indicating family size;
# - Create a new variable indicating whether the passenger is single.

# ### Remove `PassengerId`, `Cabin`, and `Ticket`

# In[ ]:


full1 = full.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1)
full1.describe(include="all")


# 
# ### One-hot encode `Pclass`, `Sex` and `Embarked`

# In[ ]:


pclass = pd.get_dummies(full1['Pclass'], prefix="Pclass_")
pclass.head()


# In[ ]:


full1['Sex_'] = np.where(full1.Sex == 'male', 1, 0)
full1.head()


# In[ ]:


Embarked = pd.get_dummies(full1['Embarked'], prefix="Embarked_")
Embarked.head()


# In[ ]:


full2 = pd.concat([full1, pclass, Embarked], axis=1)
full2.head()


# In[ ]:


full3 = full2.drop(["Pclass", "Sex", "Embarked"], axis=1)
full3.describe(include="all")


# ### Extract information from `Name`

# In[ ]:


Title = pd.get_dummies(full.Name.map(lambda x: x.split(',')[1].split('.')[0].split()[-1]))
Title.head()


# ### Create a new variable indicating family size and another new variable indicating whether the passenger is single

# In[ ]:


full3['FamilySize'] = full3.SibSp + full3.Parch + 1
full3['Single'] = np.where((full3.SibSp + full3.Parch) == 0, 1, 0)


# In[ ]:


full4 = pd.concat([full3, Title], axis=1)
full4.drop('Name', axis=1, inplace=True)
full4.head()


# ### Check correlations on new features

# In[ ]:


plt.figure(figsize=(26, 24))
sns.heatmap(full4.corr(), annot=True, cmap="YlOrBr")


# ### Check correlations without those titles
# Those titles are correlated with `Sex`, especially `Mr`, `Miss` and `Mrs`. So I decide not to keep these titles.

# In[ ]:


full5 = full4.loc[:,'Age':'Single'] # or: full4.iloc[:,0:13]
full5.head()


# In[ ]:


plt.figure(figsize=(16, 14))
sns.heatmap(full5.corr(), annot=True, cmap="YlOrBr")


# Drop `SibSp` and `Parch` because they are highly correlated `FamilySize`.

# In[ ]:


full6 = full5.drop(['SibSp','Parch'], axis=1)
full6.head()


# In[ ]:


plt.figure(figsize=(16, 14))
sns.heatmap(full6.corr(), annot=True, cmap="YlOrBr")


# ### Divide back to training and test sets

# In[ ]:


train_full = full6.iloc[:891]
test_full = full6.iloc[891:]


# In[ ]:


train_full.describe(include="all") # missing in Age


# In[ ]:


test_full.describe(include="all") # missing in Age and Fare


# # Impute the missing values
# Assume that the missing mechnaism is MAR (missing at random).  
# Impute missing data in training and test set separately, because we don't want any information of the test set being leaked into the training set.

# In[ ]:


# Impute Age in the training set
train_age_imputer = SimpleImputer()
train_imputed = train_full.copy()
train_imputed['Age_'] = train_age_imputer.fit_transform(train_full.iloc[:,0:1])
train_imputed['Fare_'] = train_imputed['Fare'] # not imputing Fare here, just kepp the training set consistent with the test set
train_imputed.drop(['Age', 'Fare'], axis=1, inplace=True)
train_imputed.head()


# In[ ]:


# Impute Age, Fare in the test set
test_age_imputer = SimpleImputer()
test_fare_imputer = SimpleImputer()

test_imputed = test_full.copy()
test_imputed['Age_'] = test_age_imputer.fit_transform(test_full.iloc[:,0:1])
test_imputed['Fare_'] = test_age_imputer.fit_transform(test_full.iloc[:,1:2])

test_imputed.drop(["Age","Fare"], axis=1, inplace=True)
test_imputed.head()


# ### Check again to make sure there is no missing values

# In[ ]:


train_imputed.describe(include="all")


# In[ ]:


test_imputed.describe(include="all")


# ### Is there any outliers or abnormal values?
# According to the results of `describe` above, all the values of all the variables seem reasonable.

# # More EDA

# ### Survival Condition of the Titanic Tragedy
# More people didn't survive.

# In[ ]:


sns.countplot(Survival, palette="coolwarm")


# ### Survival by Sex
# More men didn't survive.

# In[ ]:


sns.countplot(Survival, hue=train_imputed["Sex_"], palette="coolwarm")


# ### Survival by Pclass
# For those who didn't survive, most of them are in the lowest class, class 3.

# In[ ]:


sns.countplot(Survival, hue=train["Pclass"], palette="coolwarm")


# For those who survived, the numbers in the three classes are similar. Is it because of the different number of passengeres in each class? Let's check!

# In[ ]:


sns.countplot('Pclass', data=train, palette="coolwarm")


# The number of passengers in class 3 is twice as the numbers in class 1 and class 2, which makes sense.

# ### Survival by Age
# The age distributions of survived and not survived seem similar, while more kids survived and more people in their 20s didn't survive. There must be other variables influencing Survival.

# In[ ]:


plt.hist(train_imputed[Survival==1]['Age_'], label="Survived", alpha=0.7)
plt.hist(train_imputed[Survival==0]['Age_'], label="Not Survived", alpha=0.7)
plt.legend()
plt.xlabel("Age_Imputed")


# ### Survival by Fare
# It seems like those paid more are more likely to survive.   
# *Distrubution  of the imputed data set seems very similar as the original one.*

# In[ ]:


# Original data set
plt.hist(train[train["Survived"]==1]['Fare'], label="Survived", alpha=0.5)
plt.hist(train[train["Survived"]==0]['Fare'], label="Not Survived", alpha=0.5)
plt.legend()
plt.xlabel("Fare")
plt.title("Original Fare")


# In[ ]:


# Imputed data set
plt.hist(train_imputed[Survival==1]['Fare_'], label="Survived", alpha=0.5)
plt.hist(train_imputed[Survival==0]['Fare_'], label="Not Survived", alpha=0.5)
plt.legend()
plt.xlabel("Fare")
plt.title("Imputed Fare")


# ### Survival by Embarked
# Most people embarked Titanic form port "S". The survival condition seems similar for all three ports.

# In[ ]:


# Using original data set
sns.countplot('Embarked', data=train, palette="coolwarm")


# In[ ]:


# Using original data set
sns.countplot(Survival, data=train, hue="Embarked", palette="coolwarm")


# ### Survival by FamilySize
# It seems like those who had a family size of 2 or 3 or 4 are more likely to survive.

# In[ ]:


sns.countplot('FamilySize', data=train_imputed, palette="coolwarm", hue=Survival)
plt.legend(loc=1)


# ### Survival by Single
# It seems like those who are single are less likely to survive; those who have family are almost equally likely to survive as not survive.

# In[ ]:


sns.countplot('Single', data=train_imputed, palette="coolwarm", hue=Survival)
plt.legend(loc=1)


# ## WELL DONE!

# ## Let's start modeling!
# Since this is a classification problem, we will use the following six methods.   
# Divide training set into training and validation sets (5-fold cross validation).

# In[ ]:


kfold = KFold(n_splits=5, random_state=1, shuffle=True)
kfold


# In[ ]:


accuracy = {}


# ### 1. Gaussian Naive Bayes

# In[ ]:


m1_nb = GaussianNB()


# In[ ]:


accuracy['Gaussian Naive Bayes'] = np.mean(cross_val_score(m1_nb, train_imputed, Survival, scoring="accuracy", cv=kfold))


# ### 2. Logistic Regression

# In[ ]:


m2_log = LogisticRegression(solver='newton-cg') # 'lbfgs', 'sag' failed to converge


# In[ ]:


accuracy['Logistic Regression'] = np.mean(cross_val_score(m2_log, train_imputed, Survival, scoring="accuracy", cv=kfold))


# ### 3. K-nearest Neighbors

# In[ ]:


m3_knn = KNeighborsClassifier(n_neighbors = 5)


# In[ ]:


accuracy['K Nearest Neighbors'] = np.mean(cross_val_score(m3_knn, train_imputed, Survival, scoring="accuracy", cv=kfold))


# ### 4. Random Forests

# In[ ]:


m4_rf = RandomForestClassifier(n_estimators=10)


# In[ ]:


accuracy['Random Forest'] = np.mean(cross_val_score(m4_rf, train_imputed, Survival, scoring="accuracy", cv=kfold))


# ### 5. Support Vector Machines

# In[ ]:


m5_svc = SVC(gamma='scale')


# In[ ]:


accuracy['SVM'] = np.mean(cross_val_score(m5_svc, train_imputed, Survival, scoring="accuracy", cv=kfold))


# ### 6. Gradient Boosting

# In[ ]:


m6_gb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)


# In[ ]:


accuracy['Gradient Boosting'] = np.mean(cross_val_score(m6_gb, train_imputed, Survival, scoring="accuracy", cv=kfold))


# ## Seems like the best model here is **Gradient Boosting**

# In[ ]:


accuracy


# In[ ]:


max_accuracy = max(accuracy, key=accuracy.get)
print(max_accuracy, '\taccuracy:', accuracy[max_accuracy])


# # Important features

# In[ ]:


train_imputed.columns


# In[ ]:


m6_gb.fit(train_imputed, Survival)
m6_gb.feature_importances_


# In[ ]:


plot_importance(m6_gb)


# ### Grid Search for Hyperparameters

# In[ ]:


param_grid = {'max_depth': [1,3,5,10,15], 'n_estimators': [50,100,200,500,1000], 'learning_rate': [1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(XGBClassifier(), param_grid, cv=kfold)
grid.fit(train_imputed[["Fare_", "Age_", "FamilySize", "Sex_"]], Survival)  # with 4 selected features
grid.best_params_


# In[ ]:


gb = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.01)
np.mean(cross_val_score(gb, train_imputed[["Fare_", "Age_", "FamilySize", "Sex_"]], Survival, scoring="accuracy", cv=kfold))


# ### Predictions

# In[ ]:


gb.fit(train_imputed[["Fare_", "Age_", "FamilySize", "Sex_"]], Survival)
predictions = gb.predict(test_imputed[["Fare_", "Age_", "FamilySize", "Sex_"]])


# # Submit

# In[ ]:


submission = pd.DataFrame({ 'PassengerId': test.PassengerId,
                            'Survived': predictions })
submission.to_csv("TitanicSubmission.csv", index=False)

