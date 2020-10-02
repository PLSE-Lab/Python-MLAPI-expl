#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# This notebook  will  perform feature engineering and exploratory data analysis to see how they impact on titanic survival. The result of the impact analysis will be used to train a model and make predictions on titanic survival. Hyperparameter tuning and blending will be utilized to extend model performance.

# ### 1. Data Preparation
# Load the train and test data sets. Explore missing data, tidy data, and data wrangling

# In[ ]:


# load libraries for data manipulation
import numpy as np
import pandas as pd
# load visualization libraries
import matplotlib.pyplot as plt
plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb
sb.set()
# load modelling libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import  cross_val_score
# warnings
import string
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# load the train and test data sets
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train_len = len(train)
# combine train and test data sets
data_all = train.append(test, sort=False, ignore_index=True)
# no. of missing values 
data_all.isnull().sum()


# There will be no imputation of missing values for Cabin.

# In[ ]:


# impute missing values
data_all['Age'] = data_all.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
data_all['Fare'] = data_all.groupby('Pclass')['Fare'].apply(lambda x: x.fillna(x.median()))
data_all['Embarked'] = data_all['Embarked'].fillna('S')# internet search indicate 2 passengers from Southampton


# ### 2.0 EDA and Data Manipulation
# Data visualization and manipulations to gain insights for modelling.

# #### 2.1 Pclass
# Pclass is better represented as an ordinal rank data and will be converted

# In[ ]:


bar = sb.barplot(x='Pclass', y='Survived' , data=data_all)


# The above figure indicates 1st class passengers have the highest chance of survival, followed by second class passengers, and lastly third class passengers.

# In[ ]:


# convert Pclass to ordinal rank
data_all['Pclass'].replace([1, 2, 3], [3, 2, 1],inplace=True)


# #### 2.2 Age

# In[ ]:


scatter = sb.regplot(x='Age', y='Survived', data=data_all)


# The regression line for the above figure indicates decreasing rate of survival with increasing age. Younger people have a better chance of survival than old people. 

# In[ ]:


scatter = sb.lmplot(x='Age', y='Survived',  data=data_all, hue='Sex')


# There is increasing survival of females with age, that is older females are more likely to survive than younger ones. The reverse is the case for males that follow the general decreasing survival with age. A new varaible will be formed to combine the interaction between  Sex and Age.

# In[ ]:


# create age bins as Infant 1, Children 2, Adults 3 and Elderly 4.
data_all['Age'] = pd.cut(data_all.Age, bins=[0,2,17,65,99], labels=[1, 2, 3, 4]).astype(str)
data_all['Sex'].replace(['female', 'male'], ['F', 'M'],inplace=True)
data_all['SexAge'] = data_all['Sex'] + data_all['Age']
bar = sb.barplot(x='SexAge', y='Survived' , data=data_all)


# The female age groups follow the pattern of increasing survival with age while male age groups follow the pattern of decreasing survival with age. There are no elderly females as indicated by F4.

# #### 2.3 Fare

# In[ ]:


scatter = sb.regplot(x='Fare', y='Survived', data=data_all)


# The above figure indicates inreasing survival with Fare.

# In[ ]:


# create Fare bins as Low 1, 1stQuartile-median 2, median-3rdQuartile 3, High 4.
data_all['Fare'] = pd.cut(data_all.Fare, bins=[-1.0,7.8958,14.4542,31.2750,512.3292], labels=[1, 2, 3, 4]).astype(int)
bar = sb.barplot(x='Fare', y='Survived' , data=data_all)


# There is increasing survival in the fare bins from low to high.

# #### 2.4 Embarked

# In[ ]:


bar = sb.barplot(x='Embarked', y='Survived' , data=data_all)


# There is increasing survival from Southampton - Queenstown - Cherbourg. 

# #### 2.5 FamilySize
# Two variables SibSp(sibling/spouse) and Parch(parent/child) refer to the number of SibSp and Parch present in the titanic. Both variables are closely related that they can be pulled together to form a new column named FamilySize. FamilySize will contain the number of family members present in the titanic plus the passenger.

# In[ ]:


# add the columns SibSp and Parch to create FamilySize
data_all['FamilySize'] = data_all['SibSp'] + data_all['Parch'] + 1
# add families with 4 or more members together
for i in  np.where(pd.notnull(data_all.FamilySize) & (data_all['FamilySize'] >  3 )  ):
    data_all.at[i, 'FamilySize'] = 4
bar = sb.barplot(x='FamilySize', y='Survived' , data=data_all)


# The graph above indicates that alone passengers indicated by bin 1 have the least survival.

# #### 2.6 Sex

# In[ ]:


bar = sb.barplot(x='Sex', y='Survived' , data=data_all)


# The above figure indicates average survival between male and female passengers. It indicates females have a higher chance of survical than males.

# #### 2.7 Title
# Extract title from the name field and group passengers with the titles Mr, Master, Miss and Mrs. Title is a derived feature based on Sex.

# In[ ]:


# extracting titles and storing in Title field
for i in data_all:
    data_all['Title'] = [x[1].split(".")[0].strip(" ") for x in data_all['Name'].str.split(",")]

print(pd.crosstab(data_all['Title'], data_all['Sex']))


# In[ ]:


# reducing the titles to Mr, Mrs, Master and Miss
for i in data_all:
    data_all['Title'] = data_all['Title'].replace(['Capt', 'Col', 'Don', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Mr')
    data_all['Title'] = data_all['Title'].replace(['Mlle', 'Ms'], 'Miss')
    data_all['Title'] = data_all['Title'].replace(['Dona', 'Lady', 'the Countess', 'Mme'], 'Mrs')
for i in np.where(pd.notnull(data_all.Title) & (data_all['Title'] ==  'Dr') & (data_all['Sex'] == 'M' )  ):
     data_all.at[i, 'Title'] = 'Mr'
for i in np.where(pd.notnull(data_all.Title) & (data_all['Title'] ==  'Dr') & (data_all['Sex'] == 'F' )  ):
     data_all.at[i, 'Title'] = 'Mrs'
bar = sb.barplot(x='Title', y='Survived' , data=data_all)


# #### 2.8 Data Transformations
# Select numerical and and categorical features for modeling, encode variables and prepare train and test sets for modeling.

# In[ ]:


# check correlation of numeric features with the target
data_all_corr = data_all.corr().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
data_all_corr.rename(columns={'level_0': 'Feature 1', 'level_1': 'Feature 2', 0: 'Correlation Coefficient'}, inplace=True)
data_all_corr[data_all_corr['Feature 1'] == 'Survived']


# Select numeric features

# In[ ]:


numeric = ['Pclass', 'Fare', 'FamilySize']
data_num = data_all[numeric]


# Select categorical features

# In[ ]:


categorical = ['SexAge', 'Embarked', 'Sex', 'Title']
data_cat = data_all[categorical]
data_cat = pd.get_dummies(data_cat)


# Separate train and test sets for modeling

# In[ ]:


y = train.Survived
df_all = pd.concat([data_num, data_cat], axis = 1)
features = (data_num.columns).append(data_cat.columns)
df_train = df_all[:891]
df_test = df_all[891:]
target = 'Survived'
X = df_train[features]
X_test = df_test[features]


# ### 3. Modeling
# Cross validation and model evaluation

# In[ ]:


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5)
def auc_cv(model):
    roc_auc = cross_val_score(model, X, y, scoring='roc_auc', cv=kfold)
    return (roc_auc)


# #### 3.1.1  Logistic Regression

# In[ ]:


# LogisticRegression CV and model fitting
logreg2 = LogisticRegression(C = .4, penalty='l2')
auc_cv_logreg = auc_cv(logreg2)
logreg2.fit(X, y)
print('LogisticRegression CV score min: ' + str(auc_cv_logreg.min()) + ' mean: ' + str(auc_cv_logreg.mean()) 
      + ' max: ' + str(auc_cv_logreg.max()) )


# In[ ]:


# LogisticRegression L2 features
coef_table = pd.DataFrame(list(X.columns)).copy()
coef_table.insert(len(coef_table.columns), 'Coefs', logreg2.coef_.transpose())
coef_table = coef_table[abs(coef_table['Coefs']) > 0.0]
coef_table.sort_values('Coefs')


# Survival increases as we move down the table from top to bottom. 

# #### 3.1.2 GradientBoosting

# In[ ]:


# GradientBoosting CV and model fitting
gbct = GradientBoostingClassifier(n_estimators=1000, learning_rate=.01, max_depth=3, max_features=3, min_samples_split=4,
                                 min_samples_leaf=7, loss='exponential', subsample=.6, random_state=0)
auc_cv_gbct = auc_cv(gbct)
gbct.fit(X, y)
print('GBoost CV score min: ' + str(auc_cv_gbct.min()) + ' mean: ' + str(auc_cv_gbct.mean()) 
      + ' max: ' + str(auc_cv_gbct.max()) )


# ### 3.2  Model Evaluation

# In[ ]:


from datetime import datetime
#model blending 
def blend_models(X):
    return ((logreg2.predict(X)) + (gbct.predict(X)))/2

logreg2_preds =  logreg2.predict(X_test)
gbct_preds =  gbct.predict(X_test)
logreg2_gbct_preds = blend_models(X_test).astype(int)
titanic_logreg2_gbct = pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':logreg2_gbct_preds})
titanic_logreg2_gbct.to_csv('solution.csv', index = False)
print('Version 3 submitted on', datetime.now())


# LB score of 0.78947, top 17%
