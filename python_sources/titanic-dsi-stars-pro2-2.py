#!/usr/bin/env python
# coding: utf-8

#    
#    
#    

# ---

# # Titanic: Machine Learning from Disaster.

#   ---   

# <img alt="Someone live-tweeted the Titanic sinking and it is epic!" src="https://cdn.techjuice.pk/wp-content/uploads/2016/04/titanic-sinking.jpg" data-noaft="1" jsname="HiaYvf" jsaction="load:XAeZkd;" style="float: center;width:500px;height:400px;border:2;">

# ---

# #                Predicting the Survival of Titanic Passengers     

# The Titanic was a ship disaster that on its maiden voyage sunk in the northern Atlantic on April 15, 1912, resulting in the death of 1502 out of 2224 passengers and crew. While there exists conclusions regarding the cause of the sinking, the analysis of the data on what impacted the survival of passengers continues to this date. The approach taken is utilize a publically  available data set from a web site Kaggle.

# ### High Light:
# 
# - Create a model that predicts which passengers survived the Titanic shipwreck.

# ### Data Set Column Descriptions
# ---
# - **pclass**: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# 
# - **survived**: Survival (0 = No; 1 = Yes)
# 
# - **name**: Name
# 
# - **sex**: Sex
# 
# - **age**: Age
# 
# - **sibsp**: Number of siblings/spouses aboard
# 
# - **parch**: Number of parents/children aboard
# 
# - **fare**: Passenger fare (British pound)
# 
# - **embarked**: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# - **cabin**:Cabin number
# 
# - **ticket**:Ticket number
# 

# ---

# ## Group  member :
# - Saad Alsharef 
# 
# - Mohammed Saud
# 
# - Howida Saeed

# ---

# ## Import libraries
# 

# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from scipy.stats import norm
import warnings
import datetime
import time
# Importing libraries for Modeling
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
plt.style.use('ggplot')
sns.set(font_scale=1.5)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Reading the data

# ---

# #### Step 1.1 - Create an iPython notebook and load the csv into pandas.

# load the data

# In[ ]:


#import data file for kaggle
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


#import data file for local work
#train = pd.read_csv('../data/train.csv')
#test = pd.read_csv('../data/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# #### 1.2 how many columns and rows in train and test data

# In[ ]:


train.shape, test.shape


# #### 1.3 summary of a DataFrame.

# In[ ]:


train.info()


# In[ ]:


test.info()


# #### 1.4 print colunms with object type values

# In[ ]:


train.select_dtypes(include=object).head()


# In[ ]:


test.select_dtypes(include=object).head()


# #### 1.5 print colunms with numeric type values

# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

train.select_dtypes(include=numerics)


# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

test.select_dtypes(include=numerics)


# # 2.  Cleaning the data
# 

# ---

# #### 2.1 Create these heatmaps, yellow are the missing data. Hint: cmap='viridis'

# In[ ]:


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))

# train data 
sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')
ax[0].set_title('Train data')

# test data
sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')
ax[1].set_title('Test data');


# ###  2.2 - Which column has the most `NaN` values? How many cells in that column are empty?

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# The `Cabin` column has the most NaN values. The `Age` column has the next most. All other columns seem to be complete.

# #### 2.3 How many ports are in Embarked column?(train data)

# In[ ]:


train.Embarked.value_counts()


# #### 2.3 What is the `Pclass` of missing fare in test dataset. Print the complete row here

# In[ ]:


x =pd.isnull(test['Fare'])
test[x]


# Compute the average of `Fare` of the missing `Pclass` , you should be able to identify this from above question

# In[ ]:


mean_Fare = test.groupby('Pclass')['Fare'].mean()
mean_Fare
print("The mean fare for the Pclass (for missing fare data) is: {}".format(mean_Fare[3]))


# #### 2.4 Now we got the mean `Fare`, and we will fill the missing value of `Fare` with everyone from the same `Pclass`,the mean we have computed above.

# In[ ]:


cc =test['Fare'].replace(np.nan , mean_Fare[3], inplace=True )


# #### 2.5 What is the mean age of each Pclass in the train data.**

# In[ ]:


mean_age = train.groupby('Pclass')[['Age']].mean()
mean_age


# #### 2.6 Function impute_age to fill the mean age with respect to each Pclass.
# 

# In[ ]:


#defining a function 'impute_age'
def impute_age(age_pclass): # passing age_pclass as ['Age', 'Pclass']
    
    # Passing age_pclass[0] which is 'Age' to variable 'Age'
    Age = age_pclass[0]
    
    # Passing age_pclass[2] which is 'Pclass' to variable 'Pclass'
    Pclass = age_pclass[1]
    
    #applying condition based on the Age and filling the missing data respectively 
    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 30

        else:
            return 25

    else:
        return Age


# #### 2.7 grab age and apply the impute_age, our custom function

# In[ ]:


train.Age = train.apply(lambda x :impute_age(x[['Age', 'Pclass']] ) , axis = 1)

test.Age = test.apply(lambda x :impute_age(x[['Age', 'Pclass']] ) , axis = 1)


# #### 2.8 If there originally was a value for Cabin -- put 1, If the value is missing/null -- put 0

# In[ ]:


test['Cabin']= test['Cabin'].apply(lambda x :0 if pd.isnull(x)else 1)
train['Cabin']=train['Cabin'].notnull().astype('int')


# #### 2.9 replace nan values in embarded with must value in the column

# In[ ]:


train.Embarked.value_counts()


# In[ ]:


train.Embarked.replace(np.nan ,'S', inplace= True )


# #### cheking agin if ther any missing value 

# In[ ]:


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))

# train data 
sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')
ax[0].set_title('Train data')

# test data
sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')
ax[1].set_title('Test data');


# ---

# # 3  Exploratory analysis
# 

# ---

# ###  3.1 - What was the survival rate overall?
# 

# In[ ]:


survivors = train[train['Survived'] == 1]
train['Survived'].value_counts(normalize=True)


# ### 3.2 - Which gender fared the worst? What was their survival rate?

# In[ ]:


train.groupby(['Sex','Survived']).size().reset_index(name='Frequency')


# In[ ]:


pd.crosstab(train['Sex'],train['Survived']).apply(lambda x: 100*(x/x.sum()), axis=1)


# Female Survival Rate = 231/(231+81) = 74.04%
# 
# 
# Male Survival Rate = 109/(109+468) = 18.89%
# 
# #### Male's fared the worst; their survival rate was 18.89%.

# ---

# ###  3.3 - What was the survival rate for each `Pclass`?

# In[ ]:


train.groupby(['Pclass','Survived']).size().reset_index(name='Frequency')


# In[ ]:


pd.crosstab([train.Sex,train.Survived],train.Pclass,margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


pd.crosstab(train['Pclass'],train['Survived'],margins=True).style.background_gradient(cmap='PuBu')


# In[ ]:


pd.crosstab(train['Pclass'],train['Survived']).apply(lambda x: 100*(x/x.sum()), axis=1)


# Passanger Class 1 Survival Rate = 231/(231+81) = 62.62%
# 
# Passanger Class 2 Survival Rate = 109/(109+468) = 47.28%
# 
# Passanger Class 3 Survival Rate = 109/(109+468) = 24.24%

# ---

# ###  3.4 - What is the survival rate for each port of embarkation?

# In[ ]:


pd.crosstab(train['Embarked'],train['Survived']).apply(lambda x: 100*(x/x.sum()), axis=1).tail()


# Survival Rate, Embarkation Port C = 55.36%
# 
# Survival Rate, Embarkation Port Q = 38.96%
# 
# Survival Rate, Embarkation Port S = 33.70%

# ### 3.5 - What is the survival rate for children (under 12) in each `Pclass`?

# In[ ]:


age_less_12 = train[ train['Age'] < 12 ]
pd.crosstab(age_less_12['Pclass'],age_less_12['Survived']).apply(lambda x: 100*(x/x.sum()), axis=1).tail()


# Survival Rate, Children Under 12, Passanger Class 1 = 75.00%
# 
# Survival Rate, Children Under 12, Passanger Class 2 = 100.00%
# 
# Survival Rate, Children Under 12, Passanger Class 3 = 40.43%

# In[ ]:


print('Oldest Passenger was of:',round(train['Age'].max()),'Years')
print('Youngest Passenger was of:',round(train['Age'].min(),1),'Years')
print('Average Age on the ship:',round (train['Age'].mean()),'Years')


# ### 3.6 - Did the captain of the ship survive? Is he on the list?

# In[ ]:


train[ train['Name'].str.contains('Cap') ]


# #### The Captain of the ship, Capt. Edward Gifford Crosby, did not survive.  He is on the list.

# ---

# # 4.  visualizations

# ---

# ###  4.1 Surivival rates

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(18,8))

train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# The overall survival rate was 38.25%.
# 

# ## 4.2 Surivival rates vs Sex

# ---

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('Sex : Survived vs Dead')
plt.show()


# Men are less likely to survive than women. This is quite logical, because women were also allowed to go forward on rescue boats. Perhaps this will play a key role later.

# # 4.3 Surivival rates vs Pclass

# ---

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train['Pclass'].value_counts().plot.bar(color=['#CD7F33','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
plt.show()


# The passenger survival is not the same in the 3 classes. First class passengers have more chance to survive than second class and third class passengers.

# ### 4.4 Surivival rates vs Pclass and Sex and age

# ---

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# It show a peak in survival among young people. Unfortunately, elderly people have less chances to survive. A jump in survival in children is also visible, which is quite logical. Most likely they were saved in the first place.

# ---

# # 5 - Modeling part :

# ---

# ### 5.1 Dummy the Sex and Embarked columns. 

# In[ ]:


train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# ### 5.2 Model Prep: 

# In[ ]:


selected_features = ['PassengerId','Pclass','Age', 'SibSp', 'Parch',
                     'Fare','Cabin','Sex_male', 'Embarked_Q',
                     'Embarked_S']
selected_features


# ### 5.3 (train data) Now, separate the selected_column in `X_train` and `Survived` in `y_train`.
# 
# #### For Titanic, the score is calculated using categorization accuracy (the closer to 1 the better)

# In[ ]:


X = train[selected_features]
y = train['Survived']

X_test  = test.drop(["Name",'Ticket'], axis=1).copy()


# In[ ]:


X.shape,X_test.shape


# In[ ]:


set(X_test.columns).symmetric_difference(set(X.columns))


# In[ ]:


sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X, y)


sgd.score(X, y)

acc_sgd = round(sgd.score(X, y) * 100, 2)


# ### RandomForestClassifier

# In[ ]:


cv = StratifiedKFold(n_splits=10, shuffle=True)
scaler = StandardScaler()
forest = RandomForestClassifier()
forest_pipeline = Pipeline([('transformer', scaler), ('estimator', forest)])

forest_params = {'estimator__n_estimators': [5,50,80,100],
              'estimator__max_depth':[1,2,3,4,5,6,7],
                'estimator__max_features':[2,5,7,9]}

forest_grid = GridSearchCV(forest_pipeline, forest_params,
                           n_jobs=-1, cv=cv, verbose=2)
forest_grid.fit(X, y);
best_forest = forest_grid.best_estimator_
print(f' GridSearch best score: {forest_grid.best_score_}')
print(f' GridSearch best params : {forest_grid.best_params_}')


#  
#  ### AdaBoostClassifier

# In[ ]:


cv = StratifiedKFold(n_splits=10, shuffle=True)
scaler = StandardScaler()
ada = AdaBoostClassifier()
ada_pipeline = Pipeline([('transformer', scaler), ('estimator', ada)])

ada_params = {'estimator__base_estimator': [None, DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)],
              'estimator__n_estimators': [10,50, 70],
              'estimator__learning_rate':[.01, .1, .5, 1]}

ada_grid = GridSearchCV(ada_pipeline, ada_params, n_jobs=-1, cv=cv, verbose=2)
ada_grid.fit(X, y);
best_ada = ada_grid.best_estimator_
print(f' GridSearch best score: {ada_grid.best_score_}')
print(f' GridSearch best params : {ada_grid.best_params_}')


#  
#  ### KNN Classifier

# In[ ]:


#KNN to gid search CV

n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True,n_jobs=-1,
                cv=10, scoring = "roc_auc")
gd.fit(X, y)
best_knn = ada_grid.best_estimator_
print(gd.best_score_)
print(gd.best_estimator_)


# ### Print File of The Result : 

# In[ ]:


submit = pd.DataFrame({'PassengerId':X_test.PassengerId, 
                    'Survived':best_knn.predict(X_test).astype(int)})


submit.to_csv("gender_submission.csv", index=False)


# In[ ]:


submit.head(5)


# In[ ]:


submit.tail(5)


# * in this params we has 0.78 score

# ![%D8%AA%D8%B9%D9%84%D9%8A%D9%82%20%D8%AA%D9%88%D8%B6%D9%8A%D8%AD%D9%8A%202020-04-02%20093227.png](attachment:%D8%AA%D8%B9%D9%84%D9%8A%D9%82%20%D8%AA%D9%88%D8%B6%D9%8A%D8%AD%D9%8A%202020-04-02%20093227.png)

# In[ ]:




