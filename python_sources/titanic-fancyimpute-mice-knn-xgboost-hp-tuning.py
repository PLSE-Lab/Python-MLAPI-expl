#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing Libraries

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Preprocessing the data

# In[ ]:


#Loading the dataset

df_train = pd.read_csv('../input/titanic/train.csv') 
df_test  = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


#Viewing the dataset

display(df_train.head())
display()
display(df_test.head())


# In[ ]:


#Preprocessing needs to be done on both test and train data. So we will concatenate both test and train data.
#Creating a new column in test data
df_test['Survived'] = 999      #Assigning some random value to the test dataset


# In[ ]:


#Now both test and train can be concatenated

df = pd.concat([df_train , df_test] , axis = 0)
print(df_train.shape)
print(df_test.shape)
print('Combined dataframe shape :',df.shape)


# In[ ]:


df.info()


# **Passenger Class**

# In[ ]:


plt.style.use('ggplot')
print('Null Values :',df['Pclass'].isnull().sum())  #To find the number of null values
df['Pclass'].value_counts(sort = True).plot(kind = 'barh' ,title = 'Passenger Class')


# There are no null values in the passenger class column .
# 
# We can see that most passengers travelled in 3rd Class , and 1st class comes second and least people travelled in second class.
# 
# So no preprocessing is required

# **AGE**

# In[ ]:


#Null values
print('The number of null values in age columns',df['Age'].isnull().sum())
print('The % of null values in age columns',round(df['Age'].isnull().mean()* 100,2)) 


# In[ ]:


#Age column has lots of null values. 
#Around 20 % of the values are missing.


# In[ ]:


df['Age'].plot(kind='box')
print('Range of age is between',df['Age'].min() , 'and' ,df['Age'].max())


# In[ ]:


#Eventhough the box plot shows outlier we wont remove any outliers in age column , because maximum age of the person in titanic was 80


# **CABIN : **
# 
# C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:


#Null values
print('The number of null values in cabin columns',df['Cabin'].isnull().sum())
print('The % of null values in cabin columns',round(df['Cabin'].isnull().mean()* 100,2)) 


# Around 77 % of the values are missing .So its better to drop Cabin column than imputing the values

# In[ ]:


#Dropping Cabin column
df.drop(columns = 'Cabin' , inplace = True)


# **EMBARKED :**

# In[ ]:


df['Embarked'].value_counts().plot(kind=  'bar' , rot = 0 )


# In[ ]:


print('The number of null values in Embarked is ', df['Embarked'].isnull().sum())


# Since most people embarked in Southampton , the two values will be imputed with Southampton.

# In[ ]:


#NaN values are replaced with Southampton
df['Embarked'].replace({np.nan:'S'} , inplace = True)


# **FARE : **

# In[ ]:


#Only one value of fare is missing.
df[df['Fare'].isnull()]


# In[ ]:


plt.figure(figsize = (10,4))
plt.subplot(1,2,1)
sns.distplot(df['Fare'])
plt.subplot(1,2,2)
sns.boxplot(df['Fare'])


# **NAME : **

# In[ ]:


df['Name'].isnull().sum()
#There are no missing values in name


# In[ ]:


df['Name'].head()
#Name is useless feature , but Title in the name can give some additional information for prediction


# In[ ]:


def GetTitle_temp(name):
    fname_title = name.split(',')[1]
    title = fname_title.split('.')[0]
    title = title.strip().lower()
    return title

df.Name.map(GetTitle_temp).value_counts()


# There are around 18 titles in the dataset. It can be combined to 4 titles Master,Mr,Miss and Mrs

# In[ ]:


def GetTitle(name):
    titles = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Mr',
               'rev' : 'Mr',
               'dr' : 'Mr',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Mr',
               'lady' : 'Miss',
               'sir' : 'Mr',
               'mlle' : 'Miss',
               'col' : 'Mr',
               'capt' : 'Mr',
               'the countess' : 'Miss',
               'jonkheer' : 'Mr',
               'dona' : 'Miss'
                 }
    fname_title = name.split(',')[1]
    title = fname_title.split('.')[0]
    title = title.strip().lower()
    return titles[title]

df['Name'] = df.Name.map(GetTitle)


# In[ ]:


sns.countplot(df['Name'])


# **PARCH :**
# 
# Number of parents / children aboard the Titanic

# In[ ]:


df['Parch'].value_counts()


# **SibSp :**
#     
# Number of siblings / spouses aboard the Titanic

# In[ ]:


df['SibSp'].value_counts()


# In[ ]:


#Combining Sibsp and Parch columns to form number of people accompanying
df['Accomp'] = df['SibSp'] + df['Parch']
df.drop(columns = ['SibSp' , 'Parch'] , inplace = True)


# **SEX : **

# In[ ]:


df['Sex'].value_counts()


# In[ ]:


sns.countplot(df['Sex'])


# In[ ]:


#Encoding the Sex column
df['Sex'] = df['Sex'].map({'female':0 , 'male':1 })


# **TICKET :**

# In[ ]:


#Dropping the ticket number
df.drop(columns = ['Ticket'] , inplace = True)


# In[ ]:


#Viewing the Data
df.head()


# **Encoding Categorical Variables :**

# In[ ]:


cat = pd.get_dummies(df[['Embarked' , 'Name']] , drop_first=True)


# In[ ]:


df = pd.concat([df,cat] , axis = 1)
df.drop(columns = ['Embarked' , 'Name'] , inplace = True)    #Dropping the original columns


# In[ ]:


#After Encoding
df.head()


# ### MISSING VALUE IMPUTATION :

# In[ ]:


#Checking the percentage of missing values:
df.isnull().mean()*100


# Importing missingno package to find the missing values and its patterns.
# 
# Missing values can be classified into three types :
# 
# 1. Missing Completely at Random (MCAR)
# 2. Missing at Random (MAR)
# 3. Missing Not at Random (MNAR)
# 
# In our dataset we can find that Age column comes under the category of Missing completely at random
# 

# **MISSINGNO :**
# 
# Missingno package is imported. It helps to easily visualize missing values
# 
# Missing values can be visualized using 3 ways : 
# 
# 1. Heatmap 
# 2. Finding Missing pattern using Matrix
# 3. Dendrogram

# In[ ]:


import missingno as msno
msno.matrix(df)


# In[ ]:


msno.heatmap(df)   #Heatmap to find any correlation of missing values between other missing columns
#From this heatpmap we can see that there is no correlation between missing values


# In[ ]:


msno.dendrogram(df)


# **FANCYIMPUTE:**
# 
# Fancyimpute imputation techniques,
# 
# 1. KNN or K-Nearest Neighbor
# 2. MICE or Multiple Imputation by Chained Equation
# 
# KNN finds most similar points for imputing
# 
# MICE performs multiple regression for imputing

# In[ ]:


#Imputing using KNN : 

from fancyimpute import KNN
knn_imputer = KNN()
df_knn = df.copy()
df_knn.iloc[:,:] = knn_imputer.fit_transform(df_knn)


# In[ ]:


#Imputing using MICE

from fancyimpute import IterativeImputer
MICE_imputer = IterativeImputer()
df_mice = df.copy()
df_mice.iloc[:,:] = knn_imputer.fit_transform(df_mice)


# In[ ]:


sns.kdeplot(df['Age'] , c = 'r' , label = 'No imputation')
sns.kdeplot(df_knn['Age'] , c = 'g' , label = 'KNN imputation')
sns.kdeplot(df_mice['Age'] , c = 'b' , label = 'MICE imputation')
sns.kdeplot(df['Age'].fillna(df['Age'].mean()) , c = 'k' , label = 'Fillna_Mean')
#Distribution of the columns are maintained while using this fancy imputation techniques
#The black kde plot shows how distribution when when we fill the null values with the mean value


# In[ ]:


df_mice.head()


# In[ ]:


df_mice.Survived.unique()


# In[ ]:


#Spiltting Test and Train Datas:
dfm_train = df_mice[df_mice['Survived'] != 999]
dfm_test  = df_mice[df_mice['Survived'] == 999]

print('Train Shape :',dfm_train.shape)
print('Test Shape :',dfm_test.shape)

dfm_test.drop(columns = 'Survived' , inplace = True)


# In[ ]:


dfm_train.head()


# ### Applying XGBoost, (Without Hyperparameter Tuning)

# In[ ]:


import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
X, y = dfm_train.drop(columns = ['Survived','PassengerId']) , dfm_train['Survived']
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=123)
xg_cl = xgb.XGBClassifier(objective='binary:logistic',
n_estimators=20, seed=123)

xg_cl.fit(X_train, y_train)
preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test , preds)


# In[ ]:


y_pred =  xg_cl.predict(dfm_test.drop(columns='PassengerId')).astype('int')

results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_pred})
results.to_csv('Titanic Prediction_XGB.csv', index=False)


# ### **XGBoost with Hyperparameter Tuning using RandomizedSearchCV**

# In[ ]:


import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
X, y = dfm_train.drop(columns = ['Survived','PassengerId']) , dfm_train['Survived']
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=123)

# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 3,5,7 , 10],
        'gamma': [0.5, 1, 1.5, 2,3,4, 5],
        'subsample': [0.6,0.7, 0.8,0.9, 1.0],
        'colsample_bytree': [0.6,0.7, 0.8,0.9, 1.0],
        'max_depth': [3, 4, 5 , 6]
        }

xgb = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )

random_search.fit(X, y)


# In[ ]:


random_search.best_params_


# **Predicting the values on Test Dataset**

# In[ ]:


y_test = random_search.predict(dfm_test.drop(columns='PassengerId')).astype('int')

results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_test})
results.to_csv('Titanic Prediction_XGB_hp.csv', index=False)

#Got final score of 0.79904


# ### Checking the accuracy on Fancyimpute (Knn) imputed dataframe :

# In[ ]:


#Spiltting Test and Train Datas:
dfm_train = df_knn[df_mice['Survived'] != 999]
dfm_test  = df_knn[df_mice['Survived'] == 999]

print('Train Shape :',dfm_train.shape)
print('Test Shape :',dfm_test.shape)

dfm_test.drop(columns = 'Survived' , inplace = True)


# In[ ]:


import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
X, y = dfm_train.drop(columns = ['Survived','PassengerId']) , dfm_train['Survived']
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=123)

# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 3,5,7 , 10],
        'gamma': [0.5, 1, 1.5, 2,3,4, 5],
        'subsample': [0.6,0.7, 0.8,0.9, 1.0],
        'colsample_bytree': [0.6,0.7, 0.8,0.9, 1.0],
        'max_depth': [3, 4, 5 , 6]
        }

xgb = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )

random_search.fit(X, y)


# In[ ]:


y_test = random_search.predict(dfm_test.drop(columns='PassengerId')).astype('int')

results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_test})
results.to_csv('Titanic Predictionkn.csv', index=False)

#Got a score of 0.7666


# Mice score was better compared to Knn

# ### STACKING

# In[ ]:


#Spiltting Test and Train Datas:
dfm_train = df_mice[df_mice['Survived'] != 999]
dfm_test  = df_mice[df_mice['Survived'] == 999]

print('Train Shape :',dfm_train.shape)
print('Test Shape :',dfm_test.shape)

dfm_test.drop(columns = 'Survived' , inplace = True)


# In[ ]:


#KNN randomized search

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
from scipy.stats import randint as sp_randint

knn = KNeighborsClassifier()

params = {
    'n_neighbors' : sp_randint(1 , 20) ,
    'p' : sp_randint(1 , 5) ,
}

rsearch_knn = RandomizedSearchCV(knn , param_distributions = params , cv = 3 , random_state= 3  , n_jobs = -1 , return_train_score=True)

rsearch_knn.fit(X , y)


# In[ ]:


rsearch_knn.best_params_


# In[ ]:


#Random Forest randomized search
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=3)
params = { 'n_estimators' : sp_randint(50 , 200) , 
           'max_features' : sp_randint(1 , 12) ,
           'max_depth' : sp_randint(2,10) , 
           'min_samples_split' : sp_randint(2,20) ,
           'min_samples_leaf' : sp_randint(1,20) ,
           'criterion' : ['gini' , 'entropy']
    
}

rsearch_rfc = RandomizedSearchCV(rfc , param_distributions= params , n_iter= 200 , cv = 3 , scoring='roc_auc' , random_state= 3 , return_train_score=True , n_jobs=-1)

rsearch_rfc.fit(X,y)


# In[ ]:


rsearch_rfc.best_params_


# In[ ]:


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = 'liblinear')
knn = KNeighborsClassifier(**rsearch_knn.best_params_)
rfc = RandomForestClassifier(**rsearch_rfc.best_params_)

clf = VotingClassifier(estimators=[('lr' ,lr) , ('knn' , knn) , ('rfc' , rfc)] , voting = 'soft')

clf.fit(X , y)


# In[ ]:


y_test = clf.predict(dfm_test.drop(columns='PassengerId')).astype('int')

results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_test})
results.to_csv('Titanic Prediction_Stack.csv', index=False)


# ### **RANDOM FOREST :**

# In[ ]:


X, y = dfm_train.drop(columns = ['Survived','PassengerId']) , dfm_train['Survived']
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=123)


# In[ ]:


RFM = RandomForestClassifier(criterion='gini',
                                           n_estimators=1750,
                                           max_depth=7,
                                           min_samples_split=6,
                                           min_samples_leaf=6,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=123,
                                           n_jobs=-1,
                                           verbose=1) 

RFM.fit(X,y)


# In[ ]:


y_pred = RFM.predict(X_test)


# In[ ]:


y_test = RFM.predict(dfm_test.drop(columns='PassengerId')).astype('int')

results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_test})
results.to_csv('Titanic PredictionRFM.csv', index=False)


# **EXTRA TREE CLASSIFIER :**

# In[ ]:


X, y = dfm_train.drop(columns = ['Survived','PassengerId']) , dfm_train['Survived']
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2, random_state=123)


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
import sklearn.model_selection as model_selection

clf_ET = ExtraTreesClassifier(random_state=0, bootstrap=True, oob_score=True)

sss = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state= 0)
sss.get_n_splits(X, y)

parameters = {'n_estimators' : np.r_[10:210:10],
              'max_depth': np.r_[1:6]
             }

grid = model_selection.GridSearchCV(clf_ET, param_grid=parameters, scoring = 'accuracy', cv = sss, return_train_score=True, n_jobs=4, verbose=2)
grid.fit(X,y)


# In[ ]:


y_test = RFM.predict(dfm_test.drop(columns='PassengerId')).astype('int')

results = pd.DataFrame(data={'PassengerId':dfm_test['PassengerId'].astype('int'), 'Survived':y_test})
results.to_csv('Titanic PredictionETC.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




