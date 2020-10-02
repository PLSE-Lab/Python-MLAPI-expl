#!/usr/bin/env python
# coding: utf-8

# # Checking the Dataset Available

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Loading Module

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # setting seaborn default for plots

# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")


# # Loading Dataset

# In[ ]:


train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')


# # General Data Exploration

# ## Explore Training Dataset

# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.dtypes


# In[ ]:


train.describe()


# In[ ]:


train.describe(include=['O'])


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# ## Explore Testing Dataset

# In[ ]:


test.shape


# In[ ]:


test.head()


# In[ ]:


test.describe()


# In[ ]:


test.describe(include=['O'])


# In[ ]:


test.info()


# In[ ]:


test.isnull().sum()


# # Exploratory Data Analysis (EDA)

# In[ ]:


survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

print ('Survived: %i (%.1f%%)' %(len(survived), float(len(survived))/len(train)*100.0))
print ('Not Survived: %i (%.1f%%)' %(len(not_survived), float(len(not_survived))/len(train)*100.0))
print('Total: %i' %(len(train)))


# ## Pclass vs Survival

# > Higher Pclass have better survival chance

# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean()


# In[ ]:


sns.barplot(x='Pclass',y='Survived', data=train, ci=None)


# ## Sex vs Survival

# > Females have better survival chance

# In[ ]:


train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()


# In[ ]:


sns.barplot(x='Sex', y='Survived',data=train, ci=None)


# ## Pclass & Sex vs Survival

# > - There are more males among the 3rd Pclass
# > - Women from 1st and 2nd Pclass have almost 100% survival chance
# > - Male from 2nd and 3rd Pclass have only around 10% survival chance

# In[ ]:


tab = pd.crosstab(train['Pclass'], train['Sex'])
print(tab)

tab.div(tab.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')


# In[ ]:


sns.factorplot('Sex','Survived',hue='Pclass', size=4, aspect=2, data=train)


# ## Embarked vs Survival

# > Passengers embarked from C (Cherbourg) have better survival chance than other ambarkation places (Southampton and Queenstown)

# In[ ]:


train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()


# In[ ]:


sns.barplot(x='Embarked',y='Survived',data=train, ci=None)


# ## Pclass, Sex & Embarked vs Survival

# > - almost all females from Pclass 1 and 2 survived
# > - Female dying were mostly from 3rd Pclass
# > - Males from Pclass 1 only have slightly higher survival chance than 2nd and 3rd class

# In[ ]:


sns.factorplot(x='Pclass', y='Survived',hue='Sex',col='Embarked',data=train)


# ## Parch vs Survival

# In[ ]:


train[['Parch','Survived']].groupby(['Parch'], as_index=False).mean()


# In[ ]:


sns.barplot(x='Parch',y='Survived',data=train, ci=None)


# ## SibSp vs Survival

# In[ ]:


train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()


# In[ ]:


sns.barplot(x='SibSp',y='Survived',data=train, ci=None)


# ## Age vs Survival

# > from above **Pclass** violinplot:
# - 1st Pclass has very few children as compared to other two classes
# - 1st Pclass has more old people as compared to other two classes
# - Almost all children (between age 0 to 10) of 2nd Pclass survived
# - Most children of 3rd Pclass survived
# - Younger people of 1st Pclass survived as compared to its older people

# >from above **Sex** violinplot:
# - Most male children (age 0 to 14) survived
# - Female with age between 18 to 40 have better survival chance

# In[ ]:


fig=plt.figure(figsize=(8,11))
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)

sns.violinplot(x='Embarked', y='Age', hue='Survived', data=train, ax=ax1, split=True)
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=train, ax=ax2, split=True)
sns.violinplot(x='Sex', y='Age', hue='Survived', data=train, ax=ax3, split=True)


# > - Combining both male and female , we can see that children with age between 0 to 5 have better survival chance
# > - Females with age "18 to 40" and "50 and above" have higher survival chance
# > - Males with age between 0 to 14 have better chance of survival

# In[ ]:


total_survived = train[train['Survived']==1]
total_not_survived =  train[train['Survived']==0]
plt.figure(figsize=(12,12))
plt.subplot(111)
sns.distplot(total_survived['Age'].dropna().values, bins=range(0,81,1), kde=False, color='blue')
sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0,81,1), kde=False, color='red', axlabel='Age')


# In[ ]:


male_survived = train[(train['Survived']==1) & (train['Sex']=='male')]
female_survived = train[(train['Survived']==1) & (train['Sex']=='female')]
male_not_survived = train[(train['Survived']==0) & (train['Sex']=='male')]
female_not_survived = train[(train['Survived']==0) & (train['Sex']=='female')]

plt.figure(figsize=(15,15))
plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values, bins=range(0,81,1),kde=False, color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0,81,1),kde=False, color='red', axlabel='Female Age')

plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values, bins=range(0,81,1),kde=False, color='blue')
sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0,81,1),kde=False, color='red', axlabel='Female Age')


# ## Correlating Features

# > The features that are strongly correlated with 'Survived' so far are:
# - Pclass
# - Fare
# 
# > But at the same time those features (Pclass and Fare) are having correlation with:
# - Age
# - Parch
# - SibSp

# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)


# ## Features Extraction

# In[ ]:


train_test_data=[train, test] # combining train and test set


# ## Title

# In[ ]:


# extract title from Name column
for dataset in train_test_data:
    dataset['Title']=dataset.Name.str.extract(' ([A-Za-z]+)\.')


# In[ ]:


for dataset in train_test_data:
    dataset['Title']=dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


## Distribution Age by Title (Master, Miss, Mr, Mrs, Other)
dist_age_title=train['Age'].hist(by=train['Title'], bins=np.arange(0,81,1))


# In[ ]:


# max Age passengers with Title Master (training)
train[train['Title']=='Master']['Age'].max()


# In[ ]:


# max Age passengers with Title Master (testing)
test[test['Title']=='Master']['Age'].max()


# It will be better to simplify based on the Title and create new feature named 'Person' containing categorical value:
# - *Child* (passenger with Age less than 16) - value:0
# - *Female* (passenger with Age greater than to 16 and Sex='Female') - value: 1
# - *Male* (passenger with Age greater than to 16 and Sex='Male') - value: 2

# In[ ]:


def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
for dataset in train_test_data:
    dataset['Person']=dataset[['Age','Sex']].apply(get_person, axis=1)


# In[ ]:


# Impute Person column for missing value in Age
for dataset in train_test_data:
    dataset['Person'][ np.isnan(dataset['Age']) & (dataset['Title']=='Master') ] = 'child'
    dataset['Person'][ np.isnan(dataset['Age']) & (dataset['Title']!='Master') ] = dataset['Sex']


# In[ ]:


# convert Title into numerical value

title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Other":5}
for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset["Title"]=dataset["Title"].fillna(0)


# In[ ]:


# convert Person to numerical value

for dataset in train_test_data:
    dataset['Person']=dataset['Person'].map({'child':0,"female":1,"male":2}).astype(int)


# In[ ]:


train[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()


# ## Sex

# In[ ]:


# convert Sex to numerical value
for dataset in train_test_data:
    dataset['Sex']=dataset['Sex'].map({'female':0,"male":1}).astype(int)


# ## Embarked

# In[ ]:


train.Embarked.value_counts()


# In[ ]:


# Impute the missing value by the mode value which is 'S'
for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].fillna('S')


# In[ ]:


# convert Embarked to numerical value
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# ## Age

# In[ ]:


# impute the missing value with random integer between (mean +/- std)
for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['AgeBand'] = pd.cut(train['Age'], 5)


# In[ ]:


# map the ageBand
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


# ## Fare

# In[ ]:


# impute the missing value with median
for dataset in train_test_data:
    dataset['Fare']=dataset['Fare'].fillna(dataset['Fare'].median())


# In[ ]:


# create FareBand divided into 4 parts
train['FareBand']=pd.qcut(train['Fare'],4)
print(train[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean())


# In[ ]:


# Map the Fare according to FareBand
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[ ]:


train.head()


# ## FamilySize vs Survival

# > - Having FamilySize upto 4 (from 2 to 4) has better survival chance
# > - FamilySize = 1, i.e. travelling alone has less survival chance
# > - Large FamilySize (size of 5 and above) also have less survival chance

# In[ ]:


# Create New Feature named FamilySize by combining SibSp & Parch
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# ## IsAlone

# > travelling alone has only 30% survival chance

# In[ ]:


# Create Feature named IsAlone
for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# ## Relationship between Features

# In[ ]:


cols = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','Person','FamilySize','IsAlone']
g = sns.pairplot(data=train, vars=cols, size=1.5,
                 hue='Survived', palette=['red',"blue"])
g.set()


# # Feature Engineering (Feature Selection)

# > The features that are strongly correlated with 'Survived' so far are:
# - Pclass
# - Fare
# 
# > But at the same time those features (Pclass and Fare) are having correlation with:
# - Age
# - Parch
# - SibSp

# In[ ]:


fig=plt.figure(figsize=(15,15))
sns.heatmap(train.corr(), vmax=0.6, square=True, annot=True)


# In[ ]:


# drop unnecessary features
features_drop=['Name','Age','SibSp','Parch','Ticket','Cabin','FamilySize','Sex','Title']
train=train.drop(features_drop,axis=1)
train=train.drop(['PassengerId','AgeBand','FareBand'],axis=1)
test=test.drop(features_drop,axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Model Classification

# ## Classification & Accuracy

# > Several classifier algorithms used as following:
# - Logistic Regression
# - Support Vector Machine (SVM)
# - Linear SVM
# - K-Nearest Neigbor (KNN)
# - Decision Tree
# - Random Forest
# - Naive Bayes (Gaussian NB)
# - Perceptron
# - Stochastic Gradient Descent (SGD)

# In[ ]:


# defining the training and testing dataset
X_train = train.drop('Survived',axis=1)
y_train = train['Survived']
X_test=test.drop('PassengerId',axis=1).copy()
X_train.shape, y_train.shape, X_test.shape


# > Training and Testing Procedurs:<br>
# 1. First, we train these classifiers with our training dataset
# 2. Using trained classifier, predict the survival outcome of test data
# 3. Calculate the accuracy score (in %) of trained classifier

# In[ ]:


# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score


# ### Logistic Regression

# In[ ]:


clf_log_reg = LogisticRegression()
clf_log_reg.fit(X_train, y_train)
acc_log_reg=round((cross_val_score(clf_log_reg,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print(str(acc_log_reg)+"%")


# ### Support Vector Machine (SVM)

# In[ ]:


clf_svc=SVC() # Support Vector Classification
clf_svc.fit(X_train, y_train)
acc_svc=round((cross_val_score(clf_svc,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print(str(acc_svc)+"%")


# ### Linear SVM

# In[ ]:


clf_linear_svc = LinearSVC() # Linear Support Vector Classification
clf_linear_svc.fit(X_train, y_train)
acc_linear_svc=round((cross_val_score(clf_linear_svc,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print(str(acc_linear_svc)+"%")


# ### K-Nearest Neighbors

# In[ ]:


clf_knn=KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train, y_train)
acc_knn=round((cross_val_score(clf_knn,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print(str(acc_knn)+"%")


# ### Decision Tree

# In[ ]:


clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)
acc_decision_tree=round(clf_dt.score(X_train, y_train)*100,2)
acc_decision_tree=round((cross_val_score(clf_dt,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print(str(acc_decision_tree)+"%")


# ### Random Forest

# In[ ]:


clf_rf = RandomForestClassifier(n_estimators=100)
clf_rf.fit(X_train, y_train)
acc_random_forest=round((cross_val_score(clf_rf,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print(str(acc_random_forest)+"%")
#clf_rf.score(X_train, y_train)


# In[ ]:


y_train.head()


# In[ ]:


X_train.head()


# ### Extremely Randomised Trees

# In[ ]:


clf_ext_rf = ExtraTreesClassifier(
    max_features='auto',
    bootstrap=True,
    oob_score=True,
    n_estimators=1000,
    max_depth=None,
    min_samples_split=10
    #class_weight="balanced",
    #min_weight_fraction_leaf=0.02
    )
clf_ext_rf.fit(X_train,y_train)
acc_ext_rf=round(clf_ext_rf.score(X_train, y_train)*100,2)
acc_ext_rf=round((cross_val_score(clf_ext_rf,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print(str(acc_ext_rf)+"%")


# ### Bagging (with estimator KNN)

# In[ ]:


clf_bagging = BaggingClassifier(
    KNeighborsClassifier(
        n_neighbors=3,
        weights='distance'
        ),
    oob_score=True,
    max_samples=0.5,
    max_features=1.0
    )
clf_bagging.fit(X_train,y_train)
acc_bagging=round((cross_val_score(clf_bagging,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print(str(acc_bagging)+"%")


# ### Gaussian Naive Bayes

# In[ ]:


clf_gnb=GaussianNB()
clf_gnb.fit(X_train, y_train)
acc_gnb=round((cross_val_score(clf_gnb,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print(str(acc_gnb)+"%")


# ### Perceptron

# In[ ]:


clf_perceptron=Perceptron(max_iter=5, tol=None)
clf_perceptron.fit(X_train, y_train)
acc_perceptron=round((cross_val_score(clf_perceptron,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print(str(acc_perceptron)+"%")


# ### Stochastic Gradient Descent (SGD)

# In[ ]:


clf_sgd = SGDClassifier(max_iter=5, tol=None)
clf_sgd.fit(X_train, y_train)
acc_sgd=round((cross_val_score(clf_sgd,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print (str(acc_sgd)+"%")


# ### Gradient Boosting

# In[ ]:


#import warnings
#warnings.filterwarnings("ignore")

clf_gb = GradientBoostingClassifier(
            #loss='exponential',
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.5,
            random_state=0)
clf_gb.fit(X_train,y_train)
acc_gb=round((cross_val_score(clf_gb,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print (str(acc_gb)+"%")


# ### XGBoost

# In[ ]:


clf_xgb = xgb.XGBClassifier(
    max_depth=2,
    n_estimators=500,
    subsample=0.5,
    learning_rate=0.1
    )
clf_xgb.fit(X_train,y_train)
acc_xgb=round((cross_val_score(clf_xgb,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print (str(acc_xgb)+"%")


# ### LightGBM

# In[ ]:


clf_lgb = lgb.LGBMClassifier(
    max_depth=2,
    n_estimators=500,
    subsample=0.5,
    learning_rate=0.1
    )
clf_lgb.fit(X_train,y_train)
acc_lgb=round((cross_val_score(clf_lgb,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print (str(acc_lgb)+"%")


# ### Ada Boosting

# In[ ]:


clf_ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
clf_ada.fit(X_train,y_train)
acc_ada=round((cross_val_score(clf_ada,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print (str(acc_ada)+"%")


# ### Stacking Method (Voting Mechanism Principal)

# In[ ]:


clf_vote=VotingClassifier(
estimators=[
    #('tree',clf_dt),
    #('knn',clf_knn),
    ('svm',clf_svc),
    #('extra',clf_ext_rf),
   #('gb',clf_gb),
    ('xgb',clf_xgb),
    ('ada',clf_ada),
    #('bagging',clf_bagging),
    #('percep',clf_perceptron),
    #('logistic',clf_log_reg),
    ('lightgbm', clf_lgb),
    ('RF',clf_rf)
],
weights=[3,2,1,3,3],
voting='hard')

clf_vote.fit(X_train,y_train)
acc_vote=cross_val_score(clf_vote,X_train, y_train, cv=5,scoring='accuracy')
print("Voting Accuracy: {:.2%}".format(acc_vote.mean())+" (+/-{:.2%})".format(acc_vote.std()))


# # Model Validation

# ## Comparing Model

# > Decided to use *Stacking Methods* Classifier as it provides high accuracy and more stability

# In[ ]:


models=pd.DataFrame({'Model':['Logistic Regression', "Support Vector Machine",'Linear SVM','KNN','Decision Tree','Random Forest','Extremely Randomised Trees','Bagging','Naive Bayes','Perceptron','Stochastic Gradient Descent'
                             ,'Gradient Boosting','XGBoost','LightGBM','Ada Boosting','Stacking'],
                     'Score':[acc_log_reg, acc_svc,acc_linear_svc,acc_knn,acc_decision_tree,acc_random_forest, acc_ext_rf ,acc_bagging,acc_gnb, acc_perceptron,acc_sgd,acc_gb,acc_xgb,acc_lgb,acc_ada,(acc_vote.mean()*100)]})
models.sort_values(by='Score',ascending=False)


# ## Confusion Matrix

# In[ ]:


# As the example using the Random Forest Classifier
from sklearn.metrics import confusion_matrix
import itertools


# In[ ]:


# We use Stacking Method as an example because the classifier provides good accuracy and stability
clf=VotingClassifier(
estimators=[
    #('tree',clf_dt),
    #('knn',clf_knn),
    ('svm',clf_svc),
    #('extra',clf_ext_rf),
   #('gb',clf_gb),
    ('xgb',clf_xgb),
    ('ada',clf_ada),
    #('bagging',clf_bagging),
    #('percep',clf_perceptron),
    #('logistic',clf_log_reg),
    ('lightgbm', clf_lgb),
    ('RF',clf_rf)
],
weights=[3,2,1,3,3],
voting='hard')
clf.fit(X_train, y_train)
y_pred_vote_training_set = clf.predict(X_train)
acc_vote_training_set=round((cross_val_score(clf,X_train, y_train, cv=5,scoring='accuracy').mean())*100,2)
print(str(acc_vote_training_set)+"%")


# In[ ]:


# Confusion matrix in number
true_class_names = ['True Survived','True Not Survived']
pred_class_names = ['Predicted Survived','Predicted Not Survived']
cnf_matrix = confusion_matrix(y_train, y_pred_vote_training_set)
df_cnf_matrix = pd.DataFrame(cnf_matrix, index=true_class_names, columns=pred_class_names)
df_cnf_matrix


# In[ ]:


# Confusion matrix in percentage
true_class_names = ['True Survived','True Not Survived']
pred_class_names = ['Predicted Survived','Predicted Not Survived']
cnf_matrix_percent = cnf_matrix.astype('float')/cnf_matrix.sum(axis=1)[:,np.newaxis]
df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, index=true_class_names, columns=pred_class_names)
df_cnf_matrix_percent


# In[ ]:


plt.figure(figsize = (10,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)


# In[ ]:


# Check the contribution of each features

summary_df=pd.DataFrame(list(zip(X_test.columns,
                      #clf_log_reg.feature_importances_,
                      #clf_svc.feature_importances_,
                      #clf_linear_svc.feature_importances_,
                      #clf_knn.feature_importances_,
                      clf_dt.feature_importances_,
                     clf_rf.feature_importances_,
                     clf_ext_rf.feature_importances_,
                     #clf_bagging.feature_importances_,
                     #clf_gnb.feature_importances_,
                     #clf_perceptron.feature_importances_,
                     #clf_sgd.feature_importances_,
                     clf_gb.feature_importances_,
                     clf_xgb.feature_importances_,
                     clf_lgb.feature_importances_,
                                 #clf_vote.feature_importances_,
                     clf_ada.feature_importances_
                     )),
             columns=['Feature','Tree','RF','Ext RF','GBoost','XGBoost','lightGB','AdaBoost'])
summary_df['Median']=summary_df.median(1)
summary_df.sort_values('Median',ascending=False)


# # Submission File Output

# In[ ]:


test.head()


# In[ ]:


y_pred_vote=clf_vote.predict(X_test)


# In[ ]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],
                           'Survived':y_pred_vote})


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission_output.csv', index=False)

