# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
#For Notebook
#%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.cross_validation import KFold

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_train=pd.read_csv('../input/train.csv',sep=',')
df_test=pd.read_csv('../input/test.csv',sep=',')
df_data = df_train.append(df_test) # The entire data: train + test.

PassengerId = df_test['PassengerId']
Submission=pd.DataFrame()
Submission['PassengerId'] = df_test['PassengerId']

df_data.info

df_train.head(5)

grid = sns.FacetGrid(df_train, col = "Pclass", row = "Sex", hue = "Survived", palette = 'seismic')
grid = grid.map(plt.scatter, "PassengerId", "Age")
grid.add_legend()
grid

NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Fare']

# create test and training data
data_to_train = df_train[NUMERIC_COLUMNS].fillna(-1000)
y=df_train['Survived']
X=data_to_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=21, stratify=y)

from sklearn.svm import LinearSVC

clf = SVC()
clf.fit(X_train, y_train)
linear_svc = LinearSVC()

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))

test = df_test[NUMERIC_COLUMNS].fillna(-1000)
Submission['Survived']=clf.predict(test)
print(Submission.head())
print('predictions generated')

# write data frame to csv file
#Submission.set_index('PassengerId', inplace=True)
#Submission.to_csv('myfirstsubmission.csv',sep=',')
print('file created')

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=(12,6))
sns.boxplot(data = df_train, x = "Pclass", y = "Fare",ax=ax1);
plt.figure(1)
sns.boxplot(data = df_train, x = "Embarked", y = "Fare",ax=ax2);
plt.show()

#Fare
# Fill the na values in Fare based on embarked data
#Could have improved using Pclass & Embarked
embarked = ['S', 'C', 'Q']
for port in embarked:
    fare_to_impute = df_data.groupby(['Embarked','Pclass'])['Fare'].median()[embarked.index(port)]
    df_data.loc[(df_data['Fare'].isnull()) & (df_data['Embarked'] == port), 'Fare'] = fare_to_impute
# Fare in df_train and df_test:
df_train["Fare"] = df_data['Fare'][:891]
df_test["Fare"] = df_data['Fare'][891:]
print('Missing Fares Estimated')

#fill in missing Fare value in training set based on mean fare for that Pclass 
for x in range(len(df_train["Fare"])):
    if pd.isnull(df_train["Fare"][x]):
        pclass = df_train["Pclass"][x] #Pclass = 3
        df_train["Fare"][x] = round(df_train[df_train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#fill in missing Fare value in test set based on mean fare for that Pclass         
for x in range(len(df_test["Fare"])):
    if pd.isnull(df_test["Fare"][x]):
        pclass = df_test["Pclass"][x] #Pclass = 3
        df_test["Fare"][x] = round(df_test[df_test["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
df_data["FareBand"] = pd.qcut(df_data['Fare'], 4, labels = [1, 2, 3, 4]).astype('int')
df_train["FareBand"] = pd.qcut(df_train['Fare'], 4, labels = [1, 2, 3, 4]).astype('int')
df_test["FareBand"] = pd.qcut(df_test['Fare'], 4, labels = [1, 2, 3, 4]).astype('int')
df_train[["FareBand", "Survived"]].groupby(["FareBand"], as_index=False).mean()
print('FareBand feature created')

#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
df_data["Embarked"] = df_data["Embarked"].map(embarked_mapping)
# split Embanked into df_train and df_test:
df_train["Embarked"] = df_data["Embarked"][:891]
df_test["Embarked"] = df_data["Embarked"][891:]
print('Embarked feature created')
df_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()

# Fill the na values in Embanked based on fareband data
fareband = [1,2,3,4]
for fare in fareband:
    embark_to_impute = df_data.groupby('FareBand')['Embarked'].median()[fare]
    df_data.loc[(df_data['Embarked'].isnull()) & (df_data['FareBand'] == fare), 'Embarked'] = embark_to_impute
# Fare in df_train and df_test:
df_train["Embarked"] = df_data['Embarked'][:891]
df_test["Embarked"] = df_data['Embarked'][891:]
print('Missing Embarkation Estimated')

#Gender
# convert categories to Columns
dummies=pd.get_dummies(df_train[['Sex']], prefix_sep='_') #Gender
df_train = pd.concat([df_train, dummies], axis=1) 
testdummies=pd.get_dummies(df_test[['Sex']], prefix_sep='_') #Gender
df_test = pd.concat([df_test, testdummies], axis=1) 
print('Gender Feature added ')

print(dummies)

#map each Gendre value to a numerical value
gender_mapping = {"female": 0, "male": 1}
df_data["Sex"] = df_data['Sex'].map(gender_mapping)
df_data["Sex"]=df_data["Sex"].astype('int')

# Family_Survival in TRAIN_DF and TEST_DF:
df_train["Sex"] = df_data["Sex"][:891]
df_test["Sex"] = df_data["Sex"][891:]
print('Gender Category created')

# Title Feature
#Get titles
df_data["Title"] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#Unify common titles. 
df_data["Title"] = df_data["Title"].replace('Mlle', 'Miss')
df_data["Title"] = df_data["Title"].replace('Master', 'Master')
df_data["Title"] = df_data["Title"].replace(['Mme', 'Dona', 'Ms'], 'Mrs')
df_data["Title"] = df_data["Title"].replace(['Jonkheer','Don'],'Mr')
df_data["Title"] = df_data["Title"].replace(['Capt','Major', 'Col','Rev','Dr'], 'Millitary')
df_data["Title"] = df_data["Title"].replace(['Lady', 'Countess','Sir'], 'Honor')

# Age in df_train and df_test:
df_train["Title"] = df_data['Title'][:891]
df_test["Title"] = df_data['Title'][891:]

# convert Title categories to Columns
titledummies=pd.get_dummies(df_train[['Title']], prefix_sep='_') #Title
df_train = pd.concat([df_train, titledummies], axis=1) 
ttitledummies=pd.get_dummies(df_test[['Title']], prefix_sep='_') #Title
df_test = pd.concat([df_test, ttitledummies], axis=1) 
print('Title categories added')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Millitary": 5, "Honor": 6}
df_data["TitleCat"] = df_data['Title'].map(title_mapping)
df_data["TitleCat"] = df_data["TitleCat"].astype(int)
df_train["TitleCat"] = df_data["TitleCat"][:891]
df_test["TitleCat"] = df_data["TitleCat"][891:]
print('Title Category created')

#Fill in missing ages
#Could use more wise way
titles = ['Master', 'Miss', 'Mr', 'Mrs', 'Millitary','Honor']
for title in titles:
    age_to_impute = df_data.groupby('Title')['Age'].median()[title]
    df_data.loc[(df_data['Age'].isnull()) & (df_data['Title'] == title), 'Age'] = age_to_impute
# Age in df_train and df_test:
df_train["Age"] = df_data['Age'][:891]
df_test["Age"] = df_data['Age'][891:]
print('Missing Ages Estimated')

# Visualise Age Data 
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Training Age values - Titanic')
axis2.set_title('Test Age values - Titanic')

# plot original Age values
df_train['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
#df_test['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
        
# plot new Age Values
#df_train['Age'].hist(bins=70, ax=axis2)
df_test['Age'].hist(bins=70, ax=axis2)

# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(df_train, hue="Survived",palette = 'seismic',aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, df_train['Age'].max()))
facet.add_legend()

#Lone Travellers
df_train["Alone"] = np.where(df_train['SibSp'] + df_train['Parch'] + 1 == 1, 1,0) # People travelling alone
df_test["Alone"] = np.where(df_test['SibSp'] + df_test['Parch'] + 1 == 1, 1,0) # People travelling alone
print('Lone traveller feature created')

#Family Size
df_train["Family Size"] = (df_train['SibSp'] + df_train['Parch'] + 1)
df_test["Family Size"] = df_test['SibSp'] + df_test['Parch'] + 1
print('Family size feature created')

#Family Survival
# get last name
df_data["Last_Name"] = df_data['Name'].apply(lambda x: str.split(x, ",")[0])
# Set survival value
DEFAULT_SURVIVAL_VALUE = 0.5
df_data["Family_Survival"] = DEFAULT_SURVIVAL_VALUE

# Find Family groups by Fare
for grp, grp_df in df_data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      df_data.loc[df_data['Family_Survival']!=0.5].shape[0])

# Find Family groups by Ticket
for _, grp_df in df_data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(df_data[df_data['Family_Survival']!=0.5].shape[0]))

# Family_Survival in df_train and df_test:
df_train["Family_Survival"] = df_data['Family_Survival'][:891]
df_test["Family_Survival"] = df_data['Family_Survival'][891:]

#Cabin
# check if cabin inf exists
df_data["HadCabin"] = (df_data["Cabin"].notnull().astype('int'))
# split Embanked into df_train and df_test:
df_train["HadCabin"] = df_data["HadCabin"][:891]
df_test["HadCabin"] = df_data["HadCabin"][891:]
print('Cabin feature created')

#Deck
# Extract Deck
df_data["Deck"] = df_data.Cabin.str.extract('([A-Za-z])', expand=False)
df_data["Deck"] = df_data["Deck"].fillna("N")
# Map Deck
deck_mapping = {"N":0,"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
df_data['Deck'] = df_data['Deck'].map(deck_mapping)
#Split to training and test
df_train["Deck"] = df_data["Deck"][:891]
df_test["Deck"] = df_data["Deck"][891:]
print('Deck feature created')

#Other Missing Data
#check for any other unusable values
print(pd.isnull(df_test).sum())


#Age Visulization
# Groupby title
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# plot age distribution by title
facet = sns.FacetGrid(data = df_train, hue = "Title", legend_out=True, size = 5)
facet = facet.map(sns.kdeplot, "Age")
facet.add_legend();

# Re-evaluate with new features

NUMERIC_COLUMNS=['Alone','Family Size','Sex','Pclass','Fare','FareBand','Age','TitleCat','Embarked'] #72
ORIGINAL_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #83
REVISED_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

# create test and training data
data_to_train = df_train[REVISED_NUMERIC_COLUMNS].fillna(-1000)
y=df_train['Survived']
X=data_to_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=21, stratify=y)
RandomForest = RandomForestClassifier(random_state = 0)
RandomForest.fit(X_train, y_train)
print('Evaluation complete')


RandomForest_checker = RandomForestClassifier()
RandomForest_checker.fit(X_train, y_train)
importances_df = pd.DataFrame(RandomForest_checker.feature_importances_, columns=['Feature_Importance'],
                              index=X_train.columns)
importances_df.sort_values(by=['Feature_Importance'], ascending=False, inplace=True)
print(importances_df)

# Print the accuracy# Print  
print("Accuracy: {}".format(RandomForest.score(X_test, y_test)))

test = df_test[REVISED_NUMERIC_COLUMNS].fillna(-1000)
Submission['Survived']=RandomForest.predict(test)
print(Submission.head())
print('Submission created')

# write data frame to csv file
#Submission.set_index('PassengerId', inplace=True)
#Submission.to_csv('finalrevised01.csv',sep=',')
print('file created')

#Other Models

#Split
from sklearn.model_selection import train_test_split
NUMERIC_COLUMNS=['Alone','Family Size','Sex','Pclass','Fare','Age','TitleCat','FareBand','Embarked']
ORIGINAL_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked']
REVISED_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

# create test and training data
predictors = df_train.drop(['Survived', 'PassengerId'], axis=1)
data_to_train = df_train[REVISED_NUMERIC_COLUMNS].fillna(-1000)
y = df_train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(data_to_train, y, test_size = 0.3,random_state=21, stratify=y)
print('Data split')

#SVC
clf = SVC()

clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)
acc_clf = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_clf)

# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)

# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(random_state = 0)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)

#ExtraTreesClassifier
# Gradient Boosting Classifier
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier()
et.fit(x_train, y_train)
y_pred = et.predict(x_val)
acc_et = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_et)

# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)

# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)

# xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=10)
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_val)
acc_xgb = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_xgb)

#Comparing
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Classifier','Extra Trees','Stochastic Gradient Descent','Perceptron','xgboost'],
    'Score': [acc_clf, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian,acc_linear_svc, acc_decisiontree,
             acc_gbk,acc_et,acc_sgd,acc_perceptron,acc_xgb]})
models.sort_values(by='Score', ascending=False)

#Reforcast based on best:XGBoost
test = df_test[REVISED_NUMERIC_COLUMNS].fillna(-1000)

#Submission['Survived']=xgb.predict(test)
print(Submission.head(5))
print('Prediction complete')

# write data frame to csv file
#Submission.set_index('PassengerId', inplace=True)
#Submission.to_csv('finalsubmission.csv',sep=',')
#Submission.to_csv('xgbbasicsubmission01.csv',sep=',')
print('File created')

#Hyper Tune
from sklearn.model_selection import train_test_split
NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Services','FareBand','Embarked','Alone','Family Size']
ORIGINAL_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Military','Embarked']
REVISED_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

# create test and training data
predictors = df_train.drop(['Survived', 'PassengerId'], axis=1)
data_to_train = df_train[REVISED_NUMERIC_COLUMNS].fillna(-1000)
X=data_to_train
y = df_train["Survived"]
X_train, X_val, y_train, y_val = train_test_split(data_to_train, y, test_size = 0.3,random_state=21, stratify=y)
print('Data Split')

#Linear Regression SVC
from sklearn.model_selection import GridSearchCV

# Support Vector Classifier parameters 
param_grid = {'C':np.arange(1, 7),
              'degree':np.arange(1, 7),
              'max_iter':np.arange(0, 12),
              'kernel':['rbf','linear'],
              'shrinking':[0,1]}

clf = SVC()
svc_cv=GridSearchCV(clf, param_grid, cv=10)
svc_cv.fit(X_train, y_train)

print("Tuned SVC Parameters: {}".format(svc_cv.best_params_))
print("Best score is {}".format(svc_cv.best_score_))
acc_svc_cv = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc_cv)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# create parameter grid as a dictionary where the keys are the hyperparameter names and the values are lists of values that we want to try.
param_grid = {"solver": ['newton-cg','lbfgs','liblinear','sag','saga'],'C': [0.01, 0.1, 1, 10, 100]}

# instanciate classifier
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

logreg_cv = GridSearchCV(logreg, param_grid, cv=30)
logreg_cv.fit(X_train, y_train)

y_pred = logreg_cv.predict(X_val)
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))
acc_logreg_cv = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg_cv)

# KNN or k-Nearest Neighbors with GridSearch
#Too slow, didn't do
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# create parameter grid as a dictionary where the keys are the hyperparameter names and the values are lists of values that we want to try.
param_grid = {"n_neighbors": np.arange(1, 50),
             "leaf_size": np.arange(20, 40),
             "algorithm": ["ball_tree","kd_tree","brute"]
             }
# instanciate classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_cv = GridSearchCV(knn, param_grid, cv=10)
knn_cv.fit(X_train, y_train)
y_pred = knn_cv.predict(X_val)
print("Tuned knn Parameters: {}".format(knn_cv.best_params_))
print("Best score is {}".format(knn_cv.best_score_))
acc_knn_cv = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn_cv)
'''

# DecisionTree with RandomizedSearch

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"random_state" :  np.arange(0, 10),
              "max_depth": np.arange(1, 10),
              "max_features": np.arange(1, 10),
              "min_samples_leaf": np.arange(1, 10),
              "criterion": ["gini","entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=30)

# Fit it to the data
tree_cv.fit(X_train,y_train)
y_pred = tree_cv.predict(X_val)
# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
acc_tree_cv = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_tree_cv)

# Random Forrest

# Import necessary modules
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"random_state" :  np.arange(0, 10),
              "n_estimators" :  np.arange(1, 20),
              "max_depth": np.arange(1, 10),
              "max_features": np.arange(1, 10),
              "min_samples_leaf": np.arange(1, 10),
              "criterion": ["gini","entropy"]}

# Instantiate a Decision Tree classifier: tree
randomforest = RandomForestClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
randomforest_cv = RandomizedSearchCV(randomforest, param_dist, cv=30)

# Fit it to the data
randomforest_cv.fit(X_train,y_train)
y_pred = randomforest_cv.predict(X_val)
# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(randomforest_cv.best_params_))
print("Best score is {}".format(randomforest_cv.best_score_))
acc_randomforest_cv = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest_cv)

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold;

# Setup the parameters and distributions to sample from: param_dist
param_dist = {'max_depth':np.arange(1, 7),
              'min_samples_leaf': np.arange(1, 6),
              "max_features": np.arange(1, 10),
             }

# Instantiate Classifier
gbk = GradientBoostingClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
gbk_cv = RandomizedSearchCV(gbk, param_dist, cv=30)

gbk_cv.fit(x_train, y_train)
y_pred = gbk_cv.predict(x_val)

print("Tuned Gradient Boost Parameters: {}".format(gbk_cv.best_params_))
print("Best score is {}".format(gbk_cv.best_score_))
acc_gbk_cv = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk_cv)

# xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {'learning_rate': [.01, .03, .05, .1, .25], #default: .3
            'max_depth': np.arange(1, 10), #default 2
            'n_estimators': [10, 50, 100, 300], 
            'booster':['gbtree','gblinear','dart']
            #'seed': 5  
             }
# Instantiate Classifier
xgb = XGBClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
xgb_cv = RandomizedSearchCV(xgb, param_dist, cv=20)

# Fit model
xgb_cv.fit(X_train, y_train)

# Make prediction
y_pred = xgb_cv.predict(X_val)

# Print results
print("xgBoost Parameters: {}".format(xgb_cv.best_params_))
print("Best score is {}".format(xgb_cv.best_score_))
acc_xgb_cv = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_xgb_cv)

#Didn't do KNN
optmodels = pd.DataFrame({
    'optModel': ['SVC','Decision Tree','Gradient Boost','Logistic Regression','xgboost'],
    'optScore': [svc_cv.best_score_,tree_cv.best_score_,gbk_cv.best_score_,logreg_cv.best_score_,xgb_cv.best_score_]})
optmodels.sort_values(by='optScore', ascending=False)

optmodels = pd.DataFrame({
    'optModel': ['Linear Regression','Decision Tree','Gradient Boost','Logistic Regression','xgboost'],
    'optScore': [acc_svc_cv,acc_tree_cv,acc_gbk_cv,acc_logreg_cv,acc_xgb_cv]})
optmodels.sort_values(by='optScore', ascending=False)

#Make Submission
# Select columns
test = df_test[REVISED_NUMERIC_COLUMNS].fillna(-1000)
# select classifier
tree = DecisionTreeClassifier(random_state=0,max_depth=5,max_features=7,min_samples_leaf=2,criterion="entropy") #85,87
#knn = KNeighborsClassifier(algorithm='kd_tree',leaf_size=20,n_neighbors=5)
#logreg = LogisticRegression(solver='newton-cg')
#xgboost=XGBClassifier(n_estimators= 300, max_depth= 10, learning_rate= 0.01)
#Tuned Decision Tree Parameters: {'random_state': 0, 'min_samples_leaf': 2, 'max_features': 7, 'max_depth': 5, 'criterion': 'entropy'} #85,87
#Tuned Decision Tree Parameters: {'random_state': 1, 'min_samples_leaf': 9, 'max_features': 8, 'max_depth': 6, 'criterion': 'entropy'}#84,88
#gbk=GradientBoostingClassifier(min_samples_leaf=1,max_features=4,max_depth=5)
# train model
tree.fit(X,y)
# make predictions
Submission['Survived']=tree.predict(test)
print(Submission.head(5))

# write data frame to csv file
#Submission.set_index('PassengerId', inplace=True)
#Submission.to_csv('treesubmission03.csv',sep=',')
print('File created')


#Hyper tuning with confusing matrix

#KNN Hyper tuning
# knn Hyper Tunning with confusion Matrix
'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics #accuracy measure
from sklearn.neighbors import KNeighborsClassifier

REVISED_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

# create test and training data
data_to_train = df_train[REVISED_NUMERIC_COLUMNS].fillna(-1000)
X_test2= df_test[REVISED_NUMERIC_COLUMNS].fillna(-1000)
prediction  = df_train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(data_to_train, prediction, test_size = 0.3,random_state=21, stratify=prediction)
print('Data Split')

hyperparams = {'algorithm': ['auto'], 'weights': ['uniform', 'distance'] ,'leaf_size': list(range(1,50,5)), 
               'n_neighbors':[6,7,8,9,10,11,12,14,16,18,20,22]}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, cv=10, scoring = "roc_auc")
gd.fit(X_train, y_train)

gd.best_estimator_.fit(X_train,y_train)
y_pred=gd.best_estimator_.predict(X_test)
Submission['Survived']=gd.best_estimator_.predict(X_test2)

# Print the results
print('Best Score')
print(gd.best_score_)
print('Best Estimator')
print(gd.best_estimator_)
acc_gd_cv = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy')
print(acc_gd_cv)

# Generate the confusion matrix and classification report
print('Confusion Matrrix')
print(confusion_matrix(y_test, y_pred))
print('Classification_report')
print(classification_report(y_test, y_pred))
#Submission.set_index('PassengerId', inplace=True)
print('Sample Prediction')
print(Submission.head(10))
Submission.to_csv('knngridsearch02.csv',sep=',')
print('KNN prediction created')
'''

# Decision Tree Hyper Tunning with confusion Matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics #accuracy measure
from sklearn.tree import DecisionTreeClassifier

REVISED_NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Family_Survival','Alone','Sex_female','Sex_male','Title_Master', 'Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Millitary','Embarked'] #84

# create test and training data
data_to_train = df_train[REVISED_NUMERIC_COLUMNS].fillna(-1000)
X_test2= df_test[REVISED_NUMERIC_COLUMNS].fillna(-1000)
prediction  = df_train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(data_to_train, prediction, test_size = 0.3,random_state=21, stratify=prediction)
print('Data Split')

hyperparams = {"random_state" :  np.arange(0, 10),
              "max_depth": np.arange(1, 10),
              "max_features": np.arange(1, 10),
              "min_samples_leaf": np.arange(1, 10),
              "criterion": ["gini","entropy"]}

gd=GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = hyperparams, verbose=True, cv=10, scoring = "roc_auc")
gd.fit(X_train, y_train)

gd.best_estimator_.fit(X_train,y_train)
y_pred=gd.best_estimator_.predict(X_test)
Submission['Survived']=gd.best_estimator_.predict(X_test2)

# Print the results
print('Best Score')
print(gd.best_score_)
print('Best Estimator')
print(gd.best_estimator_)
acc_gd_cv = round(accuracy_score(y_pred, y_val) * 100, 2)
print('Accuracy')
print(acc_gd_cv)

# Generate the confusion matrix and classification report
print('Confusion Matrrix')
print(confusion_matrix(y_test, y_pred))
print('Classification_report')
print(classification_report(y_test, y_pred))
#Submission.set_index('PassengerId', inplace=True)
print(Submission.head(10))
Submission.to_csv('Treegridsearch02.csv',sep=',')
print('Decision Tree prediction created')