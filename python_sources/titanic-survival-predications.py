#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Required Libraries

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from scipy import stats
import seaborn as sns
import matplotlib as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LassoCV, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb


# # 2. Loading and Reading Dataset

# Here we combine the test and training sets together to make cleaning the data easier.

# In[ ]:


train_path = '/kaggle/input/titanic/train.csv'
gender_path = '/kaggle/input/titanic/gender_submission.csv'
test_path = '/kaggle/input/titanic/test.csv'
train = pd.read_csv(train_path)
gender = pd.read_csv(gender_path)
test =pd.read_csv(test_path)
test['Survived'] = np.empty((len(test), 0)).tolist()
Total = train.append(test)
Total.set_index('PassengerId', inplace = True)
Total


# # 3. Data Processing

# First thing I'm doing is searching for any NaN values.

# In[ ]:


print('Number of missing data points in Survived:',Total['Survived'].isnull().sum())
print('Number of missing data points in Pclass:',Total['Pclass'].isnull().sum())
print('Number of missing data points in Name:',Total['Name'].isnull().sum())
print('Number of missing data points in Sex:',Total['Sex'].isnull().sum())
print('Number of missing data points in Age:',Total['Age'].isnull().sum())
print('Number of missing data points in SibSp:',Total['SibSp'].isnull().sum())
print('Number of missing data points in Parch:',Total['Parch'].isnull().sum())
print('Number of missing data points in Ticket:',Total['Ticket'].isnull().sum())
print('Number of missing data points in Fare:',Total['Fare'].isnull().sum())
print('Number of missing data points in Cabin:',Total['Cabin'].isnull().sum())
print('Number of missing data points in Embarked:',Total['Embarked'].isnull().sum())


# The missing values for survival are from the test set so we can disregard those. To fill in the missing values for age we construct a regression. Cabin is missing the majority of its data points, so it should be dropped. Since there is only one value missing for Fare we will just use the mode of Fare to fill in the NaNs. The 2 rows that have missing embarked will be replaced with the mode of Embarked. Although ticket has all its values and could be used to determine family survival, it will be dropped for this project.

# In[ ]:


Total = Total.drop(['Ticket','Cabin'], axis = 1)
fare_mode = Total['Fare'].mode().iat[0]
Total['Fare'].fillna(fare_mode,inplace = True)
embarked_mode = Total['Embarked'].mode().iat[0]
Total['Embarked'].fillna(embarked_mode, inplace = True)
print('Number of missing data points in Fare:',Total['Fare'].isnull().sum())
print('Number of missing data points in Embarked:',Total['Embarked'].isnull().sum())


# Our Fare and Embarked columns now have no Nan values. Now we will start working towards creating an age regression to predict the missing age values. To do this we need to process some of the remaning columns. We will be creating dummy variables for Sex, Embarked, and Title.

# In[ ]:


sex_dummy = pd.get_dummies(Total['Sex'])
Total = Total.drop('Sex', axis = 1)
Total = pd.concat([Total, sex_dummy], axis = 1)
embark_dummy = pd.get_dummies(Total['Embarked'])
Total = Total.drop('Embarked', axis = 1)
Total = pd.concat([Total, embark_dummy], axis = 1)
Total


# The original columns for Sex and Embarked have been dropped, and their respective dummy variable columns have been added. Now for Title, we will need to seperate the names from the actual titles in the Title column before we can create fdummy variables.

# In[ ]:


title = lambda x: x.split(',')[1].split('.')[0].strip()
Total['Title']=Total['Name'].map(title)
Total = Total.drop('Name', axis = 1)
Total


# Now that we just have the titles of each of the passengers we can create dummy variable columns for each title. Also, since some titles overlap one another, similar titles will be added together.

# In[ ]:


title_dummy = pd.get_dummies(Total['Title'])
title_dummy
title_dummy['Mil'] = title_dummy['Capt']+title_dummy['Col']+title_dummy['Major']
title_dummy = title_dummy.drop(['Capt','Col','Major'], axis = 1)
title_dummy['Senior Male Honorific'] = title_dummy['Don']+title_dummy['Sir']
title_dummy = title_dummy.drop(['Don','Sir'], axis = 1)
title_dummy['Senior Female Honorific'] = title_dummy['Dona']+title_dummy['Mme']+title_dummy['Lady']
title_dummy = title_dummy.drop(['Dona','Mme','Lady'], axis = 1)
title_dummy['Ms+Mlle'] = title_dummy['Ms']+title_dummy['Mlle']
title_dummy = title_dummy.drop(['Ms','Mlle'], axis = 1)
title_dummy['Fancy'] = title_dummy['Jonkheer']+title_dummy['the Countess']
title_dummy = title_dummy.drop(['Jonkheer','the Countess'], axis = 1)
Total = Total.drop('Title', axis = 1)
Total = pd.concat([Total, title_dummy], axis = 1)
Total


# The only missing data points we have now are some of the ages of the passengers, so we will now be creating a regression to find those points. First thing we need to do is get rid of the survival column, since it wont be used to predict age. Next we are also getting rid of all the rows with missing age values, since those are the valus we will be predicting.

# In[ ]:


Age_Total = Total
Age_Total = Age_Total.dropna()
Age_Total
def nans(df): return df[df.isnull().any(axis=1)]
Age_Total_Predict = nans(Total)
Age_Total_Predict


# The dataframe above(Age_Total_Predict) will act as our testing set. And the dataframe below(Age_Total) will act as our training set.

# In[ ]:


col = list(Age_Total.columns)
col.remove('Age')
col.remove('Survived')
Age_Total[col]


# Now we can start to create a regression for age. We will test 3 types of regressions and score them using RMSE. The model with the lowest RMSE will be used for the age prediction.

# In[ ]:


x = Age_Total[col]
x_test = Age_Total_Predict[col]
y_train = Age_Total['Age']
x_train = preprocessing.StandardScaler().fit(x).transform(x)
x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)


# In[ ]:


lasso = LassoCV(alphas = [0.01, 0.05, 0.1, 0.15,0.5]).fit(x_train, y_train)
alpha = 0.1
lasso_tuned = LassoCV(alphas = [0.8*alpha, 0.9*alpha, alpha, 1.1*alpha, 1.2*alpha]).fit(x_train, y_train)


# In[ ]:


svm_age = svm.SVR(kernel = 'rbf')
svm_age.fit(x_train, y_train)


# In[ ]:


ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
ridge_age = GridSearchCV(ridge, parameters, scoring = 'neg_root_mean_squared_error', cv=5)
ridge_age.fit(x_train, y_train)
print(ridge_age.best_params_)
print(-ridge_age.best_score_)


# In[ ]:


print('The average R squared score for the lasso model is:', -cross_val_score(lasso_tuned, x_train, y_train, scoring ='neg_root_mean_squared_error').mean())
print('The average R squared score for the svm model is:', -cross_val_score(svm_age, x_train, y_train, scoring ='neg_root_mean_squared_error').mean())
print('The average R squared score for the ridge model is:', -ridge_age.best_score_)
#Gonna use the lasso model, since it has the lowest RSME


# Out of the Lasso, SVM, and Ridge models, the Lasso model provided the lowest RMSE so it will be used to predict the missing ages. Below we will be predicting the ages, and adding them back to the Total dataframe to fill in the NaN values.

# In[ ]:


predicted_ages = pd.DataFrame(lasso_tuned.predict(x_test), columns=['Age']) 
Age_Total_Predict.drop('Age',axis = 1)
Age_Total_Predict = Age_Total_Predict.assign(Age = lasso_tuned.predict(x_test).round())
Age_Total_Predict['PassengerId'] = Age_Total_Predict.index
Age_Total_Predict = Age_Total_Predict.reset_index(drop=True)
Age_Total_Predict


# The predicted ages have been rounded to match the original age data. The lasso model has predicted some ages that are negative so to combat this we will be adding the RSME value of the model to any age value that is below 0. We will also be adding the testing and training dataframes back together to make the Total_Processed dataframe.

# In[ ]:


Age_Total['PassengerId'] = Age_Total.index
Age_Total = Age_Total.reset_index(drop=True)
Total_Processed = pd.concat([Age_Total_Predict,Age_Total])
Total_Processed.set_index('PassengerId', inplace = True)
Total_Processed = Total_Processed.sort_index()
fix_neg = lambda x: x + 11.148738330664404 if x < 0 else x
Total_Processed['Age'] = Total_Processed['Age'].apply(fix_neg)
Total_Processed


# # 4. Model Testing
#    Now that we have filled all the missing Nan values, we can get to creating a model to predict Titanic survival. The first thing we need to do is resplit the processed dataframe back into the original training and testing sets. We will also be processing the data using standard scaler. A KNN, a SVM, a GaussianNB, a XGBC, and an AdaBoost model will all be tested and the one with the highest accuracy will be used for our final survival predictions. Additionally all prediction outputs for each model will be saved in their own csv file.

# In[ ]:


Train_Processed = Total_Processed[:len(train)]
Test_Processed = Total_Processed[len(train):]


# In[ ]:


col = list(Train_Processed.columns)
col.remove('Survived')
x_train = Train_Processed[col]
y_train = Train_Processed['Survived']
y_train = y_train.astype('int')
x_final = Test_Processed[col]
x_train = preprocessing.StandardScaler().fit(x_train).transform(x_train)
x_final= preprocessing.StandardScaler().fit(x_final).transform(x_final)


# ### Creating a K Nearest Neighbors Model
#    A KNN model will be testing and optimized to find the best K to use.

# In[ ]:


k_score = []
k_range = range(1,16)
for k in k_range:
    neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
    
    score =  -cross_val_score(neigh, x_train, y_train, scoring ='neg_root_mean_squared_error').mean()
    
    k_score.append(score)

Scores = pd.DataFrame(k_score, columns = ['Score'])
Scores.index += 1
Scores.sort_values('Score')


# At a K of 8 the KNN model seems to perform the best so when we test the KNN model against others we will use a K of 8. Below is the model with a K of 8. 

# In[ ]:


neigh = KNeighborsClassifier(n_neighbors = 8).fit(x_train,y_train)
y_KNN = pd.DataFrame(neigh.predict(x_final), columns = ['Survived'])
y_KNN = y_KNN.assign(PassengerId = Test_Processed.index)
cols = y_KNN.columns.tolist()
cols = cols[-1:] + cols[:-1]
y_KNN = y_KNN[cols]
filename = 'Titanic Predictions KNN.csv'
y_KNN.to_csv(filename,index=False)
print('Saved file: ' + filename)


# ### Creating a SVM Model
#    Kernel, gamma, and c will all try to be optimized.

# In[ ]:


svm_lin = svm.SVC(kernel='linear').fit(x_train, y_train)
print('The average R squared score for the svm_lin is:', cross_val_score(svm_lin, x_train, y_train, scoring ='accuracy').mean())
svm_rbf = svm.SVC(kernel='rbf').fit(x_train, y_train)
print('The average R squared score for the svm_rbf is:', cross_val_score(svm_rbf, x_train, y_train, scoring ='accuracy').mean())


# In[ ]:


gammas = [0.01,0.1, 1, 10, 100]
for gamma in gammas:
    svm_rbf = svm.SVC(kernel='rbf').fit(x_train, y_train)
    print('The average R squared score for the svm_rbf with a gamma of',gamma,' is:', cross_val_score(svm_rbf, x_train, y_train, scoring ='accuracy').mean())


# In[ ]:


cs = [0.65,0.67]
for c in cs:
    svm_rbf = svm.SVC(kernel='rbf', C=c).fit(x_train, y_train)
    print('The average R squared score for the svm_rbf with a C of',c,' is:', cross_val_score(svm_rbf, x_train, y_train, scoring ='accuracy').mean())


# In[ ]:


svm_rbf = svm.SVC(kernel='rbf').fit(x_train, y_train)
y_SVM = pd.DataFrame(svm_rbf.predict(x_final), columns = ['Survived'])
y_SVM = y_SVM.assign(PassengerId = Test_Processed.index)
cols = y_SVM.columns.tolist()
cols = cols[-1:] + cols[:-1]
y_SVM = y_SVM[cols]
filename = 'Titanic Predictions SVM.csv'
y_SVM.to_csv(filename,index=False)
print('Saved file: ' + filename)


# ### Creating a GaussianNB Model

# In[ ]:



gauss = GaussianNB().fit(x_train,y_train)
y_nb = pd.DataFrame(gauss.predict(x_final), columns = ['Survived'])
y_nb = y_nb.assign(PassengerId = Test_Processed.index)
cols = y_nb.columns.tolist()
cols = cols[-1:] + cols[:-1]
y_nb = y_nb[cols]
filename = 'Titanic Predictions Gauss.csv'
y_nb.to_csv(filename,index=False)
print('Saved file: ' + filename)


# ### Creating a XGBC Model

# In[ ]:


xgbc = xgb.XGBClassifier().fit(x_train,y_train)
y_xgb = pd.DataFrame(xgbc.predict(x_final), columns = ['Survived'])
y_xgb = y_xgb.assign(PassengerId = Test_Processed.index)
cols = y_xgb.columns.tolist()
cols = cols[-1:] + cols[:-1]
y_xgb = y_xgb[cols]
filename = 'Titanic Predictions XGBC.csv'
y_xgb.to_csv(filename,index=False)
print('Saved file: ' + filename)


# ### Creating an AdaBoost Model

# In[ ]:


ada = AdaBoostClassifier(n_estimators=400,learning_rate=1,algorithm='SAMME')
ada.fit(x_train, y_train)
y_ada = pd.DataFrame(ada.predict(x_final), columns = ['Survived'])
y_ada = y_ada.assign(PassengerId = Test_Processed.index)
cols = y_ada.columns.tolist()
cols = cols[-1:] + cols[:-1]
y_ada = y_ada[cols]
filename = 'Titanic Predictions ADA.csv'
y_ada.to_csv(filename,index=False)
print('Saved file: ' + filename)


# In[ ]:


print('The average accuracy for the KNN model is:', cross_val_score(neigh, x_train, y_train, scoring ='accuracy').mean())
print('The average accuracy score for the SVM model is:', cross_val_score(svm_rbf, x_train, y_train, scoring ='accuracy').mean())
print('The average accuracy score for the GaussianNB model is:', cross_val_score(gauss, x_train, y_train, scoring ='accuracy').mean())
print('The average accuracy score for the XGBC model is:', cross_val_score(xgbc, x_train, y_train, scoring ='accuracy').mean())
print('The average accuracy score for the AdaBoost model is:', cross_val_score(ada, x_train, y_train, scoring ='accuracy').mean())


# The SVM model will be used since it has the highest accuracy.
