#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from time import time
import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)


# Reading the train and test csv

# In[ ]:


train_csv = pd.read_csv('/kaggle/input/titanic/train.csv')
test_csv = pd.read_csv('/kaggle/input/titanic/test.csv')


# Checking the train and test dataframes

# In[ ]:


train_csv.head()


# In[ ]:


test_csv.head()


# The aim here is to predict the Survived column value for the test data.
# 
# 1. For this we will merge the train and test data ,so that any data preprocessing done on train data reflects to the test data
# 2. Then we will perform the preprocessing together and later split the train and test data back to the original form
# 
# Let's first add the survived column to the test data so that the shape of train and test is same

# In[ ]:


test_csv['Survived'] = -1


# In[ ]:


test_csv.head()


# Let's check the shape for both the train and test datasets jow, they show have same number of columns

# In[ ]:


print(train_csv.shape[1])
print(test_csv.shape[1])


# Creating the titanic_df , with the data merged from train_csv and test_csv

# In[ ]:


titanic_df = pd.concat([train_csv,test_csv], ignore_index=True, sort = True)


# In[ ]:


# html = titanic_df.to_html()
# with open('data_1.html', 'w') as f:
#     f.write(html)


# Now we have a dataframe that has the data from both the train_csv and test_csv

# In[ ]:


titanic_df.shape


# Let's begin with analysing the data to see what pre processing can be done

# In[ ]:


titanic_df.head()


# Let's first begin by checking if there are any missing values in the database

# In[ ]:


titanic_df.isnull().sum()


# As we can see there are columns Age, Cabin , Embarked and Fare with missing values.
# 1. The largest number of missing values are in Cabin Column
# 2. Out of total 1309, 1014 values are missing, hence cabin is a potential candidate to be dropped
# 

# In[ ]:


titanic_df.drop(columns=['Cabin'], inplace=True)


# In[ ]:


titanic_df.head()


# In[ ]:


titanic_df.isnull().sum()


# In[ ]:


titanic_df['Age'].value_counts()


# In[ ]:


age_df = train_csv[['Age','Survived']]


# In[ ]:


age_gb = age_df.groupby('Age').agg(pd.Series.mode)


# In[ ]:


h =  age_gb.to_html()
with open('t.html','w') as f:
    f.write(h)


# On Investigating further, it looks like young people are more likely to survive. Hence we will impute it with the Mean

# In[ ]:


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(titanic_df[['Age']])
titanic_df['Age'] = imputer.transform(titanic_df[['Age']])


# Lets check if the Age column is now correct

# In[ ]:


titanic_df.isnull().sum()


# In[ ]:


titanic_df.head()


# We can now use the SumpleImputer on Fare column as before to generate the missing value

# In[ ]:


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(titanic_df[['Fare']])
titanic_df['Fare'] = imputer.transform(titanic_df[['Fare']])


# In[ ]:


titanic_df.isnull().sum()


# Now we are only left with the Embarkd column , which we can correct by using SimpleImputer based on Mode

# In[ ]:


imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(titanic_df[['Embarked']])
titanic_df['Embarked'] = imputer.transform(titanic_df[['Embarked']])


# In[ ]:


titanic_df.isnull().sum()


# Look's great, now that we do not have any empty values let explore more on the fetuere relationship

# In[ ]:


titanic_df.head()


# Lets's explore the SibSp and Parch columns with respect to survival

# In[ ]:


sns.catplot(x='SibSp', y='Survived', data=train_csv, kind='point', aspect=2)


# In[ ]:


sns.catplot(x='Parch', y='Survived', data=train_csv, kind='point', aspect=2)


# It can be seen that with the more members the survival rate is less, hence let's create a new column 

# In[ ]:


titanic_df['Family'] = titanic_df['SibSp'] + titanic_df['Parch']


# In[ ]:


titanic_df.head()


# Now let's remove the SibSp and Parch columns

# In[ ]:


titanic_df.drop(['SibSp','Parch'],axis=1, inplace=True)


# In[ ]:


titanic_df.head()


# Now let's clean the Sex column by mapping values for male and female

# In[ ]:


enc = OrdinalEncoder()
sex = [['male',0],['female',1]]
enc.fit(sex)
titanic_df[['Sex']] = enc.transform(titanic_df[['Sex']])


# In[ ]:


titanic_df.head()


# Now , the column Embarked need to be changed . Lets use OneHotEncoding on this

# In[ ]:


enc = OneHotEncoder(sparse = False)
Embarked_encode = enc.fit_transform(titanic_df[['Embarked']])


# In[ ]:


df_embarked = pd.DataFrame(data=Embarked_encode, columns=['E_0','E_1','E_2'])


# In[ ]:


df_embarked.head()


# In[ ]:


titanic_df = pd.concat([titanic_df,df_embarked], axis=1)


# In[ ]:


titanic_df.head()


# Now we have a better data with us , but still there are certain column which we need to check
# 1. The Name column won't affect the survival , hence we can remove this
# 2. The Ticket will also have no effect on survial, hence we can remove this as well

# In[ ]:


titanic_df.drop(['Ticket','Name'], axis=1,inplace = True)


# In[ ]:


titanic_df.head()


# Also, now we do not need the Embarked column, hence we can remove that as well

# In[ ]:


titanic_df.drop(['Embarked'], axis=1,inplace = True)


# In[ ]:


titanic_df.head()


# Finally, we are ready to go but one last thing is to remove the test data set which we added from the dataset

# In[ ]:


train_data = titanic_df[titanic_df['Survived'] != -1]
test_data = titanic_df[titanic_df['Survived'] == -1]


# In[ ]:


print(train_data.shape)
print(test_data.shape)


# Now we will remove the Survived column from the test_data as that is what we want to predict

# In[ ]:


test_data.drop(['Survived'],axis=1, inplace=True)


# In[ ]:


test_data.shape


# Let's Split the train data into features and labels

# In[ ]:


features = train_data.drop(['Survived'], axis=1)
labels = train_data['Survived']


# Time to split the data into test and train

# In[ ]:


X_train,X_test, y_train,y_test = train_test_split(features,labels, test_size = 0.2, random_state = 42)


# Now since we have the train and test data with us
# 1. We will take the test data and split it into Validation and Test dataset
# 2. We use validation data in order to work with hyper parameters

# In[ ]:


X_val,X_test, y_val, y_test = train_test_split(X_test,y_test, test_size=0.5, random_state = 42)


# We will use different models on the validation test set in order to see the accuracy
# 
# Let us create a function that will display all the perfromance parameters

# In[ ]:


def print_results(results):
    print(f'Best params: {results.best_params_} \n')
    
    means = results.cv_results_['mean_test_score']
    stds =  results.cv_results_['std_test_score']
    
    for mean,std,params in zip(means,stds, results.cv_results_['params']):
        print(f'{round(mean,3)}(+/-{round(std,3)}) for {params}')


# In[ ]:


clf = LogisticRegression()
parameters = {
    
    'C': [0.001,0.01, 0.1, 1, 10, 100, 1000]
}

cv = GridSearchCV(clf,parameters, cv = 5)
cv.fit(X_train,y_train)
print_results(cv)


# In[ ]:


joblib.dump(cv.best_estimator_,'LR_model.pkl')


# In[ ]:


clf = RidgeClassifier()
parameters = {
    'alpha': [0.01,0.1,1,10,100,1000],
    'normalize' : [True, False]
}
cv = GridSearchCV(clf,parameters,cv=5)
cv.fit(X_train,y_train)
print_results(cv)


# In[ ]:


joblib.dump(cv.best_estimator_,'RC_model.pkl')


# In[ ]:


clf = DecisionTreeClassifier()
parameters = {
    'max_depth' : [5,10,15,50,100,None],
}
cv = GridSearchCV(clf, parameters, cv = 5)
cv.fit(X_train,y_train)
print_results(cv)


# In[ ]:


joblib.dump(cv.best_estimator_,'DC_model.pkl')


# In[ ]:


clf = KNeighborsClassifier()
parameters = {
    'n_neighbors' : [3,5,7,10,12],
    'metric' : ['euclidean', 'manhattan' , 'chebyshev']

}
cv = GridSearchCV(clf, parameters, cv = 5)
cv.fit(X_train,y_train)
print_results(cv)


# In[ ]:


joblib.dump(cv.best_estimator_,'KNN_model.pkl')


# In[ ]:


clf = RandomForestClassifier()
parameters = {
    'n_estimators' : [5, 50, 250],
    'max_depth' : [2, 4, 8, 16, 62, None]
}
cv = GridSearchCV(clf, parameters, cv = 5)
cv.fit(X_train,y_train)
print_results(cv)


# In[ ]:


joblib.dump(cv.best_estimator_,'RFC_model.pkl')


# In[ ]:


clf = AdaBoostClassifier()
parameters = {
    'n_estimators' : [5, 50, 250],
}
cv = GridSearchCV(clf, parameters, cv = 5)
cv.fit(X_train,y_train)
print_results(cv)


# In[ ]:


joblib.dump(cv.best_estimator_,'ABC_model.pkl')


# In[ ]:


clf = BaggingClassifier()
parameters = {
    'n_estimators' : [5, 50, 100, 250],
}
cv = GridSearchCV(clf, parameters, cv = 5)
cv.fit(X_train,y_train)
print_results(cv)


# In[ ]:


joblib.dump(cv.best_estimator_,'BC_model.pkl')


# In[ ]:


clf = GradientBoostingClassifier()
parameters = {
    'n_estimators' : [5, 50, 250],
    'max_depth' : [1, 3, 5, 7, 9],
    'learning_rate' : [0.01, 0.1, 1, 10, 100]
}
cv = GridSearchCV(clf, parameters, cv = 5)
cv.fit(X_train,y_train)
print_results(cv)


# In[ ]:


joblib.dump(cv.best_estimator_,'GBC_model.pkl')


# In[ ]:





# In[ ]:





# ###### From all the computation above the :
# 1. Let's read the generated model and evaluate the perfromance on the validation dataset

# In[ ]:


models = {}

for mdl in ['LR', 'RC', 'DC', 'KNN','RFC', 'ABC', 'BC' , 'GBC']:
    models[mdl] = joblib.load(f'{mdl}_model.pkl')


# In[ ]:


models


# In[ ]:


def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels,pred),3)
    precision = round(precision_score(labels,pred),3)
    recall = round(recall_score(labels,pred),3)
    print(f'{name} === Accuracy : {accuracy} , Precision : {precision} , Recall : {recall} , Latency : {round((end-start)*1000,1)}ms')


# In[ ]:


for name, mdl in models.items():
    evaluate_model(name, mdl, X_val, y_val)


# ### As now the best accuracy is done by Random Forest Classifier
# 
# Let's try the model on the Test data

# In[ ]:


for name, mdl in models.items():
    evaluate_model(name, mdl, X_test, y_test)


# On Test data, the best model turns out to be Bagging classifier, GBC, RFC and DC.
# We will now use the actual test data with us and predct the results, we will submit and see which model is the best
# 
# Initially we had the actual training data , we will use this to train the model <br>
# features = train_data.drop(['Survived'], axis=1)<br>
# 
# labels = train_data['Survived']

# In[ ]:





# In[ ]:


bagginclassifier = models['BC']
bagginclassifier.fit(features,labels)
pred  = bagginclassifier.predict(test_data)
test_data_pid = test_data['PassengerId']
results = pd.Series(data = pred, name = 'Survived', dtype='int64')
df = pd.DataFrame({"PassengerId":test_data_pid, "Survived":results})
df.to_csv("submission_BC.csv", index=False, header=True)


# In[ ]:


GradientBoostingClassifier = models['GBC']
GradientBoostingClassifier.fit(features,labels)
pred  = GradientBoostingClassifier.predict(test_data)
test_data_pid = test_data['PassengerId']
results = pd.Series(data = pred, name = 'Survived', dtype='int64')
df = pd.DataFrame({"PassengerId":test_data_pid, "Survived":results})
df.to_csv("submission_GBC.csv", index=False, header=True)


# In[ ]:


RandomForestClassifier = models['RFC']
RandomForestClassifier.fit(features,labels)
pred  = RandomForestClassifier.predict(test_data)
test_data_pid = test_data['PassengerId']
results = pd.Series(data = pred, name = 'Survived', dtype='int64')
df = pd.DataFrame({"PassengerId":test_data_pid, "Survived":results})
df.to_csv("submission_RFC.csv", index=False, header=True)


# In[ ]:


DecisionTreeClassifier = models['DC']
DecisionTreeClassifier.fit(features,labels)
pred  = DecisionTreeClassifier.predict(test_data)
test_data_pid = test_data['PassengerId']
results = pd.Series(data = pred, name = 'Survived', dtype='int64')
df = pd.DataFrame({"PassengerId":test_data_pid, "Survived":results})
df.to_csv("submission_DC.csv", index=False, header=True)


# #### Thanks for having a look, please upvote  :)
