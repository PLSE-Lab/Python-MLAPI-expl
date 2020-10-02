#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#The goal is to predict rain in Austrilia based on the pastinformation or features given in the dataset,Always remember below check list
#Any machine learning problem should be tackled with below steps. 

#1. Look at the big picture.
#2. Get the data.
#3. Discover and visualize the data to gain insights.
#4. Prepare the data for Machine Learning algorithms.
#5. Select a model and train it.
#6. Fine-tune your model.
#7. Present your solution.


# In[ ]:


#Read the data into Pandas Data frame
df=pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
df.head()


# In[ ]:



df.info()
df.columns


# In[ ]:


numberical=[var for var in df.columns if df[var].dtype !='O']
print(" There are {} categorical variables\n".format(len(numberical)))
print('The numberical variables are:',numberical)


# In[ ]:


df['Date'].value_counts()


# In[ ]:


df['Date']=pd.to_datetime(df['Date'])


# In[ ]:


df['Year']=df['Date'].dt.year
df['Month']=df['Date'].dt.month
df['Day']=df['Date'].dt.day


# In[ ]:


df.drop('Date',inplace=True,axis=1)


# In[ ]:


categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# In[ ]:


f,ax=plt.subplots(figsize=(6,8))
ax=sns.countplot(y='RainTomorrow',data=df)
plt.show()


# In[ ]:


df[categorical].head()


# In[ ]:


#percentage of missing values in each of the catagorical variables
df[categorical].isnull().sum().sort_values(ascending=False)/len(df)*100


# In[ ]:


for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')


# In[ ]:


# exporeing missing values in catagorical variables

df[categorical].isnull().sum().sort_values(ascending=False)


# In[ ]:


#expore Location variable
print('location contains',len(df['Location'].unique()),'Labels')


# In[ ]:


df.Location.value_counts()


# In[ ]:


#lets to one hot encoding of location to eliminate catagorical futures and get dummy variable using pandas

pd.get_dummies(df.Location,drop_first=True).head()


# In[ ]:


#now lets expore windgustdir variable

print('WindGustDir contains',len(df['WindGustDir'].unique()),'labels')
print("these are the values for WindGustDir",df['WindGustDir'].unique())


# In[ ]:


#lets to one hot encoding of location to eliminate windgustdir catagorical futures and get dummy variable using pandda

pd.get_dummies(df.WindGustDir,drop_first=True).head()


# In[ ]:


pd.get_dummies(df.WindGustDir,drop_first=True,dummy_na=True).sum(axis=0)


# In[ ]:


#there are 9330 missing values for WindGustDir


# In[ ]:


print('WinDir9am contains',len(df['WindDir9am'].unique()),'Lables')
print('WinDir9am has values:',df['WindDir9am'].unique())


# In[ ]:


df['WindDir9am'].value_counts()


# In[ ]:


#lets to one hot encoding of location to eliminate windgustdir catagorical futures and get dummy variable using pandda

pd.get_dummies(df.WindDir9am,drop_first=True,dummy_na=True).head()


# In[ ]:


pd.get_dummies(df.WindDir9am,drop_first=True,dummy_na=True).sum(axis=0)


# In[ ]:


# there are 10013 missing values for WindDir9AM 


# In[ ]:


#lets expore WindDir3PM variable

print('WindDir3PM variable contains',len(df['WindDir3pm'].unique()),'Lables')
print("WindDir3PM Values contains",df.WindDir3pm.unique())


# In[ ]:


#lets get dummy variables for these WindDir3Pm variable via one hot encoding

pd.get_dummies(df.WindDir3pm,drop_first=True,dummy_na=True).sum(axis=0)


# In[ ]:


# there are 3778 missing values for WindDir3pm variable


# In[ ]:


#Lets expore RainToday variable 
print('RainToday Contains',len(df['RainToday'].unique()),'Lables')
df['RainToday'].unique()


# In[ ]:


#lets do one hot encoding of RainToday variable and get dummy variables 
pd.get_dummies(df.RainToday,drop_first=True,dummy_na=True).sum(axis=0)


# In[ ]:


#there are 1406 missing values for RainToday variable

#NOW LETS expore numarical variables


# In[ ]:


print("There are {} numerical variables\n".format(len(numberical)))
print("These numerical varibales are :",numberical)


# In[ ]:


df[numberical].head()


# In[ ]:


# exporing missing values in numerical variables 

df[numberical].isnull().sum().sort_values(ascending=False)/len(df)*100


# In[ ]:


#view outliers in numerical variables

print(round(df[numberical].describe()),2)


# In[ ]:


#lets draw box plot to figure outliers 
#as above statitics shows that there may be outliers in Rainfall, Evaporation, WindSpeed9am and WindSpeed3pm columns


# In[ ]:


plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig=df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')

plt.subplot(2,2,2)
fig=df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')

plt.subplot(2,2,3)
fig=df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2,2,4)
fig=df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')


# In[ ]:


# above box plots confirms that there are outliers for these variables

#Check the distribution of variable
plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')


# In[ ]:


df.head()


# In[ ]:


corr_mat=df.corr()


# In[ ]:


plt.figure(figsize=(16,12))
sns.heatmap(corr_mat,annot=True,square=True,linecolor='White',fmt='.2f')


# In[ ]:


num_var = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'WindGustSpeed', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']

sns.pairplot(df[num_var],kind='scatter',diag_kind='hist')


# In[ ]:


#declar X,y

X=df.drop(['RainTomorrow'],axis=1)
y=df['RainTomorrow']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


#Feature engineering

X_train.dtypes


# In[ ]:


X_train.isnull().sum().sort_values(ascending=False)/len(X_train)*100


# In[ ]:


# Impute missing values in X_train,X_test

for df1 in [X_train,X_test]:
    for col in numberical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median,inplace=True)
        
for df2 in [X_train,X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True) 


# In[ ]:


X_train.isnull().sum()


# In[ ]:


import category_encoders as ce
encoder=ce.BinaryEncoder(cols=['RainToday'])

X_train=encoder.fit_transform(X_train)
X_test=encoder.fit_transform(X_test)


# In[ ]:


X_train = pd.concat([X_train[numberical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)


# In[ ]:


X_train.head()


# In[ ]:


X_test = pd.concat([X_test[numberical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)


# In[ ]:


# Feature scaling 

cols=X_train.columns

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[ ]:


X_train=pd.DataFrame(X_train,columns=[cols])
X_test=pd.DataFrame(X_test,columns=[cols])


# In[ ]:


#Model Training

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression(solver='liblinear',random_state=0)


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


#Predict results


# In[ ]:


pred=logmodel.predict(X_test)
pred


# In[ ]:


logmodel.predict_log_proba(X_test)[:,0]


# In[ ]:


# checking the accurary of the score

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print("the accuraty of the model is ",accuracy_score(y_test,pred))


# In[ ]:


y_pred_train = logmodel.predict(X_train)

y_pred_train


# In[ ]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[ ]:


print('Training set score: {:.4f}'.format(logmodel.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logmodel.score(X_test, y_test)))


# In[ ]:


logreg100=LogisticRegression(C=100,solver='liblinear',random_state=0)
logreg100.fit(X_train,y_train)


# In[ ]:


print('Train set score:{:4f}'.format(logreg100.score(X_train,y_train)))
print('Train set score:{:4f}'.format(logreg100.score(X_test,y_test)))


# In[ ]:


logreg001=LogisticRegression(C=0.01,solver='liblinear',random_state=0)
logreg001.fit(X_train,y_train)


# In[ ]:


# Verify that we are not overfitting the model by checking the scores againt train set and test test


# In[ ]:


print('Train set score:{:4f}'.format(logreg001.score(X_train,y_train)))
print('Train set score:{:4f}'.format(logreg001.score(X_test,y_test)))


# In[ ]:


y_test.value_counts()


# In[ ]:


null_accracy=(22067/(22067+6372))
print('Null accuracy score: {0:0.4f}'. format(null_accracy))


# In[ ]:


cm=confusion_matrix(y_test,pred)
print('Confusion Matrix\n\n',cm)
print('\nTrue Positive (TP)=',cm[0,0])
print('\nTrue Negative (TN)=',cm[1,1])
print('\nFalse Positive aka Type-1 Error (TN)=',cm[0,1])
print('\nFalse Negative aka Type-2 Error (TN)=',cm[1,0])


# In[ ]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[ ]:


#Classification report to understand the metrics and evaluate our model performance

print(classification_report(y_test,pred))


# In[ ]:


#Now lets try to imporvoe our model using famous GridSearch 


# In[ ]:



from sklearn.model_selection import GridSearchCV

 
parameters=[{'penalty':['l1','l2']},{'C':[1,10]}]

grid_search=GridSearchCV(estimator=logmodel,param_grid=parameters,scoring='accuracy',cv=5,verbose=1)

grid_search.fit(X_train,y_train)


# In[ ]:


print('Grid Search CV Best score:{:.4f}\n\n'.format(grid_search.best_score_))


# In[ ]:


# We see that the best score for Grid search is 0.975 which is very good but is this reliable ??/


# In[ ]:


# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))


# In[ ]:


#Lets try using popular XGBoost ensemble to see if we can get more accurate predictions for this classification


# In[ ]:


from xgboost import XGBClassifier
xgbclassifier = XGBClassifier(n_estimators=100,learning_rate=0.05)


# In[ ]:


X_train=X_train.loc[:,~X_train.columns.duplicated()]
X_test=X_test.loc[:,~X_test.columns.duplicated()]


# In[ ]:


xgbclassifier.fit(X_train,y_train)


# In[ ]:


predict=xgbclassifier.predict(X_test)


# In[ ]:


# Lets try to evaluate our model now to see if we did any good after using XGBclassifer with few hyperparametes. The goal is not to overfit our mode
print(classification_report(y_test,predict))
cm1=confusion_matrix(y_test,predict)
print('Confusion Matrix\n\n',cm1)
print('\nTrue Positive (TP)=',cm1[0,0])
print('\nTrue Negative (TN)=',cm1[1,1])
print('\nFalse Positive aka Type-1 Error (TN)=',cm1[0,1])
print('\nFalse Negative aka Type-2 Error (TN)=',cm1[1,0])
cm_matrix = pd.DataFrame(data=cm1, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])


# In[ ]:


#XBG classifer with hyperparameters(n_estimators=100,learning_rate=0.05) showing that type 1 error eliments 743 and type to error ???
#Is this overfitting????? that what matters , we need to make sure we are not overfitting the model.


# In[ ]:


y_pred_train1 = xgbclassifier.predict(X_train)

y_pred_train1
print('Train set score:{:4f}'.format(xgbclassifier.score(X_train,y_train)))
print('Test set score:{:4f}'.format(xgbclassifier.score(X_test,y_test)))


# In[ ]:


# as we can see train set score is showing as 100% accurate and test set score is 0.97.. can you validate what went wrong?
#A good data scientist would know exactly if we are overfitting our model , while tacking realworld business problesm real would come in tacking overfitting situation


# In[ ]:


# We have seen that model is too imporved after Grid search and also using XGBClassifer with few hyperparameters n_estimators and learning rate


# In[ ]:


# when using classical logistic regression we seen that our accuracy was around 85% .
#When we try to improve our model using famous Grid search and ensemble XGBclassifer we tend to get great predictions but we need to make we are not overfitting ..
#Can you identify if the model is overfitting ??? i will leave it up to you for open discussion?? 
#is this classic example of overfitting our model while trying to improve the same


# In[ ]:





# In[ ]:





# In[ ]:




