#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#reading the csv file
df=pd.read_csv('../input/train.csv')


# In[ ]:


df.head()


# In[ ]:


#dropping irrelivent columns from dataframe.
df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


#for getting information about different columns
df.info()


# In[ ]:


#Rounding off Fare column to one decimal place
df.Fare=df.Fare*10
df.Fare=df.Fare.astype(int)
df.Fare=df.Fare/10


# In[ ]:


df.head()


# In[ ]:


#converting text data of sex column into numeric
pd.get_dummies(df.Sex,prefix='Sex').head()


# In[ ]:


#converting text data of embarked column into numeric
pd.get_dummies(df.Embarked).head()


# In[ ]:


#creating new dataframe with numeric features for sex and embarked
df_new=pd.concat([df,pd.get_dummies(df.Sex,prefix='Sex'),pd.get_dummies(df.Embarked,prefix='Embarked')],axis=1)


# In[ ]:


df_new.head()


# In[ ]:


#dropping the text column for sex and embarked along with one extra column from sex and embarked, 
#as they does not give any additional information i.e they are highly correlated
df_new.drop(['Sex','Embarked','Sex_female','Embarked_Q'],axis=1,inplace=True)


# In[ ]:


df_new.head()


# In[ ]:


#filling the NaN value of cabin with 0
df_new.Cabin.fillna(value=0,inplace=True)


# In[ ]:


df_new.head()


# In[ ]:


#replacing all other values with 1 usin regular expression
df_new.Cabin=df_new.Cabin.str.replace('[A-Z].*','1')


# In[ ]:


df_new.Cabin.fillna(value=0,inplace=True)
df_new.head()


# In[ ]:


#converting the value of cabin from string to int
df_new.Cabin=df_new.Cabin.astype(int)


# In[ ]:


#getting info about all column.we can see that all column is now converted into int
df_new.info()


# In[ ]:


#seperating the response and feature vector,here X is feature
X=df_new.drop(['Survived'],axis=1)


# In[ ]:


X.head()


# In[ ]:


#here y is response
y=df_new.Survived


# In[ ]:


y.head()


# In[ ]:


#importing train_test_split from sklearn.linear model
from sklearn.model_selection import train_test_split


# In[ ]:


#splitting into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


#getting info about training set
X_train.info()


# In[ ]:


#filling the value of nan value in age column with mean of age.
temp=X_train.Age.mean()
temp


# In[ ]:


X_train.Age.fillna(value=29,inplace=True)


# In[ ]:


#new X_trtain
X_train.head()


# In[ ]:


X_train.info()


# In[ ]:


#filling the nan value in test data with mean of age
temp=X_test.Age.mean()
temp


# In[ ]:


X_test.Age.fillna(value=31,inplace=True)


# In[ ]:


X_test.info()


# In[ ]:


#converting age from float to int
X_train.Age=X_train.Age.astype(int)
X_test.Age=X_test.Age.astype(int)


# In[ ]:


#final x_train with no NaN and all numeric value
X_train.head()


# In[ ]:


#final x_train with no NaN and all numeric value
X_test.head()


# In[ ]:


#importing random forest classifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#instantiating random forest classifier
rn=RandomForestClassifier()


# In[ ]:


#fitting the random forest classifier
rn.fit(X_train,y_train)


# In[ ]:


#predicting the result
pred=rn.predict(X_test)


# In[ ]:


#importing accuracy score from metrics to evaluate the accuracy of the model
from sklearn.metrics import accuracy_score


# In[ ]:


score=accuracy_score(y_test,pred)


# In[ ]:


#accuracy given by randomforest
score


# ### Now preparing the original dataframe to be used for k_fold_crossvalidation

# In[ ]:


X.info()


# In[ ]:


#filling the nan value of age
X.Age.mean()


# In[ ]:


X.Age=X.Age.fillna(value=30)


# In[ ]:


X.head()


# In[ ]:


#converting age to int from float
X.Age=X.Age.astype(int)
X.info()


# In[ ]:


y.shape


# In[ ]:


#calculating the accuracy of random forest foe different value of hyperparameter n_estimator
from sklearn.model_selection import cross_val_score
n_estimator_list=list(range(5,50,5))
a=[]
for i in n_estimator_list:
    rn=RandomForestClassifier(n_estimators=i)
    scores=cross_val_score(rn,X,y,cv=10)
    a.append(scores.mean())
print(a)    


# In[ ]:


#plotting a graph to show relation between n_estimator and accuracy score
plt.plot(n_estimator_list,a)


# In[ ]:


#calculating the accuracy of random forest for different value of hyperparameter max_depth
depth_list=list(range(2,10,1))
b=[]
for i in depth_list:
    rn=RandomForestClassifier(n_estimators=10,max_depth=i)
    scores=cross_val_score(rn,X,y,cv=10)
    b.append(scores.mean())
print(b)   


# In[ ]:


#plotting a graph to show relation between n_estimator and accuracy score
plt.plot(depth_list,b)


# In[ ]:


#instantiating random forest with optimal value of n_estimator and max_depth found by two upper plots
rn=RandomForestClassifier(n_estimators=10,max_depth=6)


# In[ ]:


#fitting random forest
rn.fit(X,y)


# In[ ]:


#reading the test data to which we have to submit the results
test_df=pd.read_csv('../input/test.csv')


# In[ ]:


#first five rows of test data
test_df.head()


# In[ ]:


#shape of test data
test_df.shape


# In[ ]:


temp=test_df


# In[ ]:


#dropping irrelevent columns which might not be useful in predicting the response
test_df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


#getting info about test data
test_df.info()


# In[ ]:


#we have some inf value in fare and age columns,which can't be handled by fillna function of pandas 
#so first we will have to convert that inf into nan,that's what "use_inf_as_null" does
with pd.option_context('mode.use_inf_as_null', True):
   test_df.Fare.fillna(value=test_df.Fare.mean(),inplace=True)


# In[ ]:


test_df.info()


# In[ ]:


#doing the same thing toage column
with pd.option_context('mode.use_inf_as_null', True):
   test_df.Age.fillna(value=test_df.Age.mean(),inplace=True)


# In[ ]:


test_df.info()


# In[ ]:


#rounding fare to one decimal place
test_df.Fare=test_df.Fare*10
test_df.Fare=test_df.Fare.astype(int)
test_df.Fare=test_df.Fare/10


# In[ ]:


#converting age from float to int
test_df.Age=test_df.Age.astype(int)


# In[ ]:


test_df.head()


# In[ ]:


##creating new dataframe with numeric features for sex and embarked
test_df=pd.concat([test_df,pd.get_dummies(test_df.Sex,prefix='Sex'),pd.get_dummies(test_df.Embarked,prefix='Embarked')],axis=1)


# In[ ]:


test_df.head()


# In[ ]:


##dropping the text column for sex and embarked along with one extra column from sex and embarked, 
#as they does not give any additional information i.e they are highly correlated
test_df.drop(['Sex','Sex_female','Embarked','Embarked_Q'],axis=1,inplace=True)


# In[ ]:


test_df.head()


# In[ ]:


##replacing all other values with 1 usin regular expression
test_df.Cabin=test_df.Cabin.str.replace('[A-Z].*','1')
test_df.Cabin=test_df.Cabin.fillna(value=0)


# In[ ]:


test_df.head(10)


# In[ ]:


#cheking the no. of one and zeros
test_df.Cabin.value_counts()


# In[ ]:


#predicting result using random_forest
pred_class_rn=rn.predict(test_df)


# In[ ]:


temp=pd.read_csv('../input/test.csv')


# In[ ]:


#converting the predicted response into detaframe for submission
predictions=pd.DataFrame({'PassengerId':temp.PassengerId,'Survived':pred_class_rn}).to_csv('predictions_rn_11.csv',index=False)


# In[ ]:


#importing gradient boosting clasifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


X.head()


# In[ ]:


X.info()


# In[ ]:


#creating a new feature named family by adding parents,children ans siblings
X['family']=X.SibSp+X.Parch


# In[ ]:


X.head()


# In[ ]:


#predicting results by gradientboosting using k_fold cross validation
gb=GradientBoostingClassifier()
scores=cross_val_score(gb,X,y,cv=5)
print(scores.mean())


# In[ ]:


#predicting the result excluding family feature to check if accuracy is incresing or not
gb=GradientBoostingClassifier()
scores=cross_val_score(gb,X.drop(['family'],axis=1),y,cv=5)
print(scores.mean())


# In[ ]:


#trying different combinations of features this time keeping family and dropping sibsp and parch.
gb=GradientBoostingClassifier()
scores=cross_val_score(gb,X.drop(['SibSp','Parch'],axis=1),y,cv=5)
print(scores.mean())


# In[ ]:


#dropping three columns and then finding the accuracy
gb=GradientBoostingClassifier()
scores=cross_val_score(gb,X.drop(['SibSp','Parch','Cabin'],axis=1),y,cv=5)
print(scores.mean())


# In[ ]:


#splitting into training and testing data.
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


print(train_x.shape)
print(test_x.shape)
print(train_y.shape)


# In[ ]:


#using Grid searchCV to find the best hyperparameter learning rate for Gradient Boosting
lr=[0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.85,0.90,0.95,1.0]
from sklearn.model_selection import GridSearchCV
gb=GradientBoostingClassifier(n_estimators=100)
param_grid=dict(learning_rate=lr)
grid=GridSearchCV(gb,param_grid,cv=10,scoring='accuracy',return_train_score=True)
grid.fit(train_x.drop(['SibSp','Parch'],axis=1),train_y)


# In[ ]:


#finding the best score
grid.best_score_


# In[ ]:


#finding the best parameter used in finding that best score
grid.best_params_


# In[ ]:


#finding best value of two hyperparameters this time using grid search.learning_rate and max_depth
max_depth=list(range(2,8,1))
param_grid=dict(learning_rate=lr,max_depth=max_depth)
gb=GradientBoostingClassifier(n_estimators=100)
grid=GridSearchCV(gb,param_grid,cv=10,scoring='accuracy',return_train_score=True)
grid.fit(train_x.drop(['SibSp','Parch'],axis=1),train_y)


# In[ ]:


#finding the best score
grid.best_score_


# In[ ]:


#finding the best parameter used to calculate this best score
grid.best_params_


# In[ ]:


#evaluating the result on testing set using the best parameters learned.
gb=GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,max_depth=3)
gb.fit(train_x.drop(['SibSp','Parch'],axis=1),train_y)
pred_clas=gb.predict(test_x.drop(['SibSp','Parch'],axis=1))
accuracy_score(test_y,pred_clas)


# In[ ]:


#first five of the testing data to which we have to submit the results
test_df.head()


# In[ ]:


#creating family feature from sibsp and parch
test_df['family']=test_df.SibSp+test_df.Parch


# In[ ]:


# before predicting on the test dataframe we should first train our best model on whole data available.
X.head()


# In[ ]:


#training our GradientBoosing model on whole dataset
gb.fit(X.drop(['SibSp','Parch'],axis=1),y)


# In[ ]:


#test df with new added feature 'family'.
test_df.head()


# In[ ]:


#predicting the final response from gradientboosting 
final_pred=gb.predict(test_df.drop(['SibSp','Parch'],axis=1))


# In[ ]:


#converting the response into Dataframe or csv file to submit it to kaggle.
predictions=pd.DataFrame({'PassengerId':temp.PassengerId,'Survived':final_pred}).to_csv('predictions_gb_13.csv',index=False)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


X.head()


# In[ ]:


#checking the accuracy of logisticregression on training data
lg=LogisticRegression(C=1.0)
scores=cross_val_score(lg,X.drop(['SibSp','Parch'],axis=True),y,cv=10)
print(scores.mean())


# In[ ]:


#training lg on whole data
lg.fit(X.drop(['SibSp','Parch'],axis=1),y)


# In[ ]:


#predicting the final response using logistic regression
pred_class_lg=lg.predict(test_df.drop(['SibSp','Parch'],axis=1))


# In[ ]:


#predicted response by logisticregression
pred_class_lg


# In[ ]:


#predicted response by Gradient Boosting
pred_class_gb=final_pred
pred_class_gb


# In[ ]:


#predictyed response by random forest
pred_class_rn


# In[ ]:


#converting all the responses from three models into dataframe to apply maxVoting ensambling technique
predicted_df=pd.DataFrame({'pred_class_gb':pred_class_gb,'pred_class_rn':pred_class_rn,'pred_class_lg':pred_class_lg})


# In[ ]:


predicted_df.head(5)


# In[ ]:


#generating new predicted response by takin response which is in majority.i.e takin the mode value of the predicted response
final_pred_clas=predicted_df.mode(axis=1,numeric_only=True)
final_pred_clas.head()  


# In[ ]:


#finally converting into numpy array to submit it to kaggle
final_pred_arr=np.resize(final_pred_clas,(418,))
final_pred_arr


# In[ ]:


#converting numpy array into csv file or dataFrame.this file we can submit as our final response to kaggle.
predictions=pd.DataFrame({'PassengerId':temp.PassengerId,'Survived':final_pred_arr}).to_csv('predictions_max_voting_14.csv',index=False)


# #### After the submission i got an accuracy of 0.77990 which is top 50% score.
# #### rank comes out to be 5126 out of 10,416 teams.

# #### ---

# #### some guidelines and suggestion on how to increse the accuracy score will be very helpful.
# #### thanks

# In[ ]:




