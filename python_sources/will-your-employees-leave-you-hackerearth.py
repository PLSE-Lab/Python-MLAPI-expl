#!/usr/bin/env python
# coding: utf-8

# Today the competition on the HackerEarth has finished at 7:30 A.M and this is my first competition. I am at the 109th position with 81.289 accuracy. I have gained moreknowledge on many things such as feature engineering, model selection, feature selection methods, and thinking from every possible angle to make the model better and better. I have also applied all the regression techniques in which random forest, xgboost, and ridge regression is best fit for this dataset. I gained this accuracy using ridge regression.
# 
# IT WAS REALLY GOOD EXPERIENCE

# i have used almost all regression techniques such as ANN,SVR,RF,Xgboost,ridgeregression,lasso,Decisiontree,gradientboosting etc

# In[ ]:


import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_df = pd.read_csv("/kaggle/input/Train.csv")


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
corr = train_df.corr()
sns.heatmap(corr)


# In[ ]:


X = train_df.drop(
    columns={'Employee_ID', 'Gender', 'Age',
        'Hometown', 'Unit', 'Relationship_Status',
         'Travel_Rate','Decision_skill_possess','Time_since_promotion',
        'Post_Level', 'Compensation_and_Benefits','Education_Level',
         'Attrition_rate','VAR1','VAR2','VAR3','VAR4','VAR5','VAR6','VAR7'})
y = train_df['Attrition_rate']


# here i am adding all the methods with i used

# the below code is used for converting the categorical variable into numeric one
# but here i am only taking some columns which is alredy numeric so i am not applying labelencoder for this

# i am not using labelencoder but only for knowledge i am adding this code.

# In[ ]:


'''from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
X['Gender'] = labelencoder.fit_transform(X['Gender'])
X['Relationship_Status'] = labelencoder.fit_transform(X['Relationship_Status'])
X['Hometown'] = labelencoder.fit_transform(X['Hometown'])
X['Unit'] = labelencoder.fit_transform(X['Unit'])
X['Decision_skill_possess'] = labelencoder.fit_transform(X['Decision_skill_possess'])
X['Compensation_and_Benefits'] = labelencoder.fit_transform(X['Compensation_and_Benefits'])'''


# In[ ]:


X


# In[ ]:


k=0
col = ['Time_of_service', 'growth_rate', 'Pay_Scale', 'Work_Life_balance']
fig,ax = plt.subplots(2,2,figsize=(20,20))
for i in np.arange(2):
    for j in np.arange(2):
        chart = sns.countplot(x = X[col[k]],ax = ax[i][j],order = X[col[k]].value_counts().index)
        chart.set_xticklabels(rotation=90,labels = chart.get_xticklabels())
        k+=1
plt.show()


# below code is use to fill missing values you can also use 0 , mean() , mode() to fill missing values
# but here i am applying median because this is good for my dataset

# In[ ]:


X['Pay_Scale'].fillna((X['Pay_Scale'].median()), inplace=True)
X['Time_of_service'].fillna((X['Time_of_service'].median()), inplace=True)
X['Work_Life_balance'].fillna((X['Work_Life_balance'].median()), inplace=True)


# below code is used to fill missing values by 0.

# below code is use to put values which are max in particular column for example in age column 27 years encountered more times compare to other years

# In[ ]:


'''X_train['Time_of_service'] = X_train['Time_of_service'].fillna(value=0)
X_train['Work_Life_balance'] = X_train['Work_Life_balance'].fillna(value=0)
X_train['Pay_Scale'] = X_train['Pay_Scale'].fillna(value=0)'''


# In[ ]:


X['Time_of_service'].value_counts()


# as you see in above output the value 6 is encountered more times compare to other so i can add this value to fill missing values

# In[ ]:


'''X['Time_of_service']=X['Time_of_service'].fillna(value=6.0)
X['Work_Life_balance']=X['Work_Life_balance'].fillna(value=1.0)
X['Pay_Scale']=X['Pay_Scale'].fillna(value=8.0)'''


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


X_train.info(),
X_test.info()


# In[ ]:


X_train.columns


# In[ ]:


X_test.columns


# In[ ]:


X_train['Time_of_service'] = X_train['Time_of_service'].astype(np.int64)
X_train['Pay_Scale'] = X_train['Pay_Scale'].astype(np.int64)
X_train['Work_Life_balance'] = X_train['Work_Life_balance'].astype(np.int64)

X_test['Time_of_service'] = X_test['Time_of_service'].astype(np.int64)
X_test['Pay_Scale'] = X_test['Pay_Scale'].astype(np.int64)
X_test['Work_Life_balance'] = X_test['Work_Life_balance'].astype(np.int64)


# In[ ]:


X_train.info(),
X_test.info()


# applying feature scaling and ravel method to convert series into ndarray

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_test = y_test.ravel()
y_train = y_train.ravel()


# # **NOW APPLYING ALL THE REGRESSION METHODS IN WHICH THE RIDGE REGRESSION IS THE BEST**

# # 1)ridge regression

# In[ ]:


#ridge
from sklearn.linear_model import Ridge
reg = Ridge(alpha=1.0)
reg.fit(X_train,y_train)

# Predicting a new result
from sklearn.metrics import mean_squared_error
from math import sqrt
pred1 = reg.predict(X_test)
score1 = 100 * max(0,1 - sqrt(mean_squared_error(y_test,pred1)))
print('ridge regression:',score1)


# # 2)random forest

# In[ ]:


# random forest pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

model = RandomForestRegressor(n_jobs=-1, criterion="mse", max_features='auto', min_samples_split=5, max_depth=3,
                              random_state=0, min_samples_leaf=1090, bootstrap=True, verbose=0)
estimators = 110
model.set_params(n_estimators=estimators)

pipeline = Pipeline([('scaler2', StandardScaler()),
                     ('RandomForestRegressor: ', model)])
pipeline.fit(X_train, y_train)
pred2 = pipeline.predict(X_test)
score2 = 100 * max(0,1 - sqrt(mean_squared_error(y_test, pred2)))
print('random forest:', score2)


# # 3)bayesian ridge

# In[ ]:


#BR
from sklearn.linear_model import BayesianRidge
reg2 = BayesianRidge(compute_score=True)
reg2.fit(X_train,y_train)

# Predicting a new result
from sklearn.metrics import mean_squared_error
from math import sqrt
pred3 = reg2.predict(X_test)
score3 = 100 * max(0,1 - sqrt(mean_squared_error(y_test,pred3)))
print('bayesian regression:',score3)


# # 4)decision tree

# In[ ]:


#fitting the decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
reg4 = DecisionTreeRegressor(random_state=0)
reg4.fit(X_train,y_train)

# Predicting a new result
from sklearn.metrics import mean_squared_error
from math import sqrt
pred4 = reg4.predict(X_test)
score4 = 100 * max(0,1 - sqrt(mean_squared_error(y_test,pred4)))
print('decision tree regression',score4)


# # 5)ANN

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from math import sqrt
# Initialising the ANN
model = Sequential()
model.add(Dense(output_dim=10, kernel_initializer='normal', activation='relu', input_dim=4))
model.add(Dense(output_dim=10, kernel_initializer='normal', activation='relu'))
model.add(Dense(output_dim=1, kernel_initializer='normal',activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse','mae'])

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size=5, nb_epoch=10)
pred5 = model.predict(X_test)
score5 = 100 * max(0, 1 - sqrt(mean_squared_error(y_test, pred5)))
print('KNN:', score5)


# # 6)gradientboostingregressor

# In[ ]:


#GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
reg5 = GradientBoostingRegressor(random_state=0)
reg5.fit(X_train, y_train)

# Predicting a new result

from sklearn.metrics import mean_squared_error
from math import sqrt
pred6 = reg5.predict(X_test)
score6 = 100 * max(0,1 - sqrt(mean_squared_error(y_test,pred6)))
print('GradientBoostingRegressor:',score6)


# # 7)KNN regressor

# In[ ]:


#fitting KNN regression to the training set
from sklearn.neighbors import KNeighborsRegressor
reg6 = KNeighborsRegressor(n_neighbors=5)
reg6.fit(X_train,y_train)

# Predicting a new result

from sklearn.metrics import mean_squared_error
from math import sqrt
pred7 = reg6.predict(X_test)
score7 = 100 * max(0,1 - sqrt(mean_squared_error(y_test,pred7)))
print('KNN regression',score7)


# # 8)Lasso regression

# In[ ]:


# lasso regression
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
clf2 = linear_model.Lasso(alpha=0.1, normalize=True,max_iter=1e5)
from sklearn.pipeline import Pipeline
pipe2 = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', linear_model.Lasso())
        ])
pipe2.fit(X_train, y_train)
pred8 = pipe2.predict(X_test)
# Predicting a new result
from sklearn.metrics import mean_squared_error
from math import sqrt
score8 = 100 * max(0, 1 - sqrt(mean_squared_error(y_test, pred8)))
print('lasso regression:', score8)


# # 9)SVR

# In[ ]:


#SVR
#fitting svr to dataset
from sklearn.svm import SVR
reg7 = SVR(kernel='rbf')
reg7.fit(X_train,y_train)

# Predicting a new result

from sklearn.metrics import mean_squared_error
from math import sqrt
pred9 = reg7.predict(X_test)
score9 = 100 * max(0,1 - sqrt(mean_squared_error(y_test,pred9)))
print('SVR',score9)


# In[ ]:


print('ridge regression:',score1)
print('random forest:', score2)
print('bayesian regression:',score3)
print('decision tree regression',score4)
print('KNN:', score5)
print('GradientBoostingRegressor:',score6)
print('KNN regression',score7)
print('lasso regression:', score8)
print('SVR',score9)


# In[ ]:


# testing on actual data
test_df = pd.read_csv("/kaggle/input/Test.csv")

EID = test_df['Employee_ID']


# In[ ]:


test_df = test_df.drop(
    columns={'Employee_ID', 'Gender', 'Age',
        'Hometown', 'Unit', 'Relationship_Status',
         'Travel_Rate','Decision_skill_possess','Time_since_promotion',
        'Post_Level', 'Compensation_and_Benefits','Education_Level',
         'VAR1','VAR2','VAR3','VAR4','VAR5','VAR6','VAR7'})


# In[ ]:


test_df.info()


# In[ ]:


test_df.columns


# In[ ]:


test_df['Time_of_service'].fillna((test_df['Time_of_service'].median()), inplace=True)
test_df['Work_Life_balance'].fillna((test_df['Work_Life_balance'].median()), inplace=True)
test_df['Pay_Scale'].fillna((test_df['Pay_Scale'].median()), inplace=True)


# In[ ]:


# converting float variables columns in int64 training set
test_df['Time_of_service'] = test_df['Time_of_service'].astype(np.int64)
test_df['Work_Life_balance'] = test_df['Work_Life_balance'].astype(np.int64)
test_df['Pay_Scale'] = test_df['Pay_Scale'].astype(np.int64)


# In[ ]:


test_df.info()


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test_df = sc.fit_transform(test_df)


# In[ ]:


main_pred = reg.predict(test_df)
main_submission = pd.DataFrame({'Employee_ID': EID, 'Attrition_rate': main_pred})

main_submission.to_csv("submission.csv", index=False)
main_submission.head()


# # **EXTRA WORK**

# In[ ]:


w = train_df.drop(columns={'Employee_ID','Attrition_rate'})

z = train_df['Attrition_rate']


# In[ ]:


k=0
col = ['Hometown','Unit','Decision_skill_possess','Compensation_and_Benefits','Gender','Relationship_Status',
      'Education_Level','Travel_Rate','Post_Level','Work_Life_balance']
fig,ax = plt.subplots(5,2,figsize=(20,20))
for i in np.arange(5):
    for j in np.arange(2):
        chart = sns.countplot(x = w[col[k]],ax = ax[i][j],order = w[col[k]].value_counts().index)
        chart.set_xticklabels(rotation=90,labels = chart.get_xticklabels())
        k+=1
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
w['Gender'] = labelencoder.fit_transform(w['Gender'])
w['Relationship_Status'] = labelencoder.fit_transform(w['Relationship_Status'])
w['Hometown'] = labelencoder.fit_transform(w['Hometown'])
w['Unit'] = labelencoder.fit_transform(w['Unit'])
w['Decision_skill_possess'] = labelencoder.fit_transform(w['Decision_skill_possess'])
w['Compensation_and_Benefits'] = labelencoder.fit_transform(w['Compensation_and_Benefits'])

w['Age'].fillna((w['Age'].median()), inplace=True)
w['Time_of_service'].fillna((w['Time_of_service'].median()), inplace=True)
w['Work_Life_balance'].fillna((w['Work_Life_balance'].median()), inplace=True)
w['VAR2'].fillna((w['VAR2'].median()), inplace=True)
w['VAR4'].fillna((w['VAR4'].median()), inplace=True)
w['Pay_Scale'].fillna((w['Pay_Scale'].median()), inplace=True)


# In[ ]:


w.corr()


# as you can see on the above output the age , time_of_service are more correlated with each other and also time_since_pramotion,pay_scale somewhat correlated with age.

# In[ ]:


#this code is use to check correlatoin between two features
print(w[["Age","Time_of_service"]].corr())
print(w[["Age","Decision_skill_possess"]].corr())
print(w[["Age","Time_since_promotion"]].corr())


# # **please do upvote if you like**
