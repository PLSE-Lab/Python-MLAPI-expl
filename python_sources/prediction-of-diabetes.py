# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
diabetes=pd.read_csv('../input/diabetes.csv')
diabetes.head()
diabetes.columns
diabetes.isnull().sum()
cor=diabetes.corr()
cor["Outcome"].sort_values(ascending=False)
#Lets visualize our data
import seaborn as sns
plt.figure(figsize=(15,15))
sns.heatmap(cor)
fig1,ax1=plt.subplots(1,1)
ax1.pie(diabetes.Outcome.value_counts(),autopct='%1.1f%%',labels=['0','1'])
sns.scatterplot(x='Age',y='BMI',hue='Outcome',data=diabetes)
#We plot a distance plot for al the columns present to iunderstand their nature
import matplotlib.gridspec as gridspec
gs=gridspec.GridSpec(8,1)
plt.figure(figsize=(15,15))
for i,col in enumerate(diabetes[diabetes.iloc[:,0:8].columns]):
    ax2=plt.subplot(gs[i])
    sns.distplot(diabetes[col])
plt.show()  
#We found from the plot that some parameters has 0 value which they shouldnt have so we remove the 0 value
diabetes[['Glucose','BMI','BloodPressure','Insulin','SkinThickness']]=diabetes[['Glucose','BMI','BloodPressure','Insulin','SkinThickness']].replace(0,np.NaN)
diabetes['Glucose'].fillna(diabetes['Glucose'].mean(),inplace=True)
diabetes['BMI'].fillna(diabetes['BMI'].mean(),inplace=True)
diabetes['BloodPressure'].fillna(diabetes['BloodPressure'].mean(),inplace=True)
diabetes['Insulin'].fillna(diabetes['Insulin'].mean(),inplace=True)
diabetes['SkinThickness'].fillna(diabetes['SkinThickness'].mean(),inplace=True)
diabetes.isnull().sum()
diabetes.isna().sum()
sns.scatterplot(x='Age',y='BMI',data=diabetes,hue='Outcome')
#Scale the data for better accuracy
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=diabetes.drop(['Outcome'],axis=1)
y=diabetes['Outcome']
x=sc.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#accuracy of 72%
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
pred=rf.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)
#Best score of 77.36%
from sklearn.model_selection import GridSearchCV
param_grid=[{"n_estimators":np.arange(1,50)}]
forest_reg=RandomForestClassifier(random_state=30)
rfc_cv_grid=GridSearchCV(forest_reg,param_grid,cv=3)
rfc_cv_grid.fit(x_train,y_train)
rfc_cv_grid.best_estimator_
print("Best Score {}".format(str(rfc_cv_grid.best_score_)))
print("Best Parameters {}".format(str(rfc_cv_grid.best_params_)))
#accuracy score of 68.18
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
pred1=dt.predict(x_test)
accuracy_score(pred1,y_test)
#accuracy score of 75.24 when we apply GridSearchCV and tune the hyper parameters
depth=np.arange(1,20)
num_leafs=[1,5,10,20,50,100]
dt1=DecisionTreeClassifier(random_state=5)
params=[{"max_depth":depth,"min_samples_leaf":num_leafs}]
cv_grid=GridSearchCV(dt1,params,cv=5)
cv_grid.fit(x_train,y_train)
cv_grid.best_estimator_
print("Best Score {}".format(str(cv_grid.best_score_)))
print("Best Parameters {}".format(str(cv_grid.best_params_)))
#accuracy score of 69.4%
import xgboost as xgb
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(x_train, y_train)
predictions = gbm.predict(x_test)
accuracy_score(predictions,y_test)
#accuracy score of 72.07%
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=23)
knn.fit(x_train,y_train)
pred2=knn.predict(x_test)
accuracy_score(pred2,y_test)
knn.predict_proba(x_test)[:25]
#accuracy score of 77.68% when tuned the hyper parameter with GridSearchCV
grid_params={
    'n_neighbors':[10,15,20,23],
    'weights':['uniform','distance'],
    'metric':['euclidean','manhattan']
    }
knn=KNeighborsClassifier()    
gs=GridSearchCV(knn,grid_params,cv=3)  
gs_results=gs.fit(x_train,y_train)
gs_results.best_estimator_
gs_results.best_score_
gs_results.best_params_
#accuracy of 70.42%
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
clasifier=Sequential()
clasifier.add(Dense(128,activation='relu',input_dim=np.shape(x_train)[1]))
clasifier.add(Dropout(0.25))
clasifier.add(Dense(32,activation='relu'))
clasifier.add(Dropout(0.25))
clasifier.add(Dense(1,activation='sigmoid'))
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
clasifier.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
clasifier.fit(x_train,y_train,batch_size=10,epochs=100)
pred3=clasifier.predict(x_test)
pred3[1]
accuracy_score(pred3.round(),y_test)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.