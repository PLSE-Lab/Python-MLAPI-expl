"""
Created on Mon Mar 12 21:39:13 2018
Predict survival on the Titanic and get familiar with ML basics
@author: Jenny
"""

#Import basic packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

#Read in the file
file="../input/train.csv"
data=pd.read_csv(file, index_col=0)

#Understand the data
print(data.head(5))
data.info()
data.describe()
data['Survived'].value_counts()

#review test file 
testfile="../input/test.csv"
testdata=pd.read_csv(testfile, index_col=0)
testdata.info()
testdata.head()

#Data cleaning for training dataset

#Drop features that I think won't carry too much info
data=data.drop(["Name","Ticket"],axis=1)
#fill NAN age with median age and change it to integer 
data['Age']=data['Age'].fillna(data['Age'].median()).astype(int)
#fill NAN cabin value with NoCabinnote
data['Cabin']=data['Cabin'].fillna("X")
#Keep Cabin Letter Only
data["CabinL"]=data["Cabin"].apply(lambda x: x[0])
#fill the rest NAN with its corresponding column's most freq value
data=data.apply(lambda x: x.fillna(x.value_counts().index[0]))
#Drop Cabin Column
data=data.drop(["Cabin"],axis=1)
#convert object to dummy variables
data=pd.get_dummies(data, columns = ["Sex", "CabinL", "Embarked"])

print(data.head(5))
data.info()
data.describe()


#Data cleaning for testing dataset

#Drop features that I think won't carry too much info
testdata=testdata.drop(["Name","Ticket"],axis=1)
#fill NAN age with median age and change it to integer 
testdata['Age']=testdata['Age'].fillna(data['Age'].median()).astype(int)
#fill NAN cabin value with NoCabinnote
testdata['Cabin']=testdata['Cabin'].fillna("X")
#Keep Cabin Letter Only
testdata["CabinL"]=testdata["Cabin"].apply(lambda x: x[0])
#fill the rest NAN with its corresponding column's most freq value
testdata=testdata.apply(lambda x: x.fillna(x.value_counts().index[0]))
#Drop Cabin Column
testdata=testdata.drop(["Cabin"],axis=1)
#convert object to dummy variables
testdata=pd.get_dummies(testdata, columns = ["Sex", "CabinL", "Embarked"])

# to match cleaned training dataset
testdata.insert(loc=13, column = "CabinL_T", value=0)

print(testdata)

#Check Correlation
Correlation=data.corr()
pd.DataFrame(Correlation)
correlation_Y = pd.DataFrame(Correlation["Survived"])
correlation_Y.sort_values(by = "Survived", ascending = False)
print(correlation_Y)



#data Visualization
#histogram
data.hist()
plt.figure(figsize=(10.8,7.6))
plt.show()

#Multimodal Data Visualizations
scatter_matrix(data)
plt.figure(figsize=(21.6,15.2))
plt.show()

#correlation matrix
# matshow: Plot a matrix or array as an image
fig=plt.figure(figsize=(21.6,15.2))
ax=fig.add_subplot(111)
cax=ax.matshow(data.corr(),vmin=-1,vmax=1, interpolation = "none")
fig.colorbar(cax)
ticks=np.arange(0,20,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
#set names 
names = ["Survived","Pclass","Age","SibSp","Parch","Fare" ,"Female" ,"Male" ,"Cab_A" ,"Cab_B" ,"Cab_C","Cab_D","Cab_E" ,"Cab_F","Cab_G" ,"Cab_T","Cab_X","Em_C","Em_Q","Em_S"]
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()




#Start Working on Machine Learning
#http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
#Start Working on Machine Learning


#Start Working on Machine Learning
#import packages for machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#split out valudation dataset
X= data.iloc[:,1:]
Y= data.iloc[:,[0]]
validation_size=0.2
seed=7
X_train,X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size=validation_size, random_state=seed)

#test option and evaluation matrixs
num_folds=10
scoring='accuracy'


#evaulate different models
pipelines=[]
pipelines.append(("ScaledLSVC", Pipeline([('Scaler',StandardScaler()), ("LSVC", LinearSVC())]))) 
pipelines.append(("ScaledSVC", Pipeline([('Scaler',StandardScaler()), ("SVC", SVC())]))) 
pipelines.append(("ScaledRF", Pipeline([('Scaler',StandardScaler()), ("RF", RandomForestClassifier())]))) 
pipelines.append(("ScaledBC", Pipeline([('Scaler',StandardScaler()), ("BC", BaggingClassifier())]))) 
pipelines.append(("ScaledABC", Pipeline([('Scaler',StandardScaler()), ("RF", AdaBoostClassifier())]))) 

names=[]
results=[]

for name, model in pipelines:
    kfold=KFold(n_splits=num_folds, random_state=seed)
    cv_results=cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print( "model:%s cv_results_avg:%f cv_results_std:(%f) " % (name, cv_results.mean(), cv_results.std()))


    
#compare Algorithms

fig=plt.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#SVM and RF Performs well

#Use gridsearch to fine tune SVM
scaler=StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C':Cs, 'gamma': gammas}
model=SVC()
kfold=KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result=grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



#Use gridsearch to fine tune RF
scaler=StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
n_estimators = [200, 500]
max_features = ['auto', 'sqrt']
max_depth = [10,30,50, 100]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators, 
              'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': min_samples_leaf,
             'bootstrap': bootstrap}
model=RandomForestClassifier()
kfold=KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result=grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#Best: 0.844101 using {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 500, 'min_samples_split': 10, 'max_features': 'auto', 'max_depth': 100}



#finalize
#Best RF: 0.844101 using {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 500, 'min_samples_split': 10, 'max_features': 'auto', 'max_depth': 100}
#Prepare the model

scaler=StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

model=RandomForestClassifier(n_estimators=1000,max_features= 'auto',max_depth= 100, min_samples_split= 10,min_samples_leaf= 1, bootstrap= True)

model.fit(rescaledX, Y_train)

#estimate accuracy on Validation dataset
rescaledTestdataX= scaler.transform(testdata.iloc[:,0:])
prediction = model.predict(rescaledTestdataX)


print(prediction)

