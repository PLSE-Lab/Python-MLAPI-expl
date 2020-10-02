# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:24:12 2020

@author: subham

"""
"Kaggle:Titanic: Machine Learning from Disaster"

'''Since its labeled data and needs only requires classification,
 we will be using Supervised Clasification Models
 So, we will apply the below models and sort with highest accuracy
 
1.Random Forests
2. Support Vector Machines(SVM)
3.Decision Trees
4.KNN '''

# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
score_list=[]
warnings.filterwarnings('ignore')
print('-'*50)
 
"Importing DataSet"

training_set    =pd.read_csv('../input/titanic/train.csv')
test_set        =pd.read_csv('../input/titanic/test.csv')
final_output    =pd.read_csv('../input/titanic/gender_submission.csv')


"Data Preprocessing and Analizing"
survived        = training_set[training_set['Survived']==1]
not_survived    = training_set[training_set['Survived']==0]
training_set['Family']  =training_set['SibSp']+training_set['Parch']
test_set['Family']      =test_set['SibSp']+test_set['Parch']

list_column=['Pclass','Sex','Fare','Family']

for column in list_column:
    plt.figure(figsize=[24,12])
    sns.countplot(x=column,hue= 'Survived', data=training_set)
    
"Converting names to values"
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
data = [training_set, test_set]
for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
    

"For trainind data"
X               =training_set[['Pclass','Sex','Age','Family','Fare','Title']]
Y               =training_set['Survived']
X['Sex'].update(X['Sex'].map({'male': 1, 'female': 0}))

"For Testing data"
X_T             =test_set[['Pclass','Sex','Age','Family','Fare','Title']]
X_T['Sex'].update(X_T['Sex'].map({'male': 1, 'female': 0}))

"Adding missing data without using Imputer Mothod"
average_age         =X[['Age', 'Sex']].groupby("Sex").mean()
average_age_male    =(average_age['Age'][0]).round()
average_age_female  =(average_age['Age'][1]).round()

def Age_Predicted(data):
    age = data[0]
    sex = data[1]
    if pd.isnull(age):
        if(sex==1): 
            return average_age_male
        else:
            return average_age_female
    else:
        return age
    
X['Age'].update(X[['Age','Sex']].apply(Age_Predicted,axis=1))
X_T['Age'].update(X_T[['Age','Sex']].apply(Age_Predicted,axis=1))
X_T = X_T.fillna(X.mean())
     
"Splitting dataset to test and train"
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.25, random_state=0)

"Fitting Random Forest Classification"
rfc         = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, Y_train)
Y_pred_RFC  = rfc.predict(X_T)
RFC_score   =round(rfc.score(X_train, Y_train)*100,4)
print("RFC_score:" ,RFC_score)
score_list.append(RFC_score)

"Fitting SVM"
svc         =SVC(kernel='linear', random_state=0)
svc.fit(X_train, Y_train)
Y_pred_SVC  = svc.predict(X_test)
SVM_score   =round(svc.score(X_train, Y_train)*100,4)
print("SVM_score:", SVM_score)
score_list.append(SVM_score)

"Fitting KNN"
knn         =KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
Y_pred_KNN  = knn.predict(X_test)
KNN_score   =round(knn.score(X_train, Y_train)*100,4)
print("KNN_score:", KNN_score)
score_list.append(KNN_score)

"Fitting Decision Tree"
dt          =DecisionTreeClassifier()
dt.fit(X_train,Y_train)
Y_pred_DT   =dt.predict(X_T)
DT_score    =round(dt.score(X_train, Y_train)*100,4)
print("DT_score :", DT_score)
score_list.append(DT_score)

models_list=[Y_pred_RFC, Y_pred_SVC, Y_pred_KNN, Y_pred_DT]

for index in range(0, len(score_list)):
    if(score_list[index]==max(score_list)):
        predict_name=models_list[index]
        break
           
"Conveting the output to Survided_data.csv file "
final_output['Survived']=pd.DataFrame(predict_name, columns=['Survived'])
final_output.to_csv('submission.csv', index= False)


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())





