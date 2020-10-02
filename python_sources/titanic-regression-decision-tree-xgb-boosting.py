#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[92]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # View Train Test DataSet

# In[93]:


# Print Train Data
train = pd.read_csv('../input/train.csv')
print(train.info())
train.head()


# In[94]:


# Print Test Data
test = pd.read_csv('../input/test.csv')
print(test.info())
test.head()


# In[95]:



train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
allData = pd.concat([train,test], axis = 0)

nullTrain = train.isnull().sum()/len(train)
print('----------')
print('Null Value - Train')
print(nullTrain[nullTrain > 0])

nullTest = test.isnull().sum()/len(test)
print('----------')
print('Null Value - Test')
print(nullTest[nullTest > 0])


# # Fill in NA with Means/Mode and data Transformation

# In[96]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
allData = pd.concat([train,test], axis = 0)
allDataTransform = allData
allDataTransform['Age'] = allDataTransform['Age'].fillna(allData['Age'].mean())
allDataTransform['Embarked'] = allDataTransform['Embarked'].fillna(allData['Embarked'].mode())
allDataTransform['Fare'] = allDataTransform['Fare'].fillna(allData['Fare'].mean())
allDataTransform['Fare'] = allDataTransform['Fare'].apply(lambda fare: allData['Fare'].mean() if fare == 0 else fare)
allDataTransform['Cabin'] = allDataTransform['Cabin'].fillna('N')
allDataTransform['Cabin'] = allDataTransform['Cabin'].apply(lambda cabinCode: 'N' if cabinCode == 'N' else 'C')
allDataTransform['Family'] = allDataTransform['SibSp'] +  allDataTransform['Parch']

allDataTransform=allDataTransform.drop(columns=['SibSp'])
allDataTransform=allDataTransform.drop(columns=['Parch'])


# In[97]:


allDataTransform.head()


# In[98]:


catCol = ['Pclass', 'Sex' , 'Embarked','Cabin']
numCol = ['Age','Fare','Family']


# # Check Correlations

# In[99]:


import seaborn as sns
sns.heatmap(allDataTransform[numCol].corr(),annot=True)


# # Relationship of Survived/Dead vs Age

# In[100]:


s = allDataTransform[allDataTransform['Survived'] > 0]
d = allDataTransform[allDataTransform['Survived'] == 0]
plot = sns.distplot(s['Age'], color="b")
plot = sns.distplot(d['Age'] , color="r")


# # Try to do some grouping on Age

# In[101]:


def groupAge(age):
    if age >= 50: return 'O'
    elif age >= 40 : return 'M'
    elif age >= 30 : return 'A'
    elif age >= 20 : return 'Y'
    elif age >= 10 : return 'T'
    elif age >= 0: return 'K'

allDataTransform['AgeGroup'] = allDataTransform['Age'].apply(lambda x: groupAge(x))


# # Relationship of Survived/Dead vs family Size

# In[102]:


s = allDataTransform[allDataTransform['Survived'] > 0]
d = allDataTransform[allDataTransform['Survived'] == 0]
plot = sns.distplot(s['Family'], color="b")
plot = sns.distplot(d['Family'] , color="r")


# # Try to do some grouping on family Size

# In[103]:


def familySize(n):
    if n >= 5: return 'L'
    elif n >= 3 : return 'M'
    elif n >= 2 : return 'S'
    elif n >= 1 : return 'N'
    elif n == 0 : return 'I'
allDataTransform['familySizeGroup'] = allDataTransform['Family'].apply(lambda x: familySize(x))


# # Relationship of Survived/Dead vs Fare

# In[104]:


sns.lmplot("Survived", "Fare", data=allDataTransform)


# In[105]:


s = allDataTransform[allDataTransform['Survived'] > 0]
d = allDataTransform[allDataTransform['Survived'] == 0]
plot = sns.distplot(s['Fare'], color="b")
plot = sns.distplot(d['Fare'] , color="r")


# View Para Analysis Report using pandas_profiling

# In[106]:


import pandas_profiling as pdp
pdp.ProfileReport(allDataTransform)


# # Fix Data

# In[107]:


# still have NA Value in Embarked????
allDataTransform[allDataTransform['Embarked'].isnull()==True]


# In[108]:


allDataTransform['Embarked'] = allDataTransform['Embarked'].fillna('S')
allDataTransform[allDataTransform['Embarked'].isnull()==True]


# # Get Dummy Variables for Group Para

# In[109]:


allDataTransformDummy = pd.get_dummies(allDataTransform, columns = ["Embarked"], prefix="Em")
# allDataTransformDummy = pd.get_dummies(allDataTransformDummy, columns = ["Cabin"], prefix="Cabin")
# allDataTransformDummy = pd.get_dummies(allDataTransformDummy, columns = ["Sex"], prefix="Sex")
allDataTransformDummy = pd.get_dummies(allDataTransformDummy, columns = ["AgeGroup"], prefix="Age")
allDataTransformDummy = pd.get_dummies(allDataTransformDummy, columns = ["Pclass"], prefix="Class")
allDataTransformDummy = pd.get_dummies(allDataTransformDummy, columns = ["familySizeGroup"], prefix="Family")

allDataTransformDummy.head()


# In[110]:


from sklearn.preprocessing import LabelEncoder
catCol = ['Sex','Cabin']
for col in catCol:
    lib = LabelEncoder()
    lib.fit(list(allDataTransformDummy[col].values))
    allDataTransformDummy[col] = lib.transform(list(allDataTransformDummy[col].values))

allDataTransformDummy.head()
print(allDataTransformDummy.columns)


# In[111]:


# AllCol = allDataTransformDummy.columns
# AllCol.remove


# # Check Corr again after data transformation

# In[113]:


#We have 892 Train Data
trainData = allDataTransformDummy[allDataTransformDummy['PassengerId']<892]
Related = ['Survived','Age', 'Cabin', 'Fare', 'Sex',
         'Family', 'Em_C', 'Em_Q', 'Em_S', 'Age_A',
       'Age_K', 'Age_M', 'Age_O', 'Age_T', 'Age_Y', 'Class_1', 'Class_2',
       'Class_3', 'Family_L', 'Family_M', 'Family_N', 'Family_S']

sns.heatmap(trainData[Related].corr(),annot=True)


# In[ ]:


corr = trainData[Related].corr()
corr


# # List out para with high corr

# In[114]:


trainData = allDataTransformDummy[allDataTransformDummy['PassengerId']<892]

corr = trainData[Related].corr()

corrPos = corr[corr['Survived']>=0.1]['Survived']
corrNeg = corr[corr['Survived']<=-0.1]['Survived']

corrHigh = corrPos.append(corrNeg)
print(corrHigh)
corrHigh.index

rel = []
for x in corrHigh.index:
    rel.append(x)

print(rel)


# # Select Para for model

# In[ ]:


from sklearn.linear_model import LogisticRegression
relCol = rel
# relCol = ['Age','Cabin','Embarked', 'Parch' , 'Pclass','Sex','SibSp''Fare']
trainData = allDataTransformDummy[allDataTransformDummy['PassengerId']<892]
testData = allDataTransformDummy[allDataTransformDummy['PassengerId']>=892]
trainResult = trainData['Survived']
# trainResult = trainData['Survived'].apply(lambda x: int(str(x)[:1]))

trainDataNeeded = trainData[relCol]
testDataNeeded = testData[relCol]

trainDataNeeded = trainDataNeeded.drop(columns=['Survived'])
testDataNeeded = testDataNeeded.drop(columns=['Survived'])

trainDataAll = trainData.drop(columns=['Survived'])
testDataAll = testData.drop(columns=['Survived'])

trainDataAll = trainDataAll.drop(columns=['Name','Ticket'])
testDataAll = testDataAll.drop(columns=['Name','Ticket'])

# trainDataNeeded = trainDataNeeded.drop(columns=['Em_C','Em_S','Fare'])
# testDataNeeded = testDataNeeded.drop(columns=['Em_C','Em_S','Fare'])

trainDataNeeded.head()


# In[ ]:


trainResult.head()


# # Log Regression Model

# In[ ]:


LogReg = LogisticRegression().fit(trainDataNeeded, trainResult)
modelScore = LogReg.score(trainDataNeeded, trainResult)
print('Score: ' + str(modelScore))


# In[ ]:


predict = LogReg.predict(testDataNeeded)
predict = pd.DataFrame(predict, columns =['Survived'])
predictResult = pd.concat([testData['PassengerId'], predict], axis = 1)
predictResult['Survived'] = predictResult['Survived'].apply(lambda x: str(x)[:1])

fileName = "Result_{model}_{score}.csv".format(model='LogisticRegression', score=round(modelScore,2))
predictResult.to_csv(fileName, index=False)
predictResult.head()
print(fileName)


# # Import XGB Boosting Model

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from xgboost import plot_tree


# from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from xgboost import XGBClassifier

#split test and train
# train_X, test_X, train_y, test_y = train_test_split(trainDataNeeded, trainResult, test_size=0.25)




# # Decision Tree with Boosting

# In[ ]:


XGB_Class = XGBClassifier().fit(trainDataNeeded, trainResult, verbose=False)
modelScore = XGB_Class.score(trainDataNeeded, trainResult)
print('Score: ' + str(modelScore))


# In[ ]:


predictions = XGB_Class.predict(testDataNeeded)
predictions = pd.DataFrame(predictions, columns =['Survived'])
predictResult = pd.concat([testData['PassengerId'], predictions], axis = 1)
predictResult['Survived'] = predictResult['Survived'].apply(lambda x: str(x)[:1])

fileName = "Result_{model}_{score}.csv".format(model='Tree_XGB', score=round(modelScore,2))
predictResult.to_csv(fileName, index=False)
print('File: ' + str(fileName))
predictResult.head()


# In[ ]:



plot_tree(XGB_Class)
plt.show()

print(XGB_Class.feature_importances_)
print(trainDataNeeded.columns)

#Help~~ 
#Anyone knows how to plot a larger graph? resolution is too small now
#Appreciate if you can leave me a comment.
#Thanks a lot in advance


# # Regression with Boosting

# In[ ]:


XGB_Reg = XGBRegressor().fit(trainDataNeeded, trainResult, verbose=False)
modelScore = XGB_Reg.score(trainDataNeeded, trainResult)
print('Score: ' + str(modelScore))


# In[ ]:



predictions = XGB_Reg.predict(testDataNeeded)
predictions = pd.DataFrame(predictions, columns =['Survived'])
predictResult = pd.concat([testData['PassengerId'], predictions], axis = 1)
predictResult['Survived'] = predictResult['Survived'].apply(lambda x: str(x)[:1])

fileName = "Result_{model}_{score}.csv".format(model='Regression_XGB', score=round(modelScore,2))
predictResult.to_csv(fileName, index=False)
print('File: ' + str(fileName))
predictResult.head()


# In[ ]:


print(XGB_Reg.feature_importances_)
print(trainDataNeeded.columns)


# # Decision Tree with Boosting and GridSearchCV to get the best para

# In[ ]:


myPara = {
    'n_estimators': range(10,20,1),
    'max_depth': range(5,10,1),
    'learning_rate': [.175, .178, .18 , .182, .185],
#     'colsample_bytree': [.6, .7, .8, .9 ,1]
}
modelEstimator = XGBClassifier(learning_rate=0.05, max_depth=2)
GridSearch = GridSearchCV(estimator = modelEstimator, 
                          param_grid = myPara,
                          scoring = 'accuracy',
                          cv = 6
                         )

GridSearch.fit(trainDataNeeded,trainResult)
print("best score:",GridSearch.best_score_)
print("best alpha:",GridSearch.best_params_)


# In[ ]:


predictions = GridSearch.predict(testDataNeeded)
predictions = pd.DataFrame(predictions, columns =['Survived'])
predictResult = pd.concat([testData['PassengerId'], predictions], axis = 1)
predictResult['Survived'] = predictResult['Survived'].apply(lambda x: str(x)[:1])

fileName = "Result_{model}_{score}.csv".format(model='Tree_XGB_GSCV', score=round(GridSearch.best_score_,2))
predictResult.to_csv(fileName, index=False)
print('File: ' + str(fileName))
predictResult.head()


# # Decision Tree (all Para) with Boosting and GridSearchCV

# In[ ]:


# try will all para =======

print (trainDataAll.head())

myPara = {
    'n_estimators': range(10,20,1),
    'max_depth': range(2,10,1),
    'learning_rate': [.177,.178,.179 ,.180, .182,.185],
#     'colsample_bytree': [.6, .7, .8, .9 ,1]
}

modelEstimator = XGBClassifier(learning_rate=0.1, max_depth=3)
GridSearch = GridSearchCV(estimator = modelEstimator, 
                          param_grid = myPara,
                          cv = 5
                         )

GridSearch.fit(trainDataAll,trainResult)
print("best score:",GridSearch.best_score_)
print("best alpha:",GridSearch.best_params_)


# In[ ]:


predictions = GridSearch.predict(testDataAll)
predictions = pd.DataFrame(predictions, columns =['Survived'])
predictResult = pd.concat([testData['PassengerId'], predictions], axis = 1)
predictResult['Survived'] = predictResult['Survived'].apply(lambda x: str(x)[:1])

fileName = "Result_{model}_{score}.csv".format(model='TreeALL_XGB_GSCV', score=round(GridSearch.best_score_,2))
predictResult.to_csv(fileName, index=False)
print('File: ' + str(fileName))
predictResult.head()


# # Regression with Boosting and GridSearchCV

# **The testing result so bad at 0.45, disabled**

# In[ ]:




# myPara = {
#     'n_estimators': range(10,30,2),
#     'max_depth': range(2,20,2),
#     'learning_rate': [.1, .2, .3  ],
# #     'colsample_bytree': [.6, .7, .8, .9 ,1]
# }
# modelEstimator = XGBRegressor(learning_rate=0.05, max_depth=2)
# GridSearch = GridSearchCV(estimator = modelEstimator, 
#                           param_grid = myPara,
#                           cv = 5
#                          )

# GridSearch.fit(trainDataNeeded,trainResult)
# print("best score:",GridSearch.best_score_)
# print("best alpha:",GridSearch.best_params_)


# In[ ]:


# predictions = GridSearch.predict(testDataNeeded)
# predictions = pd.DataFrame(predictions, columns =['Survived'])
# predictResult = pd.concat([testData['PassengerId'], predictions], axis = 1)
# predictResult['Survived'] = predictResult['Survived'].apply(lambda x: str(x)[:1])
# predictResult.to_csv("Result_XGB_Reg_GSCV.csv", index=False)
# predictResult.head()

