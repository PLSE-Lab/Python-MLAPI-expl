#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as r


# In[ ]:


#importing ML packages
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical


# In[ ]:


#function to generate random colors
def randomColor(n):
    color = []
    colorArr = ['00','11','22','33','44','55','66','77','88','99','AA','BB','CC','DD','EE','FF']
    for _ in range(n):
        color.append('#' + colorArr[r.randint(0,15)] + colorArr[r.randint(0,15)] + colorArr[r.randint(0,15)])
    return color


# In[ ]:


#get datasets
traindf = pd.read_csv('/kaggle/input/titanic/train.csv')
display(traindf.head())
print(traindf.shape)


# In[ ]:


testdf = pd.read_csv('/kaggle/input/titanic/test.csv')
display(testdf.head())
print(testdf.shape)


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (18,9))

#Gender vs Survuval Ratio
gender_survived = traindf[['Sex', 'Survived']].groupby('Sex').mean()
gender_survived.plot(kind='barh', 
                     color = [randomColor(len(gender_survived))], 
                     ax = axes[0,0],
                     title = 'Gender vs Survival Ratio',
                     xlim = (0,1),
                     grid = True,)

#Pclass vs Survival Ratio
pclass_survived = traindf[['Pclass', 'Survived']].groupby('Pclass').mean()
pclass_survived.plot(kind='bar', 
                     color = [randomColor(len(pclass_survived))], 
                     ax=axes[0,1],
                     title = 'Pclass vs Survival Ratio',
                     ylim = (0,1),
                     grid = True,)

#Embarked vs Survival Ratio
embarked_survived = traindf[['Embarked', 'Survived']].groupby('Embarked').mean()
embarked_survived.plot(kind='bar', 
                       color = [randomColor(len(embarked_survived))], 
                       ax=axes[0,2],
                       title = 'Embarked vs Survival Ratio',
                       ylim = (0,1),
                       grid = True,)

#Parch vs Survival Ratio
parch_survived = traindf[['Parch', 'Survived']].groupby('Parch').mean()
parch_survived.plot(kind='bar', 
                    color = [randomColor(len(parch_survived))], 
                    ax=axes[1,0], 
                    title = 'Parch vs Survival Ratio',
                    ylim = (0,1),
                    grid = True,)

#SibSp vs Survival Ratio
sibsp_survived = traindf[['SibSp', 'Survived']].groupby('SibSp').mean()
sibsp_survived.plot(kind='bar', 
                    color = [randomColor(len(sibsp_survived))], 
                    ax=axes[1,1],
                    title = 'Parch vs Survival Ratio',
                    ylim = (0,1),
                    grid = True,)

#Age Group vs Survival Ratio
agegrp = {'child':(0,13), 'teen':(13,20), 'young_adult':(20,35), 'middle_adult':(35,45), 'old_adult':(45,60), 'senior_citizen':(60,100)}
age_survival = traindf[['Age','Survived']].dropna().reset_index(drop=True)
age_survival['age_grp'] = None

for i in range(len(age_survival)):
    for grp in agegrp:
        temp = agegrp[grp]
        if age_survival.loc[i,'Age'] in range(temp[0],temp[1]):
            age_survival.loc[i,'age_grp'] = grp
            break
            
age_survival = age_survival.drop(columns=['Age']).groupby('age_grp').mean()

age_survival.plot(kind= 'barh', 
                  color = [randomColor(6)], 
                  legend=False, ax=axes[1,2],
                  title = 'Age group vs Survival Ratio',
                  xlim = (0,1),
                  grid = True,)

fig.tight_layout()
fig.show()


# In[ ]:


traindf2 = traindf[['Survived', 'Pclass', 'Sex', 'SibSp']]


# <h3>I am using only Pclass, Sex and Sibsp as features for training because taking produced the best results for me</h3>

# In[ ]:


missingdf = traindf2.transpose()
missingdf['missing values'] = missingdf.apply(lambda x: len(traindf)-x.count(), axis=1)
missingdf = missingdf[['missing values']]
missingdf


# In[ ]:


features = ["Pclass", "Sex", "SibSp"]
X = pd.concat([pd.get_dummies(traindf2[features[0]]),pd.get_dummies(traindf2[features[1:]])], axis = 1, sort = False)
display(X.head())
print(X.shape)


# In[ ]:


#All values have been converted to categorical variables
X = preprocessing.StandardScaler().fit(X).transform(X)
y = to_categorical(traindf2['Survived'])
num_features = len(X[0])
num_classes = len(y[0])
print(X[0:3])
print(y[0:3])
print(num_features)
print(num_classes)


# In[ ]:


# define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_features, activation='relu', input_shape=(num_features,)))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


# build the model
model = classification_model()

# fit the model
history = model.fit(X, y, validation_split=0.15, epochs=20, verbose=1, shuffle=True)


# In[ ]:


print('Validation Accuracy : ',round(history.history['val_accuracy'][-1]*100,4))


# In[ ]:


Xfinaltest = pd.concat([pd.get_dummies(testdf[features[0]]),pd.get_dummies(testdf[features[1:]])], axis = 1, sort = False)
display(Xfinaltest.head())


# In[ ]:


#make prediction on test dataset
Ypred = model.predict_classes(Xfinaltest)


# In[ ]:


resultdf = pd.DataFrame({'PassengerId': testdf['PassengerId'], 'Survived': Ypred})
display(resultdf.head())
resultdf.to_csv('my_submission2.csv', index=False)
print("Your submission was successfully saved!")


# <h3>This model made predictions with accuracy of 76.8%</h3>

# In[ ]:




