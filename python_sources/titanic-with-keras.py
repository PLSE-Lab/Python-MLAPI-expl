# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# preview the data
titanic_df.head()
test_df.head()

def fill_fare(data):
    #Fare is missing for passengerId 1044, which is a male, with no cabin, Embarked at S on Pclass 3
    computeFare=data.loc[(data["Embarked"]=='S')&(data["Age"]>50)&(data["Pclass"]==3)&(data["Sex"]=='male')&(data["Cabin"].isnull())]
    # i will use mean value to complete missing fare 
    #print(computeFare["Fare"].mean())
    data.loc[data["Fare"].isnull(),["Fare"]] = computeFare["Fare"].mean()
    
def fill_embarked(data):
    data.loc[data['Embarked'].isnull(),['Embarked']] = 'S'
    #Convert Embarked as int
    #data.loc[data["Embarked"]] = data['Embarked'].map({'C': 0, 'Q': 1, 'S' : 2}).astype(int) 
    data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S' : 2}).astype(int) 

def fill_title(data):
    data['Title'] = data.Name.copy().astype(str)
    data['Title'] = data['Title'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
   #this function should be called after fill_title 
def fill_age(data):
    ##### Replace missing ages, by random values compute on same title
    # for example all "Master" are young men under 13 years
    age_nan = data.loc[data["Age"].isnull()].sort_values(by='Title', ascending=True)
    age_notnan = data.loc[data["Age"].notnull()].sort_values(by='Title', ascending=True)
    title_list = ['Master','Dr','Miss','Mr','Mrs']
    for title in title_list:
        title_age = age_notnan.loc[age_notnan["Title"]==title]
        count_nan_age_title = len(age_nan[age_nan["Title"]==title])
        age_std=title_age["Age"].std()
        age_mean=title_age["Age"].mean()
        #if no std -> jus one value
        if math.isnan(age_std):
            random_age = age_mean
        else:
            random_age = np.random.randint(age_mean - age_std, age_mean + age_std, size = count_nan_age_title)

        #Replace random age values for master without age
        data.loc[((data['Age'].isnull()) & (data['Title']==title)),['Age']] = random_age

#Work on family, we will compute familly size by using Parch and SibSp
def family_group(data):
    data['Family_Size'] = data['Parch']+data['SibSp']+1
    data['Family_Group'] = data['Family_Size'].map( lambda s : 1 if s == 1 else 2 if s <4 else 3)
    data['Family_Group'] = data['Family_Group'].astype(int)

#Work on Cabin
def has_cabin(data):
    data['Has_Cabin']= data["Cabin"].apply(lambda x: 1 if type(x) == str else 0)
    
#Work on titles
def title_group(data):
    data['Title'] = data['Title'].replace(['Countess', 'Dona'],'Lady')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace(['Mme', 'Ms'], 'Mrs')
    data['Title'] = data['Title'].replace(['Capt', 'Col','Don', 'Major', 'Rev', 'Jonkheer', 'Sir'], 'Officer')
    data['Title_Group'] = data['Title'].map({"Lady": 1, "Master": 2, "Miss": 3, "Mr": 4, "Mrs": 5, "Officer": 6, "Dr":7})
    data['Title_Group'] = data['Title_Group'].astype(int)

#normalize columns
def normalized(data,columns):
    for column in columns:
        data[column] = (data[column]-data[column].mean())/(data[column].max()-data[column].min())
 
MyData = pd.concat([titanic_df,test_df])
fill_fare(MyData)
fill_embarked(MyData)
fill_title(MyData)
title_group(MyData)
fill_age(MyData)
has_cabin(MyData)
family_group(MyData)

drop_list = ['Ticket','Cabin','Name','Title','Sex','Fare','Parch','SibSp','Family_Size']
MyData = MyData.drop(drop_list, axis=1)
titanic= MyData.loc[MyData.Survived.notnull()].copy()
test= MyData.loc[MyData.Survived.isnull()].copy()

#split between titanic and a set of titanic to score results:   
titanic_data=titanic.drop(['Survived','PassengerId'], axis=1)
titanic_survived=titanic['Survived'].astype(int)
test_data = test.drop(['Survived','PassengerId'], axis=1)
normalized_list=['Age', 'Embarked', 'Pclass',  'Title_Group',  'Has_Cabin',  'Family_Group']

normalized(titanic_data,normalized_list)
normalized(test_data,normalized_list)

#create keras model
model = Sequential()
model.add(Dense(100, input_dim=titanic_data.shape[1]))
model.add(Activation('linear'))
model.add(Dense(32,input_dim=100))
model.add(Activation('relu'))
model.add(Dense(1,input_dim=32))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
batch_size = 1
nb_epoch = 20
score=[]

#loop use to compare different models
#for i in range(10):
#    X_train, X_test, y_train, y_test = train_test_split(titanic_data, titanic_survived, test_size=0.1)

#    model.fit(X_train.values, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=0)

#    scores = model.evaluate(X_test.values, y_test, verbose=0)
#    print("%s: %.2f%%" %(model.metrics_names[0], scores[0]*100))
#    print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
#    score.append(scores[1]*100)

#sc=0
#for i in range(10):
#    sc += score[i]
#print("RESULTAT : %.2f%%" %(sc/10))

#fit with all values for submission:
model.fit(titanic_data.values, titanic_survived, batch_size=batch_size, epochs=nb_epoch, verbose=1)
classes = model.predict_classes(test_data.values, batch_size=32)
df=pd.DataFrame(classes)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": df[0]
    })
submission.to_csv('titanic_keras.csv', index=False)

#create prediction

