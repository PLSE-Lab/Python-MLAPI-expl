#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#loading data
train = pd.read_csv(("../input/train.csv"))
test = pd.read_csv(("../input/test.csv"))

#Exploring data
print("Exploring data")

print("#########train")
print(train.head())

print("#########test")
print(test.head())
########################################
print("shape")
print("#########train")
print(train.shape)

print("#########test")
print(test.shape)
########################################
print("info")
print("#########train")
print(train.info())

print("#########test")
print(test.info())
########################################
print("missing values")
print("#########train")
print(train.isnull().sum())

print("#########test")
print(test.isnull().sum())
########################################

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
fig=plt.figure(figsize=(18,6))

plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts(normalize= True).plot(kind="bar", alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived,df.Age, alpha=0.1)
plt.title("Age wrt Survived")

plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts(normalize= True).plot(kind="bar", alpha=0.5)
plt.title("Class ")

#ya3mallik graphe lil les 3 classes en fonction de leurs ages
plt.subplot2grid((2,3),(1,0) , colspan=2) #colspan ya3malik el graphe 3ala 2 colonnes
for x in [1,2,3] :
    df.Age[df.Pclass == x].plot(kind="kde")
plt.title("Class wrt Age")
plt.legend(("1rst", "2nd" , "3rd"))

plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts(normalize= True).plot(kind="bar", alpha=0.5)
plt.title("Embarked ")

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

female_color = 'pink'

df = pd.read_csv("train.csv")
fig=plt.figure(figsize=(18,6))

#alla salkouha
plt.subplot2grid((3,4),(0,0))
df.Survived.value_counts(normalize= True).plot(kind="bar", alpha=0.5)
plt.title("Survived")

#el rjel alla salkouha
plt.subplot2grid((3,4),(0,1))
df.Survived[df.Sex == "male"].value_counts(normalize= True).plot(kind="bar", alpha=0.5)
plt.title("Men Survived")

#el nsa alla salkouha
plt.subplot2grid((3,4),(0,2))
df.Survived[df.Sex == "female"].value_counts(normalize= True).plot(kind="bar", alpha=0.5 , color='pink')
plt.title("Women Survived ")

#el sex mta3 alla salkouha
plt.subplot2grid((3,4),(0,3))
df.Sex[df.Survived == 1].value_counts(normalize= True).plot(kind="bar", alpha=0.5 , color=['pink','blue'])
plt.title("Sex of Survived ")

#ya3mallik graphe lil les 3 classes en fonction de leurs ages
plt.subplot2grid((2,3),(1,0) , colspan=4) #colspan ya3malik el graphe 3ala 2 colonnes
for x in [1,2,3] :
    df.Survived[df.Pclass == x].plot(kind="kde")
plt.title("Class wrt Survived")
plt.legend(("1rst", "2nd" , "3rd"))

#el rjel el kroz alla salkouha
plt.subplot2grid((3,4),(2,0))
df.Survived[(df.Sex == "male")&(df.Pclass == 1)].value_counts(normalize= True).plot(kind="bar", alpha=0.5)
plt.title("Rich Men Survived")

#el rjel el mouch kroz alla salkouha
plt.subplot2grid((3,4),(2,1))
df.Survived[(df.Sex == "male")&(df.Pclass == 3)].value_counts(normalize= True).plot(kind="bar", alpha=0.5)
plt.title("Poor Men Survived")

#el nsa el kroz alla salkouha
plt.subplot2grid((3,4),(2,2))
df.Survived[(df.Sex == "female")&(df.Pclass == 1)].value_counts(normalize= True).plot(kind="bar", alpha=0.5, color='pink')
plt.title("Rich women Survived")

#el nsa el mouch kroz alla salkouha
plt.subplot2grid((3,4),(2,3))
df.Survived[(df.Sex == "female")&(df.Pclass == 3)].value_counts(normalize= True).plot(kind="bar", alpha=0.5, color='pink')
plt.title("Poor women Survived")


plt.show()
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#loading data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#In order to modify both files , we can combine them then once we finish we split them xD
#train_test_data = [train, test] # combining train and test dataset


print("#########train")
print(train.head())
# delete unnecessary feature from dataset :

##  NAME    # puisque 3andek his ID
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

##  Ticket    # Ca sert à rien
train.drop('Ticket', axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)

train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
train_test_data = [train, test]
#Transforming String values to numerics ones
##  Sex #
sex_ = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_)

#Fill NAN
##  Age #

train["Age"] = train["Age"].fillna(train["Age"].dropna().median())
test["Age"] = test["Age"].fillna(test["Age"].dropna().median())
##  Fare #

train["Fare"] = train["Fare"].fillna(train["Fare"].dropna().median())
test["Fare"] = test["Fare"].fillna(test["Fare"].dropna().median())

##  Cabin # Cabins start with a letter that refers its values and its class
#cabin_ = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
#for dataset in train_test_data:
    #dataset['Cabin'] = dataset['Cabin'].map(cabin_)
##  Embarked #
# #most of passengers embarked from S then we will fill the missing ones with S
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
#Transforming String values to numerics ones
##  Embarked #
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
#MODELLING

# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

#IN order to verify the results we gonna split train file into train_data and target

train_data= train.drop({"PassengerId",'Survived'}, axis=1)
target = train['Survived']
print(train_data.head(100))

## Cross Validation ##
### kNN ###
print("kNN")
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100, 2))

### Decision Tree ###
print("Decision Tree")
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100, 2))

### Ramdom Forest ###
print("Ramdom Forest")
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100, 2))

### Naive Bayes ###
print("Naive Bayes")
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100, 2))

### SVM ###
print("SVM ")
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100, 2))

#essai
clf = RandomForestClassifier(n_estimators=13)
clf.fit(train_data, target)
test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)

scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100, 2))
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": prediction})

submission.to_csv('submission9.csv', index=False)

submission = pd.read_csv('submission9.csv')