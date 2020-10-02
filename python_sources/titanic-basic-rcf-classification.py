import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RCF
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
factors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
names = train.Name
y = train.Survived

for i in train.columns:
    if i not in factors:
        #Remove unrequired columns
        train = train.drop(i, 1)

#RCF can't take string data so have to change gender data accordingly
train['Male'] = [int(i) for i in train['Sex']=='male']
train['Female'] = [int(i) for i in train['Sex']=='female']
train = train.drop('Sex', 1)

#A lot of age data is NaN, so let's fix it by matching with appropriate abbreviations
abbrv = ['Mr', 'Mrs', 'Miss', 'Master']
mean = []
count = []

#There is only one name outside of this abbrv list which has NaN age
for i in range(891):
    if train.Age.isnull()[i]:
        name = names[i]
        if not (('Mr' in name) or ('Mrs' in name) or ('Master' in name) or ('Miss' in name)):
            print(name + " " + str(i))
#Output -> Brewe, Dr. Arthur Jackson 766
#His age is 45 from online resources, can be taken as 47 also otherwise, compared with the other doctor
train['Age'][766] = 45
train['Age'][766] = 45 #Just in case any error happens with pandas

null_age = train['Age'].isnull()

for i in range(4):
    mean.append(0)
    count.append(0)
    for j in range(891):
        if (abbrv[i] in names[j]) and (not null_age[j]):
            mean[-1] += train.Age[j]
            count[-1] += 1
    mean[-1] /= count[-1]

#Now we have ages, lets put them back
for i in range(891):
    if null_age[i]:
        for j in range(4):
            if abbrv[j] in names[i]:
                train['Age'][i] = mean[j]

#train data all ready, lets fit to model
model = RCF(n_estimators = 100, bootstrap = True, max_features = None)
model.fit(train, y)
#model ready

test = pd.read_csv('test.csv')
namez = test['Names']
#Test dataset has problem with one guy's fare, as well as ages
for i in test.columns:
    if i not in factors:
        #Remove unrequired columns
        test = test.drop(i, 1)

test['Male'] = [int(i) for i in test['Sex']=='male']
test['Female'] = [int(i) for i in test['Sex']=='female']
test = test.drop('Sex', 1)
null_age = test.Age.isnull()
for i in range(len(test)):
    if null_age[i]:
        #One name has Ms instead of Miss
        if 'Ms' in namez[i]:
            test['Age'][i] = mean[2]
            continue
        for j in range(4):
            if abbrv[j] in namez[i]:
                test['Age'][i] = mean[j]

for i in range(len(test)):
    if test.Fare.isnull()[i]:
        test.Fare[i] = 8 #Pclass = 3 approx average for that one guy because he's in Pclass = 3

results = model.predict(test)
df = pd.DataFrame()
df['PassengerId'] = pd.read_csv('test.csv')['PassengerId']
df['Survived'] = results
df.to_csv('predictions.csv', index = False)