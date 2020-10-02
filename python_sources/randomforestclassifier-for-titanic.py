import matplotlib.pyplot as plt
import numpy as np #linear algebra
import pandas as pd #data processing, CSV file
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train_data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_data = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

all_data = pd.concat([train_data, test_data]) 
train_data['FamilySize'] = pd.Series(1, index=train_data.index)
test_data['FamilySize'] = pd.Series(1, index=test_data.index)

#----------------------Train Median Age Calculation--------------------------
median_age_sum=[0 for x in range(4)]
median_age_count=[0 for x in range(4)]
median_age=[0 for x in range(4)]

for i in range (0, len(train_data.Age)):
    if ((pd.notnull(train_data.Age[i])) and (train_data.Age[i]!=0)):
        if ("Master" in train_data.Name[i]):
            median_age_count[0]+=1
            median_age_sum[0]+=train_data.Age[i]
        elif ("Mrs" in train_data.Name[i]) or ("Countess" in train_data.Name[i]) or ("Jonkheer" in train_data.Name[i]):
            median_age_count[1]+=1
            median_age_sum[1]+=train_data.Age[i]
        elif ("Miss." in train_data.Name[i]):
            median_age_count[2]+=1
            median_age_sum[2]+=train_data.Age[i]
        else:
            median_age_count[3]+=1
            median_age_sum[3]+=train_data.Age[i]
    train_data.FamilySize[i]=1+train_data.SibSp[i]+train_data.Parch[i]

pd.pivot_table(train_data,index=["FamilySize"],values=["Survived"]).plot(kind="bar", stacked=True)
plt.show()

for x in range(4):
    median_age[x]=median_age_sum[x]/median_age_count[x]
print (median_age)
for i in range (len(train_data.Age)):
    if (pd.isnull(train_data.Age[i])) or (train_data.Age[i]==0):
        if ("Master" in train_data.Name[i]):
            train_data.Age[i]=median_age[0]
        elif ("Mrs" in train_data.Name[i]) or ("Countess" in train_data.Name[i]) or ("Jonkheer" in train_data.Name[i]):
            train_data.Age[i]=median_age[1]
        elif ("Miss" in train_data.Name[i]):
            train_data.Age[i]=median_age[2]
        else:
            train_data.Age[i]=median_age[3]
#----------------------/Train Median Age Calculation--------------------------

data = train_data.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked', 'SibSp', 'Parch'],axis=1)
label = LabelEncoder()
dicts = {}
label.fit(data.Sex.drop_duplicates()) #setlist of values for coding
dicts['Sex'] = list (label.classes_)
data.Sex = label.transform(data.Sex) #replace values from list to codes of elements

#test set processing 
median_fare_sum=[0 for x in range(3)]
median_fare_count=[0 for x in range(3)]
median_fare=[0 for x in range(3)]

#-------Train Median Fare Calculation-----------
for i in range (len(train_data.Fare)):
    if ((pd.notnull(train_data.Fare[i])) and (train_data.Fare[i]!=0)):
        if (train_data.Pclass[i]==1):
            median_fare_count[0]+=1
            median_fare_sum[0]+=train_data.Fare[i]
        elif (train_data.Pclass[i]==2):
            median_fare_count[1]+=1
            median_fare_sum[1]+=train_data.Fare[i]
        else:
            median_fare_count[2]+=1
            median_fare_sum[2]+=train_data.Fare[i]
for x in range(3):
    median_fare[x]=median_fare_sum[x]/median_fare_count[x]

for i in range (len(train_data.Fare)):
    if (pd.isnull(train_data.Fare[i])) or (train_data.Fare[i]==0):
        if (train_data.Pclass[i]==1):
           train_data.Fare[i]=median_fare[0]
        elif (train_data.Pclass[i]==2):
           train_data.Fare[i]=median_fare[1]
        else:
           train_data.Fare[i]=median_fare[2]
#-------/Train Median Fare Calculation-----------

#-------/Test Median Age Calculation-----------
for i in range (len(test_data.Age)):
    if (pd.isnull(test_data.Age[i])) or (test_data.Age[i]==0):
        if ("Master" in test_data.Name[i]):
            test_data.Age[i]=median_age[0]
        elif ("Mrs" in test_data.Name[i]) or ("Countess" in test_data.Name[i]) or ("Jonkheer" in test_data.Name[i]):
            test_data.Age[i]=median_age[1]
        elif ("Miss" in train_data.Name[i]):
            test_data.Age[i]=median_age[2]
        else:
            test_data.Age[i]=median_age[3]
    test_data.FamilySize[i]=1+test_data.SibSp[i]+test_data.Parch[i]
    
#-------/Test Median Age Calculation-----------

for i in range (0, len(test_data.Fare)):
    if (pd.isnull(test_data.Fare[i])) or (test_data.Fare[i]==0):
        if (test_data.Pclass[i]==1):
           test_data.Fare[i]=median_fare[0]
        elif (test_data.Pclass[i]==2):
           test_data.Fare[i]=median_fare[1]
        else:
           test_data.Fare[i]=median_fare[2]

result = pd.DataFrame(test_data.PassengerId)
test = test_data.drop(['Name','Ticket','Cabin','PassengerId', 'Embarked', 'SibSp', 'Parch'],axis=1)
label.fit(dicts['Sex'])
test.Sex = label.transform(test.Sex)

target = data.Survived
train = data.drop(['Survived'], axis=1) #drop Survived flag and ID from data

#model_rfc = RandomForestClassifier(n_estimators = 95, max_features='auto', criterion='gini',max_depth=5)
model_rfc = RandomForestClassifier(n_estimators=350, min_samples_split=6, min_samples_leaf=2)
model_rfc.fit(train, target)
result.insert(1,'Survived', model_rfc.predict(test))

#Any files you save will be available in the output tab below
result.to_csv('copy_of_the_training_data.csv', index=False)