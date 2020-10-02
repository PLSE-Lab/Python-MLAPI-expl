import numpy as np
import pandas as pd
import csv as csv
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv",header =0)
test = pd.read_csv("../input/test.csv",header= 0 )


########## data pre processing train data 
train['Gender']=train['Sex'].map({'female':0,'male':1}).astype(int)



if len(train.Embarked[ train.Embarked.isnull() ]) > 0:
    train.Embarked[ train.Embarked.isnull() ] = train.Embarked.dropna().mode().values
    


Ports = list(enumerate(np.unique(train['Embarked'])))    # determine all values of Embarked,
print(Ports)
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train.Embarked = train.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int


median_age = train['Age'].dropna().median()
if len(train.Age[ train.Age.isnull() ]) > 0:
    train.Age[train.Age.isnull()] = median_age
    
    
train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 



###### data pre processing test data

test['Gender']=test['Sex'].map({'female':0,'male':1}).astype(int)



if len(test.Embarked[ test.Embarked.isnull() ]) > 0:
    test.Embarked[ test.Embarked.isnull() ] = test.Embarked.dropna().mode().values
    


Ports_test = list(enumerate(np.unique(test['Embarked'])))    # determine all values of Embarked,
print(Ports_test)
Ports_dict_test = { name : i for i, name in Ports_test }              # set up a dictionary in the form  Ports : index
test.Embarked = test.Embarked.map( lambda x: Ports_dict_test[x]).astype(int)     # Convert all Embark strings to int


median_age = test['Age'].dropna().median()
if len(test.Age[ test.Age.isnull() ]) > 0:
    test.Age[test.Age.isnull()] = median_age
    
# All the missing Fares -> assume median of their respective class
if len(test.Fare[ test.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test[ test.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test.loc[ (test.Fare.isnull()) & (test.Pclass == f+1 ), 'Fare'] = median_fare[f]    
    
ids = test['PassengerId'].values    
test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 



# Convert back to a numpy array
train_data = train.values
test_data = test.values


print ('Training...')
forest = RandomForestClassifier(max_features= 'sqrt', n_estimators=1350)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print ('Predicting...')
output = forest.predict(test_data).astype(int)

print (output)

predictions_file = open("prediction.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

