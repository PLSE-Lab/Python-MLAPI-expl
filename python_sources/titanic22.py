import numpy as np
import pandas as pd
import seaborn as sea

#Print you can execute arbitrary python code
train_data = pd.read_csv("../input/train.csv")
test_data  = pd.read_csv("../input/test.csv")

print(train_data.info()) 

train_data = train_data.drop(['PassengerId','Name','Ticket'],axis=1)
train_data['Embarked'] = train_data['Embarked'].fillna('S')
sea.factorplot('Embarked','Survived', data=train_data,size=4,aspect=3)








