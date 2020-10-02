import numpy as np
import pandas as pd
import seaborn as sns

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

train_df = pd.read_csv("../input/train.csv", header=0)
print(train_df.info())
train_df['Gender'] = train_df['Sex'].map({'female':0, 'male':1})

print(np.zeros(1))

print(train_df['Embarked'][train_df.Embarked.isnull()])
print(train_df.Embarked.dropna().mode().values)
if len(train_df.Embarked[train_df.Embarked.isnull()]) > 0:
    train_df.Embarked[train_df.Embarked.isnull()] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked']))) #list of tuples
print(Ports)

Ports_dict = { name : i for i, name in Ports }
print(Ports_dict)

train_df['Embarked'] = train_df['Embarked'].map(lambda x: Ports_dict[x]).astype(int)

median_age = train_df.Age.dropna().median()
print(median_age)

if len(train_df.Age[train_df.Age.isnull()]) > 0:
    train_df.Age[train_df.Age.isnull()] = median_age

train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

print(train_df.info())
print(train_df.head(10))
print(train_df.describe())
# Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

# print("\n\nSummary statistics of training data")
print(train.describe())
test = pd.DataFrame(test, columns=["PassengerId","Survived"])
#Any files you save will be available in the output tab below
test.to_csv('copy_of_the_training_data.csv', index=False)