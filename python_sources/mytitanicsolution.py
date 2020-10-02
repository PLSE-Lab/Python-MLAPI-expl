import numpy as np
import pandas as pd
from sklearn import linear_model


#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())


#train.to_csv('copy_of_the_training_data.csv', index=False)

print('After removing the non numerical value')
train['Age'] = train['Age'].fillna(train['Age'].median())

print(train.describe())
#print(train)


new_col = ['Survived']
collist = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

new_train = train[collist]
pass_sur = train[new_col]
print(new_train.describe())
print(new_train.head(3))



new_train.loc[new_train["Sex"] == "male", "Sex"] = 0
new_train.loc[new_train["Sex"]=="female","Sex"]=1

print("After cleaning the data .....................")
print(new_train["Sex"].unique())
print(new_train.head(3))


print("After converting the Embarked Data **********")
new_train.loc[new_train["Embarked"] == "S", "Embarked"] = 0
new_train.loc[new_train["Embarked"] == "C", "Embarked"] = 1

new_train.loc[new_train["Embarked"] == "Q", "Embarked"] = 2
print(new_train.head(3))


reg = linear_model.LinearRegression()

reg.fit(new_train,pass_sur)

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

