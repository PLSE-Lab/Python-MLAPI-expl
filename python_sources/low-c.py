import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

def createInput(dataTable):
    # num of examples m
    m = dataTable["PassengerId"].size
    
    # get pclass
    pclass = dataTable["Pclass"].values
    # get sex
    dataTable["Sex"] = (dataTable["Sex"] == "female").astype(int)
    sex = dataTable["Sex"].values
    # get age
    age = dataTable["Age"].values
    # set unknown age to median
    age[pd.isnull(dataTable["Age"]).values] = dataTable.Age.median()
    
    # get sib
    sib = dataTable["SibSp"].values
    # get parch
    parch = dataTable["Parch"].values
    # get fare
    fare = dataTable["Fare"].values
    fare[pd.isnull(dataTable["Fare"]).values] = dataTable.Fare.median()
    # get embarked
    embarked = np.zeros(m)
    #embarked[(dataTable["Embarked"] == "S").values] = 0 # is already 0
    embarked[(dataTable["Embarked"] == "Q").values] = 1
    embarked[(dataTable["Embarked"] == "C").values] = 2
    #print("Embarked shape: " + str(embarked.shape))
    
    x_zero = np.ones(m)
    #print("X_Zero shape: " + str(x_zero.shape))
    
    X = np.transpose(np.concatenate((x_zero, pclass, sex, age, sib, parch, fare, embarked)).reshape(8,m))
    print("X total shape: " + str(X.shape))
    
    return X


print("Creating training X ...")
trainX = createInput(train)
#print(trainX.shape)

trainy = train["Survived"].values
#print(trainy.shape)

print("Creating testing X ...")
testX = createInput(test)
#print(testX.shape)


###########################################################################################


### MACHINE LEARNING FUNCTIONALITY ###
# parameters for model
my_C = 10

# create model
logreg = linear_model.LogisticRegression(C = my_C)

# fit model
logreg.fit(trainX, trainy)

# testing model
prediction = logreg.predict(testX)
print(prediction)


#########################################################################################################


### CREATE SOLUTION CSV FILE ###

passengerId = test["PassengerId"].values

solution = pd.DataFrame(prediction, passengerId, columns = ["Survived"])

solution.to_csv("log_reg_solution1.csv", index_label = ["PassengerId"])

# print(solution)

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)
