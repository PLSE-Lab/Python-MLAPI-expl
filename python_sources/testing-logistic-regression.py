import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

############# HELPER METHODS ##################
def preprocess(train_data, includes_labels=True):
    # fill in missing data
    #it's 3 because the only problematic missing value is "Embarcked"
    #and the most common value in the training set is S (3)
    train_data.fillna(3, inplace=True)
    
    # prep for one hot encoding
    # Sex
    train_data.replace("male", 1, inplace=True)
    train_data.replace("female", 2, inplace=True)
    
    # Embarked port
    train_data.replace("C", 1, inplace=True)
    train_data.replace("Q", 2, inplace=True)
    train_data.replace("S", 3, inplace=True)
    
    ########## TO DELETE, CHECK FEATURES!!!
    #sex = []
    #port = []
    #port_count = [0, 0, 0, 0]
    #for i, row in train_data.iterrows():
    #    sex.append(row["Sex"])
    #    port.append(row["Embarked"])
    #    port_count[row["Embarked"]] += 1
    #print("SEX VALUES: ", set(sex))
    #print("PORT VALUES: ", set(port))
    #print("PORT COUNT: ", port_count)
    #########################
    
    # remove unimportant features
    train_data.drop("Name", 1, inplace=True)
    train_data.drop("Ticket", 1, inplace=True)
    train_data.drop("Cabin", 1, inplace=True)
    
    # separate features from labels
    if includes_labels:
        train_data.drop("PassengerId", 1, inplace=True)
        preprocessed_x = train_data.drop("Survived", 1)
        output = train_data["Survived"]
    else:
        preprocessed_x = train_data.drop("PassengerId", 1)
        output = train_data["PassengerId"]
    
    # one hot encode the training features
    enc = OneHotEncoder(categorical_features=[
        False, #Pclass
        True,  #Sex
        False, #Age
        False, #SibSp
        False, #Parch
        False, #Fare
        True  #Embarked
        ], handle_unknown='ignore')
    train_x = enc.fit_transform(preprocessed_x).toarray()

    return train_x, output

#Print you can execute arbitrary python code
train_data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_data = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
##print("\n\nTop of the training data:")
##print(train_data.head())

##print("\n\nSummary statistics of training data")
##print(train.describe())

#Any files you save will be available in the output tab below
##train.to_csv('copy_of_the_training_data.csv', index=False)

######### PREPROCESSING ########################
train_x, train_y = preprocess(train_data)

######### TRAINING ###########################
model = LogisticRegression()
model = model.fit(train_x, train_y)

######### VALIDATING #########################
print(model.score(train_x, train_y))

################ TEST ##################
test_x, ids = preprocess(test_data, includes_labels=False)
test_y = model.predict(test_x)
#ids = ids.astype(int)
#test_y = test_y.astype(int)
result = np.dstack((ids, test_y))[0]
print(result)
print(type(result))
#pd.DataFrame(result, columns=["PassengerId","Survived"]).to_csv("output.csv")
f = open("output.csv", 'w')
f.write("PassengerId,Survived\n")
for row in result:
    f.write(str(row[0]) + "," + str(row[1]) + "\n")