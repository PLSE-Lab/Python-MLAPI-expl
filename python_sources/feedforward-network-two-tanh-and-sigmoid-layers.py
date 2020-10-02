import numpy as np
import pandas as pd
import string
# scikit imports
from sklearn.metrics import accuracy_score
# keras imports
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *
from keras.utils import to_categorical

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

EMBARKED = lambda x: [np.nan, 'C', 'Q', 'S'].index(x)

def to_int(x):
    """transform single cabin number to index"""
    l = string.ascii_letters.upper().index(x[0])
    if x[1:]:
        return l + int(x[1:])
    else:
        return l


def cabin_to_index(x):
    """transform cabin numbers to index"""
    if type(x) == float:
        return 0
    elif " " in x:
        return np.average([to_int(v) for v in x.split()])
    else:
        return to_int(x)


def load_transform(dataframe, y=False):
    """load and transform CSV data"""
    dataframe['Embarked'] = dataframe['Embarked'].map(EMBARKED)
    dataframe['Cabin'] = dataframe['Cabin'].map(cabin_to_index)
    dataframe['Sex'] = dataframe['Sex'].map(lambda x: 0 if x == "male" else 1)
    dataframe['Age'] = dataframe['Age'].fillna(0)
    data = np.array([dataframe['Embarked'], dataframe['Pclass'], dataframe['Sex'], dataframe['Cabin'], dataframe['Age'], dataframe['Parch']]).T
    if y:
        Y = np.array(dataframe['Survived'])
        return (data, Y)
    return (dataframe, data)


def from_categorical(y):
    Y_new = np.zeros(shape=y.shape[0], dtype=np.int8)
    for i in range(len(y)):
        if y[i][0] < y[i][1]:
            Y_new[i] = 1
    return Y_new
 
# load train data   
X, Y = load_transform(train, y=True)

# Create and train model
model = Sequential([
    Dense(110, activation="tanh", input_dim=X.shape[1]),
    Dense(103, activation="tanh"),
    Dense(36, activation="sigmoid"),
    Dense(2, activation="softmax")
])
model.compile(RMSprop(0.001), 'categorical_crossentropy', metrics=['accuracy'])
model.fit(X, to_categorical(Y), epochs=120)

#load test data
test, data = load_transform(test, y=False)
pred = from_categorical(model.predict(data))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
})
submission.to_csv("predicted.csv", index=False)