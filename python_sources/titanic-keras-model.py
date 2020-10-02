import os.path
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, Dropout
from keras.models import Model, load_model

base_path = '/kaggle/input/titanic/'
model_file = 'mod.h5'
trainset = base_path + 'train.csv'

def read_dataset(filename):
    df = pd.read_csv(filename, low_memory=False)
    msk = np.random.rand(len(df)) < 0.85
    dev = df[~msk]
    train = df[msk]

    X_train = process_X(train)
    Y_train = train['Survived'].values

    X_dev = process_X(dev)
    Y_dev = dev['Survived'].values

    return X_train, Y_train, X_dev, Y_dev


def process_X(df):
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    X['Sex'].replace('female', 0, inplace=True)
    X['Sex'].replace('male', 1, inplace=True)
    X['Embarked'].replace('S', 0, inplace=True)
    X['Embarked'].replace('C', 1, inplace=True)
    X['Embarked'].replace('Q', 2, inplace=True)

    X['Embarked'].fillna(-99, inplace=True)
    X['Age'].fillna(-99, inplace=True)
    X = (X - X.mean()) / X.std()
    return X


def gen_model(input_shape):
    # Train set accuracy =  0.8726591467857361
    # Dev set accuracy =  0.8888888955116272

    X_input = Input(shape=input_shape)
    X = Dense(30, activation='relu')(X_input)

    X = Dropout(0.5)(X)
    X = Dense(15, activation='relu')(X)
    X = Dense(10, activation='relu')(X)
    X = Dense(5, activation='relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(5, activation='relu')(X)
    X = Dense(5, activation='relu')(X)
    X = Dense(1, activation='sigmoid')(X)

    m = Model(inputs=X_input, outputs=X)
    m.summary()
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    return m


def generateSubmission(m):
    df = pd.read_csv(base_path + 'test.csv', low_memory=False)
    X_test = process_X(df)
    Y = m.predict(X_test)
    ids = range(892, 1310)

    submission = open("submission.csv", "w+")
    submission.write("PassengerId,Survived\n")

    for i in range(0, len(ids)):
        id = ids[i]
        if Y[i] > 0.5:
            survived = 1
        else:
            survived = 0

        line = str(id) + ',' + str(survived) + '\n'
        submission.write(line)


X_train, Y_train, X_dev, Y_dev = read_dataset(trainset)

model = None
if os.path.exists(model_file):
    print("Going to load model")
    model = load_model(model_file)
else:
    print("Going to generate a new model")
    model = gen_model(input_shape=(X_train.shape[1],))
    model.fit(X_train, Y_train, batch_size=64, epochs=100)
    model.save(model_file)

_, dev_acc = model.evaluate(X_dev, Y_dev)
_, train_acc = model.evaluate(X_train, Y_train)

print("Train set accuracy = ", train_acc)
print("Dev set accuracy = ", dev_acc)

generateSubmission(model)
