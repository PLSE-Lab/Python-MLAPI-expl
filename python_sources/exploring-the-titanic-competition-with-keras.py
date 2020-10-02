import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
train = train.replace(['male', 'female'], [0, 1])
train = train.replace(['C', 'Q', 'S'], [0, 1, 2])
train = train.fillna(0)
train = (train-train.min())/(train.max()-train.min())
train.to_csv('cleaned_training_data.csv', index=False)

ids = test[['PassengerId']]
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
test = test.replace(['male', 'female'], [0, 1])
test = test.replace(['C', 'Q', 'S'], [0, 1, 2])
test = test.fillna(0)
test = (test-test.min())/(test.max()-test.min())
test.to_csv('cleaned_test_data.csv', index=False)

model = Sequential()
model.add(Dense(output_dim=100, input_dim=7))
model.add(Activation("tanh"))
model.add(Dense(output_dim=100))
model.add(Activation("tanh"))
model.add(Dense(output_dim=100))
model.add(Activation("tanh"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

train_ys = train[['Survived']]
train_ys['Killed'] = 1 - train[['Survived']]
model.fit(train.drop(['Survived'], axis = 1).as_matrix(), train_ys.as_matrix(), nb_epoch=500, batch_size=32)

predictions = model.predict_classes(test.as_matrix(), batch_size=32)
print(predictions)
predictions = pd.DataFrame(1 - predictions, columns = ['Survived'])
#predictions = predictions.round().apply(np.int64)
predictions['PassengerId'] = ids
print(predictions.head())
predictions.to_csv('predictions.csv', index=False)
