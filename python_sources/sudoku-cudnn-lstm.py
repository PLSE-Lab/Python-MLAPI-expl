

# Edit: Marius Trollmann

# CSV handling
import csv

# Array handling
import numpy as np

# Machine learning framework: Keras

from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Dense, CuDNNLSTM, Input, Flatten
from tensorflow.python.keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers.merge import concatenate

# Read in data

quest_train = []

label_train = []

quest_test = []

label_test = []

with open("../input/test", "r") as dataset:
    reader = csv.reader(dataset, delimiter=",")

    for i, line in enumerate(reader):
        label_test.append(np.array(list(line[0])))

        quest_test.append(np.array(list(line[1])))

with open("../input/split_sudokus", "r") as dataset:
    reader = csv.reader(dataset, delimiter=",")

    for i, line in enumerate(reader):
        label_train.append(np.array(list(line[0])))

        quest_train.append(np.array(list(line[1])))
        
quest_train = np.array(quest_train, dtype="float32")

label_train = np.array(label_train, dtype="float32")

quest_test = np.array(quest_test, dtype="float32")

label_test = np.array(label_test, dtype="float32")

print("Loading accomplished!")

########################################################################################################

# Build the LSTM Model

# Initialize keras model

#model = Sequential()
# Add specific layers
#model.add(CuDNNLSTM((150), input_shape=(81, 10), return_sequences=True))
#model.add(Flatten())

LSTM_1 = CuDNNLSTM((75), input_shape=(81, 10), return_sequences=True)
LSTM_2 = CuDNNLSTM((75), input_shape=(81, 10), return_sequences=True)
LSTM = concatenate([LSTM_1, LSTM_2], axis=-1)
LSTM = Flatten()(LSTM)
solver = [Dense(9, activation="softmax")(LSTM) for i in range(81)]



# define one Dense layer for each of the digits we want to predict (adopted from kaggle)
#grid = Input(shape=(81, 10))
#features = model(grid)
#digit_placeholders = [Dense(9, activation="softmax")(features) for i in range(81)]

#Stack the two models together

#solver = Model(grid, digit_placeholders)

solver.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

#epoch = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10]
#epoch = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7]
epoch = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8]
#Define the parameters for the fitting iterations

old_train = 0
new_train = 140000
old_test = 0
new_test = 10000

for i in range(50):
    
    #One hot encoding
    
    d1 = to_categorical(quest_train[old_train:new_train]).astype("float32")
    d2 = to_categorical(label_train[old_train:new_train]-1).astype("float32")
    d3 = to_categorical(quest_test[old_test:new_test]-1).astype("float32")
    d4 = to_categorical(label_test[old_test:new_test]).astype("float32")
    
    solver.fit(
        
        d1,
        
        [d2[:, j, :] for j in range(81)],

        batch_size=200, 
        
        epochs=int(epoch[i]),

        validation_data=(d4, [d3[:, j, :] for j in range(81)]),
        
        verbose = 2

    )

    #Update the iteration parameters

    old_train = new_train
    new_train += 140000
    old_test = new_test
    new_test += 10000

# serialize model to JSON
model_json = solver.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
solver.save_weights("Sudoku_Solver.h5")
print("Saved model to disk")

























































