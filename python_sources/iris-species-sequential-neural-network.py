from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import sys

# Fix random seed for reproducibility
np.random.seed(7)

#
# 1. Load Iris species dataset
#

# Split into input (X) and output (Y) variables
dataframe = pd.read_csv("../input/Iris.csv")

#
# 2. Feature engineering
#

# Deletes first column "Id"
dataframe = dataframe.drop("Id", 1)

# Convert Species to numeric id
dataframe["Species"] = dataframe["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).astype(int)

# Converts data frame to np matrix
dataset = dataframe.as_matrix()

#
# 3. Preparing Cross Validation
#

training_idx = np.random.randint(dataset.shape[0], size=105)
test_idx = np.random.randint(dataset.shape[0], size=45)

X_train = dataset[training_idx, :4]
Y_train = to_categorical(dataset[training_idx, 4], num_classes=3)

X_test = dataset[test_idx, :4]
Y_test = to_categorical(dataset[test_idx, 4], num_classes=3)

#
# 4. Define Model
#

model = Sequential()  # Creating Sequential Model
model.add(Dense(32, input_dim=4, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(3, activation="sigmoid"))

#
# 5. Compile Model
#

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

#
# 6. Fit Model
#

model.fit(
    X_train,
    Y_train,
    epochs=105,
    batch_size=10
)

#
# 7. Evaluate Model
#

scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#
# 8. Saving predictions
#

predictions = pd.DataFrame({
    "Ids": range(1, 151),
    "Species": [p.argmax() for p in model.predict(dataset[:, :4])]
})

predictions["Species"] = dataframe["Species"].map({
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}).astype(str)

predictions.to_csv("predictions.csv", index=False)
