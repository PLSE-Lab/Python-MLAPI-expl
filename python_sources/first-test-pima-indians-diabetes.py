import keras as k
from keras.layers import Dense
import pandas
import numpy as np



data = pandas.read_csv("../input/diabetes.csv")

X = data.drop([], axis=1)
y = data["Outcome"]

optimizer = k.optimizers.Adam()
model = k.models.Sequential()

model.add(Dense(64, input_dim=len(X.columns), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
