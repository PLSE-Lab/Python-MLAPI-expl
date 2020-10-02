import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd

# Input data files are available in the "../input/" directory.
train_data = pd.read_csv('../input/train.csv')
X_train, y_train = train_data.drop(['label'], axis=1) / 255,  train_data['label']

X_test =  pd.read_csv('../input/test.csv') / 255

model = keras.Sequential([
    keras.layers.Dense(units=784, input_shape=(784,), activation="relu"),
    keras.layers.Dense(units=392, activation="relu"),
    keras.layers.Dense(units=10, activation="softmax"),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

predictions = model.predict_classes(X_test)
pd.DataFrame({
    "ImageId": list(range(1,len(predictions)+1)), 
    "Label": predictions
}).to_csv("submission.csv", index=False, header=True)
