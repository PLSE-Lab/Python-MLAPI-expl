import pandas as pd
import keras
from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

y_train = train['label'].as_matrix()
X_train = train.drop('label', axis=1).as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10)

model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_dim=28*28, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
model.fit(X_train, to_categorical(y_train, 10), nb_epoch=5, batch_size=600)

score = model.evaluate(X_test, to_categorical(y_test, 10), batch_size=700)
print(score)

print(model.predict(X_test)[0])
print(y_test[0])