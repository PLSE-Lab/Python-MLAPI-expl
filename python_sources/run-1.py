import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

in_file = '../input/data.csv'
data = pd.read_csv(in_file, header=None)
data_arr = np.array(data)
data_arr = data_arr[1:, :]
num_of_features = len(data_arr[1, :])
X = data_arr[:, 2:num_of_features]
Y = data_arr[:, 1]
# Encoding the ground truth
Y = LabelEncoder().fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

model = Sequential()
model.add(Dense(30, input_dim=30, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test,y_test), nb_epoch=700, batch_size=35)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))