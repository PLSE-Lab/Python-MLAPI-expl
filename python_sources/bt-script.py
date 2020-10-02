import numpy as np
import csv
import keras
import gzip


def read_train_data():
    i = 0
    with gzip.open('../input/train.csv.gz', 'rt', newline='') as csv_file:
        reader = csv.reader(csv_file)
        row_count = 500000
        X = np.zeros((row_count, 5))
        Y = np.zeros((row_count, 1))
        for row in reader:
            if row[0] == 'id':
                continue
            X[i] = row[1:6]
            Y[i] = row[6]
            i = i+1
    return X, Y


def write_data(result):
    with open('my_photoz_test.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['id', 'redshift'])
        for i in range(len(result)):
            writer.writerow([i, result[i][0]])


def read_test_data():
    i = 0
    with gzip.open('../input/test.csv.gz', 'rt', newline='') as csv_file:
        reader = csv.reader(csv_file)
        row_count = 456227
        X = np.zeros((row_count, 5))
        for row in reader:
            if row[0] == 'id':
                continue
            X[i] = row[1:6]
            i = i + 1
    return X


(x, y) = read_train_data()
z = read_test_data()

solver = keras.models.Sequential()

solver.add(keras.layers.Dense(units=128, activation='sigmoid', input_dim=5))
solver.add(keras.layers.Dropout(0.05))
solver.add(keras.layers.Dense(units=8, activation='sigmoid'))
solver.add(keras.layers.Dense(units=1, activation='linear'))

solver.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

solver.fit(x, y, epochs=500, batch_size=1000, validation_split=0.2)

result = solver.predict(z, batch_size=100)
write_data(result)