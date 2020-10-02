import numpy as np
from sklearn.ensemble import RandomForestRegressor

x_train = []
y_train = []
x_test = []

with open('../input/data.txt') as data_file:
    t = int(next(data_file))
    n, m = tuple(map(int, next(data_file).split()))
    for line_num, line in enumerate(data_file):
        hse = line_num // n # hours since epoch
        hod = hse % 24 # hour of day
        row = line_num % n
        for col, demand in enumerate(map(int, line.split())):
            features = np.array((hse, row, col, hod), dtype=np.int32)
            if demand == -1:
                x_test.append(features)
            else:
                x_train.append(features)
                y_train.append(demand)

x_train = np.array(x_train, dtype=np.int32)
y_train = np.array(y_train, dtype=np.int32)
x_test = np.array(x_test, dtype=np.int32)

regr = RandomForestRegressor(n_estimators=10000, n_jobs=-1)
regr.fit(x_train, y_train)
y_test_pred = regr.predict(x_test)

with open('result.csv', 'w') as result_file:
    result_file.write('id,demand\n')
    for x, y in zip(x_test, y_test_pred):
        result_file.write('{}:{}:{},{}\n'.format(*x[:3], y))
