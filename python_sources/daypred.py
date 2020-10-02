import numpy as np
import os
from sklearn.utils.extmath import weighted_mode

data = []
with open('../input/train.csv', 'r') as train:
    train.readline()
    for line in train:
        line = line.strip()
        days = list(map(int, line.split(',')[1].strip().split()))
        v = []
        for x in days:
            v.append((x-1)%7)
        data.append(v)

weights = []
for v in data:
    w = []
    weight = 1.0
    for i, x in enumerate(v):
        weight += 0.67
        w.append(weight)
    weights.append(w)

with open('solution.csv', 'w') as f:
    print('id,nextvisit', file=f)
    for i in range(300000):
        day, tmp = weighted_mode(data[i], weights[i])
        print('%d, %d' % (i+1, day+1), file=f)