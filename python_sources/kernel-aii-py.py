import pandas as pd
import numpy as np
from sklearn.utils.extmath import weighted_mode
train = pd.read_csv('../input/train.csv')
nextvisit = []
for visits in train['visits']:
    visits = [(int(day) - 1) % 7 for day in visits.split()]
    weights = np.logspace(1.0, 10.0, num=len(visits), base=2.0)
    a, b = weighted_mode(visits, weights)
    nextvisit.append(int(a[0])+1)
with open('solution.csv', 'wt') as file:
    print('id,nextvisit', file=file)
    for i, x in enumerate(nextvisit):
        print(str(i+1) + ', ' + str(x), file=file)
