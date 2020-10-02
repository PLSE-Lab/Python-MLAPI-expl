import pandas as pd
import numpy as np
from sklearn.utils.extmath import weighted_mode

def to_dow(s):
    days = np.array(s.strip().split(' '), dtype = np.int)
    week_days = np.apply_along_axis(lambda d: (d-1)%7 + 1, 0, days)
    return week_days

# modeling weighted mode
def prediction(weights, dows_ex):
    n = len(dows_ex)
    lw = weights[:n]
    val,_ = weighted_mode(dows_ex, lw)
    return int(val[0])

path = '../input/train.csv'
train = pd.read_csv(path, converters = {'id' : int, 'visits' : str})

visits = list(train.get('visits'))
dows = list(map(lambda v: to_dow(v), visits))
weights = np.arange(1.0,50.0,0.1)

ids = train.get('id')
preds = np.array(list(map(lambda v: prediction(weights,v), dows)))

with open('./sol.csv', 'w') as wf:
    wf.write('id,nextvisit\n')
    for p_id, p_pred in zip(ids,preds):
        wf.write('{}, {}\n'.format(p_id, p_pred))
