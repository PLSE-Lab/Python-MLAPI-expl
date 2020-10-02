# -*- coding: utf-8 -*-
## Kernel submitted on April 18th
import pandas as pd
import os
import numpy as np
from sklearn.utils.extmath import weighted_mode

def predict(visits):
    visits = [int(s) for s in visits.split(' ') if s != '']
    visits_by_day = [0 for _ in range(7)]
    for v in visits:
        day = (v - 1) % 7
        visits_by_day[day] += 1
    l = len(visits)
    probs = [v / l for v in visits_by_day]

    answer = [0 for _ in range(7)]
    for i in range(7):
        result = 1.0
        for v in probs[:i]:
            result *= (1 - v)
        result *= probs[i]
        answer[i] = result

    return np.array(answer).argmax() + 1

data = pd.read_csv('../input/train.csv')

answer = pd.DataFrame()
answer['id'] = data['id']
answer['nextvisit'] = data['visits'].apply(predict)
answer['nextvisit'] = answer['nextvisit'].apply(lambda v: ' {}'.format(v))

answer.to_csv('solution.csv', index=False)