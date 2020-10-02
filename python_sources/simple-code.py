import pandas as pd
import math
import numpy as np
import warnings

def main(train):
    mas = []
    for i in train['id']:
        user = np.zeros(7, dtype=np.float)
        flag = 0
        for k in train['visits'][i - 1][1:].split(' '):
            day = int(k)
            if day > 1032:
                flag = 1
            user[(day - 1) % 7] += day
        if flag == 1:
            mas.append(np.argmax(user))
        else:
            mas.append(0)
    return mas



test = pd.read_csv('../input/train.csv')
answer = main(test)
with open("solution.csv", "w") as f:
    print("id,nextvisit", file=f)
    for idx, day in enumerate(answer):
        print("{}, {}".format(idx + 1, day + 1), file=f)