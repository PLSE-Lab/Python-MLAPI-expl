import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/solutionex.csv')

from tqdm import tqdm
from sklearn.utils.extmath import weighted_mode

days = []

for id, row in tqdm(train.iterrows()):
    #cnt = np.zeros((7))
    lst = np.asarray(row)[1].split(' ')
    all = []
    week_w = []
    for el in lst:
        if (el == ''):
            continue
        el = (int)(el)
        all.append((el - 1) % 7 + 1)
        week_w.append((el // 7) ** 1.5)
    
    w = weighted_mode(all, week_w)
    days.append(' ' + str((int)(w[0][0])))
    #print(w[0][0])
test_rewrite = pd.DataFrame({'id': test['id'], 'nextvisit': days})
test_rewrite.to_csv("answers4.csv", index=False, sep=',')