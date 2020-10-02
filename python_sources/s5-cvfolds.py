import os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import StratifiedKFold
print(os.listdir('../input'))
print('loading data...')
with open('../input/X_train.pickle','rb') as fx, open('../input/y_train.pickle','rb') as fy:
    X_train = pkl.load(fx)
    y_train = pkl.load(fy)
print('done.')

five_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

train = []
test = []
for train_ind, test_ind in five_fold.split(X_train, y_train):
    train.append(train_ind)
    test.append(test_ind)
#train -> list of arrays, contain training set for each iteration of 5 fold
#test -> list of arrays, contan testing set for each iteration of 5 fold

folds = list(zip(train, test))
print(folds)
print(f'Number of folds: {len(folds)}')
with open('folds_list.pickle', 'wb') as f:
    pkl.dump(folds, f)