import numpy as np
import pandas as pd
import os
import pickle as pkl
from sklearn.metrics import confusion_matrix

with open('../input/y_train.pickle', 'rb') as f:
    y_train = pkl.load(f)
    
with open('../input/y_test.pickle', 'rb') as f:
    y_test = pkl.load(f)


total = y_train.shape[0]
infected = y_train.sum()
clean = total - infected

inf_perc = infected / total
cle_perc = clean / total

y_pred = [1 if rn < inf_perc else 0 for rn in np.random.sample(size=y_test.shape[0])]
print('----------------------------------')
print('Baseline Model - Confusion Matrix')
print(confusion_matrix(y_test, y_pred))