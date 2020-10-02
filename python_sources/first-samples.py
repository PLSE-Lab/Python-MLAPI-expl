import pandas as pd
import scipy as sp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv 

# The competition datafiles are in the directory ../input
# Read competition data files:
trains = pd.read_csv("../input/train.csv")
tests  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(trains.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(tests.shape))
# Any files you write to the current directory get shown as outputs

print(trains.head())


labels = trains['label'] 

del trains['label']

X_train_datasarr = trains.as_matrix()
#X_train_datasarr = np.array(trains)

X_train_norm = X_train_datasarr > 0
X_train = X_train_norm.astype(int) 

X_test_datasarr = tests.as_matrix()
X_test_norm = X_test_datasarr > 0
X_test = X_test_norm.astype(int) 

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,labels)

Y_test = rfc.predict(X_test)

headers = ['ImageId','Label']

with open('digit_submission.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    rowid = 1
    for y in Y_test:
        row = [rowid,y]
        rowid += 1
        f_csv.writerow(row)


