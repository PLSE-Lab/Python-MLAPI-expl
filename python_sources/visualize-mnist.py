import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs


X_train = train.ix[:,1:].values.astype(float)
y_train = train.ix[:,0].values

X_test = test.values.astype(float)

fig = plt.figure()

Nsample = 10
labels = list(range(10))
for y in labels:
    d = train.ix[train['label']==y, 1:].sample(n=Nsample).values
    for idx in range(Nsample):
        a = fig.add_subplot(Nsample, len(labels), idx * len(labels) + y + 1)
        plt.imshow(d[idx].reshape((28,28)))
        plt.axis('off')
        if idx == 0:
            a.set_title('label={}'.format(y))

plt.show()