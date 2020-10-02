import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
trX = train.values
trY = trX[:, 0]
trX = trX[:, 1:]
trX = trX > 127.5
counts = np.sum(trX, axis=0)
print(counts)
print(counts.shape)
print(counts[100])
neighbors = trX[trX[:, 100] == 1]
print(neighbors.shape)
plt.imshow(np.mean(neighbors, axis=0).reshape(28, 28), cmap='gray')
plt.savefig('test.png')