import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# The competition datafiles are in the directory ../input
# Read competition data files:
#train = pd.read_csv("../input/train.csv")
#test  = pd.read_csv("../input/test.csv")

# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("../input/train.csv")
train = dataset.iloc[:,1:].values
#print(dataset.head())
print(dataset.iloc[:,1:].head())
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
plt.figure()

plt.imshow(train[1729][0], cmap=cm.binary) # draw the picture
plt.savefig("one.png")

