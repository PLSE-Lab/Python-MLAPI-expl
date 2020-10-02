import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab
from matplotlib import pyplot as plt


trainFile = "../input/train.csv"
testFile = "../input/test.csv"

# Load train data
trainData = pd.read_csv(trainFile)

# Load test data
testData = pd.read_csv(testFile)

trainData.YrSold.value_counts().plot(kind='barh')

print(trainData.YrSold.value_counts())
plt.figure()
plt.plot(trainData.YrSold.value_counts())
plt.show