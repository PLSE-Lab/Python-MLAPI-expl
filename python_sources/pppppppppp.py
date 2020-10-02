import numpy as np
import pandas as pd
import nltk
from nltk.classify import NaiveBayesClassifier as nbc

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

print("\n\nSummary statistics of training data")
print(train.describe())

