import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv")

plt.hist([train[train.OpenStatus==0].ReputationAtPostCreation.values,train[train.OpenStatus==1].ReputationAtPostCreation.values],label = [0, 1],alpha=.5);
plt.yscale("log")
plt.legend()
plt.savefig("open_by_length1.png")