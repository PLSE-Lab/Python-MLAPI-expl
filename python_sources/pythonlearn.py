import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

# The competition datafiles are in the directory ../input
# Read competition data files:
data_dir = "../input/"
train_data = open(data_dir + "train.csv").read()
train_data = train_data.split("\n")[1:-1]
print(train_data)
# Write to the log:

# Any files you write to the current directory get shown as outputs