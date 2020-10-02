# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 




import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Store start time to check script's runtime
scriptStartTime = time.time()

# Read file
df = pd.read_csv("../input/Crime1.csv")

df["Dates"] = pd.to_datetime(df["Dates"])


print(df.columns)