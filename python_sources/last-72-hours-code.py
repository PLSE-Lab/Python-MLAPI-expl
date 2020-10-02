# Python 3.6 
# Jupyter Notebook with Anacoda 
# all data come from - http://www.gunviolencearchive.org/last-72-hours

import pandas as pd
import numpy as np

df = pd.read_csv("...last72hours.csv") # include the location of csv file 

df.head()

import matplotlib.pyplot as plt 
import seaborn as sns

plt.figure(figsize=(14,10))
ax = sns.countplot(x="State", data=df)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=90, ha="center")

plt.show()
# it will show a plot of count data by states for this categorical data set
