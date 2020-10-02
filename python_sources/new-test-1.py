# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 




# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from subprocess import check_output

import matplotlib as inline
print(check_output(["ls", "../input"]).decode("utf8"))
iris_df = pd.read_csv("../input/Iris.csv")
print(iris_df.describe())
iris_df.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")

plot_iris = sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris_df, size=5)
cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
corr_matrix = iris_df[cols].corr()
heatmap = sns.heatmap(corr_matrix,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols,cmap='Dark2')