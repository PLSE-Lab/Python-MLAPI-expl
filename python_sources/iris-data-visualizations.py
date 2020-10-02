# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# load iris data
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/Iris.csv")
df.head()
iris = df.drop("Id",axis = 1)

# Number of samples for each species
g = sns.catplot(x = 'Species',data = df,kind='count',palette = sns.color_palette("Set2"))
g.savefig("1_Samples.png")

# distribution of SepalLengthCm
plt.figure(figsize = (10,6))
g = sns.catplot(data = iris,palette = 'Set2',x = "Species",y = "SepalLengthCm",kind='violin',inner = None);
sns.swarmplot(data = iris,x = "Species",y = "SepalLengthCm",color = 'white', size=5,alpha = 0.7,ax = g.ax)
plt.title('SepalLengthCm')
g.savefig("2_SepalLengthCm.png")

# distribution of data for each species
melted_iris = pd.melt(iris,id_vars = ['Species'],var_name = 'Stat')
melted_iris.head()
g = sns.catplot(data = melted_iris,palette = 'Set2',x = 'Stat',y = 'value',hue = 'Species',dodge = True,kind='violin',inner = None);
plt.xticks(rotation = 45)
plt.xlabel('')
g.savefig("3_Distibutions.png")

# pairplot of data for each species
g = sns.pairplot(iris,hue='Species',kind='scatter',diag_kind='hist',palette='Set2')
g.savefig("4_Pairs.png")

# heatmap of correlation
corr = iris.drop('Species',axis = 1).corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, cmap=sns.diverging_palette(220, 20, s=85, l=50, n=14), annot=True, fmt=".2f")
fig.savefig("5_Heatmap.png")