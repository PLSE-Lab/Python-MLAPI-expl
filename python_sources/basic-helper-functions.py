# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sea
import os

def create_stacked_bargraph(train, attributes):
    df2 = train.groupby(attributes).size().unstack()
    for index, row in df2.iterrows():
        print(row[0], row[1])
    df2.plot(kind='bar', stacked=True)
    return plt.show()

def bargraph_percentage_split(train,attribute):
    import matplotlib.patches as mpatches
    result = pd.value_counts(train[attribute])
    total = sum(result)
    result = result.astype('float64')
    result = result.sort_index()

    print(result)
    for city in result.index:
        result[city] /= total
    
    result.plot(kind='bar')
    return plt.show()

def dark_display_preferences():
    mpl.rcParams['text.color'] = 'grey'
    mpl.rcParams['axes.labelcolor'] = 'grey'
    mpl.rcParams['xtick.color'] = 'grey'
    mpl.rcParams['ytick.color'] = 'grey'
    mpl.rcParams["axes.facecolor"] = "#262B35"
    mpl.rcParams["figure.facecolor"] ="#262B35"
    
if __name__ == "__main__":
    data = pd.read_csv("../input/renfe.csv", nrows=1000)
    dark_display_preferences()
    bargraph_percentage_split(data,"origin")
    #print(data.head())