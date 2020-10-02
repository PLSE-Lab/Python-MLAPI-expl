# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

### %matplotlib inline

df=pd.read_csv("../input/diabetes.csv")

def plot_corr(df, size=11):
    '''function to plots a graphical correlation
    Input:
        df: pandas dataframe
        size: size of the plot
    Displays:
        matrix of correlation between columns.
        Blue-cyan-yellow-red-darkred >>> less to more correlated
        expect cols to correlate perfectly with itself
    '''
    corr=df.corr()
    fig,ax=plt.subplots(figsize=(size,size))
    ax.matshow(corr)    #color code the correlation value
    plt.xticks(range(len(corr.columns)),corr.columns)   #draw x tick marks
    plt.yticks(range(len(corr.columns)),corr.columns)   #draw y tick marks
    
df.head(5)
