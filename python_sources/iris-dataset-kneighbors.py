# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing ,cross_validation,neighbors #for training the dataset
from pandas.tools.plotting import radviz,andrews_curves,scatter_matrix #for getting insights via plotting graphs
from bokeh.charts import Scatter ,show 
from bokeh.io import output_notebook
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df=pd.read_csv("../input/Iris.csv")
df.head(5)
df['Species'].value_counts().to_frame()
df.drop('Id',1,inplace=True)#since there's no need for id 
andrews_curves(df,'Species')
scatter_matrix(df,alpha=0.2,figsize=(6,6),diagonal='kde')
