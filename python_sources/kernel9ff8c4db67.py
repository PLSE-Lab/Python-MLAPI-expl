# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

dt=pd.read_csv("../input/heart-disease-uci/heart.csv")
dt
dt.info()# Any results you write to the current directory are saved as output.
dt[dt.oldpeak<1]
from pandas.plotting import parallel_coordinates
from matplotlib import pyplot as plt
plt.figure(figsize=(20,10))
parallel_coordinates(dt, 'target',color=('#556270', '#4ECDC4'))
plt.show()