# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

dt = pd.read_csv("../input/Rate.csv")
dt1 = dt[['Age','IndividualRate']]
dt1.Age = dt1.Age._convert(numeric = True)
dt1 = dt1.dropna()
dt1.Age = dt1.Age.astype(int)


plt.xlim(105, 1500)
plt.ylim(0, 175000)
plt.title('Individual Rate Histogram (105-1500)')
plt.ylabel('Frequency/Count')
plt.xlabel('Individual Rate')

dt1 = dt1[dt1.IndividualRate < 1600]
#Lets look at a histogram of our Individual Rate data
dt1.IndividualRate.hist(bins = 300)