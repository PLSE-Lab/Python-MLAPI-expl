# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

#Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/MicrosoftCompanyStock.csv")
tester=pd.read_csv("../input/test.csv")

print(df.head())
fig,ax=plt.subplots(ncols=1)
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot(df["year"], df["high"], 'o-',label="High price")
plt.title('Microsoft stock Visuals')
plt.ylabel(' Prices')
plt.ticklabel_format(style='plain',axis='x',useOffset=False)
plt.plot(df["year"], df["low"], '.-',label="Low Price")
plt.legend()
plt.xlabel("Years")

plt.show()