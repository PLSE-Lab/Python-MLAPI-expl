# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
"""
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()"""
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/HR_comma_sep.csv")
print(df.shape)
#print(df.head())

plt.plot([1,2,3,4])
plt.ylabel('test')
plt.pause(.001)
plt.ion()
plt.show()


"""
for dept in df['sales'].unique():
    
    print('department:',dept)
    print(df[df['sales'] == dept].mean())
    print('\n')
    
for s in df['salary'].unique():
    print('salary:',s)
    print(df[df['salary'] == s].mean())
    print('\n')
  """  
correlation = df.corr()
f = plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different features')
