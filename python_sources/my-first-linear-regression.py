# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#!/bin/python
import numpy as np
timeCharged = float(input())
    #!/bin/python
import matplotlib as ml
import matplotlib.pyplot as plt
x=[]
y=[]
x1=[]
y1=[]
k=[]
l=[]
with open('../input/train.md','r') as f:
    array=[map(float,line.split(',')) for line in f.readlines()]
for i in range (100):
    x.append(array[i][0])
    y.append(array[i][1])
for i in range (100):
    if x[i]<4:
        x1.append(x[i])
        y1.append(y[i])
plt.scatter(x1, y1, c="r", alpha=1)
plt.show()
n=len(x1)
for i in range (n):
    k.append(1)
g=np.vstack((x1,k))
b=np.linalg.pinv(g.T)
c=np.matmul(b,y1)
l.append(timeCharged)
l.append(1)
a=np.matmul(c,l)
print("%.2f"%(a))
f.close()
