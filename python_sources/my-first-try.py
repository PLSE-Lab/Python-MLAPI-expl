# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/6_differe_classes_.csv")
df = df.drop(['Unnamed: 0'],axis=1)
df1 = df.loc[:,'delta':'ZC']
df2 = df.loc[:,'Eye_state']
data = {'data':df1,'target':df2}

X , y = data["data"] , data["target"]
X = X.as_matrix()
y = y.as_matrix()
X_featurea = X[:,0:23]
