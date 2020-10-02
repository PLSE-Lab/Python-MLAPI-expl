# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))
import os
print(os.listdir("../input/digit-recognizer"))


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
train = pd.read_csv("../input/digit-recognizer/train.csv")
train.shape 
test  = pd.read_csv("../input/digit-recognizer/test.csv")
test.shape

train.head()
test.head()

train_x = train.values[:,1:]
train_y = train.iloc[:,0]
test_x = test.values
p = PCA(n_components=0.8)
train_x = p.fit_transform(train_x)

test_x = p.transform(test_x)
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(train_x, train_y)

test_y = neigh.predict(test_x)
pd.DataFrame({"ImageId": range(1,len(test_y)+1), "Label": test_y}).to_csv('output.csv', index=False, header=True)
