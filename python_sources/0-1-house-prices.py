#!/usr/bin/env python
# coding: utf-8

# Let's try

# In[ ]:


# For example, here's several helpful packages to load in 

import seaborn as sns
import seaborn.matrix as smatrix
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import seaborn as sns


# ## Panic - data exploration##

# In[ ]:


#data imported
train = pd.read_csv ('../input/train.csv')
train.head()


# In[ ]:


# dtypes: float64(3), int64(35), object(43) = 79 features and 1460 samples
train.info()


#  *Step by Step (slowly)*

# In[ ]:


#have been selected only the int and the float
Train_numeric = train [['SalePrice','MSSubClass', 'LotArea','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']]
Train_numeric.head()


# In[ ]:


Train_numeric.describe()


# In[ ]:


sns.set()

sns.pairplot(Train_numeric, hue= 'OverallQual')


# In[ ]:


sns.set(style="white")
corr = Train_numeric.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True



# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.99, square=True, linewidths=.5, cbar_kws={"shrink": .5}, )


# In[ ]:


corr


# In[ ]:


sns.stripplot(x="OverallQual", y= 'SalePrice', data=Train_numeric);


# In[ ]:


sns.jointplot(x="OverallQual", y="SalePrice", data=Train_numeric, kind="hex", );


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# X = the other data
# Y = SalePrice


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


n_neighbors = 15

# import some data to play 
X = train [['MSSubClass', 'LotArea','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = train ['SalePrice']


# In[ ]:


#just a quick test with scikit learn

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5, weights='distance' ) #5 closest neighbors 
neigh.fit(X, y) 


# In[ ]:


print (neigh.predict([[60,11250,10,5,2001,2002]])) # index 2 with OverallQual = 10 (and not 7)


# In[ ]:


print (neigh.predict_proba([[60,11250,10,5,2001,2002]]))


# In[ ]:





# In[ ]:




