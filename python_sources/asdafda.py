# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_csv('../input/300k.csv', delimiter=',',usecols=["pokemonId", "latitude", "longitude","city"])
chicago = df[(df['city'] == 'Chicago')]
y = chicago['pokemonId'].tolist()
X = np.column_stack((chicago['latitude'].tolist(), chicago['longitude'].tolist()))
print(len(set(y)))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
print(len(set(y_train)))
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(4)
neigh.fit(X_train, y_train) 
print(neigh.predict([X[3]]))
print(neigh.predict_proba([X[3]]))
print(y[3])
print(neigh.score(X_train,y_train))
print(neigh.score(X_test,y_test))
# Any results you write to the current directory are saved as output.