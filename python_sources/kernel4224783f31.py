import os
print(os.listdir("../input"))


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/iris/Iris.csv')
df.head()

types = df.Species.unique()
lookup = dict(zip(types, range(len(types))))
df['SpeciesLabel'] = df['Species'].replace(lookup)
df.head()

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['SpeciesLabel']
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

