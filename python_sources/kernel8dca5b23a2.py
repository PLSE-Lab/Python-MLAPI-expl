                              # Importing the libraries
import numpy as np
import matplotlib.pyplot as pd
import pandas as pd
                                  # Importing the Dataset

dataset = pd.read_csv('Finance_Applications.csv')
print(dataset)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

                                # Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)
print(X)

                                # Training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

                                # Visualising the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5, markers[y[i]],
         markeredgecolor=colors[y[i]], markerfacecolor= 'None', markersize=10, markeredgewidth=2)
show()



