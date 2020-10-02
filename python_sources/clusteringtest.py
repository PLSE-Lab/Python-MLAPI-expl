import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

df = pd.DataFrame(pd.read_csv("../input/College.csv"))

# Dataframe shape
print(df.shape)

# Total Columns
print(len(df.columns.values))

# Columns
print(df.columns.values)

# HEAD
print(df.head(2))

# Factorize - if Private = 1 else 0
df["Private"] = 1- pd.factorize(df["Private"])[0]
print(df.head(1))

Y = list(df["Private"])

df = df.drop(["Private", "Unnamed: 0"], axis=1)

X = df.values

print(X.shape)

model = KMeans(n_clusters=2, random_state=1)
model.fit(X)
y_kmeans = model.predict(X)
print(model.labels_)
print(model.cluster_centers_)
print(accuracy_score(Y, model.labels_))

# Cluster Plot
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);