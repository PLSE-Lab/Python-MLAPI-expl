import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

voice = pd.read_csv('../input/voice.csv')
headers = voice.columns
X, y = voice[headers[:-1]], voice['label']
y = y.replace({'male':1,'female':0})
# print(y)
# print(voice.corr())
# Rescale data before TSNE

# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)

pca = PCA(n_components=2, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
# pca.fit(X)
X_new=pca.fit_transform(X)
print(X_new)
