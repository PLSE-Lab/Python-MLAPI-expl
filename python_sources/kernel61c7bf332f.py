# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout, BatchNormalization
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

df = pd.read_csv("../input/corona-virus-latest-dataset/corona_latest.csv")
df.head()
samples = list(df.pop('Province/State'))
samples_1 = list(df.pop('Country/Region'))
samples_2 = list(df.pop("Last Update"))
corona = df.values

feature_1 = corona[:,0]
feature_2 = corona[:,1]
plt.scatter(feature_1,feature_2)
plt.axis('equal')
plt.show()

correlation, pvalue = pearsonr(feature_1,feature_2)
print("Correlation is:", correlation)

#Identify the principal components
normal = Normalizer()
pca = PCA()
kmeans = KMeans(n_clusters = 2)
pipeline = make_pipeline(normal,pca,kmeans)
pipeline.fit(corona)

features = range(pca.n_components_)
plt.bar(features,pca.explained_variance_)
plt.xlabel('PCA Features')
plt.ylabel('Variance')
plt.xticks(features)
plt.show()
