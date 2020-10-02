import pandas as pd

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

iris = pd.read_csv("../input/Iris.csv")

iris_matrix = pd.DataFrame.as_matrix(iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']])

for n_clusters in range(2,11):
    cluster_model = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = cluster_model.fit_predict(iris_matrix)
    silhouette_avg = silhouette_score(iris_matrix,cluster_labels,metric='euclidean')

