import pandas as pnd

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score


iris_dataset = pnd.read_csv("../input/Iris.csv")


matrix = pnd.DataFrame.as_matrix(iris_dataset[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']])


for no_of_clusters in (2,3,5):
    cluster_algo = KMeans(n_clusters=no_of_clusters)
    labels = cluster_algo.fit_predict(matrix)
    silhouette_coeff = silhouette_score(matrix,labels,metric='euclidean')
   
   
    print("For cluster =", no_of_clusters, 
          "The average silhouette coefficient is:", silhouette_coeff)


