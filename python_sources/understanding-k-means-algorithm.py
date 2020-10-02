######################### Visualising the Clusters  ############################
import matplotlib.pyplot as plt  

def graph_img(kmeans, X):
    plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')  
    plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black') # Black dots are centroids
    plt.savefig("graph.png")
    

######################### Utilising Algorithm Code ###########################

from sklearn.cluster import KMeans
    
def Clustering(X):   
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)
    return kmeans

############################## Main Code #######################################

from numpy import *

points = genfromtxt("../input/data.csv", delimiter=",")

cluster = Clustering(points)

graph_img(cluster,points)

################################################################################