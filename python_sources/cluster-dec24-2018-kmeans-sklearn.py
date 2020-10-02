'''
Jan04-2018 : Experimenting with multiple clusters
'''


import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.cluster import KMeans  

X = np.array([[5,3],  
     [10,15],
     [15,12],
     [24,10],
     [30,45],
     [85,70],
     [71,80],
     [60,78],
     [55,52],
     [80,91],]) 
     
     
     
plt.scatter(X[:,0],X[:,1], label='True Position')  
# kmeans = KMeans(n_clusters=2)  
kmeans = KMeans(n_clusters=3)  

kmeans.fit(X)  

print("The centroids of the clusters are ", kmeans.cluster_centers_)  
print("The point labels after clustering are ", kmeans.labels_)  

print("predicting cluster for test points")
print(kmeans.predict([[55,65], [61,71], [80, 90], [4, 4]]))

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')  
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black') 

plt.show()
plt.savefig("out.png", bbox_inches='tight')
plt.clf() 

''' version 2
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.predict([[0, 0], [4, 4]]))
print(kmeans.cluster_centers_) '''
