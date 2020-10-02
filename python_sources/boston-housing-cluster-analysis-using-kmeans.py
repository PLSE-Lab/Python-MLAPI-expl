#BOSTON HOUSING Data Classification Using KMeans Cluster Analysis Algorithim ( Distance based -partitional Clustering Algo )

import pandas as pd
from sklearn.datasets import load_boston
boston=load_boston()

ds=pd.DataFrame(boston.data,columns=boston.feature_names)
ds.head()

#1-hot encoding of RAD variable; because its categorical variable
#representing it as categorical variable
ds["RAD"]=ds["RAD"].astype("category")
#datatype of the ds
ds.dtypes

#now using df.get_dummies(); it will drop the original column also
#this method will automatically pick the categorical variable and apply 1-hot encoding
ds=pd.get_dummies(ds,prefix="RAD")
ds.head()

#now doing Scaling on AGE,TAX,B or on entire Dataset
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler();
scaler=scaler.fit(ds)

scaledData=scaler.transform(ds)

#now create the scaled dataframe from it
dss=pd.DataFrame(scaledData,columns=ds.columns)

#now perform the clusetring 
#step 1  cluster configuration to kind the k
#step 2 using the value of 'k', generate the cluster

#now to know the best value of 'k' 
# wss/bss vs k

#That is when k=2, wss=sum of all point with theri 2 centeroid individually 
#        i.e within clusterdistance ( this is inertia )
#    and   bwss means distance between centroid c1 and c2

#now when k=3, wss= sum of distance all point of culter and their centroid 
# the above wss is given by inertia of the cluster configuration
## but for bwss the sum of distance between 3 centroid.
## c1 to c2, c1 to c3 and c2 to c3

###when cluster configuration=4
##the bss= dist(c1,c2)+dist(c1,c3) +dist(c1,c4) + dist(c2,c3) +dist(c2,c4) +dist(c3,c4)

#so all possible combination we need to find out for all values of k


from sklearn.cluster import KMeans
from itertools import combinations_with_replacement

from itertools import combinations 
from scipy.spatial import distance
print(list(combinations_with_replacement("ABCD", 2)))

wss=[]
bss=[]
pairmap={}
dis=[]
d=0
distanceMap={}
for k in range(2,16):
    #perforiming  the cluster configuration
    clust=KMeans(n_clusters=k,random_state=0).fit(dss)
    wss.append(clust.inertia_)
    c=list(combinations(range(0,k), 2))
    print("Combinations ----------->",c)
    print("ClusterCenters Are Below----------->")
    dataFrameClusterCenter=pd.DataFrame(clust.cluster_centers_)
    print(pd.DataFrame(clust.cluster_centers_))
    print("The above are clusterCenters are for k==",k)
    pairmap[k]={"pairs":c}
    for i in c:
        #converting the tuple() to list using the list() method
        pair=list(i)
        print("pair is",pair)
        #extracting the index from the pair
        index1=pair[0]
        index2=pair[1]
        #print("row 1"); print(dataFrameClusterCenter.iloc[index1,:])
        #print("row 2"); print(dataFrameClusterCenter.iloc[index2,:])
        d=distance.euclidean(dataFrameClusterCenter.iloc[index1,:],
                             dataFrameClusterCenter.iloc[index2,:])
        print("distance",d)
        #appending the calculated distance between each pair of the cluster centers in a list
        dis.append(d)  
        distanceMap[k]={"distance":dis}
    #making the list empty for next k
    dis=[]
        
print("disstacne map for each k ")
print(distanceMap)   
print("wss for all k ")
print(wss)     


#sum the distance of between every cluster 
#summedDistance storing to bss list
bss=[]
import math
for i in range(2,16):
    value=distanceMap.get(i)
    print(value)
    list=value['distance']
    print(math.fsum(list))
    summedDistance=math.fsum(list)
    bss.append(summedDistance)
    
bss
#1. now we have bss for all the k 
bss
#2. now we have wss for all the k
wss
#but wss shal be sqrt(wss[i])
len(wss)
len(bss)
sqrtwss=[]
for i in range(0,len(wss)):
    sqrt=math.sqrt(wss[i])
    print(sqrt)
    sqrtwss.append(sqrt)

#so this sqrtwss shall be used
sqrtwss


#final ratio =sqrtwss/bss
ratio=[]
for i in range(0,len(sqrtwss)):
    #ratio.append(sqrtwss[i]/wss[i])
    ratio.append(sqrtwss[i]/bss[i])
    
    #So finally perforimg scatter plot of ratio vs k plot
#########################   ratio=(sqrtwss/bss) vs k plot ############################
ratio
del list
k=range(2,16)
k
k=list(k)
k
from matplotlib import pyplot as plt
plt.plot(k,ratio)
plt.xlabel("No of cluster k")
plt.ylabel("Ratio of sqrtwss/bss")
plt.show()


#plot of sqrtwss vs k
plt.plot(k,sqrtwss)
plt.xlabel("No of cluster k")
plt.ylabel("wss or sqrtwss")
plt.show()


#plot of bss vs k
plt.plot(k,bss)
plt.xlabel("No of cluster k")
plt.ylabel("bss")
plt.show()




############# Now as we knoe the optiomal value of k is 4, so 
############# So we now perform actual clustering of 506 observations and there scaled 
############ scaled and linear independence dataset

#our scaled dataset is represented by dss
dss.shape
#to find corelation matrix 
dss.corr()


#now performing the clustering
clust=KMeans(n_clusters=4,max_iter=500,random_state=0).fit(dss)

#now extract the clusterCenters
clusterCenter=clust.cluster_centers_

#convert clusterCenter to dataframe to do the cluster profilin
ccd=pd.DataFrame(clusterCenter,columns=dss.columns)

#ccd for cluster profilin
ccd
#so profiling details
#clusterId 1 is having the highest crime rate
# industry are more in clusterId 1              


#to see the labels i.e clusterId for each observation
labels=clust.labels_

#total labes;
len(labels)
clusterIds=list(labels)

#now perform the inverse Scaling
originalDataAsNumpy=scaler.inverse_transform(dss)
#converting numpy to dataset
originalDataset=pd.DataFrame(originalDataAsNumpy,columns=dss.columns)

#adding the labelled column to the originalDataset
originalDataset["Label"]=labels

#saving data on the system as OriginalData.csv
originalDataset.to_csv("yoursystem path\\originalData.csv")
#to see whether data contains the label or not
originalDataset.Label[0]

##### Now plotting the Classfication 
import pylab as pl
len=originalDataset.shape[0]
len
for i in range(0, len):
   if originalDataset.Label[i] == 0:
      c1 = pl.scatter(originalDataset.iloc[i,2],originalDataset.iloc[i,4],c='r', marker='+')
   elif originalDataset.Label[i]  == 1:
      c2 = pl.scatter(originalDataset.iloc[i,2],originalDataset.iloc[i,4],c='g',marker='o')
   elif originalDataset.Label[i]  == 2:
      c3 = pl.scatter(originalDataset.iloc[i,2],originalDataset.iloc[i,4],c='b',marker='*')
   elif originalDataset.Label[i] == 3:
      c4 = pl.scatter(originalDataset.iloc[i,2],originalDataset.iloc[i,4],c='y',marker='^')
pl.legend([c1, c2, c3,c4], ['c1','c2','c3','c4'])  
pl.title('Boston Data classification')
pl.show()

