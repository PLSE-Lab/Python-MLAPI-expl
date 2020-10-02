# Import necessary modules
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split 
from multiprocessing.dummy import Pool as ThreadPool
from timeit import default_timer as timer
from sklearn.cluster import KMeans
import pandas as pd 
import numpy as np

start= timer()

labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
X=images
y=labels
X[X>0]=1

isValid=[np.sum(X.loc[:,x])>0 for x in X.columns]
Clean_X= X.loc[:,isValid]
new_x= Clean_X.T.drop_duplicates(keep="first").T
Clean_X.columns.difference(new_x.columns)

from sklearn.decomposition import PCA
pca = PCA(n_components=500)
pca.fit(Clean_X)
features = range(pca.n_components_)
plt.bar(features,pca.explained_variance_)
print(max(pca.explained_variance_))
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
print(pca.explained_variance_ratio_) 
print(np.sum(pca.explained_variance_ratio_)) 
X_train,y_train,X_test, y_test= train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=21)



