import pandas as pd
import numpy as np
def load_data(format='np'):
        train=np.genfromtxt('../input/train.csv',delimiter=',',skip_header=1)
        test_x=np.genfromtxt('../input/test.csv',delimiter=',',skip_header=1)
        train_y=train[:,0]
        train_x=train[:,1:]
        return train_x,train_y,test_x
        
train_x,train_y,test_x=load_data()

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
pca_choose=PCA().fit(train_x)
plt.plot(np.cumsum(pca_choose.explained_variance_ratio_))
plt.xlabel('n_compoents')
plt.ylabel('Overall variance')
plt.axhline(0.9,ls='dashed',color='r')

pca=PCA(0.9)
pca.fit(train_x)
train_trans=pca.transform(train_x)
print("There are {} dimension remaining".format(len(pca.components_)))

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(train_trans,train_y)
test_trans=pca.transform(test_x)

out=open('predict.csv','w')
out.write("ImageId,Label\n")
count=0
for prediction in clf.predict(test_trans):
    count+=1
    out.write("{},{}\n".format(count,int(prediction)))