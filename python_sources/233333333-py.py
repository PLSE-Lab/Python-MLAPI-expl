#encoding:utf-8
import pandas,numpy,os,sys,csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import RandomizedPCA
train_path="../input/train.csv"
test_path="../input/test.csv"
train_data=pandas.read_csv(train_path,header=0).values
test_data=pandas.read_csv(test_path,header=0).values
knn_object=KNeighborsClassifier()
pca=RandomizedPCA(n_components=30,whiten=True).fit(train_data[0::,1::])
#print(train_data)
print("Learning")
knn_object.fit(pca.transform(train_data[0::,1::]),train_data[0::,0])
print("Complete")
print("Predicting")
knn_out=knn_object.predict(pca.transform(test_data))
print("Complete")
prediction=open("out.csv","w")
pre_open_file=csv.writer(prediction)
pre_open_file.writerow(["ImageId","Label"])
n,m=test_data.shape
pre_open_file.writerows(zip(range(1,n+1),knn_out))
prediction.close()
