#encoding:utf-8
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
import pandas,numpy,os,sys,csv
train_path="../input/train.csv"
test_path="../input/test.csv"
train_data=pandas.read_csv(train_path,header=0).values
test_data=pandas.read_csv(test_path,header=0).values
svc=SVC(C=1.0,kernel='rbf')
pca=RandomizedPCA(n_components=16,whiten=True).fit(train_data[0::,1::])
print("learning")
svc.fit(pca.transform(train_data[0::,1::]),train_data[0::,0])
print("complete")
print("predicting")
svc_out=svc.predict(pca.transform(test_data))
prediction=open("out.csv","w")
pre_open_file=csv.writer(prediction)
pre_open_file.writerow(["ImageId","Label"])
n,m=test_data.shape
pre_open_file.writerows(zip(range(1,n+1),svc_out))
prediction.close()
