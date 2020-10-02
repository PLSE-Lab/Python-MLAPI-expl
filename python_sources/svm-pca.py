import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

COMPONET_NUM=35

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

x_train=train.values[:,1:].astype(float)
y_train=train.values[:,0]

pca=PCA(n_components=COMPONET_NUM,whiten=True)
pca.fit(x_train)
x_train=pca.transform(x_train)

print("train svm")
svc=SVC()
svc.fit(x_train,y_train)

test=pca.transform(test)

predict=svc.predict(test)


subm=np.empty((len(predict),2))
subm[:,0]=np.arange(1,len(predict)+1)
subm[:,1]=predict
np.savetxt('submission.csv',subm,fmt='%d',delimiter=',',header='ImageId,Label',comments='')