import pandas as pd
from numpy import *
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.svm import SVC 

# The competition datafiles are in the directory ../input
# Read competition data files:
df = pd.read_csv('../input/train.csv', header = 0)
df_test = pd.read_csv('../input/test.csv',header = 0)

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(df.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(df_test.shape))
# Any files you write to the current directory get shown as outputs
df_sample = df.head(42000)
data = df_sample.values
data_test = df_test.values
x = data[:,1::]
y = data[:,0]
#x_test1 = data_test[0:15000,:]
#x_test2 = data_test[15000::,:]
x_test = data_test
#x_test = data[990:1000,1::]
#y_test = data[990:1000,0]

# normalize data 
x_mean = mean(x)
x = x - x_mean
x_std = std(x)
x = x/x_std
#x_test1 = (x_test1 - x_mean) / x_std
#x_test2 = (x_test2 - x_mean) / x_std
x_test = (x_test - x_mean) / x_std
# decomposition with PCA
num_components = 100
pca = PCA(n_components = num_components, whiten = True)
pca.fit(x)
x = pca.transform(x)
x_test = pca.transform(x_test)
# fit the training data
#logr = linear_model.LogisticRegression()
#logr.set_params(C=0.01)
#logr.fit(x,y)
svc = SVC(kernel='rbf')
svc.fit(x,y)

#score = logr.score(x_test,y_test)
# prediction
#output1 = logr.predict(x_test1)
#output2 = logr.predict(x_test2)
#output = concatenate((output1,output2),axis = 0)
#output = logr.predict(x_test)
output = svc.predict(x_test)
###############################################################################
# submission
output = output.astype(int)
imageId = arange(1,28001)
submission = pd.DataFrame({"ImageId":imageId,"label":output})
submission.to_csv("submission.csv", index=False)