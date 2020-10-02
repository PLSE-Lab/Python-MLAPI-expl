import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
#print("Training set \n", train.label.describe())

# print("Plot handwritten digit ", train.iat[3,0])
# plt.imshow(train.ix[0,1:].reshape(28,28))
#eig_vals,eig_vects = np.linalg.eig(np.cov(train.ix[:,1:].values)) # killed by out of memory
#print("First ten eigenvalues: ", eig_vals[0:10]) 
N_COMPONENTS = 50
pca = PCA(n_components = N_COMPONENTS, whiten = True)
pca.fit(train.ix[:,1:].values)
train_label = train.ix[:,0].values
train_data = pca.transform(train.ix[:,1:].values)

svc = SVC()
svc.fit(train_data, train_label)

test_label = svc.predict(pca.transform(test.values))

with open("predict.csv", "w") as writer:
    writer.write('ImageID,Label\n')
    for i, p in enumerate(test_label):
        writer.write(str(i+1) + "," + str(p) + "\n")