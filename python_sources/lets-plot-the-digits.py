import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

### Import Statements
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import RandomizedPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding


### end of import statements

# Create list of features in the dataSet
features_list = train.columns.tolist()

train = train.as_matrix( columns = None).astype(float)

#First run 5000 rows only
features, labels = train[:,1:], train[:,0]

pca = LocallyLinearEmbedding(n_components=3, n_neighbors=10, method='modified')

fig = plt.figure(1, figsize=(14, 4))

ax = Axes3D(fig, elev=-150, azim=110)

projection = pca.fit_transform(features)
ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], c=labels, cmap=plt.cm.Paired)
ax.set_xlabel("First Eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Second Eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Third Eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.title(pca.__class__.__name__)
plt.savefig('LocallyLinearEmbedding.png', bbox_inches='tight')