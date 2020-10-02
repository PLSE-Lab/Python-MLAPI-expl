import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:

# Any files you write to the current directory get shown as outputs
"""print(train.shape)
print(test.shape)
print(train.head())
"""
import matplotlib.pyplot as plt
plt.hist(train["label"])
plt.title("Frequency Histogram of Numbers in Training Data")
plt.xlabel("number Values")
plt.ylabel("Frequency")
plt.show()
plt.savefig("Histo.png")

import math
# plot the first 25 digits in the training set. 
f, ax = plt.subplots(5, 5)
# plot some 4s as an example
for i in range(1,26):
    #print (train["label"][i-1])
    # Create a 1024x1024x3 array of 8 bit unsigned integers
    data = train.iloc[i,1:785].values #this is the first number
    nrows, ncols = 28, 28
    grid = data.reshape((nrows, ncols))
    n=math.ceil(i/5)-1
    m=[0,1,2,3,4]*5
    img=ax[m[i-1], n].imshow(grid)
    plt.show()
    plt.savefig("test.png")
    
label_data=train["label"]
train.drop("label",axis=1)
train=train/255
test=test/255
train["label"]=label_data

from sklearn import decomposition
from sklearn import datasets

pca=decomposition.PCA(n_components=50)
pca.fit(train.drop('label',axis=1))
PCTrain=pd.DataFrame(pca.transform(train.drop('label',axis=1)))
PCTrain["label"]=train["label"]

PCtest = pd.DataFrame(pca.transform(test))
print("PCA done")
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
x=PCTrain[0]
y=PCTrain[1]
z=PCTrain[2]

colors=[int(i % 9) for i in PCTrain['label']]
ax.scatter(x,y,z,c=colors,marker='o',label=colors)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')


plt.show()
plt.savefig("3deee.png")


from sklearn.neural_network import MLPClassifier
y=PCTrain["label"][:20000]
x=PCTrain.drop("label",axis=1)[:20000]
clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(2500,),random_state=1)
clf.fit(x,y)

"""from sklearn import metrics
predicted=clf.predict(PCTrain.drop("label",axis=1)[20001:420000])
expected=PCTrain["label"][20001:420000]
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))"""

output=pd.DataFrame(clf.predict(PCtest),columns=["label"])
output.reset_index(inplace=True)
output.rename(columns={'index': 'ImageId'}, inplace=True)
output['ImageId']=output['ImageId']+1
output.to_csv('output.csv',index=False)





    
    
    
    
