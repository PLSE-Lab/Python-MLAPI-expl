
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

# The competition datafiles are in the directory ../input
# Read competition data files:
labeled_images = pd.read_csv("../input/train.csv")
test_data  = pd.read_csv("../input/test.csv")

# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(labeled_images.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test_data.shape))
# Any files you write to the current directory get shown as outputs

images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

test_images[test_images>0]=1
train_images[train_images>0]=1

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
test_data[test_data>0]=1
results=clf.predict(test_data)
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
#df
df.to_csv('results.csv', header=True)

