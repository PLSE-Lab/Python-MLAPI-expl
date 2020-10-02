import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.cross_validation import train_test_split
from sklearn import svm

# Big shout out to Charlie.H who provides me, a ML beginner, a way to approach this question
#https://www.kaggle.com/archaeocharlie/digit-recognizer/a-beginner-s-approach-to-classification/notebook


labeled_images = pd.read_csv("../input/train.csv")
sample = 10000 
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

#using SVM method
train_images[train_images>0] = 1
test_images[test_images>0] = 1
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images,test_labels))

#write out the output
test_data = pd.read_csv("../input/test.csv")
test_data[test_data>0] = 1
results = clf.predict(test_data)

df = pd.DataFrame(results)
df.index.name='ImageId' #Error the ImageId doesn't show up in Excel
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)