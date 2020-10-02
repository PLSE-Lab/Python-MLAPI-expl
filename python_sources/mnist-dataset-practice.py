#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


labeled_images = pd.read_csv('../input/train.csv')
print(labeled_images.head());


# In[37]:


print(labeled_images.describe())


# In[10]:


images = labeled_images.iloc[0:5000,1:]
print(images.head())
labels = labeled_images.iloc[0:5000,:1]
print(labels.head())


# In[11]:


train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, test_size=0.2,random_state=0)


# In[12]:


for i in range(1,2):
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.subplot(2, 4, i + 1)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(train_labels.iloc[i,0])
    plt.show()


# In[13]:


train_images = train_images/255
test_images /=255
for i in range(1,2):
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[i,0])
    plt.show()


# In[14]:


plt.hist(train_images.iloc[i])


# In[15]:


clf = svm.SVC(gamma='auto')
clf.fit(train_images, train_labels.values.ravel())
expected = test_labels
predicted = clf.predict(test_images)
score = clf.score(test_images, test_labels);
print("Score of classifier %s:\n %s\n"% (clf, score))
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# In[16]:


clf = svm.SVC(gamma='scale')
clf.fit(train_images, train_labels.values.ravel())
expected = test_labels
predicted = clf.predict(test_images)
score = clf.score(test_images, test_labels);
print("Score of classifier %s:\n %s\n"% (clf, score))
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# In[17]:


clf = svm.SVC(kernel="linear")
clf.fit(train_images, train_labels.values.ravel())
expected = test_labels
predicted = clf.predict(test_images)
score = clf.score(test_images, test_labels);
print("Score of classifier %s:\n %s\n"% (clf, score))
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# In[18]:


clf = svm.SVC(kernel="linear",gamma=0.001)
clf.fit(train_images, train_labels.values.ravel())
expected = test_labels
predicted = clf.predict(test_images)
score = clf.score(test_images, test_labels);
print("Score of classifier %s:\n %s\n"% (clf, score))
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# In[19]:


clf = svm.SVC(kernel="linear",gamma=0.001, C=10)
clf.fit(train_images, train_labels.values.ravel())
expected = test_labels
predicted = clf.predict(test_images)
score = clf.score(test_images, test_labels);
print("Score of classifier %s:\n %s\n"% (clf, score))
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# In[20]:


clf = svm.SVC(kernel="linear",gamma=0.001, C=100)
clf.fit(train_images, train_labels.values.ravel())
expected = test_labels
predicted = clf.predict(test_images)
score = clf.score(test_images, test_labels);
print("Score of classifier %s:\n %s\n"% (clf, score))
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# In[38]:


images = labeled_images.iloc[0:42000,1:]
print(images.head())
labels = labeled_images.iloc[0:42000,:1]
print(labels.head())


# In[45]:


train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.33, test_size=0.6667,random_state=42)


# In[46]:


test_images = test_images[2:]
test_labels= test_labels[2:]


# In[34]:


clf = svm.SVC(kernel="linear",gamma=0.001, C=1000)
clf.fit(train_images, train_labels.values.ravel())
expected = test_labels
predicted = clf.predict(test_images)
score = clf.score(test_images, test_labels);
print("Score of classifier %s:\n %s\n"% (clf, score))
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# In[22]:


resultDf = pd.DataFrame(predicted)
resultDf.index += 1
resultDf.index.name = 'ImageId'
resultDf.columns=['Label']
resultDf.to_csv('results.csv', header=True)


# In[ ]:




