#!/usr/bin/env python
# coding: utf-8

# # Accuracy=100% using kNN k=1 and MNIST 70k images
# This kernel is an example of what NOT to do. I did not submit the result from this kernel but if I did, it would score 100% on Kaggle's leaderboard. I performed kNN k=1 with Kaggle's 28,000 "test.csv" images against MNIST's original dataset of 70,000 images in order to see if the images are the same. The result verifies that Kaggle's unknown "test.csv" images are entirely contained unaltered within MNIST's original dataset with known labels. Therefore we CANNOT train with MNIST's original data, we must train our models with Kaggle's "train.csv" 42,000 images, data augmentation,  and/or non-MNIST image datasets.
# 
# # Load MNIST's 70,000 original images

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data('../input/mnist-numpy/mnist.npz')


# In[ ]:


x_train = x_train.reshape(x_train.shape[0],784)
y_train = y_train.reshape(y_train.shape[0],1)
x_test = x_test.reshape(x_test.shape[0],784)
y_test = y_test.reshape(y_test.shape[0],1)
MNIST_image = np.vstack( (x_train,x_test) )
MNIST_label = np.vstack( (y_train,y_test) )


# # Load Kaggle's 28,000 competition test images

# In[ ]:


Kaggle_test_image = pd.read_csv("../input/digit-recognizer/test.csv")
Kaggle_test_image = Kaggle_test_image.values.astype("uint8")
Kaggle_test_label = np.empty( (28000,1), dtype="uint8" )


# # Perform kNN k=1
# For each of Kaggle's 28000 test images, we search within MNIST's original 70000 image dataset for the closest image. If the closest image has distance zero that means the images are exactly the same and the label of the unknown Kaggle test image is precisely the label of the known MNIST image.

# In[ ]:


c1=0; c2=0;
print("Classifying Kaggle's 'test.csv' using kNN k=1 and MNIST 70k images")
for i in range(0,28000): # loop over Kaggle test
    for j in range(0,70000): # loop over MNIST images
        if np.absolute(Kaggle_test_image[i,] - MNIST_image[j,]).sum()==0:
            Kaggle_test_label[i] = MNIST_label[j]
            if i%1000==0:
                print("  %d images classified perfectly" % (i))
            if j<60000:
                c1 += 1
            else:
                c2 += 1
            break
if c1+c2==28000:
    print("  28000 images classified perfectly")
    print("Kaggle's 28000 test images are fully contained within MNIST's 70000 dataset")
    print("%d images are in MNIST-train's 60k and %d are in MNIST-test's 10k" % (c1,c2))


# # Result 100% classification accuracy !!
# Every Kaggle "test.csv" image was found unaltered within MNIST's 70,000 image dataset. Therefore we CANNOT use the original 70,000 MNIST image dataset to train models for Kaggle's MNIST competition. Since MNIST's full dataset contains labels, we would know precisely what each unknown Kaggle test image's label is. We must train our models with Kaggle's "train.csv" 42,000 images, data augmentation,  and/or non-MNIST image datasets. The following csv file would score 100% on Kaggle's public and private leaderboard if submitted.

# In[ ]:


results = pd.Series(Kaggle_test_label.reshape(28000,),name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("Do_not_submit.csv",index=False)


# # Examples

# In[ ]:


# find examples
index = np.zeros(6,dtype='uint')
for i in range(0,6): # loop over Kaggle test
    for j in range(0,70000): # loop over MNIST images
        if np.absolute(Kaggle_test_image[i,] - MNIST_image[j,]).sum()==0:
            index[i] = j
            break
            
# plot examples
import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
for i in range(6):  
    plt.subplot(2, 6, 2*i+1)
    plt.imshow(Kaggle_test_image[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.title("Kaggle test #%d\nLabel unknown" % i,y=0.9)
    plt.axis('off')
    plt.subplot(2, 6, 2*i+2)
    plt.imshow(MNIST_image[index[i]].reshape((28,28)),cmap=plt.cm.binary)
    plt.title("MNIST #%d\nLabel = %d" % (index[i],MNIST_label[index[i]]),y=0.9)
    plt.axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()


# Using kNN k=1, we know with 100% accuracy that Kaggle's first six test images are digits 2, 0, 9, 0, 3, 7 respectively. Similarly we know the next 27994 test images perfectly too.

# # What accuracy is actually possible?
# Here are the best published MNIST classifiers found on the internet:
# * 99.79% [Regularization of Neural Networks using DropConnect, 2013][1]
# * 99.77% [Multi-column Deep Neural Networks for Image Classification, 2012][2]
# * 99.77% [APAC: Augmented PAttern Classification with Neural Networks, 2015][3]
# * **99.76% [Kaggle published kernel, 2018][12]**
# * 99.76% [Batch-normalized Maxout Network in Network, 2015][4]
# * 99.71% [Generalized Pooling Functions in Convolutional Neural Networks, 2016][5]
# * More examples: [here][7], [here][8], and [here][9]  
#   
# On Kaggle's website, there is only one published kernel more accurate than 99.70%. The few other high scoring published kernels train with the full MNIST 70000 image dataset and therefore have actual accuracy under 99.70%.
# [1]:https://cs.nyu.edu/~wanli/dropc/
# [2]:http://people.idsia.ch/~ciresan/data/cvpr2012.pdf
# [3]:https://arxiv.org/abs/1505.03229
# [4]:https://arxiv.org/abs/1511.02583
# [5]:https://arxiv.org/abs/1509.08985
# [7]:http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html
# [8]:http://yann.lecun.com/exdb/mnist/
# [9]:https://en.wikipedia.org/wiki/MNIST_database
# [10]:https://www.kaggle.com/cdeotte/mnist-perfect-100-using-knn/
# [12]:https://www.kaggle.com/cdeotte/35-million-images-0-99757-mnist
