#!/usr/bin/env python
# coding: utf-8

# **First import the necessary modules:**

# In[ ]:


import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
from sklearn.decomposition import PCA
import math 
from sklearn.model_selection import train_test_split


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# **The procedures we will need later in the scope of the project:**

# In[ ]:


bin_n = 16 # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img,SZ):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


def show_digits(lst,features,labels):
    bn=np.array(lst)

    bnm=bn.reshape(math.ceil(len(bn)/2),2)
    x = np.asarray(features)
    y = np.asarray(labels)
    n_col = 8.0
    n_row = np.ceil(len(lst)/n_col/2 )
    
    n_col=int(n_col)
    n_row = int(n_row)
    
    aspect = 1
    n = n_row # number of rows
    m = n_col # numberof columns
    bottom = 0.1; left=0.05
    top=1.-bottom; right = 1.-left
    fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
    #widthspace, relative to subplot size
    wspace=0.15  # set to zero for no spacing
    hspace=wspace/float(aspect)*3
    #fix the figure height
    figheight= 1*n_row # inch
    figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp
    
    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
                        wspace=wspace, hspace=hspace)
    
    i=0
    for ax in axes.flatten():
        if i < len(bnm):
            ax.imshow(x[bnm[i][0],:].reshape(56,56))
            ax.set_title(str(y[bnm[i][0]])+' as '+str(bnm[i][1]),fontsize=8, y=0.97)
            ax.axis('off')
            i+=1
        else: break
    plt.show()

def prep_image(pict,oldSize,newSize):
    #raw = np.asarray(pict)
    image = pict.reshape(oldSize,oldSize)
    rsimage = cv2.resize(image, (0,0), fx=2, fy=2)     
    skewed_image = deskew(rsimage,newSize)
    gray = cv2.GaussianBlur(skewed_image, (3, 3), 0)
    thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            cv2.THRESH_BINARY,11,0)  
    return thresh.flatten()


# **Import the raw data and set feature data and labels for the first 10000 rows:**

# In[ ]:


t0 = time()
print ("read data,resize, skew, blur and threshold it...")
data = pd.read_csv('../input/train.csv')[0:10000] # Read csv file in pandas dataframe
labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

print ("read data, skew, blur and threshold done in [sec]: %0.3fs" % (time() - t0))


# **At that form the feature data would not give out anything, if we do not do any thresholding. So we will do something very simple like the one below:**

# In[ ]:


data[data>0]=1


# **Split the features and labels from each other, again split those in training and test data sets:**

# In[ ]:


from sklearn.cross_validation import train_test_split
t0 = time()
train_images,test_images,train_labels,test_labels=train_test_split(data,labels,train_size=0.8,random_state=0,stratify=labels)

len_test_labels=len(test_labels)
len_train_labels=len(train_labels)

print ("read data, skew, blur and threshold done in [sec]: %0.3fs" % (time() - t0))
print (train_images.shape)
print (test_images.shape)
print (train_labels.shape)
print (test_labels.shape)
print (len_test_labels)
print (len_train_labels)


# **do one hot encoding to categorize the labels(digits). To do that, first we have to merge the splitted train and test labels**

# In[ ]:


merged_labels=np.concatenate((train_labels,test_labels),axis=0)


# **do the one hot encoding and split it back:**

# In[ ]:


ohe = OneHotEncoder()
labels3 = np.array(ohe.fit_transform(merged_labels.reshape(-1, 1)).todense(),dtype = 'uint8') 
train_labels1= labels3[:len_train_labels,:]
test_labels1= labels3[len_train_labels:,:]


# **At the very end we will need to reverse the one hot encoding for some visualization reasons, so below a mapping table is generated:**

# In[ ]:


testlabels_concat=np.append(test_labels1,test_labels[:, np.newaxis], axis=1)
testlabels_concat_unique=unique_rows(testlabels_concat)


# **Let's do a simple linear SVM to find out if the data linear seperable or not. In Scikit-Learn we are not allowed to use svm directly for multiclass classification, so we pack it into "OneVsRestClassifier"**

# In[ ]:


t0 = time()

clf = OneVsRestClassifier(svm.LinearSVC())
clf.fit(train_images, train_labels1)
acc= clf.score(test_images,test_labels1)  

print ("Linear SVC done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)


# **At that point it is obvious that the data is NOT linear seperable since we have a error rate >26%. Lets try it as supposed wth 'rbf' and 'poly' **

# In[ ]:


t0 = time()

clf = OneVsRestClassifier(svm.SVC(kernel='rbf'))
clf.fit(train_images, train_labels1)
acc= clf.score(test_images,test_labels1)  

print (" SVC with 'rbf' kernel done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)


# **SVC with RBF kernel looks good!, andwithout parameter optimization.
# and with polynomial kernel (default degree of polynomial):**

# In[ ]:


t0 = time()

clf = OneVsRestClassifier(svm.SVC(kernel='poly'))
clf.fit(train_images, train_labels1)
acc= clf.score(test_images,test_labels1)  

print (" SVC with 'poly' kernel done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)


# **It is very low. So we will go with 'rbf' kernel.<br> As next we will do dimensionality reduction iterations over PCA  with  SVM(kernel=rbf) to find out the optimal reduced number of features. We are gonna do 14 iterations between explained variance min max values of 0.65 and 0.85 respectively**

# In[ ]:


from sklearn.decomposition import PCA

t0 = time()
n_s = np.linspace(0.65, 0.85, num=14)
accuracy = []
for n in n_s:
    pca = PCA(n_components=n)
    pca.fit(train_images)
    train_images_new = pca.transform(train_images)
    test_images_new = pca.transform(test_images)
    clf = OneVsRestClassifier(svm.SVC(kernel='rbf'))
    clf.fit(train_images_new, train_labels1)
    acc= clf.score(test_images_new,test_labels1)   
    accuracy.append(acc)
    accuracy.append(pca.n_components_)
print ("PCA iterations with SVM (kernel:rbf) takes[sec]: %0.3fs" % (time() - t0))


# **Let us plot the generated accuracies and pick the highest accuracy**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
accuracyarr = np.array(accuracy) 
accuracyarr=accuracyarr.reshape(math.ceil(len(accuracyarr)/2),2)
plt.plot(accuracyarr[:,1],accuracyarr[:,0], 'b-')
print (accuracyarr)


# **by 29 features we reach the highest accuracy of 0.946.So let's fix it by 29 :**

# **In SVM there are an additional parameter we have to look at ,"class_weight". Class weight is per default set to equal but in balanced mode, it adjusts the class weights inversely proportional to class frequencies. Remember the class histogram we plotted, the classes(digits) were a bit unbalanced. So balancing may improve the accuracy.**

# In[ ]:


from sklearn.decomposition import PCA

t0 = time()
n_s = np.linspace(0.65, 0.85, num=21)
accuracy = []
for n in n_s:
    pca = PCA(n_components=n)
    pca.fit(train_images)
    train_images_new = pca.transform(train_images)
    test_images_new = pca.transform(test_images)
    clf = OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced'))
    clf.fit(train_images_new, train_labels1)
    acc= clf.score(test_images_new,test_labels1)   
    accuracy.append(acc)
    accuracy.append(pca.n_components_)
print ("PCA iterations with SVM (kernel:rbf) takes[sec]: %0.3fs" % (time() - t0))


# **Let us plot the generated accuracies and pick the highest accuracy**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
accuracyarr = np.array(accuracy) 
accuracyarr=accuracyarr.reshape(math.ceil(len(accuracyarr)/2),2)
plt.plot(accuracyarr[:,1],accuracyarr[:,0], 'b-')
print (accuracyarr)


# **by 29 features we reach the highest accuracy of 0.9515, which is better than 0.946 by unbalanced class weights. So let's continue with that class_weight='balanced'. But first fix the 29 features: **

# In[ ]:


print ("PCA starts...")
t0 = time()
max_n_components=29
pca = PCA(n_components=max_n_components)
pca.fit(train_images)
train_images_new = pca.transform(train_images)
test_images_new = pca.transform(test_images)
print ("PCA done in %0.3fs" % (time() - t0))


# **As next in order to improve the accuracy we will perform a gridsearch across a bunch of 'C' and 'gamma' parameters to optimize them as discussed before**

# In[ ]:


param_grid = {
    "estimator__C": [.1, 1, 10, 100, 1000],
    "estimator__gamma":[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
}
clf = GridSearchCV(OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced')), param_grid)
clf.fit(train_images_new, train_labels1)
acc= clf.score(test_images_new,test_labels1)  
print (" Gridsearch done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
print (clf.best_params_)


# **let's narrow the search once again around the best_params:**

# In[ ]:


param_grid = {
    "estimator__C": [1,3,7,9, 10,11,13,16,19,22,25,30],
    "estimator__gamma":[0.005,0.007,0.008,0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015]
}
clf = GridSearchCV(OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced')), param_grid)
clf.fit(train_images_new, train_labels1)
acc= clf.score(test_images_new,test_labels1)  
print (" Gridsearch done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
print (clf.best_params_)


# **so now we can fix our C and Gamma as 13 and 0.015 respectively, At which gridsearch seems to bring less since we already had that accuracy 0.9515 at the beginning!<br> Let's do the whole thing with complete data set:**

# In[ ]:


t0 = time()
print ("read data,skew, blur and threshold done in...")
data = pd.read_csv('../input/train.csv') # Read csv file in pandas dataframe
labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

data[data>0]=1

from sklearn.cross_validation import train_test_split
t0 = time()
train_images,test_images,train_labels,test_labels=train_test_split(data,labels,train_size=0.8,random_state=0,stratify=labels)

len_test_labels=len(test_labels)
len_train_labels=len(train_labels)
print ("read data skew, blur and threshold done in : %0.3fs" % (time() - t0))
merged_labels=np.concatenate((train_labels,test_labels),axis=0)
ohe = OneHotEncoder()
labels3 = np.array(ohe.fit_transform(merged_labels.reshape(-1, 1)).todense(),dtype = 'uint8') 
train_labels1= labels3[:len_train_labels,:]
test_labels1= labels3[len_train_labels:,:]

print ("PCA starts...")
t0 = time()
max_n_components=29
pca = PCA(n_components=max_n_components)
pca.fit(train_images)
train_images_new = pca.transform(train_images)
test_images_new = pca.transform(test_images)
print ("PCA done in %0.3fs" % (time() - t0))

testlabels_concat=np.append(test_labels1,test_labels[:, np.newaxis], axis=1)
testlabels_concat_unique=unique_rows(testlabels_concat)

clf = OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced',C=13,gamma=0.015))
clf.fit(train_images_new, train_labels1)
acc= clf.score(test_images_new,test_labels1)  
print (" Classifiying and fitting done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)


# **<br> As next we are gonna go back study image preprocessing to improve accuracy. We have the target 97.5% ! a lot to do!<br> let us randomly look at our digits again:**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
data = shuffle(pd.read_csv('../input/train.csv'))[0:5000]

features=np.array(data.ix[:,1:])
labels = np.array(data.ix[:,0]) 

x = np.array(data,dtype = 'uint8') 

samples=[]
for i in range(0,10):
    samples.append( x[x[:,0] ==i][0:10])

sa =[]
sa = np.vstack(samples)

x=features[0:40]
n_col=10
n_row = 10

aspect = 1
print (aspect)
n = n_row # number of rows
m = n_col # numberof columns
bottom = 0.1; left=0.05
top=1.-bottom; right = 1.-left
fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
#widthspace, relative to subplot size
wspace=0.15  # set to zero for no spacing
hspace=wspace/float(aspect)*3
#fix the figure height
figheight= 1.5*n_row # inch
figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp


fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
                    wspace=wspace, hspace=hspace)

i=0
for ax in axes.flatten():
    if i < len(sa):
        ax.imshow(sa[i,1:].reshape(28,28), cmap='gray_r')
        ax.set_title(str(sa[i,0]),fontsize=8, y=0.97)
        #ax.axis('off')
        ax.spines['bottom'].set_color(None)
        ax.spines['top'].set_color(None)
        ax.spines['right'].set_color(None)
        ax.spines['left'].set_color(None)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        i+=1
    else: break

plt.show()


# **We detect from the digit pictures that they are not uniform and some skewed. Let's look this time the roughly thresholded data before(data[data>0]=1)  **

# In[ ]:


sar=sa
sar[:,1:][sar[:,1:]>0]=1

x=features[0:40]

n_col=10
n_row = 10

aspect = 1
print (aspect)
n = n_row # number of rows
m = n_col # numberof columns
bottom = 0.1; left=0.05
top=1.-bottom; right = 1.-left
fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
#widthspace, relative to subplot size
wspace=0.15  # set to zero for no spacing
hspace=wspace/float(aspect)*3
#fix the figure height
figheight= 1.5*n_row # inch
figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp


fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
                    wspace=wspace, hspace=hspace)

i=0
for ax in axes.flatten():
    if i < len(sar):
        ax.imshow(sar[i,1:].reshape(28,28), cmap='gray_r')
        ax.set_title(str(sar[i,0]),fontsize=8, y=0.97)
        #ax.axis('off')
        ax.spines['bottom'].set_color(None)
        ax.spines['top'].set_color(None)
        ax.spines['right'].set_color(None)
        ax.spines['left'].set_color(None)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        i+=1
    else: break

plt.show()


# **As supposed, it is a very rough thresholding, now the question is, kann we do preprocessing smarter, that will lead to a improvement in accuracy?<br> I tried a lot of computer vision methods to improve the quality of the images as deskewing, resizing, blurring, thresholding, dilating, eroding etc. Dilating, eroding and some others do not help. At the end the following four methods are used: deskewing, resizing, blurring, thresholding.    **

# In[ ]:


bin_n = 16 # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img,SZ):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def prep_image(pict,oldSize,newSize):
    raw = np.asarray(pict)
    image = raw.reshape(oldSize,oldSize)
    rsimage = cv2.resize(image, (0,0), fx=2, fy=2)     
    skewed_image = deskew(rsimage,newSize)
    gray = cv2.GaussianBlur(skewed_image, (3, 3), 0)
    thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            cv2.THRESH_BINARY,11,0)  
    return thresh.flatten()


# **Let's look at the new preprocessed images **

# In[ ]:


x = np.array(data,dtype = 'uint8') 

samples=[]
for i in range(0,10):
    samples.append( x[x[:,0] ==i][0:10])

sa =[]
sa = np.vstack(samples)


newSize=56 #size after resizing, from 28 to 56
newdata_imageProcessed = []
for row in sa[:,1:]:     
     temp =prep_image(row,28,newSize)
     newdata_imageProcessed.append(temp)

sas = np.array(newdata_imageProcessed ).astype(int)
sas[sas>0]=1


x=features[0:40]

n_col=10
n_row = 10

aspect = 1
print (aspect)
n = n_row # number of rows
m = n_col # numberof columns
bottom = 0.1; left=0.05
top=1.-bottom; right = 1.-left
fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
#widthspace, relative to subplot size
wspace=0.15  # set to zero for no spacing
hspace=wspace/float(aspect)*3
#fix the figure height
figheight= 1.5*n_row # inch
figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp


fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
                    wspace=wspace, hspace=hspace)

i=0
for ax in axes.flatten():
    if i < len(sas):
        ax.imshow(sas[i,:].reshape(56,56), cmap='gray_r')
        
        #ax.axis('off')
        ax.spines['bottom'].set_color(None)
        ax.spines['top'].set_color(None)
        ax.spines['right'].set_color(None)
        ax.spines['left'].set_color(None)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        i+=1
    else: break

plt.show()


# **they look better!<br> Let's do the whole thing this time with that image preprocessing:**

# In[ ]:


t0 = time()
print ("read data,resize, skew, blur and threshold it...")
data = pd.read_csv('../input/train.csv')[0:10000] # Read csv file in pandas dataframe
labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

x = np.array(data,dtype = 'uint8') 
SZ=56
newdata_imageProcessed = []
for row in x:     
     temp =prep_image(row,28,SZ)
     newdata_imageProcessed.append(temp)

newdata = np.array(newdata_imageProcessed ).astype(int)
newdata[newdata>0]=1
print ("read data, skew, blur and threshold done in [sec]: %0.3fs" % (time() - t0))



train_images, test_images,train_labels, test_labels = train_test_split(newdata, labels, train_size=0.8, random_state=0,stratify=labels)
len_test_labels=len(test_labels)
len_train_labels=len(train_labels)

merged_labels=np.concatenate((train_labels,test_labels),axis=0)

ohe = OneHotEncoder()
labels3 = np.array(ohe.fit_transform(merged_labels.reshape(-1, 1)).todense(),dtype = 'uint8') 
train_labels= labels3[:len_train_labels,:]
test_labels1= labels3[len_train_labels:,:]


testlabels_concat=np.append(test_labels1,test_labels[:, np.newaxis], axis=1)
testlabels_concat_unique=unique_rows(testlabels_concat)


# **Let's find out again the PCA parameters, this time after image preprocessing:**

# In[ ]:


t0 = time()
n_s = np.linspace(0.65, 0.85, num=14)
accuracy = []
for n in n_s:
    pca = PCA(n_components=n)
    pca.fit(train_images)
    train_images_new = pca.transform(train_images)
    test_images_new = pca.transform(test_images)
    clf = OneVsRestClassifier(svm.SVC(kernel='rbf', class_weight='balanced'))
    clf.fit(train_images_new, train_labels)
    acc= clf.score(test_images_new,test_labels1)   
    accuracy.append(acc)
    accuracy.append(pca.n_components_)
    print(n)
print ("PCA iterations with SVM (kernel:rbf) takes[sec]: %0.3fs" % (time() - t0))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
accuracyarr = np.array(accuracy) 
accuracyarr=accuracyarr.reshape(math.ceil(len(accuracyarr)/2),2)
plt.plot(accuracyarr[:,1],accuracyarr[:,0], 'b-')
print (accuracyarr)


# **since we cannot see accuracy comes to a steady state as the number of features increases, let's increase the sample size by factor 2**

# In[ ]:


t0 = time()
print ("read data,resize, skew, blur and threshold it...")
data = pd.read_csv('../input/train.csv')[0:20000] # Read csv file in pandas dataframe
labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

x = np.array(data,dtype = 'uint8') 
SZ=56
newdata_imageProcessed = []
for row in x:     
     temp =prep_image(row,28,SZ)
     newdata_imageProcessed.append(temp)

newdata = np.array(newdata_imageProcessed ).astype(int)
newdata[newdata>0]=1
print ("read data, skew, blur and threshold done in [sec]: %0.3fs" % (time() - t0))



train_images, test_images,train_labels, test_labels = train_test_split(newdata, labels, train_size=0.8, random_state=0,stratify=labels)
len_test_labels=len(test_labels)
len_train_labels=len(train_labels)

merged_labels=np.concatenate((train_labels,test_labels),axis=0)

ohe = OneHotEncoder()
labels3 = np.array(ohe.fit_transform(merged_labels.reshape(-1, 1)).todense(),dtype = 'uint8') 
train_labels= labels3[:len_train_labels,:]
test_labels1= labels3[len_train_labels:,:]


testlabels_concat=np.append(test_labels1,test_labels[:, np.newaxis], axis=1)
testlabels_concat_unique=unique_rows(testlabels_concat)


# In[ ]:


t0 = time()
n_s = np.linspace(0.65, 0.85, num=11)
accuracy = []
for n in n_s:
    pca = PCA(n_components=n)
    pca.fit(train_images)
    train_images_new = pca.transform(train_images)
    test_images_new = pca.transform(test_images)
    clf = OneVsRestClassifier(svm.SVC(kernel='rbf', class_weight='balanced'))
    clf.fit(train_images_new, train_labels)
    acc= clf.score(test_images_new,test_labels1)   
    accuracy.append(acc)
    accuracy.append(pca.n_components_)
    print("done Iteration for: ",n)
print ("PCA iterations with SVM (kernel:rbf) takes[sec]: %0.3fs" % (time() - t0))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
accuracyarr = np.array(accuracy) 
accuracyarr=accuracyarr.reshape(math.ceil(len(accuracyarr)/2),2)
plt.plot(accuracyarr[:,1],accuracyarr[:,0], 'b-')
print (accuracyarr)


# **by around 150 features the accuracy seems to reach a steady state, so set it by 150 and fix it!**

# In[ ]:


print ("PCA starts...")
t0 = time()
max_n_components=150
pca = PCA(n_components=max_n_components)
pca.fit(train_images)
train_images_new = pca.transform(train_images)
test_images_new = pca.transform(test_images)
print ("PCA done in %0.3fs" % (time() - t0))

t0 = time()
#n_estimators=10
print ("modelling starts...")
clf = OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced'))
clf.fit(train_images_new, train_labels)
acc= clf.score(test_images_new,test_labels1)  
print (" Classifiying and fitting done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)


# **As next in order to improve the accuracy we will perform a gridsearch across a bunch of 'C' and 'gamma' parameters again to optimize them as discussed before**

# In[ ]:


param_grid = {
    "estimator__C": [5, 10, 50],
    "estimator__gamma":[ 0.001, 0.005, 0.01,0.05]
}
clf = GridSearchCV(OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced')), param_grid)
clf.fit(train_images_new, train_labels)
acc= clf.score(test_images_new,test_labels1)  
print (" Gridsearch done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
print (clf.best_params_)


# **further narrow the search:**

# In[ ]:


param_grid = {
    "estimator__C": [8, 10, 15,20],
    "estimator__gamma":[ 0.003, 0.005, 0.007]
}
clf = GridSearchCV(OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced')), param_grid)
clf.fit(train_images_new, train_labels)
acc= clf.score(test_images_new,test_labels1)  
print (" Gridsearch done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
print (clf.best_params_)


# **narrowing the search did not help so let's stick with C=10 and Gamma=0.005 and try with the whole data set.. over 98% accuracy is possible, if you send the whole dataset to kaggle get the result from there. The result we got is as a result of splitting the training data set.**

# In[ ]:


t0 = time()
print ("read data,resize, skew, blur and threshold it...")
data = pd.read_csv('../input/train.csv') # Read csv file in pandas dataframe
labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

x = np.array(data,dtype = 'uint8') 
SZ=56
newdata_imageProcessed = []
for row in x:     
     temp =prep_image(row,28,SZ)
     newdata_imageProcessed.append(temp)

newdata = np.array(newdata_imageProcessed ).astype(int)
newdata[newdata>0]=1
print ("read data, skew, blur and threshold done in [sec]: %0.3fs" % (time() - t0))


t0 = time()
train_images, test_images,train_labels, test_labels = train_test_split(newdata, labels, train_size=0.8, random_state=0,stratify=labels)
len_test_labels=len(test_labels)
len_train_labels=len(train_labels)

merged_labels=np.concatenate((train_labels,test_labels),axis=0)

ohe = OneHotEncoder()
labels3 = np.array(ohe.fit_transform(merged_labels.reshape(-1, 1)).todense(),dtype = 'uint8') 
train_labels= labels3[:len_train_labels,:]
test_labels1= labels3[len_train_labels:,:]


testlabels_concat=np.append(test_labels1,test_labels[:, np.newaxis], axis=1)
testlabels_concat_unique=unique_rows(testlabels_concat)


print ("train-test split and one hot encoding done in [sec]: %0.3fs" % (time() - t0))


# In[ ]:


print ("PCA starts...")
t0 = time()
max_n_components=150
pca = PCA(n_components=max_n_components)
pca.fit(train_images)
train_images_new = pca.transform(train_images)
test_images_new = pca.transform(test_images)
print ("PCA done in %0.3fs" % (time() - t0))
print ("Classifiying and fitting starts...")
t0 = time()
clf = OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced',C=10,gamma=0.005))
clf.fit(train_images_new, train_labels)
acc= clf.score(test_images_new,test_labels1)  
predicted_test_labels1=clf.predict(test_images_new)
print (" Classifiying and fitting done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)


# **results**
