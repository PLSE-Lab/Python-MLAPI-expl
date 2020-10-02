#!/usr/bin/env python
# coding: utf-8

# I am trying to  go about with the MNIST data classification task using Python.
# 
# Although I could have forked existing kernels for this task, I really wanted to build a kernel ground up and this seemed the easiest data to work off. Also I am using [this great kernel as my source](https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification). 
# I started with creating a new kernel and then adding the data under 'Draft Environment' and selecting the Digit Recognizer data. 
# 
# 
# 

# In[ ]:



import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt,matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#%matplotlib is a magic function in IPython
#%matplotlib inline sets the backend of 
#matplotlib to the 'inline' backend: 
#With this backend, the output of plotting commands
#is displayed inline within frontends like the Jupyter notebook, 
#directly below the code cell that produced it.
get_ipython().run_line_magic('matplotlib', 'inline')

#constants
NUMBER_OF_TRAINING_IMGS=5000
IMG_HEIGHT=28
IMG_WIDTH=28
#list all the contents of the ../input directory
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Load the training data in and create a dataframe from the csv. 
# The first column in the ground truth label. The remaining are pixel values
# The images are 28X28 pixels and these pixels are unrolled. See the data section to see more details.
# 
# The data is of the format: 
# Label, pixel0, pixel1, pixel2,..., pixel783
# 
# Loading the data is done as follows:
# 1. Read the csv using pandas read_csv to read into a dataframe
# 2. separate into images ( design matrix X) and labels (Y)
# 3. Split into training and test sets
# 

# In[ ]:


labeled_images=pd.read_csv('../input/train.csv')
print('shape of the dataframe ',labeled_images.shape)
labeled_images.head(n=3)


# *pandas loc is used to subset rows by labels while iloc is used to subset rows by row numbers.  
# By default(?) the dataframe rows are assigned labels set to the row numbers.   
# Labels need not always be equal to the row numbers in all cases.   
# (row_number->label   0->A,1->B,2->C  or 0->Z,1->Y,2->X and so on).   
# loc => labels  
# iloc => row numbers*

# In[ ]:


images=labeled_images.iloc[0:NUMBER_OF_TRAINING_IMGS,1:] # first NUMBER_OF_TRAINING_IMGS rows,column 2 onwards.
labels=labeled_images.iloc[0:NUMBER_OF_TRAINING_IMGS,:1] #first NUMBER_OF_TRAINING_IMGS rows, first column. 
                                                        #I could have used .iloc[0:NUMBER_OF_TRAINING_IMGS,0 ] 
                                                        #instead of        .iloc[0:NUMBER_OF_TRAINING_IMGS,:1] but the first case returns a Series and the second one a DataFrame.
                                                        #prefer the latter.
train_images,test_images,train_labels,test_labels=train_test_split(images,labels,test_size=0.2,random_state=13)


# The output type from  [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) is the same as the input type.  

# In[ ]:


type(train_images)


# 
# **Now to viewing an image**[](http://).  
# The image is unrolled into a single row that represents an image. Convert the single data point (row) to a 28X28 matrix (to view the original image)
# And then use matplotlib to plot this matrix.

# In[ ]:


ii=13
img=train_images.iloc[ii].values
print('shape of numpy array',img.shape,'reshaping to 28x28')
img=img.reshape(IMG_HEIGHT,IMG_WIDTH)
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[ii,0])


# **Statistics**  
# how are the pixel values distributed?   
# 

# In[ ]:


plt.hist(train_images.iloc[ii].values)


# More darker pixel values (close to 0 ) as compared to lighter( closer to 256).  
# Each of the pixels is a feature, so the data point is a 784-Dimensional vector.  
# We need to standardize the features .   
# 
# 

# In[ ]:


train_images.describe()


# [How to standardize a dataframe while keeping the header information](https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame)  
# The training set is assumed variable enough to be that all data (test and new) comes from a distribution similar to the training set distribution.  
# Therefore I am saving  the max  and the min values  from the training data distribution.  
# max and min will then be used to perform the standardization of all the data.

# In[ ]:


max=train_images.max()
min=train_images.min()
train_images_std=(train_images-min)/(max-min)
test_images_std=(test_images-min)/(max-min)
train_images_std.head(n=2)


# > division by 0  causes  [NaNs](https://stackoverflow.com/questions/13295735/how-can-i-replace-all-the-nan-values-with-zeros-in-a-column-of-a-pandas-datafra).  and [infinities](https://stackoverflow.com/questions/17477979/dropping-infinite-values-from-dataframes-in-pandas) that need to be replaced
# 

# In[ ]:


train_images_std=train_images_std.replace([-np.inf,np.inf],np.nan)
test_images_std=test_images_std.replace([-np.inf,np.inf],np.nan)
train_images_std=train_images_std.fillna(0)
test_images_std=test_images_std.fillna(0)
train_images_std.head(n=2)


# what does the distribution look like now? 

# In[ ]:


plt.hist(train_images_std.iloc[13])


# **Using [Support Vector Machines](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)**  
# 
# 

# In[ ]:


test_images_std.describe()


# In[ ]:


clf=SVC(kernel='rbf',C=1.0,random_state=1,gamma=0.1)   # radial basis function K(a,b) = exp(-gamma * ||a-b||^2 ) where gamma=1/(2*stddev)^2
clf.fit(train_images_std,train_labels.values.ravel())
clf.score(test_images_std,
          test_labels.values.ravel())


# **81%** accuracy with scaled data.   
# Now compare this with the case below where the data is used as is. (no scaling)

# In[ ]:


## Will not be using this classifier. This  is to test the need for scaling in SVM

clf_without_std=SVC(kernel='rbf',C=1.0,random_state=1,gamma=0.1)   # radial basis function K(a,b) = exp(-gamma * ||a-b||^2 ) where gamma=1/(2*stddev)^2
clf_without_std.fit(train_images,train_labels.values.ravel())      # unscaled original  data.
clf_without_std.score(test_images,
          test_labels.values.ravel())


# **10%** accuracy  
# This is because the radial basis function kernel aka the gaussian kernel is of the the form K(a,b) = exp(-gamma * ||a-b||^2 ) i.e the L2 norm is used between data points to determine similarity.   
# And when the L2 norm is been used, it's best to keep the scaling similar between features vectors .

# 
# **Submission**  
# 
# load the data in test.csv  
# scale it using the min and max values derived from the training set.  
# remove Nans and infinities as before.  
# 

# In[ ]:


#use the clf model. (trained on scaled data)
raw_data=pd.read_csv('../input/test.csv')
print('shape of dataframe ',raw_data.shape)
raw_data.head(n=3)


# In[ ]:


raw_data_scaled=(raw_data-min)/(max-min)
raw_data_scaled=raw_data_scaled.replace([-np.inf,np.inf],np.nan)
raw_data_scaled=raw_data_scaled.fillna(0)
raw_data_scaled.describe()


# Spot check to see if the clf predicts correctly on new data.  
# View an image from the raw data and see if the prediction matches expected value

# In[ ]:


from random import randint

jj=13

for i in range(1,3):
    jx=randint(0,raw_data_scaled.shape[0])
    smp=raw_data_scaled.iloc[[jx]]
    img=smp.values        #get a random image
    img=img.reshape(IMG_HEIGHT,IMG_WIDTH)
    for j in range(1,3):
        for k in range(1,2):
            print('plt',i,j,k)
            plt.subplot(i,j,k)
            plt.imshow(img)
            y=clf.predict(smp)
            plt.title('predicted : '+str(y[0]))
            

'''
img=raw_data_scaled.iloc[[jj]].values
img=img.reshape(IMG_HEIGHT,IMG_WIDTH)
img1=raw_data_scaled.iloc[jj+1].values
img1=img1.reshape(IMG_HEIGHT,IMG_WIDTH)
plt.subplot(121)
plt.imshow(img)
plt.title('predicted: ')
plt.subplot(122)
plt.imshow(img1)
plt.title('predicted: ')
'''


# In[ ]:


jjthSample=raw_data_scaled.iloc[[jj]]     # iloc[j] returns a Series. We need a dataframe to pass to predict. which is returned by iloc[[jj]]  .i.e. a list of rows
type(jjthSample)
y_pred_jj=clf.predict(jjthSample)
y_pred_jj


# In[ ]:


y_pred=clf.predict(raw_data_scaled)


# In[ ]:


y_pred.shape


# In[ ]:


submissions=pd.DataFrame({"ImageId":list(range(1,len(y_pred)+1)), "Label":y_pred})
submissions.head()


# In[ ]:


submissions.to_csv("mnist_svm_submit.csv",index=False,header=True)


# In[ ]:


get_ipython().system('ls')


# In[ ]:




