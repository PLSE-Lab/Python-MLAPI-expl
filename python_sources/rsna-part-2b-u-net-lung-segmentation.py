#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# To maximize our search I think it might be advantageous to perform some segmentation on the dataset before applying and algorithm to identify bounding boxes.  I know this might be tedious but coming from the classical ML approach where by features are extracted from the data domain and classification is done on the feature space.  I think this approach is far more intuitve for me at the moment.  Therefore, I propose the following steps:
# 
# 1. Pre-Processing
#     - Masking the image based on the distribution of bounding boxes
# 2. Feature Extraction
#     - Extract histogram as a feature for each of the test cases
# 3. Classification
#     - Use classification techniques to classify normal vs. ( opaque, non-opaque) cases
#  
#  **This notebook will only focus on the pre-processing aspects of this workflow using another dataset to generate the lung segmentation portion.  We can further retrain the model on our model and see if it has improved. **

# # Imports 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,sys

import pydicom
import tensorflow as tf

from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize

from glob import glob

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# # Pre-Processing Step

# ## Reading the Data
# 
# Let us use the data uploaded by Kevin Mader [kernel](https://www.kaggle.com/kmader/gaussian-mixture-lung-segmentation) to train our segmentation U-Net algorithm.  It is noted that he has also done this but uses code that I cannot follow at the moment due to my lack of inexperiance but please refer to his guide.

# In[ ]:


cxr_paths = glob(os.path.join('..', 'input', 'pulmonary-chest-xray-abnormalities','Montgomery', 'MontgomerySet', '*', '*.png'))
print("There are {} images in the dataset".format(len(cxr_paths)))


# Obtain the masks for each 138 patients. 

# In[ ]:


cxr_images = [(c_path, 
               [os.path.join('/'.join(c_path.split('/')[:-2]),'ManualMask','leftMask', os.path.basename(c_path)),
               os.path.join('/'.join(c_path.split('/')[:-2]),'ManualMask','rightMask', os.path.basename(c_path))]
              ) for c_path in cxr_paths]


# An example patient in the list has the structures below.  
# 
# cxr_images[ **#ofPatients**][**0 = Location of the Patient Original Scan**][** 0/1 = Mask for the left and right lung respectively**]

# In[ ]:


cxr_images[0]


# ## Reading Imports
# 
# We need more imports for reading the actual data

# In[ ]:


from skimage.io import imread as imread_raw
from skimage.transform import resize
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore', category=UserWarning, module='skimage') # skimage is really annoying


# These images are 512 x 512 images so let us create a global variable so that we can follow it.

# ## Function to read the images

# In[ ]:


def imread(in_path):
    OUT_DIM = (512, 512)
    
    # use the skimge function to read the file specified in the path
    img_data = imread_raw(in_path)
    
    # make sure the image data is between the range 0-255 and convert the variable into uint8
    n_img = (255*resize(img_data, OUT_DIM, mode = 'constant')).clip(0,255).astype(np.uint8)
        
    return np.expand_dims(n_img, -1)


# In[ ]:


# init empty array for images and masks or in this case segmentations
img_vol, seg_vol = [], []

for img_path, s_paths in tqdm(cxr_images):
    # first read the image paths
    img_vol += [imread(img_path)]    
    
    # read both images, stack them up, then store them    
    seg_vol += [np.max(np.stack([imread(s_path) for s_path in s_paths],0),0)]
    
img_vol = np.stack(img_vol,0)
seg_vol = np.stack(seg_vol,0)

print('Images', img_vol.shape, 'Segmentations', seg_vol.shape)


# As you can see from the above line there are 138 patients consisting of images with a 512x512 image and segmentation corresponding to the labeled lung segmentation.  Let us look at them side by side to get a better intuitive understading of the above images.

# In[ ]:


# Get a random patient
np.random.seed(64)

randomPatient = int(np.random.rand()*138)
t_img, m_img = img_vol[randomPatient], seg_vol[randomPatient]

#plot it
scan = t_img[:,:,0]
mask = m_img[:,:,0]
segmented = scan*mask

def drawImage(ax_img,img,label):
    ax_img.imshow(img,interpolation='none',cmap='bone')
    ax_img.set_title(label)
    ax_img.set_axis_off()
    

fig, (ax_img, ax_mask,ax_segmentedImage) = plt.subplots(1,3, figsize = (12, 6))
drawImage(ax_img,scan,label='CXR')
drawImage(ax_mask,mask,label='Labeled Mask')
drawImage(ax_segmentedImage,segmented,label='Segmented Image')


# ## Mask Generation

# In this section we will use the [U-Net algorithm](https://arxiv.org/abs/1505.04597) for segmenting the lungs.
# 
# For some guidance we will use the following diagram from the article and also use the working sample by Jae Duk Seo in his [article](https://towardsdatascience.com/medical-image-segmentation-part-1-unet-convolutional-networks-with-interactive-code-70f0f17f46c6).
# 
# Note the colouration as they will be useful when generating the layers.
# 
# ![img2](https://cdn-images-1.medium.com/max/1600/1*aRMefObpm7AMVOZYYiQAMQ.png)  
# 
# 

# ## Set up environment and utility functions

# In[ ]:


tf.set_random_seed(6464)


# ### Activation Functions

# In[ ]:


def tf_relu(x): 
    return tf.nn.relu(x)

def d_tf_relu(s): 
    return tf.cast(tf.greater(s,0),dtype=tf.float32)

def tf_softmax(x): 
    return tf.nn.softmax(x)

def np_sigmoid(x): 
    1/(1 + np.exp(-1 *x))


# ### Generate Classes for Convolution Layers

# In[ ]:


class conlayer_left():
    
    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.05))

    def feedforward(self,input,stride=1,dilate=1):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding='SAME')
        self.layerA = tf_relu(self.layer)
        return self.layerA

class conlayer_right():
    
    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.05))

    def feedforward(self,input,stride=1,dilate=1,output=1):
        self.input  = input

        current_shape_size = input.shape

        self.layer = tf.nn.conv2d_transpose(input,self.w,
        output_shape=[batch_size] + [int(current_shape_size[1].value*2),int(current_shape_size[2].value*2),int(current_shape_size[3].value/2)],strides=[1,2,2,1],padding='SAME')
        self.layerA = tf_relu(self.layer)
        return self.layerA


# ## Normalize the data

# In[ ]:


train_images = (img_vol - img_vol.min()) / (img_vol.max() - img_vol.min())
train_labels = (seg_vol - seg_vol.min()) / (seg_vol.max() - seg_vol.min())

print('Images', img_vol.shape, 'Segmentations', seg_vol.shape)


# ## Hyper-Parameters

# In[ ]:


num_epoch = 100
init_lr = 0.0001
batch_size = 2


# ## Generate Layers

# ### Left Side of th U-Net architecutre or Red in the above image

# In[ ]:


l1_1 = conlayer_left(3,1,3)
l1_2 = conlayer_left(3,3,3)
l1_3 = conlayer_left(3,3,3)

l2_1 = conlayer_left(3,3,6)
l2_2 = conlayer_left(3,6,6)
l2_3 = conlayer_left(3,6,6)

l3_1 = conlayer_left(3,6,12)
l3_2 = conlayer_left(3,12,12)
l3_3 = conlayer_left(3,12,12)

l4_1 = conlayer_left(3,12,24)
l4_2 = conlayer_left(3,24,24)
l4_3 = conlayer_left(3,24,24)

l5_1 = conlayer_left(3,24,48)
l5_2 = conlayer_left(3,48,48)
l5_3 = conlayer_left(3,48,24)


# ### Right Side of the U-Net Architecture or Blue in the above image
# 

# In[ ]:


# right
l6_1 = conlayer_right(3,24,48)
l6_2 = conlayer_left(3,24,24)
l6_3 = conlayer_left(3,24,12)

l7_1 = conlayer_right(3,12,24)
l7_2 = conlayer_left(3,12,12)
l7_3 = conlayer_left(3,12,6)

l8_1 = conlayer_right(3,6,12)
l8_2 = conlayer_left(3,6,6)
l8_3 = conlayer_left(3,6,3)

l9_1 = conlayer_right(3,3,6)
l9_2 = conlayer_left(3,3,3)
l9_3 = conlayer_left(3,3,3)


# ### Combining layer or bottle neck layer or Green in the above image

# In[ ]:


l10_final = conlayer_left(3,3,1)


# ## Generate architecture

# In[ ]:


x = tf.placeholder(shape=[None,512,512,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,512,512,1],dtype=tf.float32)

layer1_1 = l1_1.feedforward(x)
layer1_2 = l1_2.feedforward(layer1_1)
layer1_3 = l1_3.feedforward(layer1_2)

layer2_Input = tf.nn.max_pool(layer1_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer2_1 = l2_1.feedforward(layer2_Input)
layer2_2 = l2_2.feedforward(layer2_1)
layer2_3 = l2_3.feedforward(layer2_2)

layer3_Input = tf.nn.max_pool(layer2_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3_1 = l3_1.feedforward(layer3_Input)
layer3_2 = l3_2.feedforward(layer3_1)
layer3_3 = l3_3.feedforward(layer3_2)

layer4_Input = tf.nn.max_pool(layer3_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4_1 = l4_1.feedforward(layer4_Input)
layer4_2 = l4_2.feedforward(layer4_1)
layer4_3 = l4_3.feedforward(layer4_2)

layer5_Input = tf.nn.max_pool(layer4_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer5_1 = l5_1.feedforward(layer5_Input)
layer5_2 = l5_2.feedforward(layer5_1)
layer5_3 = l5_3.feedforward(layer5_2)

layer6_Input = tf.concat([layer5_3,layer5_Input],axis=3)
layer6_1 = l6_1.feedforward(layer6_Input)
layer6_2 = l6_2.feedforward(layer6_1)
layer6_3 = l6_3.feedforward(layer6_2)

layer7_Input = tf.concat([layer6_3,layer4_Input],axis=3)
layer7_1 = l7_1.feedforward(layer7_Input)
layer7_2 = l7_2.feedforward(layer7_1)
layer7_3 = l7_3.feedforward(layer7_2)

layer8_Input = tf.concat([layer7_3,layer3_Input],axis=3)
layer8_1 = l8_1.feedforward(layer8_Input)
layer8_2 = l8_2.feedforward(layer8_1)
layer8_3 = l8_3.feedforward(layer8_2)

layer9_Input = tf.concat([layer8_3,layer2_Input],axis=3)
layer9_1 = l9_1.feedforward(layer9_Input)
layer9_2 = l9_2.feedforward(layer9_1)
layer9_3 = l9_3.feedforward(layer9_2)

layer10 = l10_final.feedforward(layer9_3)

cost = tf.reduce_mean(tf.square(layer10-y))
auto_train = tf.train.AdamOptimizer(learning_rate=init_lr).minimize(cost)


# In[ ]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        # train
        for current_batch_index in range(0,len(train_images),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
            sess_results = sess.run([cost,auto_train],feed_dict={x:current_batch,y:current_label})
            print(' Iter: ', iter, " Cost:  %.32f"% sess_results[0],end='\r')
        print('\n-----------------------')
        train_images,train_labels = shuffle(train_images,train_labels)

        if iter % 10 == 0:
            test_example =   train_images[:2,:,:,:]
            test_example_gt = train_labels[:2,:,:,:]
            sess_results = sess.run([layer10],feed_dict={x:test_example})

            sess_results = sess_results[0][0,:,:,:]
            test_example = test_example[0,:,:,:]
            test_example_gt = test_example_gt[0,:,:,:]

            plt.figure()
            plt.imshow(np.squeeze(test_example),cmap='gray')
            plt.axis('off')
            plt.title('Original Image')  
            plt.savefig(str(iter)+"a_Original_Image.png")

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt),cmap='gray')
            plt.axis('off')
            plt.title('Ground Truth Mask')  
            plt.savefig(str(iter)+"b_Original_Mask.png")            

            plt.figure()
            plt.imshow(np.squeeze(sess_results),cmap='gray')
            plt.axis('off')
            plt.title('Generated Mask') 
            plt.savefig(str(iter)+"c_Generated_Mask.png")

            plt.figure()
            plt.imshow(np.multiply(np.squeeze(test_example),np.squeeze(test_example_gt)),cmap='gray')
            plt.axis('off')
            plt.title("Ground Truth Overlay")   
            plt.savefig(str(iter)+"d_Original_Image_Overlay.png")

            plt.figure()
            plt.axis('off')
            plt.imshow(np.multiply(np.squeeze(test_example),np.squeeze(sess_results)),cmap='gray')
            plt.title("Generated Overlay")                 

            plt.close('all')


# Look at the ouputs to view the results of the U-Net alogirthm
