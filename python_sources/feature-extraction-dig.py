#!/usr/bin/env python
# coding: utf-8

# # Draft
# This Kernel is still being developed

# # Table of Contents
# <ul >
# <li> <a href="#intro">Introduction</a></li>
# <li> <a href="#lib">Python Libraries</a></li>
# <li> <a href= "#img">Image Classification</a>
# <ul>
#          <li> <a href="#pre">Preprocessing</a>             
#                    <ul> 
#                    <li><a href="#norm">Normalisation</a></li>
#                     <li><a href="#pcaw">PCA-Whitening</a></li>
#                     </ul>
#             </li>
#     </ul>
# </li>
# <li><a href="#ft1">Feature Extraction</a>
# <ul>
# <li> Old School
# <ul>
# <li><a href="#canny">Canny Edge Detection</a></li>
# <li><a href="#hough">Hough Transform</a></li>
# <li> <a href="#hog">Histogram of Oriented Gradients (HOG)</a></li>
# </ul>
# <li> Deep Learning</li>
# </ul>
# </li>
# </ul>
# 

#  ## <a id="intro">Introduction</a>
# 
# Image classification involves assigning a label/s to an image based on its features. the traditional steps (process pipeline) carried out by an image classification system are:
# <ol>
# <li><strong>Preprocessing</strong>: highly dependant on the image/ how its captured also the features to be extracted, a very common preprocessing operation is normalisation, i.e. subtract mean and devide by standard deviation i.e <strong style="{font-size:17px;}"markdown="1">$\frac {x-\bar{x}}{\sigma}$</strong>. Another is gamma correction</li>
# <li><strong>Feature Extraction </strong>: the design of good features to extract used to play the most critical role in the design of the image classifier. That seems to have changed with the emmergance of deep learning</li>
# <li><strong>Learning Model Selection</strong>: the design of different classifiers with tools such as [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), [Support Vector Machines SVM](https://en.wikipedia.org/wiki/Support_vector_machine), and [Adaboost](https://en.wikipedia.org/wiki/AdaBoost).</li>
# <li><strong>Label Assignment</strong></li>
# </ol>

# Below is a visualisation of the classification process pipline
# <figure style="{padding-bottom:100px;}" markdown="1"><img src='http://www.learnopencv.com/wp-content/uploads/2016/11/image-classification-pipeline.jpg'><figcaption style="text-align: center;padding-bottom:20px;" markdown="1">Image classification pipeline [source](http://www.learnopencv.com/image-recognition-and-object-detection-part1/)</figcaption></figure>

# 
# the main categories of features i.e. 
# <ul>
# <li><b>Low Level Features</b> such as 
# <ul><li>GIST</li><li>SIFT</li> <li>HOG descriptors.</li></ul></li>
# <li><b>Meduim level Features</b> such as
#     <ul> 
#         <li>The Bag-of-Features (BoF) model</li> 
#          <li>The Spatial-Pyramid-Matching (SPM) model</li>
#          <li>the Oriented Pyramid Matching (OPM) model.  [TODO add refrence]</li>
#          </ul>
#    </li>
# <li><b>High-Level Features</b> such as 
# <ul>
# <li>Object Bank</li>
# </ul>
# </li>
# 
# </ul>

# ### <a i="lib">Python Libraries</a>
# [<strong>OpenCV</strong>](http://opencv.org/) is an open source library used mainly by Computer Vision practitionors. I used it extensively in the past with `C++` and briefly with `Python`. so this kernel is really an opportunity for me to trasfer those skills
# I will also be using [scikit image](http://scikit-image.org/)
# also leargist

# In[ ]:


import pandas as pd
import numpy as np
import bson
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import matplotlib.gridspec as gridspec
import cv2 #opencv library 
from skimage import data, io, filters,exposure, feature #essential when extracting features
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from sklearn.decomposition import PCA #princble component analysis
# import keras
# from keras.preprocessing import image as image_utils
from IPython.display import Javascript
# try:
#     import leargist
# except:
#     !conda install leargist -y


# In[ ]:


path = '../input/'
get_ipython().system('ls "$path"')


# ### Read Files

# #### Read bson file and convert to pandas DataFrame

# In[ ]:


with open('{}{}'.format(path,'train_example.bson'),'rb') as b:
    df = pd.DataFrame(bson.decode_all(b.read()))
#read and convert to opencv image
df['imgs'] = df['imgs'].apply(lambda rec: cv2.imdecode(
    np.fromstring(rec[0]['picture'], np.uint8), cv2.IMREAD_COLOR))
#change index to catergory id
df.set_index('category_id',inplace=True)
# Combine images and categries
df.head()


# In[ ]:


df.info()


# ## <a name="img">Image Classification</a>
# ### <a name="pre">Pre-processing</a>
# #### <a name="norm">Normalisation</a>

# Convert to gray scale: this will generate the intensity part of the image using the equation I =0.299R+0.587G+0.114B. If I recall correctly,  the weights in the equation are derived from understaning how each color receptor works in the human visual cotex see [here](https://en.wikipedia.org/wiki/Grayscale) to understand the concept

# In[ ]:


#read the image
image = df.iloc[80,-1]
gray = .299 * image[...,0] + .587 * image[...,1]+ .114* image[...,2]


# Apply <strong style="{font-size:17px;}"markdown="1">$\frac {x-\bar{x}}{\sigma}$</strong>. The mean is subtracted so that the image histogram is centralised around zero and the division by sigma performs the actual normalisation

# In[ ]:


normed_gray = (gray - gray[:].mean())/gray[:].std()


# #### <a name="pcaw">PCA-whitening</a>
# 

# In[ ]:


gray -= gray[:].mean()# zero-center
covariance_matrix = np.dot(gray.transpose(), gray) / gray.shape[0] 
# compute the SVD factorization of the data covariance matrix
U,S,V = np.linalg.svd(covariance_matrix) 
rotation = np.dot(gray, U) # decorrelate the data
whitened_gray = rotation / np.sqrt(S+1e-6)


# In[ ]:


#plot both
sns.set_style('white')
fig, axs = plt.subplots(2,2, figsize=(8,8))
axs = axs.flatten()
imgs = [image ,gray,normed_gray,whitened_gray]
titles = ['Original','GrayScale','Normalised','PCA-Whitened']
for i, ax in enumerate(axs):
    if i<3:
        ax.imshow(imgs[i],aspect='auto',interpolation='nearest')
    else:
        ax.imshow(imgs[i],aspect='auto',interpolation='nearest',cmap='gray')
    ax.set_title(titles[i])
    ax.axis('off')
plt.tight_layout()
plt.show()


# ## <a name="ft1">Feature Extraction</a>
# ### <a name="canny">Canny Edge Detection</a>
# The Process of Canny edge detection algorithm can be broken down to 5 different steps: 
# 
# <ol>
# <li>Apply Gaussian filter to smooth the image in order to remove the noise</li>
# <li>Find the intensity gradients of the image</li>
# <li>Apply non-maximum suppression to get rid of spurious response to edge detection</li>
# <li>Apply double threshold to determine potential edges</li>
# <li>Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.</li>
# </ol>
# <br/>
# Source: [https://en.wikipedia.org/wiki/Canny_edge_detector](https://en.wikipedia.org/wiki/Canny_edge_detector)

# In[ ]:


#%%javascript
#// $('#help').height(20);


# In[ ]:


edges = feature.canny(normed_gray,low_threshold =2,sigma=.5)
# help(feature.canny)
# #create refrence to style output
# Javascript('this.element.attr("id", "help")')


# ### <a name="hough">Hough Transform</a>
# The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure. This voting procedure is carried out in a parameter space, from which object candidates are obtained as local maxima in a so-called accumulator space that is explicitly constructed by the algorithm for computing the Hough transform.
# 
# The classical Hough transform was concerned with the identification of lines in the image, but later the Hough transform has been extended to identifying positions of arbitrary shapes, most commonly circles or ellipses. 
# <br/><br/>Source:  [https://en.wikipedia.org/wiki/Hough_transform](https://en.wikipedia.org/wiki/Hough_transform)

# In[ ]:





# In[ ]:


lines = probabilistic_hough_line(edges, threshold=20, line_length=10,
                              line_gap=3)
# help(probabilistic_hough_line)  
# Javascript('this.element.attr("id", "help")')


# In[ ]:


sns.set_style('white')
# Create 2x3 sub plots
gs = gridspec.GridSpec(2, 2)

plt.figure(figsize=(10,7))
ax = plt.subplot(gs[0, 0]) # row 0, col 0
ax.imshow(normed_gray, cmap=cm.gray,
          aspect=1,
          interpolation='nearest')
ax.set_title('Grayscale image')
ax.axis('off')

ax = plt.subplot(gs[1, 0]) # row 1, col 0
ax.imshow(edges, cmap=cm.gray)
ax.set_title('Canny edges')
ax.axis('off')

# ax = plt.subplot(gs[:, 1]) # col 1, span all rows
# ax.imshow(edges * 0, cmap=cm.gray,aspect='auto',interpolation='nearest')
# ax.set_title('Hough Transform')
# ax.axis('off')

ax = plt.subplot(gs[:, 1]) # col 1, span all rows
ax.imshow(gray,
          aspect=1,
          interpolation='nearest')
for line in lines:
    p0, p1 = line
    ax.plot((p0[0], p1[0]), (p0[1], p1[1]),color='r',lw=3)

ax.set_title('Hough Transform overlaid')
ax.axis('off')

plt.tight_layout()
plt.show()


# ### <a name="hog">Extract Histogram of Oriented Gradients (HOG)</a>
# 
# Compute a Histogram of Oriented Gradients (HOG) by
# <ol>
# <li>global image normalisation</li>
# <li>computing the gradient image in x and y</li>
# <li>computing gradient histograms</li>
# <li>normalising across blocks</li>
# <li>flattening into a feature vector</li>
# </ol>

# In[ ]:


points_of_intrest, hog_img = feature.hog(normed_gray,orientations=8, pixels_per_cell=(12, 12),
                    cells_per_block=(1, 1), visualise=True,block_norm='L2-Hys')

# Rescale histogram for better display
hogrescaled = exposure.rescale_intensity(hog_img, in_range=(0, 0.02))
# help(feature.hog)


# In[ ]:


sns.set_style('white')
# Create 2x3 sub plots
gs = gridspec.GridSpec(2, 2)

plt.figure(figsize=(10,7))
ax = plt.subplot(gs[0, 0]) # row 0, col 0
ax.imshow(normed_gray, cmap=cm.gray,
          aspect=1,
          interpolation='nearest')
ax.set_title('Grayscale image')
ax.axis('off')

ax = plt.subplot(gs[1, 0]) # row 1, col 0
ax.imshow(hog_img, cmap=cm.gray)
ax.set_title('hog edges')
ax.axis('off')

### display 2 images on top of each other
def func3(x, y):
    return (1 - x/2 + x**5 + y**3)*np.exp(-(x**2 + y**2))

# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

x = np.arange(-3.0, 3.0, dx)
y = np.arange(-3.0, 3.0, dy)
X, Y = np.meshgrid(x, y)

# when layering multiple images, the images need to have the same
# extent.  This does not mean they need to have the same shape, but
# they both need to render to the same coordinate system determined by
# xmin, xmax, ymin, ymax.  Note if you use different interpolations
# for the images their apparent extent could be different due to
# interpolation edge effects


xmin, xmax, ymin, ymax = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
extent = xmin, xmax, ymin, ymax
ax = plt.subplot(gs[:, 1]) # col 1, span all rows
ax.imshow(gray,
          aspect=1,
                 extent=extent)
ax.imshow((hog_img>.05)*255,
                 alpha=.4, interpolation='bilinear',
                     extent=extent,cmap=cm.Reds)

ax.set_title('HOG Transform overlaid')
ax.axis('off')

plt.tight_layout()
plt.show()

