#!/usr/bin/env python
# coding: utf-8

# # Malaria Cell Images
# ## Exploratory Data Analysis

# Our primary focus is to classify the malaria cell images as 'Parasitized' or 'Uninfected'. However, with a little experimentation, visualization and analysis, we can get a deeper meaning of the provided images, which in turn might help us in creating better features for a more efficient model...
# 
# In this notebook, much of the image analysis has been done using the 'skimage' and 'PIL' module. Let us start off by importing the essential models...

# In[ ]:


# For Managing Datasets
import numpy as np
import pandas as pd


# In[ ]:


# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Here, I've set a random seed to obtain reproducible results. However, the general results that we obtain in this case should also apply for a different random seed value...

# In[ ]:


import random
random_seed=33


# In[ ]:


# To help reduce memory usage
import gc
import time
def cleanup():
    gc.collect()
    time.sleep(10)


# # Loading Data

# Let us now load the data. Data has been provided to us in two different folders containing parasitized and uninfected data. Also, for uniformity, we reshape all the images into a 125x125x3 numpy array.

# In[ ]:


from PIL import Image
import os
from tqdm import tqdm_notebook

main_address='../input/cell_images/cell_images/Parasitized'
images_address=os.listdir(main_address)
images_address.remove('Thumbs.db')
parasitized_data=np.zeros((len(images_address),125,125,3),dtype=np.int16)

print("Importing Parasitized Data...")
for ind,img_address in tqdm_notebook(enumerate(images_address),total=len(images_address)):
    img=Image.open(main_address+'/'+img_address)
    img=img.resize((125,125),Image.ANTIALIAS)
    img=np.asarray(img)
    img=img.astype(np.int16)
    parasitized_data[ind]=img
print("Done Importing Parasitized Data!")
cleanup()


# In[ ]:


np.random.seed(random_seed)
random_10=np.random.randint(0,parasitized_data.shape[0],size=10)
fig=plt.figure(figsize=(10,5))
plt.title("Examples of Parasitized Body Cells",fontsize=20)
plt.axis('off')
for ind,rand in enumerate(random_10):
    ax_n=fig.add_subplot(2,5,ind+1)
    ax_n.imshow(parasitized_data[rand])
    ax_n.get_xaxis().set_visible(False)
    ax_n.get_yaxis().set_visible(False)
plt.show()


# In[ ]:


main_address='../input/cell_images/cell_images/Uninfected'
images_address=os.listdir(main_address)
images_address.remove('Thumbs.db')
uninfected_data=np.zeros((len(images_address),125,125,3),dtype=np.int16)

print("Importing Uninfected Data...")
for ind,img_address in tqdm_notebook(enumerate(images_address),total=len(images_address)):
    img=Image.open(main_address+'/'+img_address)
    img=img.resize((125,125),Image.ANTIALIAS)
    img=np.asarray(img)
    img=img.astype(np.int16)
    uninfected_data[ind]=img
print("Done Importing Uninfected Data!")
cleanup()


# In[ ]:


np.random.seed(random_seed)
random_10=np.random.randint(0,uninfected_data.shape[0],size=10)
fig=plt.figure(figsize=(10,5))
plt.title("Examples of Uninfected Body Cells",fontsize=20)
plt.axis('off')
for ind,rand in enumerate(random_10):
    ax_n=fig.add_subplot(2,5,ind+1)
    ax_n.imshow(uninfected_data[rand])
    ax_n.get_xaxis().set_visible(False)
    ax_n.get_yaxis().set_visible(False)
plt.show()


# One immediate observation we notice is that parasitized cells seem to have these different-coloured spots or patches, while the uninfected ones seem to be uniform and clear. It is this observation that we will work on later...

# # Creating Training & Testing Data

# Right now, we have both parasitized and uninfected data in separate datasets. However, we will try to randomly shuffle both datasets, mix them to form 'train' and 'test' datasets and proceed as if we were provided with training and testing datasets itself. I've tried to make it as fair as possible, but any fairer method to  create the training and testing datasets are appreciated...

# In[ ]:


parasitized_indices=np.arange(parasitized_data.shape[0])
uninfected_indices=np.arange(uninfected_data.shape[0])

np.random.seed(random_seed)
np.random.shuffle(parasitized_indices)
np.random.seed(random_seed)
np.random.shuffle(uninfected_indices)

parasitized_data=parasitized_data[parasitized_indices]
uninfected_data=uninfected_data[uninfected_indices]


# In[ ]:


train_test_split_ratio=3/4

parasitized_train=parasitized_data[:int(train_test_split_ratio*parasitized_data.shape[0])]
uninfected_train=uninfected_data[:int(train_test_split_ratio*uninfected_data.shape[0])]

parasitized_test=parasitized_data[int(train_test_split_ratio*parasitized_data.shape[0]):]
uninfected_test=uninfected_data[int(train_test_split_ratio*uninfected_data.shape[0]):]


# In[ ]:


train=np.append(parasitized_train,uninfected_train,axis=0)
test=np.append(parasitized_test,uninfected_test,axis=0)


# In[ ]:


train_labels=np.array([1]*parasitized_train.shape[0]+[0]*uninfected_train.shape[0])
test_labels=np.array([1]*parasitized_test.shape[0]+[0]*uninfected_test.shape[0])


# In[ ]:


fig=plt.figure(figsize=(10,5))
ax1=fig.add_subplot(121)
ax1.pie([parasitized_train.shape[0],uninfected_train.shape[0]],explode=[0,0.03],startangle=90,colors=['red','blue'])
ax1.set_title("Distribution of Parasitized & Uninfected Data")
ax1.legend(['Parasitized','Uninfected'],loc='best')
ax2=fig.add_subplot(122)
ax2.pie([train.shape[0],test.shape[0]],explode=[0,0.03],startangle=90,colors=['blue','red'])
ax2.set_title("Distribution of Train & Test Data")
ax2.legend(['Train','Test'],loc='best')
plt.show()


# As we can see, the ratio of total parasitized and uninfected data is one. Also I've used the classic 75-25 percent split between training and testing data. Now let us mix train and test further...

# In[ ]:


train_index=np.arange(train.shape[0])
np.random.seed(random_seed)
np.random.shuffle(train_index)
train=train[train_index]
train_labels=train_labels[train_index]

test_index=np.arange(test.shape[0])
np.random.seed(random_seed)
np.random.shuffle(test_index)
test=test[test_index]
test_labels=test_labels[test_index]


# In[ ]:


cleanup()


# # Final Training Data
# 
# ### Let's have a look at some random samples of train and test datasets...

# In[ ]:


np.random.seed(random_seed)
random_10=np.random.randint(0,train.shape[0],size=10)
fig=plt.figure(figsize=(10,5))
plt.title("Examples of Training Data",fontsize=20)
plt.axis('off')
for ind,rand in enumerate(random_10):
    ax_n=fig.add_subplot(2,5,ind+1)
    ax_n.imshow(train[rand])
    if(train_labels[rand]==1):
        ax_n.set_title("Parasitized")
    else:
        ax_n.set_title("Uninfected")
    ax_n.get_xaxis().set_visible(False)
    ax_n.get_yaxis().set_visible(False)
plt.show()


# In[ ]:


np.random.seed(random_seed)
random_10=np.random.randint(0,test.shape[0],size=10)
fig=plt.figure(figsize=(10,5))
plt.title("Examples of Testing Data",fontsize=20)
plt.axis('off')
for ind,rand in enumerate(random_10):
    ax_n=fig.add_subplot(2,5,ind+1)
    ax_n.imshow(test[rand])
    if(test_labels[rand]==1):
        ax_n.set_title("Parasitized")
    else:
        ax_n.set_title("Uninfected")
    ax_n.get_xaxis().set_visible(False)
    ax_n.get_yaxis().set_visible(False)
plt.show()


# # Analysis of Blood Cells

# Let us now analyze various features of the images in order to gain more insight into what exactly differentiates a parasitized cell from an infected cell...

# ## 'Average' Image
# 
# Let us find what I like to call the 'Average Image' which is basically an image where each component of each pixel is the average of the corresponding values of all images in the data...

# In[ ]:


def plot_avg_image(data,ax=None,title='',to_return=False):
    data_avg_img=np.mean(data,axis=0)
    data_avg_img=np.array([np.round(x) for x in data_avg_img])
    data_avg_img=data_avg_img.astype(np.int16)
    if(to_return==True):
        return data_avg_img
    ax.imshow(data_avg_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)
    return ax


# In[ ]:


fig=plt.figure(figsize=(7,3))
plt.title("Average Images\n\n",fontsize=25)
plt.axis('off')
ax1=fig.add_subplot(121)
ax1=plot_avg_image(parasitized_train,ax1,"Parasitized Train")
ax2=fig.add_subplot(122)
ax2=plot_avg_image(uninfected_train,ax2,"Uninfected Train")
plt.show()


# As we see, the shape of average images of both parasitized and uninfected data is almost same - a nice round circle. However, we do observe the color of the average parasitized image to be very slightly darker than that of the uninfected data. This may be due to the presence of 'darker' parasites in the parasitized cells. Also the excess darkness in the average parasitized image appears to be uniform, indicating that there is no specific position where parasites are more likely to be found within the cell...

# #### For the remaining analysis, I'd like to have a sample parasitized and sample uninfected image to apply the analysis on. The samples are again chosen randomly.

# In[ ]:


np.random.seed(random_seed)
rand=np.random.randint(parasitized_train.shape[0])
parasitized_sample=parasitized_train[rand]
np.random.seed(random_seed)
rand=np.random.randint(uninfected_train.shape[0])
uninfected_sample=uninfected_train[rand]

fig=plt.figure(figsize=(8,5))
plt.title("Sample Images for Reference",fontsize=20)
plt.axis('off')
ax1=fig.add_subplot(121)
ax1.imshow(parasitized_sample)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax1.set_title("Parasitized Sample",fontsize=15)
ax2=fig.add_subplot(122)
ax2.imshow(uninfected_sample)
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.set_title("Uninfected Sample",fontsize=15)
plt.show()


# ## Analyzing Blood Cell colour
# 
# For colour analysis, we try and plot the image histogram. The histogram plot tells us what is the most frequent intensity of each colour component...

# In[ ]:


def get_image_histogram(sample_data,title,to_return=False):
    sample=Image.fromarray(sample_data.astype(np.uint8),'RGB')
    hist=sample.histogram()
    hist_r=hist[:256]
    hist_g=hist[256:512]
    hist_b=hist[512:]
    if(to_return==True):
        return hist_r,hist_g,hist_b
    x=np.arange(256)
    
    fig=plt.figure(figsize=(12,5))
    plt.title(title+'\n',fontsize=20)
    plt.axis('off')
    ax1=fig.add_subplot(121)
    ax1.imshow(sample)
    ax1.set_title("Image",fontsize=15)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2=fig.add_subplot(122)
    ax2.plot(hist_r,color='red')
    ax2.fill_between(x,hist_r,color='red')
    ax2.plot(hist_g,color='green')
    ax2.fill_between(x,hist_g,color='green')
    ax2.plot(hist_b,color='blue')
    ax2.fill_between(x,hist_b,color='blue')
    ax2.set_ylim(0,3000)
    ax2.set_title("Image Histogram",fontsize=15)
    ax2.set_ylabel("Frequency")
    ax2.set_xlabel("Colour Intensity")
    plt.show()


# In[ ]:


get_image_histogram(parasitized_sample,"Parasitized Sample")
get_image_histogram(uninfected_sample,"Uninfected Sample")


# As we see, both images seem to have more frequency of bright red colour component. However, there isn't a lot of difference in the parasitized and uninfected histograms...

# ## Cell Outline
# 
# Let us try and extract the outline of the cell. As we know, most of the essential information will lie within the cell; so an outline might help us focus only on the interior of the cell. However, I have not been able to apply this thought with meaningful observations...

# In[ ]:


from scipy import ndimage
from skimage.color import rgb2gray
def get_outline(sample,title='',to_return=False):
    sample=sample.astype(np.int32)
    sx=ndimage.sobel(sample,axis=1)
    sy=ndimage.sobel(sample,axis=0)
    sob=np.hypot(sx,sy)
    sob=np.divide(sob,np.max(sob))
    if(to_return==True):
        return sob
    
    fig=plt.figure(figsize=(12,5))
    plt.title(title+'\n',fontsize=20)
    plt.axis('off')
    ax1=fig.add_subplot(121)
    ax1.imshow(sample)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title("Image",fontsize=15)
    ax2=fig.add_subplot(122)
    ax2.imshow(sob)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title("Outline",fontsize=15)
    plt.show()


# In[ ]:


get_outline(parasitized_sample,"Parasitized Sample")
get_outline(uninfected_sample,"Uninfected Sample")


# We do notice a very light outline of the parasite in the first image. We could perhaps use this light outline to identify parasites...

# ## Identifying Dark Spots
# 
# We will now try to identify dark spots in the image. We do this because we know that the parasitic part of the cell is, in general, darker than the rest of the cell. Thus, this parasitic part might appear as a dark spot on the cell...
# 
# We use the skimage function 'black_tophat' which returns an image containing dark spots of the image (refer to http://scikit-image.org/docs/dev/api/skimage.morphology.html ). These dark spots appear as bright spots in the returned image.

# In[ ]:


import skimage.morphology as skm
def dark_spots(sample,title='',to_return=False):
    black_tophat=skm.black_tophat(sample)
    black_tophat_refined=skm.black_tophat(black_tophat)
    black_tophat_refined=skm.black_tophat(black_tophat_refined)
    if(to_return==True):
        return black_tophat_refined
    
    fig=plt.figure(figsize=(14,5))
    plt.title(title+'\n',fontsize=20)
    plt.axis('off')
    ax1=fig.add_subplot(131)
    ax1.imshow(sample)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title("Image",fontsize=15)
    ax2=fig.add_subplot(132)
    ax2.imshow(black_tophat)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title("Dark Spots",fontsize=15)
    ax3=fig.add_subplot(133)
    ax3.imshow(black_tophat_refined)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title("Refined Dark Spots",fontsize=15)
    plt.show()


# In[ ]:


dark_spots(parasitized_sample,"Parasitized Sample")
dark_spots(uninfected_sample,"Uninfected Sample")


# When we apply black_tophat once, we realize numerous number of white spots on the outline of the image. These maybe caused because of numerous, tiny, very dark spots on top of the cell boundary. To remove these spots, we apply black_tophat twice, to ensure that the very bright spots don't appear in the final image. However, the downside of this procedure is that the brightness and size of the bright spot, which were caused by the parasitic part, decreases...
# 
# Also, the final parasitic spot is largely green in colour. This is because the original spot is almost purple in colour, which is a mixture of red and blue. The black_tophat function actually somewhat inverts the colors of the original spot, causing red and blue to become predominantly green. Applying black_tophat on this image again will cause the spot to again turn into purple.

# ## Blob Detection
# 
# Blob Detection is an algorithm to detect blobs in the image. Blobs are similar to spots. When we looked at the parasitized and uninfected data, one observation that did come to mind is that almost all parasitized cells had 'blobs' in them, while almost none of the uninfected cells had these blobs. Thus, we can simplify the problem to identifying whether a cell has a blob in it. If yes, then it might be parasitized, else it might be uninfected.

# In[ ]:


from skimage.feature import blob_dog,blob_log,blob_doh
def identify_blobs(sample,title='',to_return=False,threshold=2.5*1e-9):
    spots=dark_spots(sample,to_return=True)
    gray_spots=rgb2gray(spots)
    final_spots=skm.white_tophat(gray_spots,selem=skm.square(10))
    log=blob_doh(final_spots,threshold=threshold,min_sigma=10,max_sigma=50)
    if(to_return==True):
        return log
    
    fig=plt.figure(figsize=(12,7))
    plt.title(title,fontsize=20)
    plt.axis('off')
    ax1=fig.add_subplot(121)
    ax1.imshow(final_spots,cmap='gray')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title("Cleaned Grayscale Image",fontsize=15)
    ax2=fig.add_subplot(122)
    ax2.imshow(sample)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    for blob in log:
        y,x,r=blob
        c=plt.Circle((x,y),r,color='red',fill=False,linewidth=2)
        ax2.add_patch(c)
    ax2.set_title("Location of Blobs",fontsize=15)
    plt.show()


# In[ ]:


identify_blobs(parasitized_sample,"Parasitized Train")
identify_blobs(uninfected_sample,"Uninfected Train")


# We use the dark spots function mentioned above as it provides a very clean image where blobs, if any, can be distinguished easily. Skimage has three blob detection features - blob_dog, blob_log and blob_doh. Blob_dog is the most accurate and slowest, while blob_doh is less accurate but fast (Refer to http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html ). We'll use blob_doh now to save time.
# 
# The values of max_sigma, min_sigma and threshold have been chosen after a bit of experimentation/hit-and-trial. Max sigma and min sigma basically limit the maximum and minimum radius of a detected blob respectively while threshold is used to allow or disallow less prominent blobs to be detected. Obviously, these values will vary for different problems as the general sizes and intensity of blobs to be detected might vary from problem to problem. As we can observe in the first image, the given parameters are sufficient to approximately identify the location and radius of a blob in the image.

# ## Blob Features
# 
# We can now identify the number of blobs an image might have, also its position and size. Using this, we'll try to find the average position and size of a blob and also the very important feature 'has_blob'...

# In[ ]:


def blobs(sample,threshold=2.5*1e-9):
    blobs=identify_blobs(sample,to_return=True,threshold=threshold)
    if(len(blobs)==0):
        return 0,0,0,0
    y,x,r=np.sum(blobs,axis=0)
    return len(blobs),x,y,r


# In[ ]:


def get_blob_features(data,threshold=2.5*1e-9):
    print("Extracting Blob Features...")
    blobs_cnt=np.zeros(data.shape[0],dtype=np.int16)
    blobs_x=np.zeros(data.shape[0],dtype=np.float32)
    blobs_y=np.zeros(data.shape[0],dtype=np.float32)
    blobs_r=np.zeros(data.shape[0],dtype=np.float32)
    for ind,sample in tqdm_notebook(enumerate(data),total=data.shape[0]):
        blobs_cnt[ind],blobs_x[ind],blobs_y[ind],blobs_r[ind]=blobs(sample,threshold=threshold)
    print("Done Extracting Blob Features!")
    return blobs_cnt,blobs_x,blobs_y,blobs_r


# In[ ]:


blob_cnt,blob_x,blob_y,blob_r=get_blob_features(train)


# In[ ]:


avg_x=np.sum(blob_x)/np.sum(blob_cnt)
avg_y=np.sum(blob_y)/np.sum(blob_cnt)
avg_r=np.sum(blob_r)/np.sum(blob_cnt)

avg_train=plot_avg_image(train,to_return=True)
fig,ax=plt.subplots()
plt.title("Average Position & Size of Blobs")
plt.axis('off')
ax.imshow(avg_train)
c=plt.Circle((avg_x,avg_y),avg_r,color='red',fill=False,linewidth=2,linestyle='--')
ax.add_patch(c)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:


print("Average Length of Blobs - {rad:.2f}".format(rad=avg_r))


# The average location of blobs in cells seems to be almost near the middle, again emphasizing that there is no particular direction where blobs are more likely to be found. Also the average radius of a blob is around 11.5 pixels. Since the average shape of a cell is almost an incircle of the image frame, the diameter of this cell is almost the width of the image frame (i.e. 125 pixels). Therefore the average parasitic area to cell area is calculated to be around 3.5%. Thus on average, a parasite occupies only 3.5% of the total area of the cell. Yet these tiny parasites have the frightening potential to damage an entire human body...

# ## 'has_blob' Feature
# 
# Let us check the correlation of the 'has_blob' feature...

# In[ ]:


from scipy.stats import pearsonr
def correlation(feature,label):
    corr=np.abs(pearsonr(label,feature)[0])
    return corr


# In[ ]:


has_blob=[1 if x>0 else 0 for x in blob_cnt]
print("Correlation of 'has_blob' feature - {corr:.2f}".format(corr=correlation(has_blob,train_labels)))


# The 'has_blob' feature has an impressive 0.89 correlation with the target variable. This is, indeed, a very good feature to work with. In fact, just calculating the 'has_blob' feature for a given image gives us a very good approximation of its corresponding target. Therefore, we can actually use 'has_blob' directly to predict the presence of malaria-causing parasites in a cell, without any kind of classification model. This is similar to how a human might deal with the given classification problem - if you find a blob in the cell, it should be parasitic, else uninfected...
# 
# Therefore, let us find the 'has_blob' features for the test, which we will also use as the eventual predictions...

# In[ ]:


predictions=get_blob_features(test)[0]
predictions=[1 if x>0 else 0 for x in predictions]


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix

print("Accuracy = {acc:.4f}%".format(acc=accuracy_score(test_labels,predictions)*100))
print("\nConfusion Matrix :")
print(confusion_matrix(test_labels,predictions))


# So, we achieve around 94% accuracy using just ONE feature! Looking at the confusion matrix, we realize we've actually predicted 273 parasitized cells as uninfected and 132 uninfected cells as parasitized. Of course, such a high accuracy might also be the result of chance. Therefore, I decided to repeat the process for different random seeds. Given below are the results :
# 
# * Random Seed = 33 --> Accuracy = 94.12%
# * Random Seed = 20 --> Accuracy = 94.17%
# * Random Seed = 41 --> Accuracy = 94.25%
# 
# Based on this, we can hopefully expect this feature to be reliable for different datasets as well. Therefore, it is fair to say that we have created a reliable and simple one-feature model to tackle this classification problem...

# # Conclusion
# 
# ### We loaded the data, created our train & test datasets and analyzed the images to obtain a good idea regarding many aspects of malaria parasites. We specially utilized the blob detection algorithm to create a very useful feature that alone can contribute to around 94% accuracy on unseen data.
# 
# ### Further Scope:
# * Using blob_dog or blob_log detection techniques, hoping for better accuracy at the expense of time
# * Using the fact that the parasitic part of the cell is usually of a different colour than the cell colour
# * Trying to stack a good Keras Neural Network model with the 'has_blob' feature
# 
# ### We can obviously use Keras Neural Networks efficiently to get a much better accuracy score. However, we often use Keras as a black-box, not knowing exactly how we determine whether a cell is parasitized or not. This is my humble attempt of trying to do the same task of classification, but with a simpler model, fewer features and a more 'human' touch...
# 
# #### P.S. - This is my first public kernel, so I might make some rookie mistakes here and there. Also, any kind of suggestions are welcome. Thank You!

# In[ ]:




