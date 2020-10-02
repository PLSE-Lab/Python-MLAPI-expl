#!/usr/bin/env python
# coding: utf-8

# # LEts Start First with What Is Steganography ?
# Steganography  is  to  conceal  the  secret  data  within  multimedia contents such as file, message, image, or video. Steganography is concerned  with concealing the fact that  the  secret data  is being sent covertly as well as concealing the contents of the secret data .  Steganalysis  is  the  counter  part  of  steganography  that defined as the art of science of detecting the hidden secret data in cover objects. In other words, steganalysis is to detect secret data hidden  using  steganography,  where  steganalysis  is  to  identify suspected  packages,  determine  whether  the  secret  data  is embedded  or  not .  Machine learning  is  a  field  of  artificial intelligence  to  provide  the  ability  to  learn  without  being programmed and deep learning is a subset of machine learning .    In this paper, steganalysis and machine learning  techniques are explained and the process and possibility for steganalysis in various  machine  learning  frameworks  are  described.  Some datasets  on  stego-images  are  prepared  and  training  model  are tested. 

# # What is Actually Stegnanalysis ?
# 
# Stegnanalysis is the art of secret communication and steganalysis is the art of detecting the hidden messages embedded in digital media using steganography. Both steganography and steganalysis have received a great deal of attention from law enforcement and the media. In the past years many powerful and robust methods of steganography and steganalysis have been reported in the literature. In this paper, we classify and give an account of the various approaches that have been proposed for steganalysis. Some promising methods for statistical steganalysis have also been identified.

# 
# <img src="https://www.researchgate.net/profile/Bismita_Choudhury/publication/282889667/figure/fig1/AS:614435383152640@1523504219027/Block-diagram-of-Steganalysis.png">

# **Steganalysis** is to identify suspected data, determine hidden data, and recover the hidden data. Steganalysis can be divided into **four** categories: 
# - visual 
# - structural 
# - statistical 
# - learning steganalysis.  
# 
# **Visual steganalysis** is to investigate visual artifacts in the stego-images, where try to catch visual difference by analyzing stego-images. 
# 
# **Structural steganalysis** looks  into suspected signs in the media  format  representation  since  the  format  is  often  changed when  the  secret  message  is  embedded.  RS  analysis  and  pair analysis  are  included  in  the  structural  steganalysis.
# 
# **Statistical steganalysis**  utilizes  statistical  models  to  detect  steganography techniques.  Statistical  steganalysis  can  be  divided  into  specific statistical  and  universal  statistical  steganalysis.
# 
# **Learning steganalysis**  also  called  blind  steganalysis  is  one  of  universal statistical  steganalysis  since  cover  images  and  stego-images  are used as training datasets.  Other  classification  of  steganalysis  can  be  divided  into  six categories . It is depending on what kind of attacks a forensic examiner uses.  

# # EDA Stegnanalysis

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch


# In[ ]:


PATH = '/kaggle/input/alaska2-image-steganalysis'
model = '/kaggle/input/alaska2-image-steganalysis/e4s-srm/'

train_images = pd.Series(os.listdir(PATH + '/Cover/')).sort_values(ascending=True).reset_index(drop=True)
test_images = pd.Series(os.listdir(PATH + '/Test')).sort_values(ascending=True).reset_index(drop=True)
sample_submission = pd.read_csv(f'{PATH}/sample_submission.csv')


# In[ ]:


es4 = .alexnet(pretrained=True)


# In[ ]:


sample_submission.head()


# # Display Images

# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
k=0
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        img = mpimg.imread(PATH + '/Cover/' + train_images[k])
        col.imshow(img)
        col.set_title(train_images[k])
        k=k+1
plt.suptitle('Samples from Cover Images', fontsize=14)
plt.show()


# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
k=0
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        img = mpimg.imread(PATH + '/Test/' + test_images[k])
        col.imshow(img)
        col.set_title(test_images[k])
        k=k+1
plt.suptitle('Samples from Test Images', fontsize=14)
plt.show()


# In[ ]:


for folder in os.listdir(PATH):
    try:
        print(f"Folder {folder} contains {len(os.listdir(PATH + '/' + folder))} images.")
    except:
        print(f'{folder}')


# Displaying JMiPOD , JUNIWARD , UERD, with train

# In[ ]:


folders = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
k=0
img_cover = mpimg.imread(PATH + '/' + folders[0] + '/'+ train_images[k])
img_jmipod = mpimg.imread(PATH + '/' + folders[1] + '/'+ train_images[k])
img_juniward = mpimg.imread(PATH + '/' + folders[2] + '/'+ train_images[k])
img_uerd = mpimg.imread(PATH + '/' + folders[3] + '/'+ train_images[k])

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))

ax[0,0].imshow(img_jmipod)
ax[0,1].imshow((img_cover == img_jmipod).astype(int)[:,:,0])
ax[0,1].set_title(f'{train_images[k]} Channel 0')

ax[0,2].imshow((img_cover == img_jmipod).astype(int)[:,:,1])
ax[0,2].set_title(f'{train_images[k]} Channel 1')
ax[0,3].imshow((img_cover == img_jmipod).astype(int)[:,:,2])
ax[0,3].set_title(f'{train_images[k]} Channel 2')
ax[0,0].set_ylabel(folders[1], rotation=90, size='large', fontsize=14)


ax[1,0].imshow(img_juniward)
ax[1,1].imshow((img_cover == img_juniward).astype(int)[:,:,0])
ax[1,2].imshow((img_cover == img_juniward).astype(int)[:,:,1])
ax[1,3].imshow((img_cover == img_juniward).astype(int)[:,:,2])
ax[1,0].set_ylabel(folders[2], rotation=90, size='large', fontsize=14)

ax[2,0].imshow(img_uerd)
ax[2,1].imshow((img_cover == img_uerd).astype(int)[:,:,0])
ax[2,2].imshow((img_cover == img_uerd).astype(int)[:,:,1])
ax[2,3].imshow((img_cover == img_uerd).astype(int)[:,:,2])
ax[2,0].set_ylabel(folders[3], rotation=90, size='large', fontsize=14)

plt.suptitle('Pixel Deviation from Cover Image', fontsize=14)

plt.show()


# Total number of images

# In[ ]:


print('{} images with Cover Images '.format(train_images.nunique()))
print('{} images with Test Images '.format(test_images.nunique()))

