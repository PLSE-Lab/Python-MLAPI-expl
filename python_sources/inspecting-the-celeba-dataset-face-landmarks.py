#!/usr/bin/env python
# coding: utf-8

# # ANALIZING FACE LANDMARKS
# 
# This is a very very simple kernel whose purpose is just to check what is in the dataset regarding the given coordinates for landmarks and bounding boxes.
# 
# For that, we're going to load the data, create the bounding boxes onto the images and plot them to see where they are.

# ## Imports and helper functions

# In[ ]:


import os
import pandas as pd
import numpy as np

from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#displays for a batch of images as a numpy array basic statistics
def inspect(x, name):
    print(name + ": ", 
          'shape:', x.shape, 
          'min:', x.min(), 
          'max:',x.max(),
          'mean:',x.mean(), 
          'std:', x.std())
    
#plots the passed images side by side
def plotImages(*images, minVal =0, maxVal = 1):
    fig, ax = plt.subplots(1,len(images), figsize=(13,4))
    
    for i,image in enumerate(images):
        ax[i].imshow(image,vmin = minVal, vmax=maxVal)
    
    plt.show()


# ## Loading and checking raw data
# 
# Notice the "assert" command, to make sure all arrays that were loaded are matching rows. 

# In[ ]:


#Folder and image files
generalFolder = '../input/'
folder = generalFolder + 'img_align_celeba/img_align_celeba'
allFiles = [folder + "/" + f for f in sorted(os.listdir(folder))]

#CSV files 
allAttributes = pd.read_csv(generalFolder + 'list_attr_celeba.csv')
faceBoxes = pd.read_csv(generalFolder + 'list_bbox_celeba.csv')
featureLocations = pd.read_csv(generalFolder + 'list_landmarks_align_celeba.csv')


#Make sure our filenames and all CSV files are matching
for f, a, face, loc in zip(allFiles, 
                           allAttributes['image_id'], 
                           faceBoxes['image_id'], 
                           featureLocations['image_id']):
    assert(f.split('/')[-1] == a == face == loc)
    


# ## What is in the CSV files and how are we preprocessing them
# 
# Here, we understand the data and process it so it can be easily used in some models such as a Keras model. 
# 
# ### File: list_attr_celeba.csv (`allAtributes`) 
# 
# This is not in the scope of this kernel, so, just a quick description: it's a file that for every image, has 40 columns identifying one feature, such as shadow, certain types of eyebrows, etc. The values are -1 and 1.
# 
# In the preprocessing, they're being normalized from 0 to 1, so they can be used with sigmoid activations and loss functions specialized in classification. 
# 
# ### File: list_bbox_celeba.csv (`faceBoxes`)
# 
# For each image, contains `x`, `y`, `width` and `height`, defining a complete bounding box that should cover the entire face.
# We just create an numpy array of zeros, and in the range defined by these cordinates and sizes, we define the value as 1. 
# 
# ### File: list_landmarks_align_celeba.csv (`featureLocations`)  
# 
# For each image, contains "only the location" of these:
# 
# - right eye   
# - left eye    
# - nose    
# - right side of the mouth    
# - left side of the mouth    
# 
# For each location, we're creating a 20x20 bounding box, centered on the given coordinate pair. We use the same convention as with the face bounding boxes (0 and 1) and stack everything, these and the face bounding boxes in a 6 channel array, for using in future models.
# 

# ## Creating a batch generator
# 
# This is a working batch generator that can be used for training models. (Not strictly what we need for this test, but useful).
# This generator, if derived from the `keras.utils.Sequence` class, can be used directly with `Model.fit_generator(generator, len(generator), ....)`    

# In[ ]:


class Reader():
    #If you make this a 'class Reader(keras.utils.Sequence)', 
    #you can use it with `Model.fit_generator`   
    
    #creates the generator taking the raw data and a batch size
    def __init__(self, files, attributes, faces, locations, batchSize):

        assert len(files)==len(attributes)==len(faces)==len(locations)
        
        self.files = files
        
        #from the CSV files, make them arrays without the "image_id" column 
        #we checked before that the array rows are matching the filenames    
        self.attributes = (np.array(attributes)[:,1:] + 1)/2. #normalized from 0 to 1
        self.faces = np.array(faces)[:,1:]           #x, y, width, height
        self.locations = np.array(locations)[:,1:]   #5 coordinate pairs
        
        self.batchSize = batchSize
        
        #defining the number of batches (length) and the last batch size
        l,r = divmod(len(files), batchSize)
        self.length = l + (1 if r > 0 else 0)
        self.lastSize = r if r > 0 else batchSize
        
        
    #gets the number of batches that can be generated    
    def __len__(self):
        return self.length
    
    
    #gets one batch respective to the passed index    
    def __getitem__(self, i):
        batch = self.batchSize if i + 1 < len(self) else self.lastSize
        start = i * self.batchSize
        end = start + batch
        
        imgs = list()
        for im in self.files[start:end]:
            imgs.append(np.array(Image.open(im), dtype=np.float64)/255.)
        imgs = np.stack(imgs, axis=0)
        masks = np.zeros(imgs.shape[:3]+(6,))
        
        self.addFaceMasks(self.faces[start:end], masks)
        self.addLocations(self.locations[start:end], masks)
        
        return imgs, masks, self.attributes[start:end]
        
    #processes the face bounding boxes into mask images   
    def addFaceMasks(self,faces,addTo):
        
        #for each face in the batch
        for i, face in enumerate(faces):
            x, y, w, h = face #gets coordinates and sizes
            
            addTo[i,y:y+h, x:x+w,0] = 1. #updates the masks    
    
    #processes the other coordinates into mask images   
    def addLocations(self, locations, addTo):
        
        #for each face in the batch
        for i, locs in enumerate(locations):
            locs = locs.reshape((-1,2))    #reshapes into pairs of coords
            
            #for each pair of coords
            for ch, (posX, posY) in enumerate(locs):
                #20x20 bounding boxes for this coord pair 
                x = posX - 10
                y = posY - 10
                x2 = x + 20
                y2 = y + 20
                if x < 0: x = 0
                if y < 0: y = 0
                    
                addTo[i,y:y2, x:x2,ch+1] = 1.


# ## Getting batches and visualizing the data
# 
# Here, we're taking only the first image of each batch to plot (otherwise there would be too many images)
# 
# **Warning:** The face bounding boxes are wrong in the dataset (see details after the plots)

# In[ ]:


generator = Reader(allFiles, allAttributes, faceBoxes, featureLocations, 32)

#test 20 batches
for i in range(20):
    imgs, masks, atts = generator[i]
    
    #print the batch shape:
    if i == 0:
        print("image batch shape:", imgs.shape)
        print("masks batch shape:", masks.shape)
    
    #use these inspections to verify that the values are within the expected ranges   
#     inspect(imgs, 'imgs')
#     inspect(masks, 'masks')
#     inspect(atts, 'atts')
    
    #takes four faded images from the batch (we fade them to add with the masks)
    n = 2 #plotting n images per batch
    imgs = imgs[:n]/2.
    #plotImages(*imgs)
    
    #gets the face bounding boxes (first channel in the masks array)
    faceM = np.array(masks[:n,:,:,:3]) #we keep 3 channels for compatibility   
    faceM[:,:,:,1:] = 0      #we make the non related channels 0
    
    #gets the other bounding boxes, summing two groups of 3 channels
    #each channel is one feature (remove the face boxes that are above)
    otherM = masks[:n,:,:,:3] + masks[:n,:,:,3:] - faceM
    
    #pairs of images:
        #faces + face bounding box
        #faces + other features bounding boxes
    faceBxs = np.clip(imgs + faceM , 0,1)
    otherBxs = np.clip(imgs + otherM, 0, 1)
        
    #putting pairs side by side
    imgs = np.stack([faceBxs,otherBxs], axis=1)
    shp = imgs.shape
    imgs = imgs.reshape((-1,) + shp[2:])
    
    #plotting
    plotImages(*imgs )
    


# ## Wrong bounding boxes?
# 
# Well, we may think there was a bug in the code, that's not rare after all. 
# 
# But then we look closer at the dataset and quickly see something is clearly wrong. 
# While our images are shaped as 218 x 178, values in the CSV file contain numbers such as 500 and greater.
# 
# ![Image from the dataset showing values greater than the image shape](https://pix.toile-libre.org/upload/original/1533859338.png)
# 
# # Conclusions
# 
# Regarding the landmarks and bounding boxes we can say that this dataset:
# 
# - Is great when dealing with eyes, mouth and nose locations    
# - Has wrong bounding boxes for the faces    
# 
