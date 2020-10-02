#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#import the libraries
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
import PIL


# In[ ]:


#loading the annotations file
key = pd.read_csv('../input/input-key/training_frames_keypoints.csv')
key.head()


# In[ ]:


#UNCOMMENT BELOW COMMAND TO INSTALL imgaug lib
#!pip install imgaug
from imgaug import augmenters as iaa
import imgaug as ia


# # DATA AUGMENTATION PIPELINE USING IMGAUG LIB FROM PYPI
# 

# In[ ]:


class ImgAugTransform:
  def __init__(self):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    self.aug = iaa.Sequential([
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 2.0))),#sigma was 3.0 initially
        iaa.Fliplr(0.4),# flip was 0.5 initially
        iaa.Affine(rotate=(-20, 20), mode='symmetric'),
        iaa.Sometimes(0.3,
                      iaa.OneOf([iaa.Dropout(p=(0, 0.1)), # in place of 0.3, 0.2 was placed
                                 iaa.CoarseDropout(0.1, size_percent=0.5)])),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
         # Apply affine transformations to some of the images
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - rotate by -45 to +45 degrees
        # - shear by -16 to +16 degrees
        # - order: use nearest neighbour or bilinear interpolation (fast)
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        # - cval: if the mode is constant, then use a random brightness
        #         for the newly created pixels (e.g. sometimes black,
        #         sometimes white)
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),

        #
        # Execute 0 to 5 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((0, 5),
            [
                # Convert some images into their superpixel representation,
                # sample between 20 and 200 superpixels per image, but do
                # not replace all superpixels with their average, only
                # some of them (p_replace).
                sometimes(
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),

                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # Same as sharpen, but for an embossing effect.
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                # Search in some images either for all edges or for
                # directed edges. These edges are then marked in a black
                # and white image and overlayed with the original image
                # using an alpha of 0 to 0.7.
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),

                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),

                # Either drop randomly 1 to 10% of all pixels (i.e. set
                # them to black) or drop them on an image with 2-5% percent
                # of the original size, leading to large dropped
                # rectangles.
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),

                # Invert each image's channel with 5% probability.
                # This sets each pixel value v to 255-v.
                iaa.Invert(0.05, per_channel=True), # invert color channels

                # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 10), per_channel=0.5),

                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.5, 1.5), per_channel=0.5),


                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),

                # In some images move pixels locally around (with random
                # strengths).
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),

                # In some images distort local areas with varying strength.
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    ],
    # do all of the above augmentations in random order
    random_order=True
)

    
      
  def __call__(self, sample):
    image, key_pts = sample['image'], sample['keypoints']
    img = np.array(image)
    return {'image': self.aug.augment_image(img), 'keypoints': key_pts}
    #return self.aug.augment_image(img)


# In[ ]:



class ToTensor(object):

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 3)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}

#FUNCTION TO PERFORM NORMALIZATION
class Normalize(object):      

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        #image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
        #image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}
    
#RESCALING FUNCTION    
class Rescale(object):
    

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}
    
    
#FUNCTION TO RANDOMLY CROP IMAGE 
class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


# # PROCESSING THE DATASET AND CONVERTING INTO A PROPER FORMAT
# ****MAKING DICTIONARY CONTAINING IMAGES WITH THIER KEYPOINTS IN (X,Y) FORM
# ****HERE X AND Y ARE THE CORDINATES OF A KEYPOINT

# In[ ]:


#ist column contain name of the file image and all other columns 
#contain the x and y axis of keypoints
#so we will separate them
#create a function to make a dataset of form A sample of our dataset will be a dictionary {'image': image, 'keypoints': key_pts}
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms, utils

class facialdataset(Dataset):
    def __init__(self,csv_file , root_dir , transform  = None):
        self.key_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    #function to return the length of keypoints dictionary
    def __len__(self):
        return len(self.key_csv)
    def __getitem__(self,idx):
        #append root dir and image name from csv to fetch image
        image_name = os.path.join(self.root_dir , self.key_csv.iloc[idx,0])
        image = mpimg.imread(image_name)
        #remove last channel if image have 4 channels instead of 3
        if image.shape[2] == 4:
            image = image[:,:,0:3]
        #now convert the cords in matrix and the reshape
        key_cords = self.key_csv.iloc[idx,1:].to_numpy()
        key_cords = key_cords.astype('float').reshape(-1,2)
        dictionary = {'image':image , 'keypoints':key_cords}
        
        #applying transform is not none:
        if self.transform:
            dictionary= self.transform(dictionary)
        return dictionary
# define the data tranform
# order matters! i.e. rescaling should come before a smaller crop
train_transforms = transforms.Compose([Rescale(250),
                                       ImgAugTransform(),#add the custom augmetations in the pytorch transform pipeline
                                        RandomCrop(224),
                                         Normalize(),
                                      ToTensor()])


# # APPLY THE ABOVE DEFINED TRANSFORMATION ON THE DATASET
# HERE I AM USING A DATASET CONTAINING IMAGES OF FACES WITH THEIR RESPECTIVE KEYPOINTS AS (X,Y) FORM IN A CSV FILE

# In[ ]:


#after creating the class, we will pass the images and labels directory
#in the class to get the dictionary
face_dataset_train = facialdataset(csv_file='../input/input-key/training_frames_keypoints.csv',
                            root_dir='../input/input-key/training/training/',transform=train_transforms)
face_dataset_test = facialdataset(csv_file = '../input/input-key/test_frames_keypoints.csv',
                                 root_dir = '../input/input-key/test/test/',
                                 transform=train_transforms)


# In[ ]:


print('length of train data' , len(face_dataset_train))
print('length of test data' , len(face_dataset_test))


# In[ ]:


print("train" , face_dataset_train)


# In[ ]:


print('number of images in train' , len(face_dataset_train))
print('number of images in test' , len(face_dataset_test))
for i in range(1,5):
    sample = face_dataset_test[i]
    print(i , sample['image'].size() , sample['keypoints'].size() )


# # ******BUILD MODEL AS PER THE TASK.
# 

# In[ ]:





# In[ ]:




