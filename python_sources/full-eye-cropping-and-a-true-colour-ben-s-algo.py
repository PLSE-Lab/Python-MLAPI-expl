#!/usr/bin/env python
# coding: utf-8

# # Steal and critique my methods as you wish
# 
# I just thought I'd let others use any elements of the pre-processing functions I've used.
# This isn't to say I continue to use all elements of this in my current iteration.

# In[ ]:


import numpy  as np 
import pandas as pd 
import os
import cv2 
import matplotlib.pyplot as plt

# Global constants
IMG_DIM       = 256
NUM_CLASSES   = 5

print(os.listdir("../"))
print(os.listdir("../input/"))
print(os.listdir("../input/aptos2019-blindness-detection"))

INPUT_FOLDER = '../input/aptos2019-blindness-detection/'


# # Cropping, Mirroring and Squaring up
# 
# To crop I just assume the eye is centered, and take the image boundaries in to the first pixel that is bright enough. (Green was used because I was also using the green channel for other preprocessing functions at the time...) The percent smaller is a hacky way to deal with the fact that some times the bottom or top of pictures that are cut off are slanted to the actual bottom of the image.
# 
# For mirroring and squaring, I presume the image is a rectangle, wider than taller, so I reflect the image up and down far enough that the final image forms a square.

# In[ ]:


def crop(gray, img, percent_smaller):
    
    thresh = 8
    
    top    = 0
    left   = 0
    bottom = gray.shape[0] - 1
    right  = gray.shape[1] - 1
    
    # work in from the top and bottom along the middle collumn
    middleCol = gray[:, int(gray.shape[1]/2)] > thresh
    while middleCol[top] == 0:
        top += 1
    while middleCol[bottom] == 0:
        bottom -= 1
        
    # work in from the sides along the middle row
    middleRow = gray[int(gray.shape[0]/2)] > thresh
    while middleRow[left] == 0:
        left += 1
    while middleRow[right] == 0:
        right -= 1
        
    height = bottom - top
    width  = right - left
    
    bottom -= int(percent_smaller*height)
    top    += int(percent_smaller*height)
    right  -= int(percent_smaller*width)
    left   += int(percent_smaller*width)
        
    if height < 100 or width < 100:
        print("Error: squareUp: bottom:", bottom, "top:", top)
        print("Error: squareUp: right:", right, "left:", left)
        return img
    
    return img[top:bottom, left:right]



def reflectAndSquareUp(img):
    
    height = img.shape[0]
    width  = img.shape[1]
    
    # if its portrait mode, it's probably already kind of square. Make it properly square by cutting
    # down the height until the dimensions match
    if (height > width):
        
        offset = int((height - width)/2)
        return img[offset:offset+width]
    
    # otherwise, do the whole reflection thingo
    else:
        if len(img.shape) == 3:
            new_img = np.zeros((width, width, img.shape[2]), np.uint8)
        else:
            new_img = np.zeros((width, width), np.uint8)

        #  0  |
        #     |
        #  h1 |####
        #     |####
        #     |####
        #  h2 |
        #     |

        h1 = int((width - height)/2)
        h2 = h1 + height

        # paste the original into the center
        new_img[h1:h2,:] = img

        # paste in the reflections
        for i in range(h1):
            new_img[h1-i] = img[i]

        for i in range(width - h2):
            new_img[h2+i] = img[height - i - 1]

        return new_img


# # Bens Algorithms
# 
# I was never quite convinced that using bens algorithm applied directly to all channels independantly made much sense. I like the idea of subtracting the gaussian blur from pixels in a weighted manner as a nifty method of contrast enhancement, but I implemented it by splitting it into the YCC first and just applying it to the Y channel, before stitching it back to RGB (initial attempts to do so on the S channel in HSV space caused headaches).
# 
# I haven't made any rigorous checks that this improves performance (the NN training pipeline takes long enough, and I'm just a student chewing through my initial AWS credits atm), but just leave this here for anyone who is interested.

# In[ ]:



def circleMask(img):
    
    if (img.shape[0] != img.shape[1]):
        print("Error: circle mask assumes square image")
        return img
    
    dim = img.shape[0]
    half = int(dim/2)
    
    # crop out circle:
    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)

    return cv2.bitwise_and(img, img, mask=circle_mask)

def clahe_gray(gray, clipLimit=3.5, grid=4):

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid,grid))
    return clahe.apply(gray)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def benYCC(bgr, weight=4, gamma=12):
        
    # convert to y, cr, cb so that we can modify the image based on just the y (brightness)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    # perform bens algorithm on the y component
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0,0), gamma), -weight, 128)

    # merge the ycc back together, and recolor it
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    
    return bens 


def benSimple(img, weight=4, gamma=20):
        
    bens = cv2.addWeighted(img, weight, cv2.GaussianBlur(img, (0,0), gamma), -weight, 128)
    
    return bens 


# # Visualisation:

# In[ ]:


def test_pre_processing():
    
    images_dir  = f"{INPUT_FOLDER}train_images/"
    df          = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    df.id_code  = df.id_code.apply(lambda x: x + ".png")

    for j, filename in enumerate(df.sample(4).id_code):
        
        bgr = cv2.imread(images_dir + filename)

        # original
        ax  = figure.add_subplot(4,4, 4*j+1)
        plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # cropped
        cropped = crop(bgr[:,:,1], bgr, 0.02)
        ax      = figure.add_subplot(4,4, 4*j+2)
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        # mirrored
        reflected = reflectAndSquareUp(cropped)
        ax        = figure.add_subplot(4,4, 4*j+3)
        plt.imshow(cv2.cvtColor(reflected, cv2.COLOR_BGR2RGB))

        # bens
        circled = circleMask(reflected)
        resized = cv2.resize(circled, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
        bens    = benYCC(resized)
        ax      = figure.add_subplot(4,4, 4*j+4)
        plt.imshow(cv2.cvtColor(bens, cv2.COLOR_BGR2RGB))


figure=plt.figure(figsize=(22,20))
test_pre_processing()

