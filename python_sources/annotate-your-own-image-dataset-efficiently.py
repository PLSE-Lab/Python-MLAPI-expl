#!/usr/bin/env python
# coding: utf-8

# # Annotating your image data efficiently
# While creating your own dataset for a project, the hardest part is annotating the data, infact once you are done with the annotation, the battle is half won. If you are also looking for a way to quickly get done with the boring and yet the most important part of your project, then follow along.
# (P.S.-This is my first Kaggle notebook, so please take the time to point out my mistakes, it will help me get familiar with Kaggle)
# 
# In this notebook, I will be using basic image processing techniques available inside OpenCV to help annotate our images dataset for tasks like classification fairly easily and efficiently.
# ## Dependencies
# Before proceeding, I want to lay out the required packages and libraries:
# 1. OpenCV
# 2. imutils(this is a set of convenience function by Adrian Rosebrock, and saves a lot of time)
# 
# I used this to annotate the captcha images which I had downloaded while working on a project. The captcha images consisted of four numeric digits each. With a little tweaking it can be used to annotate your own image dataset.
# Let's get started,
# 

# In[ ]:


#import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import os
import sys

#Parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True,help="path to input directory of images")
ap.add_argument("-a","--annot",required=True,help="path to output directory of annotations")
args=vars(ap.parse_args())


# First, we import the necessary packages and then parse our command line arguments(the arguments provided at the time of running the script). This script requires two arguments:
# * --input: The input path to our raw images
# * --annot:The output path 
# 

# In[ ]:


#grab the image paths and initialize the dictionary of character counts

imagePaths=list(paths.list_images(args["input"]))
counts={}


# This code block grabs the paths to all images in the --input directory and initializes a
# dictionary named counts that will store the total number of times a given key has been pressed.

# In[ ]:


#loop over image paths 
for (i,imagePath) in enumerate(imagePaths):
    #display an update
    print(f"[PROCESSING IMAGES]Currently processing {i+1}/{len(imagePaths)}")
    try:
         #loading the images and converting them to grayscale
         #add a border to ensure the digits on the border of the image are retained
         image=cv2.imread(imagePath)
         gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
         gray=cv2.copyMakeBorder(gray,8,8,8,8,cv2.BORDER_REPLICATE)


# We loop over each individual image paths. For each image, we load it from the disk, convert it to grayscale and then pad the borders of the image. We pad the image to prevent any loss of information from the border edges while processing the images further.

# In[ ]:


#threshold image 
         thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]


# We convert the image to binary by thresholding it, to separate the foreground from the background.

# In[ ]:


#find contours in the image,keeping only the four largest ones
        cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts=cnts[0]
        cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:4]
     


# Next we find contours in the image, to detect objects in the image. In my case, this happened to be four digits, therefore I sort the contours on the basis of their area and keep the four largest ones which will represent the digits, in case more contours are detected in the image owing to noise present in the image.

# In[ ]:


#Loop over the contours
        for c in cnts:
            x,y,w,h=cv2.boundingRect(c)
            roi=gray[y-5:y+h+5,x-5:x+w+5]
            #display the character and then wait for keypress
            cv2.imshow("ROI",imutils.resize(roi,width=100))
            key=cv2.waitKey(0)


# Now we loop over the contours and extract them by computing their bounding box. We loop over each of the contours found in the thresholded image. We call cv2.boundingRect to compute the bounding box (x, y)-coordinates of the digit region. This region of interest (ROI) is then extracted from the grayscale image.
# ROI is then displayed and then it waits for a keypress, we have to be careful while pressing the key because this key will be used to label your ROI(you can pre-decide a key which represents a category in your image dataset).

# In[ ]:


#if the '`' (tilde)key is pressed then the ROI is ignored
if key==ord("~"):
                 print("[INFO] IGNORING....")
                 continue
             #grab the key that was pressed and construct the path to output directory
             key=chr(key).upper()
             dirPath=os.path.sep.join([args["annot"],key])
             #if the directory is not present then create it
             if not os.path.exists(dirPath):
                 os.makedirs(dirPath)


# If the tilde(~) key is pressed then the object(ROI) is ignored, this may happen if our script accidentally detects noise. Otherwise we assume that the key pressed was the label and the key is used to construct the directory path of our output label.
# For example if I pressed the *key 'C'* and the --annot argument was dataset, then dirPath would be:
# *dataset/C*
# Since in my case I had to label digits,
# If I pressed the *key 1*, then all the images containing the digit "1" would be stored in the *dataset/1* sub directory.
# Then, it is checked if the dirPath directory exists, if it doesn't then we create the directory.

# In[ ]:


#writing to file   
             count=counts.get(key,1)
             p=os.path.sep.join([dirPath,f"{str(count).zfill(6)}.png"])
             cv2.imwrite(p,roi)
             #increment the count for the current key
             counts[key]=count+1


# The first line in the above code block grabs the total number of examples written to disk thus far for the current key. We then construct the output path using the dirPath. We then write to the output path p and the last line updates the counter for the current key.
# Say we are building the cats and dogs classifier wherein C represents a Cat in the image then after writing the image the output path p will look like:
# *dataset/C/000001.png*
# Now you see, how all the images containing cats will be stored in the subdirectory *dataset/C*  this is an easy, convenient way to organize your datasets when labeling images.

# In[ ]:


#Handles the keyboard interrupt when we want to stop annotation         
except KeyboardInterrupt:
    print("[INFO]Ugh I dont want to do this anymore....that's what she said")
    break
#catch the unexpected exception and print its type and line number
except Exception as e:
    exception_type,exception_object,exception_traceback=sys.exc_info()
    line_number=exception_traceback.tb_lineno
    print("Exception type:",exception_type)
    print("Line Number",line_number)
    print("[INFO] skipping image......")

cv2.destroyAllWindows()      


# Our final code blocks handles if we want to quit the script or an unexpected error occurs.
# If we want to exit the script then the first line in this code block detects this and allows us to exit the program gracefully. The last block catches all other exceptions, prints the type of exception and the line number and then ignores them to let us continue with the labelling process.

# Now that you have your labelled dataset, what is topping you from training your own network.
# 
# ### If you have reached this far, please leave an upvote for a wider reach.
# Any and all questions and suggestions are always welcome in the comments.
# #### Thank You !
# 
# 
# 
# P.S.-(This notebook is based on my understanding of a chapter by Adrian Rosebrock)
