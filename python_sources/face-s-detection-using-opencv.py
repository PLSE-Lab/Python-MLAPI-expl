#!/usr/bin/env python
# coding: utf-8

# ## 1. Some basic steps

# In[ ]:


import os
import cv2
import matplotlib.pyplot as plt

dataset_path = "../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/"
filter_path = "../input/haarcascades/haarcascade_frontalface_default.xml"
print("Number of sub folder in images folder =",len(os.listdir(dataset_path)))


# ## 2. Define method to detect faces

# In[ ]:


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(filter_path)
    bboxes = face_cascade.detectMultiScale(image, 1.3, 5)
    return bboxes


# ## 3. Define method to draw bounding boxes around the face

# In[ ]:


def draw_bounding_boxes(image, bboxes):
    for box in bboxes:
        print("\nBounding box co-ordinates =",box)
        x1, y1, w, h = box
        cv2.rectangle(image, (x1, y1), (x1+w,y1+h), (0,255,0), 4)


# ## 4. Load an image, detect face and draw bounding box around the face
# 
# - Let's choose a set of images and see how face is being detected

# In[ ]:


jk_rowling_imgs = "../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/JK_Rowling/"

print("List of images in J.K.Rowling folder:\n",os.listdir(jk_rowling_imgs))


# In[ ]:


# to display images using matplotlib
plt.figure(1, figsize = (20, 5))

for index, image in enumerate(os.listdir(jk_rowling_imgs)):
    image_path = jk_rowling_imgs + image
    print("\n",index+1,"Image file path =",image)
    
    # read image as numpy array
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # get bounding box co-ordinates for the image
    bboxes = detect_faces(image)
    
    # draw bounding box around the face
    draw_bounding_boxes(image, bboxes)
    
    # display the image with bounding box drawn
    plt.subplot(1, len(os.listdir(jk_rowling_imgs)), index+1)
    plt.subplots_adjust(wspace = 0.3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)

plt.show

