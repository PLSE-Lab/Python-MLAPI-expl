#!/usr/bin/env python
# coding: utf-8

# # Performance comparison of reading a single image frame from a video using OpenCV (Python)

# ### Import packages

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


# # Reading the last frame of a sample video

# ### Path to the sample video

# In[ ]:


""" Path to the sample video """
filename = "/kaggle/input/samplevideo/SampleVideo_1280x720_5mb.mp4"


# ### Let's look at the properties of the video

# In[ ]:


# capture the video
cap = cv2.VideoCapture(filename)

# check if capture was successful
if not cap.isOpened(): 
    print("Could not open!")
else:
    print("Video read successful!")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print('Total frames: ' + str(total_frames))
    print('width: ' + str(width))
    print('height: ' + str(height))
    print('fps: ' + str(fps))


# ### Approach 1: Loop to the last frame

# In[ ]:


start = time.time()
cap = cv2.VideoCapture(filename)
for i in range(total_frames):
    success = cap.grab()
    if (i == (total_frames-1)):
        ret, image = cap.retrieve()
        end = time.time()
        plt.figure(1)
        plt.imshow(image)
print("Total time taken: " + str(end-start) + " seconds") 


# ### Approach 2: Grab exact frame using OpenCV properties

# In[ ]:


start = time.time()
cap = cv2.VideoCapture(filename)
cap.set(1,total_frames-1);
success = cap.grab()
ret, image = cap.retrieve()
end = time.time()
plt.figure(2)
plt.imshow(image)
print("Total time taken: " + str(end-start) + " seconds")  


# ### Let's compare Approach 1 and Approach 2 by running each 20 times

# In[ ]:


# Approach 1
timer_approach1 = []
for loop in range(20):
    start = time.time()
    cap = cv2.VideoCapture(filename)
    for i in range(total_frames):
        success = cap.grab()
        if (i == (total_frames-1)):
            ret, image = cap.retrieve()
            end = time.time()
            timer_approach1.append(end - start)
            
# Approach 2
timer_approach2 = []
for loop in range(20):
    start = time.time()
    cap = cv2.VideoCapture(filename)
    cap.set(1,total_frames-1);
    success = cap.grab()
    ret, image = cap.retrieve()
    end = time.time()
    timer_approach2.append(end - start)
    
# Plot the results
plt.figure(3)
plt.plot(timer_approach1)
plt.title('Approach 1 in seconds')

plt.figure(4)
plt.plot(timer_approach2)
plt.title('Approach 2 in seconds')

