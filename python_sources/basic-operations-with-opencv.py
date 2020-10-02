#!/usr/bin/env python
# coding: utf-8

# ## Hi Everyone and Welcome My Notebook
# 
# # Nowadays I was eager to learn opencv. This is the first kernel, but more to come.
# # The topics are as follows
# 
# - Reading Images 
# - Grayscaling
# - Color Spaces
# - Histograms
# - Drawing on Images
# 
# If You like, Pls upvote !
# Have a nice day!

# Let's start by importing the OpenCV libary 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# Let's now load our first image

# In[ ]:


# We don't need to do this again, but it's a good habit
import cv2 

# Load an image using 'imread' specifying the path to image
image = cv2.imread('/kaggle/input/operations-with-opencv/1elephant.jpg')
plt.imshow(image)
print("printed")


# ### Let's take a closer look at how images are stored

# In[ ]:


# Let's print each dimension of the image

print('Height of Image:', int(image.shape[0]), 'pixels')
print('Width of Image: ', int(image.shape[1]), 'pixels')


# ## Grayscaling
# 
# #### Grayscaling is process by which an image is converted from a full color to shades of grey (black & white)
# 
# In OpenCV, many functions grayscale images before processing. This is done because it simplifies the image, acting almost as a noise reduction and increasing processing time as there is less information in the image.
# 
# ### Let convert our color image to greyscale

# In[ ]:


# Load our input image
image = cv2.imread('/kaggle/input/operations-with-opencv/1elephant.jpg')
#plt.imshow(input)

# We use cvtColor, to convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_image)


# ## Let's take a closer look at color spaces
# 
# You may have remembered we talked about images being stored in RGB (Red Green Blue) color Spaces. Let's take a look at that in OpenCV.
# 
# ### First thing to remember about OpenCV's RGB is that it's BGR (I know, this is annoying)
# 
# Let's look at the image shape again. The '3L' 

# In[ ]:


# Load our input image
image = cv2.imread('/kaggle/input/operations-with-opencv/1elephant.jpg')
#plt.imshow(input)

# BGR Values for the first 0,0 pixel
B, G, R = image[10, 50] 
print(B, G, R)
print(image.shape)


# Let's see what happens when we convert it to grayscale

# In[ ]:


gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)
print(gray_img[10, 50]) 


# It's now only 2 dimensions. Each pixel coordinate has only one value (previously 3) with a range of 0 to 255

# In[ ]:


gray_img[0, 0] 


# ### Another useful color space is HSV 
# Infact HSV is very useful in color filtering.

# In[ ]:


#H: 0 - 180, S: 0 - 255, V: 0 - 255

# Load our input image
image = cv2.imread('/kaggle/input/operations-with-opencv/1elephant.jpg')

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
plt.imshow(hsv_image)


# ## Histograms are a great way to visualize individual color components

# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('/kaggle/input/operations-with-opencv/1coffee.jpg')
plt.imshow(image)


# In[ ]:


import cv2
import numpy as np

# We need to import matplotlib to create our histogram plots
from matplotlib import pyplot as plt

image = cv2.imread('/kaggle/input/operations-with-opencv/1coffee.jpg')

histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# We plot a histogram, ravel() flatens our image array 
plt.hist(image.ravel(), 256, [0, 256]); plt.show()

# Viewing Separate Color Channels
color = ('b', 'g', 'r')

# We now separate the colors and plot each in the Histogram
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color = col)
    plt.xlim([0,256])
    


# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('/kaggle/input/operations-with-opencv/1Hillary.jpg')
plt.imshow(image)


# In[ ]:


import cv2
import numpy as np

# We need to import matplotlib to create our histogram plots
from matplotlib import pyplot as plt

image = cv2.imread('/kaggle/input/operations-with-opencv/1Hillary.jpg')

histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# We plot a histogram, ravel() flatens our image array 
plt.hist(image.ravel(), 256, [0, 256]); plt.show()

# Viewing Separate Color Channels
color = ('b', 'g', 'r')

# We now separate the colors and plot each in the Histogram
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color = col)
    plt.xlim([0,256])
    
plt.show()


# ## Drawing images and shapes using OpenCV
# 
# Let's start off my making a black square

# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
# Create a black image
image = np.zeros((512,512,3), np.uint8)


plt.imshow( image)


# ### Let's draw a line over our black square
# 
# cv2.line(image, starting cordinates, ending cordinates, color, thickness)

# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
# Draw a diagonal  line of thickness of 5 pixels
image = np.zeros((512,512,3), np.uint8)
cv2.line(image, (0,0), (511,511), (255,127,0), 5)
plt.imshow(image)


# ### Let's now draw a rectangle
# 
# cv2.rectangle(image, starting vertex, opposite vertex, color, thickness)

# In[ ]:


# Draw a Rectangle in
image = np.zeros((512,512,3), np.uint8)

cv2.rectangle(image, (100,100), (300,250), (127,50,127), -1)
plt.imshow( image)


# ### Let's now draw a circle
# cv2.cirlce(image, center, radius, color, fill)

# In[ ]:


image = np.zeros((512,512,3), np.uint8)

cv2.circle(image, (350, 350), 100, (15,75,50), -1) 
plt.imshow(image)


# ### And Also Polygons..

# In[ ]:


image = np.zeros((512,512,3), np.uint8)

# Let's define four points
pts = np.array( [[10,50], [400,50], [90,200], [50,500]], np.int32)

# Let's now reshape our points in form  required by polylines
pts = pts.reshape((-1,1,2))

cv2.polylines(image, [pts], True, (0,0,255), 3)
plt.imshow(image)


# ### Let's even add text with cv2.putText
# 
# cv2.putText(image, 'Text to Display', bottom left starting point, Font, Font Size, Color, Thickness)
# 
# - FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN
# - FONT_HERSHEY_DUPLEX,FONT_HERSHEY_COMPLEX 
# - FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL
# - FONT_HERSHEY_SCRIPT_SIMPLEX
# - FONT_HERSHEY_SCRIPT_COMPLEX

# In[ ]:


image = np.zeros((512,512,3), np.uint8)

cv2.putText(image,'Hello World!', (75,290), cv2.FONT_HERSHEY_COMPLEX, 2, (100,170,0), 1)
plt.imshow(image)

