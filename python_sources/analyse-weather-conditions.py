#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/train_v2.csv")
tags = df["tags"].apply(lambda x: x.split(' '))
   
end = len(tags)
id_haze = []
id_cloudy = []
id_partly = []
id_clear = []

for i in range (0,end):
    for x in tags[i]:
        if x == 'haze':
            id_haze.append(i)
        elif x == 'cloudy':
            id_cloudy.append(i)
        elif x == 'partly_cloudy':
            id_partly.append(i)
        elif x == 'clear':
            id_clear.append(i)
print (len(id_haze))
print (len(id_cloudy))
print(len(id_partly))
print (len(id_clear))


# In[ ]:


import cv2
import matplotlib.pyplot as plt
import random

new_style = {'grid': True}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))
index = []
for i in range(0,9):
 
    if i <3:
        l = random.choice(id_cloudy)
        index.append(l)
    elif (i>=3 and i<6):
        l = random.choice(id_partly)
        index.append(l)
    elif (i>=6 and i<9):
        l = random.choice(id_haze)
        index.append(l)
    
    img = cv2.imread('../input/train-jpg/train_'+str(l)+'.jpg')
    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()
print (index)


# In[ ]:


import cv2
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
l = random.choice(id_cloudy)
im = cv2.imread('../input/train-jpg/train_'+str(l)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

r = im[:,:,0]
g = im[:,:,1]
b = im[:,:,2]
    
plt.subplot(1,2,1)
plt.imshow(im)
plt.subplot(1,2,2)
plt.hist(r.ravel(), bins=256, range=(0., 255))
plt.hist(g.ravel(), bins=256, range=(0., 255))
plt.hist(b.ravel(), bins=256, range=(0., 255))
plt.show()
    


# In[ ]:


import cv2
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
l = random.choice(id_partly)
im = cv2.imread('../input/train-jpg/train_'+str(l)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

r = im[:,:,0]
g = im[:,:,1]
b = im[:,:,2]
    
plt.subplot(1,2,1)
plt.imshow(im)
plt.subplot(1,2,2)
plt.hist(r.ravel(), bins=256, range=(0., 255),color='red')
plt.hist(g.ravel(), bins=256, range=(0., 255),color='green')
plt.hist(b.ravel(), bins=256, range=(0., 255),color='blue')
plt.show()


# In[ ]:


import cv2
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
l = random.choice(id_haze)
im = cv2.imread('../input/train-jpg/train_'+str(l)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

r = im[:,:,0]
g = im[:,:,1]
b = im[:,:,2]
    
plt.subplot(1,2,1)
plt.imshow(im)
plt.subplot(1,2,2)
plt.hist(r.ravel(), bins=256, range=(0., 255))
plt.hist(g.ravel(), bins=256, range=(0., 255))
plt.hist(b.ravel(), bins=256, range=(0., 255))
plt.show()


# In[ ]:


### PART2 : PLOT IMAGE
l = 38845
im = cv2.imread('../input/train-jpg/train_'+str(l)+'.jpg') 
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()


# In[ ]:


### SLINDING WINDOW 
l = 38845
im = cv2.imread('../input/train-jpg/train_'+str(l)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0],:])

#SIZE OF WINDOW

winW = 20
winH = 20 

# APPLY SLINDING WINDOWS
fenetre = []
for (x, y, window) in sliding_window(im, stepSize=32, windowSize=(winW, winH)):
	# if the window does not meet our desired window size, ignore it
	if window.shape[0] == winH and window.shape[1] == winW:
		fenetre.append(window)
        
L = len(fenetre)
idd = range(0,L,1)

l1 = random.choice(idd)
l2 = random.choice(idd)
l3 = random.choice(idd)
l4 = random.choice(idd)

# print all window
plt.subplot(2,2,1)
plt.imshow(fenetre[l1])
plt.subplot(2,2,2)
plt.imshow(fenetre[l2])
plt.subplot(2,2,3)
plt.imshow(fenetre[l3])
plt.subplot(2,2,4)
plt.imshow(fenetre[l4])

print (l1)
print (l2)
print (l3)
print (l4)




 

         


# In[ ]:


# take id cloudy and not cloudy
wind_cloudy = [11, 25 ,20 ,61,19,28]
wind_not= [30,50,52,63]


# In[ ]:


# feature extractor 
from skimage.feature import local_binary_pattern
from skimage.feature import hog 
H = []
S = []
V = []
HOG = []
LBP = []
P = 8
R = 4
L = len (fenetre)
for i in range(0,L):
    window = cv2.cvtColor(fenetre[i], cv2.COLOR_BGR2HSV)
    H.append (window[:,:,0])
    S.append ( window[:,:,1])
    V.append (window[:,:,2])
    


for i in range(0,L):
    window = cv2.cvtColor(fenetre[i], cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(window,P,R)
    hog_ft = hog(window, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(3, 3))
    LBP.append(lbp)
    HOG.append(hog_ft)


# In[ ]:


# Plot result
#index cloudy, not cloudy 
id1 = random.choice(wind_cloudy)
id2 = random.choice(wind_not)

# CLOUDY
plt.figure(figsize=(12,12))
col = 4
row = 2
plt.subplot(row,col,1)
plt.imshow(fenetre[id1])
plt.title('cloudy')
plt.subplot(row,col,2)
plt.hist(H[id1].ravel(), bins=256, range=(0., 255),color='red')
plt.hist(S[id1].ravel(), bins=256, range=(0., 255),color='green')
plt.hist(V[id1].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo HSV')
plt.subplot(row,col,3)
plt.hist(LBP[id1].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo LBP')
plt.subplot(row,col,4)
plt.hist(HOG[id1])
plt.title('HOG')

#NOT CLOUDY
plt.subplot(row,col,5)
plt.imshow(fenetre[id2])
plt.title('not cloudy')
plt.subplot(row,col,6)
counts, bins, bars = plt.hist(H[id2].ravel(), bins=256, range=(0., 255),color='red')
plt.hist(S[id2].ravel(), bins=256, range=(0., 255),color='green')
plt.hist(V[id2].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo HSV')
plt.subplot(row,col,7)
plt.hist(LBP[id2].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo LBP')
plt.subplot(row,col,8)
plt.hist(HOG[id2])
plt.title('HOG')

plt.show()
print (counts)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb

def overlay_labels(image, lbp, labels):
        mask = np.logical_or.reduce([lbp == each for each in labels])
        return label2rgb(mask, image=image, bg_label=0, alpha=0.5)

    
def create_texture_maps(image, radius, METHOD='uniform'):
    n_points = 8 * radius
    image = image + np.abs(np.amin(image))
    
    lbp = local_binary_pattern(image, n_points, radius, METHOD)

    w = width = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    i_14 = n_points // 4            # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                     list(range(i_34 - w, i_34 + w + 1)))
    
    edges = overlay_labels(image, lbp, edge_labels)
    flats = overlay_labels(image, lbp, flat_labels)
    corners = overlay_labels(image, lbp, corner_labels)

    return edges, flats, corners
im = cv2.imread('../input/train-jpg/train_'+str(3500)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

edges, flats, corners = create_texture_maps(im, radius=3, METHOD='uniform')
plt.subplot(1,2,1)
plt.imshow(im)

plt.subplot(1,2,2)
plt.imshow(flats[:,:,0], cmap='nipy_spectral')
flats.shape
#sns.distplot(flats.flatten(), kde=False)


# In[ ]:




