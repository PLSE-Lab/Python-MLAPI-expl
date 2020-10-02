#!/usr/bin/env python
# coding: utf-8

# ### <font color=black> Welcome to arithmetic Reconstruction technique (ART) notebook!</font>
# 

# <font size=3>here we are going to develop a simple algorithm to find the attenuation in only the 2x2 image below.</font>
# <img src="https://i.imgur.com/NdkaMmg.png" width="500px">
# 
# 

# In[ ]:


import numpy as np


# In[ ]:


img=np.zeros((2,2),dtype=float) #initialize the image to be zeros
no_cols=img.shape[1]
no_rows=img.shape[0]
print(img)


# In[ ]:


#saving the attenuation values
horizontal_projection=np.array([0.6,0.3]) #top to bottom
vertical_projection=np.array([0.4,.5])    #left to right
diagonal_projection=np.array([0.3])      #top to bottom
reverse_projection=np.array([0.6])        #top to bottom
print(horizontal_projection,vertical_projection,diagonal_projection,reverse_projection)


# In[ ]:


#horizontal phase 
#a vectorized effecient code that doesn't use loops
horizontal=np.sum(img,axis=1) #sums over rows
error=(horizontal_projection-horizontal)/no_cols
img+=error[:,np.newaxis] #broadcasting the addition to the rows
print("after horizontal phase:\n",img)


# In[ ]:


#vertical phase
##a vectorized effecient code that doesn't use loops
vertical=np.sum(img,axis=0) #sums over columns
error=(vertical_projection-vertical)/no_rows
img+=error[np.newaxis,:] #broadcasting the addition to the columns
print("after vertical phase:\n",img)


# In[ ]:


#main diagonals phase
#to be vectorized
a=0
for a in range(img.shape[0]-1):
    main=img[a,a]+img[a+1,a+1]
    error=diagonal_projection[a]-main
    img[a,a]=img[a,a]+(error/2)
    img[a+1,a+1]=img[a+1,a+1]+(error/2)
print("after main diagonal phase:\n",img)


# In[ ]:


#reverse diagonals phase
#to be vectorized
b=0
for b in range(img.shape[0]-1):
    reverse=img[b,b+1]+img[b+1,b]
    error=reverse_projection[b]-reverse
    img[b+1,b]=img[b+1,b]+(error/2)
    img[b,b+1]=img[b,b+1]+(error/2)
print("after reverse diagonal phase:\n",img)


# <font color=green, size=5>done!</font>

# ## <font color=black>future modifications</font>:
# - complete code vectorization which will generalize the program to NxM images effeciently
# - making the code modular by using the concepts of oop/functions

# ### thanks!
