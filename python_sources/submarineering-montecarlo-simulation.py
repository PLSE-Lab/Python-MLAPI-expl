#!/usr/bin/env python
# coding: utf-8

# <img src="https://lh3.googleusercontent.com/-tNe1vwwd_w4/VZ_m9E44C7I/AAAAAAAAABM/5yqhpSyYcCUzwHi-ti13MwovCb_AUD_zgCJkCGAYYCw/w256-h86-n-no/Submarineering.png">

# # MonteCarlo Simulation tutorial in Python 3 
# MonteCarlo Simulation consist in testing all the possibles outcomes in a operation before to take any decision. Normally is very useful in risk assessment.
# But here, we are going apply MC method ,in a very different approach, to understand how it works and see in what way we could take advantage for any investigation purpose.
# 
# To know more about: https://en.wikipedia.org/wiki/Monte_Carlo_method
# 
# 

# As we know, Pi number is mathematical constant  with value: 3.14159265358979323846...

# We are going to use this number and a pair of good known formulas of Geometry to show MC power.
# Imagine, you do not know the Pi value.  We are going to approximate the number as we if were many years ago.
# All we know the area of a square is side times side.
# And the area of a circle is a constant times squared Diameter and all divided by 4.
# Let's suppose we are in a beautiful beach and you can find easily as stones as you need.

# <img src="https://i.pinimg.com/originals/ea/2d/ba/ea2dba8abb17038c89fe162fbe9891a7.jpg">

# Now go to a flat place and draw a square with side equal to 2 meters.

# <img src="https://img.etsystatic.com/il/a3e897/1540403924/il_340x270.1540403924_5smb.jpg">

# After collect a lot of little stone, this is going to be the point. We are going to throw one by one stone inside of the square. Finally we will count how many stones are inside the square to see the relation to the total stones.

# In[ ]:


#import libraries
import numpy as np
import matplotlib.pyplot as plt
  


# In[ ]:


#area of the bounding box, square of side 2 meters.
box_area = 4.0


# In[ ]:


#number of samples or stones to trhow.
N_total = 1000000


# In[ ]:


#drawing random points uniform between -1 and 1
X = np.random.uniform(low=-1, high=1, size=N_total)  
Y = np.random.uniform(low=-1, high=1, size=N_total)   


# In[ ]:


# calculate the distance of the point from the center 
distance = np.sqrt(X**2+Y**2);  


# In[ ]:


# check if point is inside the circle (diameter 2 meters)   
is_point_inside = distance<1.0


# In[ ]:


# sum up the stones inside the circle
N_inside=np.sum(is_point_inside)


# In[ ]:


# estimate the circle area
circle_area = box_area * N_inside/N_total


# In[ ]:


# some nice visualization
plt.scatter(X[0:1000],Y[0:1000],  s=40,c=is_point_inside[0:1000],alpha=.6, edgecolors='None')  
plt.axis('equal')

# text output
print ('Area of the circle = ', circle_area)
print ('pi = ', np.pi)
plt.show()


# In[ ]:




