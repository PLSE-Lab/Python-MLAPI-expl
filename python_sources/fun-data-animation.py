#!/usr/bin/env python
# coding: utf-8

# # How to Make an Animation!
# The data for the "Don't Overfit! II" competition has 300 features. Therefore it resides in 300 dimensional space and we have trouble visualizing it. In this kernel, we project it into 2 dimensional space and create an animation by rotating it through a third dimension. This visualization shows us that the training data target=1 and target=0 is separable by a hyperplane.  
#   
# ![image](http://playagricola.com/Kaggle/data.gif)
# 
# # Load the data

# In[ ]:


# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from sklearn.linear_model import LogisticRegression

# LOAD THE DATA
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
train.head()


# Each of the 300 variables has a Gaussian distribution with mean 0 and standard deviation 1. We can see this by plotting histograms of the train and test data combined. Therefore the 20000 data points reside within a 300 dimensional hypersphere of approximate radius 3 centered at the origin (zero).

# In[ ]:


plt.figure(figsize=(15,15))
for i in range(5):
    for j in range(5):
        plt.subplot(5,5,5*i+j+1)
        plt.hist(test[str(5*i+j)],bins=100)
        plt.title('Variable '+str(5*i+j))
plt.show()


# If we plot only variable 33 and 65 from the 250 training data points and color target=1 yellow and target=0 blue, we see that the data may be separable. We can confirm this with logistic regression on all 300 variables below.

# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(train['33'],train['65'],c=train['target'])
plt.plot([-1.6,1.4],[3,-3],':k')
plt.xlabel('variable 33')
plt.ylabel('variable 65')
plt.title('Training data')
plt.show()


# # Logistic regression finds a hyperplane!
# The coefficients to logistic regression are the normal of (perpendicular direction to) the hyperplane that separates the data in 300 dimensional space. This direction is variable `u1` below. Next we create two random perpendicular directions `u2` and `u3`. Thus if we project all the points onto the plane created by `u1` and `u3` we will see if the train data is separable. Next we can rotate through the direction `u2` and view an animation.

# In[ ]:


# FIND NORMAL TO HYPERPLANE
clf = LogisticRegression(solver='liblinear',penalty='l2',C=0.1,class_weight='balanced')
clf.fit(train.iloc[:,2:],train['target'])
u1 = clf.coef_[0]
u1 = u1/np.sqrt(u1.dot(u1))


# In[ ]:


# CREATE RANDOM DIRECTION PERPENDICULAR TO U1
u2 = np.random.normal(0,1,300)
u2 = u2 - u1.dot(u2)*u1
u2 = u2/np.sqrt(u2.dot(u2))


# In[ ]:


# CREATE RANDOM DIRECTION PERPENDICULAR TO U1 AND U2
u3 = np.random.normal(0,1,300)
u3 = u3 - u1.dot(u3)*u1 - u2.dot(u3)*u2
u3 = u3/np.sqrt(u3.dot(u3))


# # Create an animation

# In[ ]:


# CREATE AN ANIMATION
images = []
steps = 60
fig = plt.figure(figsize=(8,8))
for k in range(steps):
    
    # CALCULATE NEW ANGLE OF ROTATION
    angR = k*(2*np.pi/steps)
    angD = round(k*(360/steps),0)
    u4 = np.cos(angR)*u1 + np.sin(angR)*u2
    u = np.concatenate([u4,u3]).reshape((2,300))
    
    # PROJECT TRAIN AND TEST ONTO U3,U4 PLANE
    p = u.dot(train.iloc[:,2:].values.transpose())
    p2 = u.dot(test.iloc[:,1:].values.transpose())
    
    # PLOT TEST DATA
    img1 = plt.scatter(p2[0,:],p2[1,:],c='gray')
    
    # PLOT TRAIN DATA (KEEP CORRECT COLOR IN FRONT)
    idx0 = train[ train['target']==0 ].index
    idx1 = train[ train['target']==1 ].index
    if angD<180:
        img2 = plt.scatter(p[0,idx1],p[1,idx1],c='yellow')
        img3 = plt.scatter(p[0,idx0],p[1,idx0],c='blue')
    else:
        img2 = plt.scatter(p[0,idx0],p[1,idx0],c='blue')
        img3 = plt.scatter(p[0,idx1],p[1,idx1],c='yellow')
        
    # ANNOTATE AND ADD TO MOVIE
    img4 = plt.text(1.5,-3.5,'Angle = '+str(angD)+' degrees')
    images.append([img1, img2, img3, img4])
    
# SAVE MOVIE TO FILE
ani = ArtistAnimation(fig, images)
ani.save('data.gif', writer='imagemagick', fps=15)


# # Conclusion
# In this kernel, we learned how to create an animation. This animation shows us that the training data target=1 and target=0 are separable. Also, it can be shown that if we use the hyperplane that separates the training data as our classification decision boundary for the test data, we will score LB 0.75 in this competition.

# In[ ]:


from IPython.display import Image
Image("../working/data.gif")


# ![image](http://playagricola.com/Kaggle/data.gif)
