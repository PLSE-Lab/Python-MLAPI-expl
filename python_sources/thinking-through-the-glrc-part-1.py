#!/usr/bin/env python
# coding: utf-8

# # What's the point?
# 
# I'm starting this challenge incredibly late, so I'm not even sure I'll get a credible submission made before it is over.  Why bother?  I guess because it's there ;^)
# 
# I am still quite new to Kaggle and Machine Learning, so this will simply be a learning exercise rather than a true competition for me.  I hope I can learn, and maybe help some others along the way.

# ## Thank-you Andrew Ng...
# 
# After taking 6 Coursera classes from him, I hope I've picked up a few reasonable ideas about how to approach a machine learning project.
# 
# ### Step 1 - What is the problem you are trying to solve?
# 
# The name gives you the general idea - "Label famous (and not-so-famous) landmarks in images.  I can think of 4 different reasons why I would want to do this - and each one would dictate a different approach
# 
# #### Academic
# 
# If I take this as an academic problem, then I would probably just focus on the computer vision aspects.  Can I find novel new algorithms, network structures, or clever reuse of solutions created for different (but similar) problems.  I would focus on the idea more than the absolute performance.  A good idea, even if it isn't fully developed, is better than a brute force solution.  YOLO when it was first introduced wasn't the most accurate solution available for Object Detection.  It was much faster than most, but not most accurate.  However, with further development, it was better and faster than other approaches.
# 
# #### Production
# 
# If I think of this as a real problem to be solved with the solution to be deployed as a product, I would have very different concerns and constraints than if it were simply an academic exercise.  I would set goals for accuracy, speed of execution, computational cost, etc. and I would measure my solution against them.  I would accept a solution that is "good enough" to satisfy my customers at a cost I can afford.  The nature of a software product or service is that you can always improve it over time.
# 
# #### Competition
# 
# This is, obviously, the approach kaggle wants us to take.  It is a competition.  That means optimizing the scoring metric is all that matters.  Beautiful, clever solution - nice to have, but if a massive brute force approach gets a better score use it!  Spend 5 times the computational cost to create an enseble model that will result in a preformance gain of a few percent - do it!  Time and compute resources available are your only real limitations!
# 
# #### Forensic
# 
# This is an interesting use case that I hadn't considered initially.  Think of a state actor (law enforcement, security services, military, etc.) that has obtained an image of an interesting subject (i.e. two people talking on the street of some city).  They want to know where this happened, to hope to take the next step in an investigation.  I think this drives behaviour most like the competition approach above.  The value of that piece of information may be large enough to justify extreme solutions.
# 
# #### So...
# 
# I'm sure I'm starting to bore you, so I'll wrap up the philosophy class and get to some code!!!
# 
# It turns out than I'm not an academic, I don't expect to win this competition, and I've never worked for a spy agency.  I think I'll approach this as though it was something I would deploy as a product (service) if I ran a large social media photo sharing operation

# ### Step 2 - Let's look at the data
# 
# Before we make any decisions on implementation plans, let just spend some time poking around...
# 
# I started at home using the excellent script provided by Anokas to download the test and train data https://www.kaggle.com/anokas/python3-dataset-downloader-with-progress-bar
# 
# However, you need to be able to see some results in this kernel, so I'll use what I learned offline to show you some things online...

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import Image
from IPython.core.display import HTML 
from urllib import request


# In[3]:


# read in the list of train and test photos

# Kaggle version
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Let's make sure that went as expected...  There should be 117703 test photos and 1225029 train photos

print("Number of photos for testing = ", test.shape[0])
print("Number of photos for training = ", train.shape[0])


# In[4]:


# A utility for displaaying thumbnails of images
# Taken from the very nice Kernel by Gabriel Preda
# "Google Landmark Recogn. Challenge Data Exploration"

def displayLandmarkImagesLarge(urls):
    
    imageStyle = "height: 150px; margin: 2px; float: left; border: 1px solid blue;"
    imagesList = ''.join([f"<img style='{imageStyle}' src='{u}' />" for _, u in urls.iteritems()])

    display(HTML(imagesList))


# ### Let's take a look at a random sample of images from the train dataset

# In[5]:


np.random.seed(42) # my favorite "random" seed, change it if you want other images

urls = [] # start with an empty list...

for i in range (20):
    urls.append(train.iloc[np.random.randint(0,1225029),1])
    
urls = pd.Series(urls)

displayLandmarkImagesLarge(urls) # Thanks Gabriel!


# Ok, this looks pretty reasonable.  A couple of waterfalls, churches, a mountain, seashore, and places with names in the photos.  I'm not sure what is going on with the girl standing on a rock - maybe it is a famous rock.

# ### Let's take a look at a random sample from the test images

# In[6]:


np.random.seed(1960) # guess how old I am..., change it if you want other images

urls = [] # start with an empty list...

for i in range (20):
    urls.append(test.iloc[np.random.randint(0,117703),1])
    
urls = pd.Series(urls)

displayLandmarkImagesLarge(urls) # Thanks again Gabriel!


# Well this is interesting... It looks like 7 of these are definitely NOT landmarks (although those are some bright white teeth!)
# 
# Of the photos that do look like specific locations, only one or two capture what I would expect to be an identifiable landmark.
# 
# Maybe this is a reasonable representation of what you might expect if you were trying to classify photos added to a social media photo sharing app.  Most of them are photos of our lunch or selfies of us with our kids and pets!
# 
# I checked a few other random seeds, and my observation is that 25-50% of the test images are definitely not landmarks.
# 
# I'm not sure what to do with that information right now, but I'll keep it locked away for future reference...

# ### A few other observations
# 
# I assume most of the participants downloaded the test and train images already.  As I mentioned above I did it using Anokas script.  It took a couple of hours to download the test images.  I was able to get 115,493 out of the total 117,703 (~98%).  I let the train images download overnight, so they took 10-12 hours.  For them I got 1,217,387 out of 1,225,029 (~99%).
# 
# The total disk space used was:
# 
#     Test - 32.5GB
#     Train - 334GB
# 
# I decided to resize them, since many of the images were 1200x1600.  Most CNN's use much smaller images.  I resized so that the shorter edge of the image was 256 pixels.
# 
# This greatly reduced the storage space required:
# 
#     Resized Test - 2.5GB
#     Resized Train - 25GB
# 
# I'll share the code I used for the resizing in post later this week
# 
# Looking at the original dimensions of the test images, none have a short side dimension less than 160px
# 
# On the other hand, in the training data, there are >15,000 with a short side dimension less than 160px.  Some are as small as 10x15.  Just how useful are these?
# 
# Lets take a look

# In[7]:


# Review some of the VERY small images in the training set

# A handful of 10x15 images

urls = []
urls.append('https://lh4.googleusercontent.com/-2aSO8nzNfeY/Tne4ZHE2ZKI/AAAAAAAAC1Y/5eRU1tfinQI/s15/')
urls.append('https://lh3.googleusercontent.com/-SXCAgqmUSCY/TKKFZqwVxxI/AAAAAAAADbw/H440k4K4rlY/w11-h15/')
urls.append('https://lh5.googleusercontent.com/-wgFxt042p-4/SkdfH8QuuWI/AAAAAAAADuw/F2hmQxBuVdc/s15/')
urls.append('https://lh4.googleusercontent.com/-qyGFiv31etQ/RzHbxBZUXkI/AAAAAAAABV8/vsSEsNKwQLM/s15/')
urls.append('https://lh4.googleusercontent.com/-1GaFiQamJmU/SYdLH0vdjBI/AAAAAAAAFck/2vwbPssj1xg/s15/')

urls = pd.Series(urls)

displayLandmarkImagesLarge(urls)


# These don't look very useful - but hey, notice that the ending for all of them is something like '/s15/'
# 
# Maybe there are larger versions available...

# In[8]:


# Review some of the VERY small images in the training set

# A handful of 10x15 images

urls = []
urls.append('https://lh4.googleusercontent.com/-2aSO8nzNfeY/Tne4ZHE2ZKI/AAAAAAAAC1Y/5eRU1tfinQI/')
urls.append('https://lh3.googleusercontent.com/-SXCAgqmUSCY/TKKFZqwVxxI/AAAAAAAADbw/H440k4K4rlY/')
urls.append('https://lh5.googleusercontent.com/-wgFxt042p-4/SkdfH8QuuWI/AAAAAAAADuw/F2hmQxBuVdc/')
urls.append('https://lh4.googleusercontent.com/-qyGFiv31etQ/RzHbxBZUXkI/AAAAAAAABV8/vsSEsNKwQLM/')
urls.append('https://lh4.googleusercontent.com/-1GaFiQamJmU/SYdLH0vdjBI/AAAAAAAAFck/2vwbPssj1xg/')

urls = pd.Series(urls)

displayLandmarkImagesLarge(urls)


# Aha!  Much more useful.  Seems like there is some work to be done to clean up the data we have.  And that will be the subject of my next post.  See you soon.

# In[ ]:




