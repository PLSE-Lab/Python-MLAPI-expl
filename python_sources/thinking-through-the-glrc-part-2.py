#!/usr/bin/env python
# coding: utf-8

# # Part 2 - Broad outline, some data clean-up, and a few decisions

# ### High level plan...
# 
# I'll pick up where we left off yesterday, but first I'd like to give a high level plan of where I think this will ultimately go...
# 
# As I said yesterday, I'm going to take an approach as though I was building a product - not trying to win a competition.
# 
# So - steps I think you should take to build this product
# 
#     1) Decide what problem you are trying to solve (see previous discussion)
#     2) Look at the data to get a feel for it.  I started this yesterday and I'll do a bit more today
#     3) Clean up the data as necessary for it to be useful
#     4) Pick an approach and get started
#     5) Implement a quick and dirty model
#     6) Evaluate the model's performance
#     7) Look at the problems and failures to get an idea where the model should be improved
#     8) Update the model
#     9) Repeat 6-8 until you are satisfied with the final results!
#     

# ### Step 2 - Look at the data (cont.)
# 
# Yesterday I found that a large number of the training images were very small with sizes like 15px x 11px
# 
# In fact, I found >23,000 images with the smaller edge less than 256px.
# 
# To "solve" that problem I took advantage of the fact that most of these images come from google user content and exist of a base image plus the ability to resize.
# 
# I just modified the url to take the base image.  You saw thoose results previously, so I won't go through it again.
# 
# However, I this is the function I used to modify the url.  I'm certain there is a more elegant way to do this, but what the heck.  It worked and took 3 minutes to write!

# In[1]:


def url_stripper(old_url):
    
    new_url = old_url
    #length = len(old_url)
    position = -1
    done = False
    change = False
    if old_url[position] == '/':
        while not done:
            position -=1
            if old_url[position] == '/':
                done = True
                change = True
    if change:
        new_url = old_url[:position+1]
    
    return new_url


# In[2]:


# A simple demonstration of url_stripper

url = 'https://lh3.googleusercontent.com/-SXCAgqmUSCY/TKKFZqwVxxI/AAAAAAAADbw/H440k4K4rlY/w11-h15/'
print('old url: ', url)
url = url_stripper(url)
print('new url: ', url)


# #### Looking some more at the data...
# 
# Let's look at the training data to understand how many unique landmarks we have...

# In[4]:


import pandas as pd
train_data = pd.read_csv('../input/train.csv') #kaggle version
#train_data = pd.read_csv('./train.csv') # local version

# find out how many unique landmark ids there are, and how many instances of each one
landmark_ids = train_data['landmark_id'].value_counts() 
print('there are', landmark_ids.shape[0], 'unique landmarks')


# In[5]:


print('the top 20 landmarks are:')
print('id       count')
print(landmark_ids.head(20))


# Notice that there is a 10x reduction in samples from the top landmark to the 20th landmark!

# In[6]:


counts = landmark_ids.values
index = landmark_ids.index
size = 1024
for i in [1024,2048,4096, 8192]:
    percentage = 100*(counts[0:i].sum()/counts.sum())
    print('the top %d landmarks account for %5.2f percent of the training samples' %(i,percentage))


# ### Time to make some choices
# 
# I want to jump into the modelling, and I want to keep it simple to begin.
# 
# I think I'll use a pre-trained ResNet50 network and retrain the output stages for a new softmax layer with 1024 categories (the top 1023 landmarks and a "don't see a landmark" categotry).
# 
# That will almost certainly not be very good, since it will only cover ~2/3 of the landmarks inthe training set, but hopefully it will be easier to set up, start training, etc.  I can use the number of landmarks / categories as a hyper-parameter later if I want to see how it improves with 2k or 4k landmarks.

# #### Next topic - setting up a ResNEt50 network and using transfer learning to start identifying landmarks

# In[ ]:




