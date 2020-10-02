#!/usr/bin/env python
# coding: utf-8

# Hello everyone, so this competition I couldn't get the idea I wanted working but I came to some interesting conclusions and felt they were worth sharing. I hope for this kernel to be a small antithesis to the last second blending kernels that screw up the rankings without offering anything to the community. One aspect I really spent a great deal of time on was the image data. I am sure the top teams have found some great way to handle the large quantity of images and an inteligent model or features to extract the information from this, but I figured I would share the research process I have gone through and possibly help someone along so they dont have to do some of the fundamental testing I went through.  

# First off lets look at the process of loading in 1.3 million images. We have many options here. 

# Speed on this is very important. You don't want to have idle gpu and be waiting on disk to load in the images. I actually never tried with the zip file. I just went for the unzipped files and only ever dealt with those. That may be the way to go, I do not know. My assumption was that repeatedly unzipping the files would be a waste. Even if that the case, it is very likely someone will come across a problem where they don't received the images zipped like that and this will be relevant to them still. Anyway, lets start by loading in just the images and seeing what various libraries can do in terms of loading in the images

# In[ ]:


#I don't think this line needs explaining
import pandas as pd


# In[ ]:


#load the train csv and grab the images
images = pd.read_csv("../input/train.csv")[["image"]]


# In[ ]:


#let's measure some time
import time


# We'll start with PIL to load the images. I am sure many of you are familiar with this library. 

# In[ ]:


#I'm sure importing open is probably a terrible pythonic sin, but yolo
from PIL.Image import open as PILRead


# Sorry but this code won't work on a kernel as it is all under the assumption that you have already unzipped the jpg folders. 
# 
# First thing we will do is measure time to iterate through the first thousand images and see how long it takes simply to load the image.

# In[ ]:


start = time.time()
for image in images[0:1000]:
        img = PILRead('./data/competition_files/train_jpg/' + str(image) + ".jpg")
print(time.time() - start)


#     ---------------------------------------------------------------------------
#     FileNotFoundError                         Traceback (most recent call last)
#     <ipython-input-41-33e0c0720396> in <module>()
#           1 start = time.time()
#           2 for image in X["image"][0:1000]:
#     ----> 3         img = PILRead('./data/competition_files/train_jpg/' + str(image) + ".jpg")
#           4 print(time.time() - start, i/1000)
# 
#     c:\users\magic\appdata\local\programs\python\python35\lib\site-packages\PIL\Image.py in open(fp, mode)
#        2546 
#        2547     if filename:
#     -> 2548         fp = builtins.open(filename, "rb")
#        2549         exclusive_fp = True
#        2550 
# 
#     FileNotFoundError: [Errno 2] No such file or directory: './data/competition_files/train_jpg/nan.jpg'
# 

# Drats. Errors. Looks like we have some missing images. We'll just return an array of zeros of our target size if we cant load an image properly I guess. Let's also keep track of how many aren't able to be loaded for whatever reason.
# Second Try:

# In[ ]:


i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = PILRead('./data/competition_files/train_jpg/' + str(image) + ".jpg")
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)

0.19799304008483887 0.921
Looks like my favorite tool to make the bad men go away, the try clause, saves the day again. Hey that's pretty fast. Missing that many images isn't perfect but that's just part of the data science gig. PIL looks like a decent candidate
# In[ ]:


from cv2 import imread as cvimread


# In[ ]:


start = time.time()
for image in X["image"][0:1000]:
    img = cvimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
print(time.time() - start)


# 2.5197324752807617. That's way slower, why is that? and why didn't it error out like the last one? Time for exploration. Lets check some shapes

# In[ ]:


start = time.time()
for image in X["image"][0:1000]:
    img = cvimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
    print(img.shape)
print(time.time() - start)


#     (480, 358, 3)
#     (480, 360, 3)
#     (360, 392, 3)
#     (360, 360, 3)
#     (360, 640, 3)
#     (480, 360, 3)
#     (360, 480, 3)
#     (480, 360, 3)
#     (480, 360, 3)
#     (480, 270, 3)
#     (480, 270, 3)
#     (480, 360, 3)
#     (463, 360, 3)
#     (360, 480, 3)
#     (360, 480, 3)
#     (480, 320, 3)
#     (360, 640, 3)
#     (480, 360, 3)
#     (480, 270, 3)
#     ---------------------------------------------------------------------------
#     AttributeError                            Traceback (most recent call last)
#     <ipython-input-48-7289f6063999> in <module>()
#           2 for image in X["image"][0:1000]:
#           3     img = cvimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
#     ----> 4     print(img.shape)
#           5 print(time.time() - start)
# 
#     AttributeError: 'NoneType' object has no attribute 'shape'
#     
# Looks like cv2 is returning None instead of throwing an error. Still doesn't explain why it is so slow. Let's make sure it is returning what we want with both cv2 and PIL by checking the type

# In[ ]:


type(img)


#     numpy.ndarray
# Perfect. cv2 seems to be returning what we expect. An array with all of the pixel values. Let's be lazy and copy and paste the code from above for PIL and check again. 

# In[ ]:


i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = PILRead('./data/competition_files/train_jpg/' + str(image) + ".jpg")
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)


#     PIL.JpegImagePlugin.JpegImageFile
# 
# Huh, you're not my mom? On further research it looks like PIL isn't really doing what I intended, but we can try to convert this object into what we need, an array, using a keras utility.
# 

# In[ ]:


from keras.preprocessing.image import img_to_array


# In[ ]:


i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = img_to_array(PILRead('./data/competition_files/train_jpg/' + str(image) + ".jpg"))
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)


#     3.4392542839050293 0.921
#     numpy.ndarray
# There we go. More what we wanted but looks like it is actually slower than cv2 now. There might exist a faster image to array tool, but I didnt do any additional research on this. Let's look and compare with some other image loaders though

# Trusty scipy has some tool that looks like it might work

# In[ ]:


from scipy.misc import imread as scimread
i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = (scimread('./data/competition_files/train_jpg/' + str(image) + ".jpg"))
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)


#     2.7287728786468506 0.921
# Pretty good but gives me a deprecation warning though. Let's try what they recommend.

# In[ ]:


from imageio import imread as ioimread

i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = ioimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)


#     2.8476409912109375 0.921
#     imageio.core.util.Image
# Did some research and seems like this bizarre type would work just fine like a numpy array, but weird type and slightly slower. Probably not worth.

# In[ ]:


from matplotlib.image import imread as matimread
i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = (matimread('./data/competition_files/train_jpg/' + str(image) + ".jpg"))
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)


# matplotlib's library looks like it does pretty well and returns what we expect but still marginally slower than cv2. We are dealing with relatively small margins but still. If we are pushing this to so many images it might be worth it to care about these small margins. 
# 2.780322313308716 0.921
# numpy.ndarray

# Now we'll try keras's tools. Those must be fast. They've surely considered and curated the fastest implementations already. We'll chain together their load_img and img_to_array to get an array like we expect

# In[ ]:


from keras.preprocessing.image import load_img
i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = img_to_array(load_img('./data/competition_files/train_jpg/' + str(image) + ".jpg"))
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)


#     3.6963675022125244 0.921
#     numpy.ndarray
# Well looks like cv2 it is. All we have to do is deal with the small quirk where it returns None sometimes. 

# In[ ]:


start = time.time()
for image in X["image"][0:1000]:
    img = cvimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
    try:
        img.shape
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start)


#     2.6547305583953857
# Not the most eloquent solution, but it's the fastest I could come up with. Started reading a book to up my python skills, mentions that throwing expection is preferable to returning none. This is a prime example. 

# Now lets rewind back to something that we have glossed over. If we look at the images they arent' the same size. The resolution is variable. I don't know of a way to deal with images of variable size in a NN. It may exist, but it'd be easiest to resize them to all be the same size. Let's do 224, 224, 3 as that seems to be what some of the major pretrained CNN's come as. 

# In[ ]:


from cv2 import resize
start = time.time()
for image in X["image"][0:1000]:
    img = cvimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
    try:
        img = resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        img.shape
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start)


#     3.6379148960113525

# Using the same previous code snippet plus the resize function from cv2. Shows that this resize process adds some time. This is a very simple way to rescale the image. I apologize for whoever wrote the kernel because I can't find it anymore but I also utilized someones code to try to pad the sides so all dimensions were the same before reducing the image to 224,224 because that causes some rectangular images to become squished. Wasn't able to test the hypothesis, but I don't think one is necessarily better than the other because padding the sides also destroys quite a bit of resolution despite keeping the same aspect ratio. 

# Anyway, is that the fastest we can do? Well no. Everyone knows for loops are slow. What if we could deal with the data in parallel instead of linearly while maintaining order? Too good to be true? Enter concurrent.futures or any other number of multithreading tools. For this one I will focus on concurrent.futures because the map function is very slick and easy. 

# Let's rewrite our previous snippet as a function so we can use map. 

# In[ ]:


import concurrent.futures
def resize_img(image):
    img = cvimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
    try:
        img = resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        img.shape
    except:
        img = np.zeros(shape = (224, 224, 3))
    return img


# In[ ]:


start = time.time()
resized_imgs = []
with concurrent.futures.ThreadPoolExecutor(max_workers = 16) as executor:
    for value in executor.map(resize_img, X["image"][0:1000]):
        resized_imgs.append(value)
print(time.time() - start)      


#    0.49677205085754395

# concurrent.futures map is very very cool and easy to use. It allows us to take a list like we have and then gobble it up in parallel with our function and then it does the work of piecing it together in the correct order. max_workers is a parameter you can toy with based on your system configuration. Things to consider when changing that: amount of cores and available memory. If you queue too many things up then you might throw too much in your memory and might spill over into disk which will greatly slow things down or it may be more cores than you have available. I show the images being appended to a list. This likely isn't what you want to do, but I'm too lazy to make a better example. 
# 
# If you want to dig into this a little more there is a very good stack overflow post looking at some of the performance considerations and how you might use submit instead of map sometimes. https://stackoverflow.com/questions/42074501/python-concurrent-futures-processpoolexecutor-performance-of-submit-vs-map
# Also, just to note if you are on Windows it is technically possible to use the ProcessPoolExecutor instead of ThreadPoolExecutor, but I'm going to suggest you spend the time just installing Linux if you are really wanting to go down that path. This is actually something Keras has decided to just totally avoid. https://github.com/keras-team/keras/pull/8662
# 
# If you are on Linux actually a lot of the problems we have been focusing on go away because you can use the multiprocessing option in Keras with a generator and number of threads and queue size all built in, but I thought it would be valuable to share anyway. So we're done, right? But what happens if we resize the images before loading them in and write them to a separate folder. How fast would that be? 

# In[ ]:


import concurrent.futures
def resize_img(image):
    img = cvimread('./data/competition_files/train_jpg_resized/' + str(image) + ".jpg")
    try:
        img.shape
    except:
        img = np.zeros(shape = (224, 224, 3))
    return img


# In[ ]:


start = time.time()
resized_imgs = []
with concurrent.futures.ThreadPoolExecutor(max_workers = 16) as executor:
    for value in executor.map(resize_img, X["image"][0:1000]):
        resized_imgs.append(value)
print(time.time() - start)    


#     0.16491961479187012
# Even better.  So if we just resized the images beforehand and put them in a new folder we can load in the files stupid quick. We took a process that could take 3+ seconds per round if we tried to load in serially and resize on the fly down to .165 seconds. This is crucial for quick prototyping and testing of new ideas. 
# 
# This is what I ended up using for most of my attempts at a model that utilized the images directly, but I also went down the rabbit hole of rewriting the images in HDF5 for faster reading. I will post that research tomorrow when it isn't 2:30am for me. Also have some work I did regarding Keras's flow_from_directory, but ultimately didn't find a good way to use that. For now I will just focus on sharing the HDF5 stuff because I think that will actually be highly relevant not just to this challenge but to many peoples and the big data they may be dealing with. 
