#!/usr/bin/env python
# coding: utf-8

# ## I am currently working through the FastAI MOOC and this NoteBook is meant to do a few things
# 
# 1. For me to practce and develop my ability using the FAST AI library. 
# 
# 2. I have used TensorFlow to build CNNs for this Competition and tested different architectures to get upto as high as the top 23% of the competitors in this competition. I intend to compare the results and accuracy of these two libraries as well as the ease of use of both. 
# My Tensorflow Notebook can be found [here](https://www.kaggle.com/sanwal092/tensorflow-and-cnn-99-accuracy). I was able to achieve 99.428% accuracy with the code in this mentioned Notebook
# 
# 
# I really appreciate [Chris Wallenwein's instructive Notebook](https://www.kaggle.com/christianwallenwein/beginners-guide-to-mnist-with-fast-ai). He does an excellent job of creating an easy to follow the fastai library
# 
# ### In this Notebook, I will be adding a lot of comments and thoughts as a way for me to virtually "think out loud" and figure out as I use the FASTAI library. Let's get to work. 
# 
# ![](https://media.giphy.com/media/l0HlxJMw7rkPTN8sg/giphy.gif)

# ## Table of Conetents
# 
# ** Model Log for my iterations is [here](#model_log) **
# 
# 1. [Setting up data for the FastAI library](#fastaipath)
# 2. [Preparing Data for FastAI](#data_prep)    
#     * [Converting raw pixels to images](#pix2img)
#     * [Plot Images](#plot_images)
# 3. [Feed Data to FastAI](#feed_fastai)
# 4. [Iterating different models](#model_log)
# 5. [Checking some predictions made by the model.](#make_preds)

# ** The Following cell of code is used everytime FASTAI library is used. They tell the notebook to reload any changes made to any libraries used. They also ensure that any graphs are plotted are shown in this notebook**

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# FOR NON-FASTAI LIBRARIES
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# FOR ALL THE FASTAI LIBRARIES

from fastai.vision import *
from fastai.metrics import *


# make sure CUDA is available and enabled
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)


# <a id = 'fastaipath'></a>
# 
# ## BUILDING THE PATH TO THE FILES TO BE FED TO THE FASTAI LIBRARY
# 
# Using the **Path** instead of the Python OS library lets us do a lot more things organically that the traditional OS library doesn't. One of things which we can do is constructing concatenated path easily. 
# 

# In[ ]:


# mainDIR = "/kaggle/input/digit-recognizer/"
# os.listdir(mainDIR)
INPUT = Path("../input/digit-recognizer")
os.listdir(INPUT)


# In[ ]:


# train_df = pd.read_csv(mainDIR+ "train.csv")
train_df = pd.read_csv(INPUT/"train.csv")
train_df.head(3)


# In[ ]:


test_df = pd.read_csv(INPUT/"test.csv")
test_df.head(3)


# <a id = 'data_prep'></a>
# ### To use FASTAI library, we need to feed our data into their **ImageDataBunch** function. However,
# 
# 1. It only accepts images, not csv pixel data as we do have right now. 
# 2. It also needs the data to be in proper labeled ImageNet format. So something like such:
# 
#     > path\
#         train\
#         0\
#             ___.jpg
#             ___.jpg
#             ___.jpg
#         1\
#             ___.jpg
#             ___.jpg
#         2\
#             ...
#         3\
#            ...
#            ...
#        test\
#            ___.jpg
#            ___.jpg
#             ...
#     
# ### So,
# 
# We will create the folder structure which matches this kind of folder structure. 
# 
#     

# In[ ]:


# TRAIN = Path("/kaggle/train/")
# TEST = Path("/kaggle/test/")

TRAIN = Path("../train")
TEST = Path("../test")


# In[ ]:


#MAKE DIRECTORIES  FOR TRAINING FOLDER

for i in range(10):    
    try:         
        os.makedirs(TRAIN/str(i))       
    except:
        pass


# In[ ]:


#CHECK IF MAKING THE DIRECTORIES WORKED!!
sorted(os.listdir(TRAIN))


# In[ ]:


#LET'S MAKE THE TEST FOLDER 

try:
    os.makedirs(TEST)
except:
    pass


# In[ ]:


os.listdir(TEST)


# In[ ]:


# os.listdir(TEST)
if os.path.isdir(TRAIN):
    print('Train directory has been created')
else:
    print('Train directory creation failed.')

if os.path.isdir(TEST):
    print('Test directory has been created')
else:
    print('Test directory creation failed.')


# <a id = 'pix2img'></a>
# ### So, the train and test directories have been created. Now, here are a few things to consider. 
# 1. The directories exist. 
# 2. The picture data exist in the csv files as pixel values for each pixels. 
# 
# ### Since FastAI only takes data in as images, not pixel values, we will have to convert this data into images for which we will use the PIL library. 
# 
# 
# We will have to reshape this into 28x28 matrices. To do this, I will use the PIL library in Python 

# In[ ]:


from PIL import Image


# In[ ]:


def pix2img(pix_data, filepath):
    img_mat = pix_data.reshape(28,28)
    img_mat = img_mat.astype(np.uint8())
    
    img_dat = Image.fromarray(img_mat)
    img_dat.save(filepath)
    
    


# In[ ]:


# SAVE TRAINING IMAGES 

for idx, data in train_df.iterrows():
    
    label, data = data[0], data[1:]
    folder = TRAIN/str(label)
    
    fname = f"{idx}.jpg"
    filepath = folder/fname
    
    img_data = data.values
    
    pix2img(img_data,filepath)


# In[ ]:


# THE SAME PROCESS FOR TESTING DATA 
for idx, data in test_df.iterrows():
    
#     label, data = data[0], data[1:]
    folder = TEST
    
    fname = f"{idx}.jpg"
    filepath = folder/fname
    
    img_data = data.values
    
    pix2img(img_data,filepath)


# <a id = 'plot_images'></a>
# ### Let's plot some of the training images to see what they are looking like
# 

# In[ ]:


def plotTrainImage():
    
    fig = plt.figure(figsize= (5,10))
    
    for rowIdx in range(1,10):
        
        foldNum = str(rowIdx)
        path = TRAIN/foldNum
        
        images = os.listdir(path)
        
        for sampleIdx in range(1,6):
            
            randNum = random.randint(0, len(images)-1)
            image = Image.open(path/images[randNum])
            ax = fig.add_subplot(10, 5, 5*rowIdx + sampleIdx)
            ax.axis("off")
            
            plt.imshow(image, cmap='gray')
            
    plt.show()      
    


# In[ ]:


print('plotting training images')
plotTrainImage()


# In[ ]:


# FUNCTION FOR PLOTTING TEST IMAGES 

def plotTestImage():
    
    fig = plt.figure(figsize=(5, 10))    
    paths = os.listdir(TEST)    
        
    for i in range(1, 51):
        randomNumber = random.randint(0, len(paths)-1)
        image = Image.open(TEST/paths[randomNumber])
        
        ax = fig.add_subplot(10, 5, i)
        ax.axis("off")
        
        plt.imshow(image, cmap='gray')
    plt.show()


# In[ ]:


print('plotting testing images')
plotTestImage()


# <a id = 'feed_fastai'></a>
# ### Now that the data is in the correct file and folder structure, we will feed it it to [DataBunch](https://docs.fast.ai/basic_data.html#DataBunch) object which is used inside the FastAI library to train the CNN Leaner class

# In[ ]:


# transforms which are a part of data augmentation
tfms = get_transforms(do_flip = False)


# In[ ]:


print('test : ',TEST)
print('train: ', TRAIN)
print(type(TEST))


# In[ ]:


data = ImageDataBunch.from_folder(

    path = ("../train"),
    test = ("../test"),
    valid_pct = 0.1,
#     bs = 16,
    bs = 256,    
    size = 28,
    num_workers = 0,
    ds_tfms = tfms
)


# In[ ]:


mnist_stats


# In[ ]:


# data.normalize(mnist_stats)
data.normalize(imagenet_stats)


# In[ ]:


print(data.classes)
print('There are', data.c, 'classes here')


# ## The data is ready. Time to feed it to a Convolutional Neural Network. 
# 
# * There are many architures to choose it from. I have tried different architectures, implemented them in Tensorflow that you can find in 
#     [this Notebook](https://www.kaggle.com/sanwal092/tensorflow-and-cnn-99-accuracy)
# * For implementation here, I will use Resnet18 which as shown in FASTAI lecture 1. It performed well on many different metrics. There are more sophiscated version of Resnet such as Resnet34 and Resnet50, but I will be sticking to Resnet18 here
# 
# * I will be using FastAI's cnn_learner fucntion to implement a ResNet architecture. This is a really handy function to skip implementing the Resnet architecture from scratch which would be a nightmare. 
# 
# 
# 
# 

# In[ ]:


#VERSION 1
# learn = cnn_learner(data, base_arch = models.resnet18, metrics = accuracy,model_dir="/tmp/models", callback_fns=ShowGraph )

#version 2
learn = cnn_learner(data, base_arch = models.resnet34, metrics = accuracy,model_dir="/tmp/models", callback_fns=ShowGraph )

# version 3
# learn = cnn_learner(data, base_arch = models.resnet50, metrics = accuracy,model_dir="/tmp/models", callback_fns=ShowGraph )


# ### In FastAI, the function to fit the CNN to the data is called fit_one_cycle which is based on the application of this [paper by Leslie Smith](https://arxiv.org/abs/1803.09820)
# 
# 

# <a id = 'model_log'></a>
# ## So far, this is what I tried:
# 
# ## VERSION 1: 
#     * This is a bit of cheating because I tried it in 2 phases. For both the phases I used Resnet18. 
#     * However, in the first phase I only trained for maybe 10 epochs. In the second, I trained for 30. The score jumped up from 10% to 99.03% 
#     at test-time. 
#     * The learning rate I used was 1e-6.
#     
# ## VERSION 2:
#     * I will be using Resnet34 here. 
#     * I used 15 epochs with learning rate of 4e-6. 
#     * The training time was about 30 minutes on Kaggle GPUs. 
#     * Score = 99.10%    
#    
# ## VERSION 3:
#     * Still Resnet34.
#     * 20 epochs, learning rate of 4e-6
#     * Took about 40 minutes of training with result 99.10%. My highest score was 99.428% using TensorFlow to build up model
#       which took close to 2 hours to train.I want to see how close I can get to that if not beat it.
#     * Score = 99.142%
#     
# ## VERSION 4: 
#     * Resnet50
#     * 15 epochs to start and see what happens
#     * It took almost 1.5 hours to train and didn't offer any significant imporvement on anything. ResNet might be a
#       bit of an overkill.
#     * Score = 97.75%. 
# 
# ## VERSION 5: 
#     * Back to Resnet34. 
#     * Will try 30 epochs this. 
#     * Instead of using mnist_stats, i am normalizing based on image_net stats and using a batch size of 256 instead of 16.
#     

# In[ ]:


doc(fit_one_cycle)


# ### train fit_one_cycle for 5 cycles to get an idea of how accurate the model is.

# In[ ]:


learn.fit_one_cycle(3)


# # Our model was able to get to very high accuracy in just a few epoch(s). 
# 
# * This is nothing to be sneezed at. Now, we will fine tune the mode 
# 
# * We will now use FastAI's lr_find function to find a range of learning rate which we could choose from. 
# 
# * We use recorder.plot to visualize this rannge.

# In[ ]:


learn.save('model1')


# In[ ]:


learn.load('model1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(30 , slice(1e-3, 1e-2))


# <a id = 'make_preds'></a>
# ### In just a few epochs, the model was able to achieve a very high accuracy.Time to evaluate results 
# 
# ### Let's see some of the predictions made by our model
# 
# 

# In[ ]:


learn.show_results(3, figsize= (7,7))


# ## Let's check the the top 6 images with highest losses. 

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(6, figsize=(7, 7))


# ## Okay, some of these are really hard to pin down as one specific integers. So, I don't feel too bad about my model getting it wrong. Let's take a look at the Confusion matrix.

# In[ ]:


interp.plot_confusion_matrix()


# ## Time to make some predictions
# 
# ![](https://media.giphy.com/media/l2JJu8U8SoHhQEnoQ/giphy.gif)
# 

# In[ ]:


class_score , y = learn.get_preds(DatasetType.Test)


# In[ ]:


probabilities = class_score[0].tolist()
[f"{index}: {probabilities[index]}" for index in range(len(probabilities))]


# ## Right now, our predictions include 10 separate predictions for each image. We will only take the top class and then save it as our result

# In[ ]:


class_score = np.argmax(class_score, axis=1)


# In[ ]:


class_score[1].item()


# In[ ]:


sample_submission =  pd.read_csv(INPUT/"sample_submission.csv")
display(sample_submission.head(2))
display(sample_submission.tail(2))


# In[ ]:


# remove file extension from filename
ImageId = [os.path.splitext(path)[0] for path in os.listdir(TEST)]
# typecast to int so that file can be sorted by ImageId
ImageId = [int(path) for path in ImageId]
# +1 because index starts at 1 in the submission file
ImageId = [ID+1 for ID in ImageId]


# In[ ]:


submission  = pd.DataFrame({
    "ImageId": ImageId,
    "Label": class_score
})
# submission.sort_values(by=["ImageId"], inplace = True)
submission.to_csv("submission.csv", index=False)
display(submission.head(3))
display(submission.tail(3))


# In[ ]:




