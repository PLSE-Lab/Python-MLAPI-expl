#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### install facenet-pytorch to implement it later on in the code
get_ipython().system('pip install facenet-pytorch')


# In[ ]:


# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm


# In[ ]:


import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("/kaggle/input/dogimages/lfw/lfw/*/*"))
dog_files = np.array(glob("/kaggle/input/dogimages/dogImages/dogImages/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))


# # Convolutional Neural Networks
# 
# ## Project: Write an Algorithm for a Dog Identification App 
# 
# ---
# 
# In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.
# 
# The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.
# 
# 
# 
# ---
# ### Why We're Here 
# 
# In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 
# 
# ![Sample Dog Output](images/sample_dog_output.png)
# 
# In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!
# 
# ### The Road Ahead
# 
# We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.
# 
# * [Step 0](#step0): Import Datasets
# * [Step 1](#step1): Detect Humans
# * [Step 2](#step2): Detect Dogs
# * [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
# * [Step 4](#step4): Create a CNN to Classify Dog Breeds (using Transfer Learning)
# * [Step 5](#step5): Write your Algorithm
# * [Step 6](#step6): Test Your Algorithm
# 
# ---
# <a id='step0'></a>
# ## Step 0: Import Datasets
# 
# Make sure that you've downloaded the required human and dog datasets:
# 
# **Note: if you are using the Udacity workspace, you *DO NOT* need to re-download these - they can be found in the `/data` folder as noted in the cell below.**
# 
# * Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in this project's home directory, at the location `/dog_images`. 
# 
# * Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the home directory, at location `/lfw`.  
# 
# *Note: If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.*
# 
# In the code cell below, we save the file paths for both the human (LFW) dataset and dog dataset in the numpy arrays `human_files` and `dog_files`.

# <a id='step1'></a>
# ## Step 1: Detect Humans
# 
# In this section, we use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  
# 
# OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.  In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.

# In[ ]:


import cv2                
import matplotlib.pyplot as plt                        
get_ipython().run_line_magic('matplotlib', 'inline')

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('/kaggle/input/library/haarcascades/haarcascade_frontalface_alt.xml')
lbp_cascade = cv2.CascadeClassifier('/kaggle/input/lbpcascade/lbpcascade_frontalface.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image (faces is a numpy array)
faces = lbp_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


# Here I used the MTCNN library from facenet to detect human face in an image

# In[ ]:


# create face detector
mtcnn = MTCNN(keep_all=True)

# load color (BGR) image
img = cv2.imread(human_files[3])

# Detect face
boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img)
ax.axis('off')

for box, landmark in zip(boxes, landmarks):
    ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
    ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
fig.show()


# Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  
# 
# In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.
# 
# ### Write a Human Face Detector
# 
# We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.

# In[ ]:


def face_detector(img_path, algorithm = "face_cascade"):
    """Function to detect human face in an image
    Args: 
        img_path: string value file path
        algorithm: string name of what type of algorithm is used to detect the face (default = "face_cascade")
                   lbp_cascade, facenet 
        
    Returns: 
        boolean: 
            True if the a humann face is detected, False if it not
    """
    # read the image from the file path
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = list()
    
    if (algorithm == "face_cascade"):             
        faces = face_cascade.detectMultiScale(gray)
        
    elif (algorithm == "lbp_cascade"):
        faces = lbp_cascade.detectMultiScale(gray)
        
    elif (algorithm == "facenet"):
        mtcnn = MTCNN(keep_all=True)
        # Detect face(s)
        faces = mtcnn(img)
        
    return (faces is not None and len(faces) > 0)


# ### (IMPLEMENTATION) Assess the Human Face Detector
# 
# __Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
# - What percentage of the first 100 images in `human_files` have a detected human face?  
# - What percentage of the first 100 images in `dog_files` have a detected human face? 
# 
# Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

# Testing the accuracy of the face-cascade and lbp-cascade algorithms 

# In[ ]:


def test_algorithm(algorithm):
    """Function to test face detection algorithm for a given images of human and dogs
    Args: 
        algorithm: string name of what type of algorithm is used to detect the face (default = "face_cascade")
                   lbp_cascade, facenet         
    Returns: 
       None
    """
    human_files_short = human_files[:100]
    dog_files_short = dog_files[:100]

    human_faces_human_files = 0
    human_faces_dog_files = 0
    for human_img in human_files_short:    
        # check if the face detector is detecting faces in the human faces dataset 
        if(face_detector(human_img, algorithm=algorithm)):
            human_faces_human_files += 1 

    for dog_img in dog_files_short:      
        # if it is not detecting faces in the dog images, then we are good to go   
        if(face_detector(dog_img, algorithm=algorithm)):
            human_faces_dog_files += 1


    print("Number of detected faces in human files : {}% \n Number of detected faces in the dog files : {}%".format(
                                                                    (human_faces_human_files/len(human_files_short))*100 ,
                                                                     (human_faces_dog_files/len(dog_files_short))*100))


# In[ ]:


test_algorithm("face_cascade")


# In[ ]:


test_algorithm("lbp_cascade")


# In[ ]:


test_algorithm("facenet")


# ### Observations
# The accuracy of the face-cascade is far more better than facenet and lbp-cascade even though it is almost a tie between face-cascade and the MTCNN

# ---
# <a id='step2'></a>
# ## Step 2: Detect Dogs
# 
# In this section, we use a [pre-trained model](http://pytorch.org/docs/master/torchvision/models.html) to detect dogs in images.  
# 
# ### Obtain Pre-trained VGG-16 Model
# 
# The code cell below downloads the VGG-16 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  

# In[ ]:


import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()


# Given an image, this pre-trained VGG-16 model returns a prediction (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image.

# ### (IMPLEMENTATION) Making Predictions with a Pre-trained Model
# 
# In the next code cell, you will write a function that accepts a path to an image (such as `'/kaggle/input/dogimages/dogImages/dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg'`) as input and returns the index corresponding to the ImageNet class that is predicted by the pre-trained VGG-16 model.  The output should always be an integer between 0 and 999, inclusive.
# 
# Before writing the function, make sure that you take the time to learn  how to appropriately pre-process tensors for pre-trained models in the [PyTorch documentation](http://pytorch.org/docs/stable/torchvision/models.html).

# In[ ]:


from PIL import Image
import torchvision.transforms as transforms

def process_img(img_path, img_size):    
    '''Function to batched an image and process it as input for a model
    
    Args:
        img_path: string value path to an image
        img_size: integer value the desired image size
        
    Returns:
        batched image
    '''
    
    ## Load and pre-process an image from the given img_path
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transforms  = transforms.Compose([transforms.CenterCrop(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
    img = Image.open(img_path)
    img_t = test_transforms(img)
    batch_t = img_t.unsqueeze(0)
    
    return batch_t
    
def predict(model, img_path, img_size):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
         model: takes a pretrained model
         img_path: path to an image
         img_size: integer value for image size
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    
    batch_t = process_img(img_path, img_size)
    if use_cuda:
        batch_t = batch_t.cuda()
    model.eval()
    output = model(batch_t)
    _, pred = torch.max(output, 1)

    ## Return the *index* of the predicted class for that image
    
    return int(pred) # predicted class index


# ### (IMPLEMENTATION) Write a Dog Detector
# 
# While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).
# 
# Use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).

# In[ ]:


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(model, img_path, img_size):
    ## TODO: Complete the function.
    pred = predict(model, img_path, img_size)
    if(pred >=151 and pred <=268):
        return True
    else:
        return False


# ### (IMPLEMENTATION) Assess the Dog Detector
# 
# __Question 2:__ Use the code cell below to test the performance of your `dog_detector` function.  
# - What percentage of the images in `human_files_short` have a detected dog?  
# - What percentage of the images in `dog_files_short` have a detected dog?

# __Answer:__ 
# 

# In[ ]:


### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
def assess_dog_detector(model, img_size):
    dog_faces_human_files = 0
    dog_faces_dog_files = 0
    
    for human_img in human_files_short:
    
        # check if the face detector is detecting faces in the hyman faces dataset 
        if(dog_detector(model, human_img, img_size)):
            dog_faces_human_files += 1 

    for dog_img in dog_files_short:      
        # if it is not detecting faces in the dog images, then we are good to go   
        if(dog_detector(model, dog_img, img_size)):
            dog_faces_dog_files += 1


    print("Number of detected dogs in human files : {}% \nNumber of detected dogs in the dog files : {}%".format(
                                                                    (dog_faces_human_files/len(human_files_short))*100 ,
                                                                     (dog_faces_dog_files/len(dog_files_short))*100))

assess_dog_detector(VGG16,  img_size=224)


# We suggest VGG-16 as a potential network to detect dog images in your algorithm, but you are free to explore other pre-trained networks (such as [Inception-v3](http://pytorch.org/docs/master/torchvision/models.html#inception-v3), [ResNet-50](http://pytorch.org/docs/master/torchvision/models.html#id3), etc).  Please use the code cell below to test other pre-trained PyTorch models.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.

# In[ ]:


### (Optional) 
### TODO: Report the performance of another pre-trained network.
### Feel free to use as many code cells as needed.

# download a pretrained ResNet model
resnet = models.resnet50(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    resnet = resnet.cuda()
    


# ### Assess ResNet-50 model

# In[ ]:


assess_dog_detector(resnet, img_size=224)


# ### Implementing Inception model to detect dogs

# In[ ]:


# download a pretrained Inception model
inception = models.inception_v3(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    inception = inception.cuda()


# In[ ]:


assess_dog_detector(inception,  img_size=224)


# ---
# <a id='step3'></a>
# ## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
# 
# Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 10%.  In Step 4 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.
# 
# We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have trouble distinguishing between a Brittany and a Welsh Springer Spaniel.  
# 
# Brittany | Welsh Springer Spaniel
# - | - 
# <img src="kaggle/input/helper/images/Brittany_02625.jpg" width="100"> | <img src="kaggle/input/helper/images/Welsh_springer_spaniel_08203.jpg" width="200">
# 
# It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  
# 
# Curly-Coated Retriever | American Water Spaniel
# - | -
# <img src="/kaggle/input/helper/images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="/kaggle/input/helper/images/American_water_spaniel_00648.jpg" width="200">
# 
# 
# Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  
# 
# Yellow Labrador | Chocolate Labrador | Black Labrador
# - | -
# <img src="/kaggle/input/helper/images/Labrador_retriever_06457.jpg" width="150"> | <img src="/kaggle/input/helper/images/Labrador_retriever_06455.jpg" width="240"> | <img src="/kaggle/input/helper/images/Labrador_retriever_06449.jpg" width="220">
# 
# We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  
# 
# Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!
# 
# ### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset
# 
# Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dog_images/train`, `dog_images/valid`, and `dog_images/test`, respectively).  You may find [this documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) to be a useful resource.  If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!

# In[ ]:


import os
from torchvision import datasets
import torch

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

import os
from torchvision import datasets

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
batch_size = 20

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(112),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std)
    ])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std)
    ])

# Choose the trainig, test, and validation set
data_dir = "/kaggle/input/dogimages/dogImages/dogImages/"
# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data =  datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
# prepare the data loaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

dataset_size = len(train_data+test_data+valid_data)
classes = train_data.classes
number_classes = len(train_data.classes)
print("data set size: ", dataset_size)
print("Number of classes: ", number_classes)


# **Question 3:** Describe your chosen procedure for preprocessing the data. 
# - How does your code resize the images (by cropping, stretching, etc)?  What size did you pick for the input tensor, and why?
# - Did you decide to augment the dataset?  If so, how (through translations, flips, rotations, etc)?  If not, why not?
# 

#     **Answer**:  
# - I decided to resize the images to 224X224 then center cropping to 112 so our network will be exposed to more details that related to dogs. The reason I chose this size because it is not too small so we end up losing importent information, neither too big since we have limited GPU.
# - Yes, I decided to use augmentation (flipping, cropping, and rotation), to provide a better training experience for our model since it will be exposed to more images of dogs in different positions and angles. 

# ### (IMPLEMENTATION) Model Architecture
# 
# Create a CNN to classify dog breed.  Use the template in the code cell below.

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        
        # image size 112X112
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   
        self.bn1 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # image size after maxpooling 56X56 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # image size after maxpooling 28X28     
        self.conv3 = nn.Conv2d(64, 64,  kernel_size=3, padding=1)  
        self.bn3 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # image size after maxpooling 14X14 
        self.conv4 = nn.Conv2d(64, 128,  kernel_size=3, padding=1)  
        self.bn4 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # image size after maxpooling 7X7 
        self.conv5 = nn.Conv2d(128, 256,  kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # define the pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        #define the linear layers (input 3 * 3 * 256)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(3 * 3 * 256, 4608)
        self.fc2 = nn.Linear(4608, 2304)
        self.fc3 = nn.Linear(2304, 1024)
        self.fc4 = nn.Linear(1024, number_classes)
    
    def forward(self, x):
        ## Define forward behavior
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = x.view(-1, 3 * 3* 256)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

#-#-# You do NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()


# In[ ]:


# apply normal distribution rule ro initialize the weights
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''
    
    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        # m.weight.data shoud be taken from a normal distribution
        n = m.in_features
        y = 1.0/np.sqrt(n)    
        m.weight.data.normal_(0, y)
        # m.bias.data should be 0
        m.bias.data.fill_(0)
        
model_scratch.apply(weights_init_normal)

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()


# __Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  

# __Answer:__ 
# - I decided that 5 convolutional layers is enogh considering my humble resources, and since I know that the number of linear layer decrease as the convolutional layers increase, I added 4 linear layers to compensate and extract more patterns and useful information from our CNN.
# 
# - padding is 1 so I keep the size of the original image. kernel size = 3, and stride = 1
# - I also initialized the weights by applying normal distrbution rule, since I have seen it's effect in the previous lesson
# 
# - I added drop out to prevent overfitting

# ### (IMPLEMENTATION) Specify Loss Function and Optimizer
# 
# Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_scratch`, and the optimizer as `optimizer_scratch` below.

# In[ ]:


import torch.optim as optim

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.005, momentum=0.9)


# ### (IMPLEMENTATION) Train and Validate the Model
# 
# Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_scratch.pt'`.

# In[ ]:


# the following import is required for training to be robust to truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, trainloader, testloader, validloader , model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # clear the gradient
            optimizer.zero_grad()
            output = model(data)
            
            ## find the loss and update the model parameters accordingly
            loss = criterion(output, target)
            loss.backward()
            # Perform the optimizer step
            optimizer.step()
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            train_loss += loss.item() * data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(validloader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()           
            # Pass data acrss the network
            output = model(data)
            loss = criterion(output, target)
            ## update the average validation loss
            valid_loss += loss.item() * data.size(0)
        
        # Calculate the average losses
        train_loss = train_loss/len(trainloader)
        valid_loss = valid_loss/len(validloader)
    
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            valid_loss_min = valid_loss
            # print the decremnet in the validation
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min, 
            valid_loss))
            torch.save(model.state_dict(), save_path)
            
    # return trained model
    return model


# In[ ]:


# train the model
model_scratch = train(30, trainloader, testloader, validloader, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')


# In[ ]:


# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))


# ### (IMPLEMENTATION) Test the Model
# 
# Try out your model on the test dataset of dog images.  Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 10%.

# In[ ]:


def test(testloader, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(testloader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


# In[ ]:


# call test function    
test(testloader, model_scratch, criterion_scratch, use_cuda)


# ---
# <a id='step4'></a>
# ## Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
# 
# You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.
# 
# ### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset
# 
# Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dogImages/train`, `dogImages/valid`, and `dogImages/test`, respectively). 
# 
# If you like, **you are welcome to use the same data loaders from the previous step**, when you created a CNN from scratch.

# In[ ]:


## TODO: Specify data loaders

batch_size = 20

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transfer_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std)
    ])

test_transfer_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std)
    ])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transfer_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transfer_transforms)
valid_data =  datasets.ImageFolder(data_dir + '/valid', transform=test_transfer_transforms)
# prepare the data loaders
trainloader_transfer = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader_transfer = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
validloader_transfer = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)


# ### (IMPLEMENTATION) Model Architecture
# 
# Use transfer learning to create a CNN to classify dog breed.  Use the code cell below, and save your initialized model as the variable `model_transfer`.

# In[ ]:


# assigning the pretrained model
model_transfer = resnet

model_transfer


# In[ ]:


# Building the network
import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 

#freeze the model calssifier
for param in  model_transfer.parameters():
    param.requires_grad = False

from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([                           
                          ('fc1', nn.Linear(2048, 1024)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(1024, 512)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(512, number_classes)),
                          ('output', nn.LogSoftmax(dim=1))]))

model_transfer.fc = classifier
model_transfer
if use_cuda:
    model_transfer = model_transfer.cuda()
model_transfer


# In[ ]:


# Initializing the weights
model_transfer.fc.apply(weights_init_normal)


# __Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

# __Answer:__ 
# - I chose resnet as my pretrained model, I found its results really good in detecting dogs and at the same time it is also detecting a very small humans as dogs too which could be fun for our app at the same type it has a simpler architectures in comparsion to the other models which I thought is enough for achieving our goal.
# 
# - Since the dataset is similar but smaller than the original, I kept all the convolutional layers, removed the classifiers and added a new 3 fully connected layers with dropout.

# ### (IMPLEMENTATION) Specify Loss Function and Optimizer
# 
# Use the next code cell to specify a [loss function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html).  Save the chosen loss function as `criterion_transfer`, and the optimizer as `optimizer_transfer` below.

# In[ ]:


from torch import optim
criterion_transfer = nn.NLLLoss()
optimizer_transfer = optim.SGD(model_transfer.fc.parameters(), lr=0.001, momentum=0.9)


# ### (IMPLEMENTATION) Train and Validate the Model
# 
# Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_transfer.pt'`.

# In[ ]:


# train the model
model_transfer = train(20, trainloader_transfer, testloader_transfer, validloader_transfer,
                       model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')


# In[ ]:


# load the model that got the best validation accuracy
model_transfer.load_state_dict(torch.load('model_transfer.pt'))


# ### (IMPLEMENTATION) Test the Model
# 
# Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.

# In[ ]:


test(testloader_transfer, model_transfer, criterion_transfer, use_cuda)


# ### (IMPLEMENTATION) Predict Dog Breed with the Model
# 
# Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan hound`, etc) that is predicted by your model.  

# In[ ]:


### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in train_data.classes]

def predict_breed_transfer(img_path):
    '''Function that takes a path to an image as input 
    and returns the dog breed that is predicted by the model
    Args:
         img_path: string path to an image
        
    Returns:
        Index corresponding to our model's prediction
    '''
    # load the image and return the predicted breed
    # pre_process the image
    t_batch = process_img(img_path, 224)
    model_transfer.eval()
    t_batch = t_batch.cuda()
    output = model_transfer(t_batch)
    _, pred = torch.max(output, 1)    
    class_name = classes[int(pred)]
    # Get the breed name
    class_name = class_name[class_name.find('.')+1 :].replace("_", " ")
    return class_name


# ---
# <a id='step5'></a>
# ## Step 5: Write your Algorithm
# 
# Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
# - if a __dog__ is detected in the image, return the predicted breed.
# - if a __human__ is detected in the image, return the resembling dog breed.
# - if __neither__ is detected in the image, provide output that indicates an error.
# 
# You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `human_detector` functions developed above.  You are __required__ to use your CNN from Step 4 to predict dog breed.  
# 
# Some sample output for our algorithm is provided below, but feel free to design your own user experience!
# 
# ![Sample Human Output](images/sample_human_output.png)
# 
# 
# ### (IMPLEMENTATION) Write your Algorithm

# ### Plot
# You are standing at the gate of a dogs only super secret club and there is a giant robot body guard standing by the gate. Let's just say it is not easy to fool that guy and enter the club unless you are a dog or it'll spot you!

# In[ ]:


### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def run_app(img_path):
    # load color (BGR) image
    img = cv2.imread(img_path)
    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ## handle cases for a dog,human face, and neither
    if dog_detector(VGG16, img_path, img_size=224) == True:      
        # display the image, along with bounding box
        plt.imshow(cv_rgb)
        plt.suptitle("Welcome!") 
        plt.title("The {} gathering is over ther..".format(predict_breed_transfer(img_path)))
                
    elif face_detector(img_path) == True:             
        plt.imshow(cv_rgb)
        plt.suptitle("Got you human!")  
        plt.title("You are trying to pose as a {}".format(predict_breed_transfer(img_path)))
                         
    else:
        plt.imshow(cv_rgb)
        plt.suptitle( "Un-Identified object Aalert!") 
        plt.title("Posing as {}".format(predict_breed_transfer(img_path)))
                     
    plt.show()    
    


# ---
# <a id='step6'></a>
# ## Step 6: Test Your Algorithm
# 
# In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that _you_ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?
# 
# ### (IMPLEMENTATION) Test Your Algorithm on Sample Images!
# 
# Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  
# 
# __Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

# __Answer:__ (Three possible points for improvement)
# - Actually the results were better than I expected :)
# - We can make a better dog and face detectors so we insure none of the images are being mis classified, even though this could provide a fun experience.
# - Use different pretrained model and see if we can get a better results for idevtifying breeds.
# - Tunning our classifier further.

# In[ ]:


## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

## suggested code, below
for file in np.hstack((human_files[-4:-1], dog_files[-4:-1])):
    run_app(file)


# In[ ]:


img_path = "/kaggle/input/testimgs/me.jpg"
run_app(img_path)


# In[ ]:


img_path = "/kaggle/input/testimgs/Rocky.jpg"
run_app(img_path)


# ### Implementing facenet-pytorch for face detecting
# 
# reference: https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch

# In[ ]:




