#!/usr/bin/env python
# coding: utf-8

# ## Installing Libraries
# Installing list of libraries needed for image classification purpose. Installing torch and torchvision which are main libraries for pytorch framework. torchvision is for handling image manipulation and also provides pretrained image classification models.
# 
# Also need numpy, pandas, matplotlib, pillow and scikit-learn. Matplotlib is needed for displaying images. Pillow is for loading images and manipulating them. Scikit-learn will be used for generating confusion matrix.

# In[ ]:


get_ipython().system('pip install torch torchvision')
get_ipython().system('pip install numpy pandas matplotlib pillow sklearn')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt ## Plotting library

import torch  ## Main torch library which provides functionalities for creating deep learning models.
import torch.nn as nn ## provides functionalities for creating neural network models and layers.
import torch.nn.functional as F
import torch.optim as optim ## Provides functionalities for optimizers like SGD, Adam, etc.
from torch.utils.data import dataloader ## Utility for loading data from folders.

import torchvision ## Library responsible for handling image related tasks.
from torchvision import models, datasets, transforms

from sklearn import metrics
from PIL import Image
import os
import sys
import shutil
import glob
import warnings
from IPython.core.display import display, HTML
print(os.listdir("../input")) ## Kaggle stores all dataset in this directory.

print(torch.__version__) 
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Cuda available : '+ str((device.type == 'cuda')))


# Defining dictionary below which is mapping from tag name to corresponding html contents which will stored as HTML file. The model will first find out tag name and then using this dictionary it'll convert tag to appropriate HTML content and save it as HTML file.

# In[ ]:


tag_to_html = {'body_tag': '<!DOCTYPE html>                            <html>                            <head>                            <title>Shape To HTML</title>                            </head>                            <body>                            This is html body.                            </body>                            </html>',
                'button_tag':'<!DOCTYPE html>\
                                <html>\
                                <body>\
                                <button type="button" onclick="alert("Hello world!")">Click Me!</button>\
                                </body>\
                                </html>',
               "text_tag": "<!DOCTYPE html>\
                                <html>\
                                <head>\
                                <title>Shape To HTML</title>\
                                </head>\
                                <body>\
                                This is html text tag.\
                                </body>\
                                </html>",
               'heading_and_image_tag': "<!DOCTYPE html>\
                                            <html>\
                                            <head>\
                                            <title>Shape To HTML</title>\
                                            </head>\
                                            <body>\
                                            <h1>Heading & Image tag</h1>\
                                            <img src='image_name.jpg'  width='500' height='333'>\
                                            </body>\
                                        </html>",
               "heading_image_and_table_tag": "<!DOCTYPE html>\
                                                <html>\
                                                <head>\
                                                <style>\
                                                table, th, td {\
                                                  border: 1px solid black;\
                                                  border-collapse: collapse;\
                                                }\
                                                th, td {\
                                                  padding: 5px;\
                                                  text-align: left;    \
                                                }\
                                                </style>\
                                                </head>\
                                                <body>\
                                                <h2>The heading, image & table Element</h2>\
                                                <img src='image_name.jpg'  width='500' height='333'>\
                                                <table style='width:100%'>\
                                                  <tr>\
                                                    <th>Name</th>\
                                                    <th colspan='2'>Telephone</th>\
                                                  </tr>\
                                                  <tr>\
                                                    <td>Bill Gates</td>\
                                                    <td>55577854</td>\
                                                    <td>55577855</td>\
                                                  </tr>\
                                                </table>\
                                                </body>\
                                                </html>",
               "image_tag": "<!DOCTYPE html>\
                                <html>\
                                <head>\
                                <title>Shape To HTML</title>\
                                </head>\
                                <body>\
                                <img src='image_name.jpg'  width='500' height='333'>\
                                </body>\
                                </html>",
               "nav_tag" : "<!DOCTYPE html>\
                                <html>\
                                <body>\
                                <nav>\
                                <a href='/html/'>HTML</a> |\
                                <a href='/css/'>CSS</a> |\
                                <a href='/js'>JavaScript</a> |\
                                <a href='/jquery/'>jQuery</a>\
                                </nav>\
                                </body>\
                                </html>",
               "table_tag":"<!DOCTYPE html>\
                            <html>\
                            <head>\
                            <style>\
                            thead {color:green;}\
                            tbody {color:blue;}\
                            tfoot {color:red;}\
                            table, th, td {\
                              border: 1px solid black;\
                            }\
                            </style>\
                            </head>\
                            <body>\
                            <table>\
                              <thead>\
                                <tr>\
                                  <th>Month</th>\
                                  <th>Savings</th>\
                                </tr>\
                              </thead>\
                              <tbody>\
                                <tr>\
                                  <td>January</td>\
                                  <td>$100</td>\
                                </tr>\
                                <tr>\
                                  <td>February</td>\
                                  <td>$80</td>\
                                </tr>\
                              </tbody>\
                              <tfoot>\
                                <tr>\
                                  <td>Sum</td>\
                                  <td>$180</td>\
                                </tr>\
                              </tfoot>\
                            </table>\
                            </body>\
                            </html>",
               'combo': """<!doctype html>
<html lang=''>
<head>\
   <meta charset='utf-8'>
   <meta http-equiv="X-UA-Compatible"content="IE=edge">
   <meta name="viewport" content="width=device-width, initial-scale=1">
   <script src="http://code.jquery.com/jquery-latest.min.js" type="text/javascript"></script>
   <script src="script.js"></script>
   <title>CSS MenuMaker</title>
   <style>
* {
  box-sizing: border-box;
}


.column {
  float: left;
  width: 50%;
  padding: 10px;
  height: 600px; 
}
.column1 {
  float: left;
  width: 50%;
  padding: 10px;
  height: 600px; 
}
.row:after {
  content: "";
  display: table;
  clear: both;
}
table, th, td {
  border: 1px solid black;
  font:solid white;
}
@import url(http://fonts.googleapis.com/css?family=Lato:300,400,700);
@charset "UTF-8";
/* Base Styles */
#cssmenu ul,
#cssmenu li,
#cssmenu a {
  list-style: none;
  margin: 0;
  padding: 0;
  border: 0;
  line-height: 1;
  font-family: 'Lato', sans-serif;
}
#cssmenu {
  border: 1px solid #133e40;
  -webkit-border-radius: 5px;
  -moz-border-radius: 5px;
  -ms-border-radius: 5px;
  -o-border-radius: 5px;
  border-radius: 5px;
  width: auto;
}
#cssmenu ul {
  zoom: 1;
  background: #36b0b6;
  background: -moz-linear-gradient(top, #36b0b6 0%, #2a8a8f 100%);
  background: -webkit-gradient(linear, left top, left bottom, color-stop(0%, #36b0b6), color-stop(100%, #2a8a8f));
  background: -webkit-linear-gradient(top, #36b0b6 0%, #2a8a8f 100%);
  background: -o-linear-gradient(top, #36b0b6 0%, #2a8a8f 100%);
  background: -ms-linear-gradient(top, #36b0b6 0%, #2a8a8f 100%);
  background: linear-gradient(top, #36b0b6 0%, #2a8a8f 100%);
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='@top-color', endColorstr='@bottom-color', GradientType=0);
  padding: 5px 10px;
  -webkit-border-radius: 5px;
  -moz-border-radius: 5px;
  -ms-border-radius: 5px;
  -o-border-radius: 5px;
  border-radius: 5px;
}
#cssmenu ul:before {
  content: '';
  display: block;
}
#cssmenu ul:after {
  content: '';
  display: table;
  clear: both;
}
#cssmenu li {
  float: right;
  margin: 0 5px 0 0;
  border: 1px solid transparent;
}
#cssmenu li a {
  -webkit-border-radius: 5px;
  -moz-border-radius: 5px;
  -ms-border-radius: 5px;
  -o-border-radius: 5px;
  border-radius: 5px;
  padding: 8px 15px 9px 15px;
  display: block;
  text-decoration: none;
  color: #ffffff;
  border: 1px solid transparent;
  font-size: 16px;
}
#cssmenu li.active {
  -webkit-border-radius: 5px;
  -moz-border-radius: 5px;
  -ms-border-radius: 5px;
  -o-border-radius: 5px;
  border-radius: 5px;
  border: 1px solid #36b0b6;
}
#cssmenu li.active a {
  -webkit-border-radius: 5px;
  -moz-border-radius: 5px;
  -ms-border-radius: 5px;
  -o-border-radius: 5px;
  border-radius: 5px;
  display: block;
  background: #1e6468;
  border: 1px solid #133e40;
  -moz-box-shadow: inset 0 5px 10px #133e40;
  -webkit-box-shadow: inset 0 5px 10px #133e40;
  box-shadow: inset 0 5px 10px #133e40;
}
#cssmenu li:hover {
  -webkit-border-radius: 5px;
  -moz-border-radius: 5px;
  -ms-border-radius: 5px;
  -o-border-radius: 5px;
  border-radius: 5px;
  border: 1px solid #36b0b6;
}
#cssmenu li:hover a {
  -webkit-border-radius: 5px;
  -moz-border-radius: 5px;
  -ms-border-radius: 5px;
  -o-border-radius: 5px;
  border-radius: 5px;
  display: block;
  background: #1e6468;
  border: 1px solid #133e40;
  -moz-box-shadow: inset 0 5px 10px #133e40;
  -webkit-box-shadow: inset 0 5px 10px #133e40;
  box-shadow: inset 0 5px 10px #133e40;
}

</style>
</head>
<body>

<div id='cssmenu'>
<ul>
   <li class='active'><a href='#'><span>Button</span></a></li>
   <li><a href='#'><span>Button</span></a></li>
   <li><a href='#'><span>Button</span></a></li>
   <li class='last'><a href='#'><span>Button</span></a></li>
</ul>
</div>
<div class="row">
  <div class="column">
  <h1 style="color:black;">This is a Heading</h1>
<img src="rtn.jpg"  width="500" height="300" border="3"><br>
<button onclick="myFunction()">Click me</button>
<p id="demo"></p>
<script>

function myFunction() {
  document.getElementById("demo").innerHTML = "Hello World";
}
</script>
  </div>
  <div class="column1">
   <br><br><br><br><br><br><table style="width:100%">
   
  <tr>
    <th>Firstname</th>
    <th>Lastname</th> 
   
  </tr>
  <tr>
    <td>Jill</td>
    <td>Smith</td>
 
  </tr>
  <tr>
    <td>Eve</td>
    <td>Jackson</td>

  </tr>
  <tr>
    <td>John</td>
    <td>Doe</td>
   
  </tr>
</table>
  </div>
</div>


</body>
<html>
"""
               
              }


# Checking if `shapes` and `image_folders` directory already exists. If they exists then delete it for fresh start.

# In[ ]:


if os.path.exists('shapes'):
    print('Deleting Shapes Directory if exists already from last run')
    shutil.rmtree('shapes')
if os.path.exists('image_folders'):
    print('Deleting image_folders Directory if exists already from last run')
    shutil.rmtree('image_folders')


# During creating dataset and attach it to this kernel then it will be available in `../input` directory. here has been used list command to check its contents.

# In[ ]:


get_ipython().run_line_magic('ls', '../input/image-folders/image_folders/')


# here has been copied `image_folders` from `../input/image-folders` to current directory for the processing purpose.

# In[ ]:


get_ipython().run_line_magic('ls', '../input')
get_ipython().system('cp -r ../input/image-folders/image_folders/ .')


# Verify that `image_folders` got copied successfully.

# In[ ]:


get_ipython().run_line_magic('ls', '')


# Rename folder named `heading_tag&image_tag` as it has special character `&` which is not valid name for directories and can cause failures in future.

# In[ ]:


get_ipython().run_line_magic('ls', '../input/image-folders/image_folders/')


# In[ ]:


get_ipython().run_line_magic('ls', 'image_folders')
os.rename('image_folders/heading_tag&image_tag','image_folders/heading_and_image_tag')
get_ipython().run_line_magic('ls', 'image_folders')


# ## Creating Folder Structure For Training/Validation/Testing Purpose Of Model.
# 
# Here have been defined below method which will be responsible for creating data structure for the model and moving files into particular folders.
# 
# Creating new directory called `shapes`. It'll have 3 folders.
# 1. train
# 2. val
# 3. test
# 
# Each of train/val/test will have same 7 subfolders corresponding to 7 shapes as that of image_folders. Then will be moved 80% of images to train folder, 10% will be moved to val folder and remaining 10% to test folder and also will  be providing method 2 input direcotries (src-refers to direcotory where all images are currently residing, dest-refers to directory where iages will be copied).

# In[ ]:


def create_ml_file_strcuture_and_move_files(src, dest):
    ## We create first 3 top level directories under dest directory.
    os.makedirs(os.path.join(dest,'train'), exist_ok=True)## Creates subdirectory like /shapes/train
    os.makedirs(os.path.join(dest,'val'), exist_ok=True)## Creates subdirectory like /shapes/val
    os.makedirs(os.path.join(dest,'test'), exist_ok=True)## Creates subdirectory like /shapes/test
    
    ## We now create 7 sub directories under main 3 directories under dest and move files according to propotion from src to dest.
    for directory in os.listdir(src):
        os.makedirs(os.path.join(dest,'train',directory), exist_ok=True) ## Creates subdirectory like /shapes/train/text_tag
        os.makedirs(os.path.join(dest,'val',directory), exist_ok=True)## Creates subdirectory like /shapes/val/text_tag
        os.makedirs(os.path.join(dest,'test',directory), exist_ok=True)## Creates subdirectory like /shapes/test/text_tag
        init_path = os.path.join(src, directory)
        all_files = os.listdir(init_path) ## Getting all files from src subdirectory of tag.
        n = len(all_files) ## Number of files for particular tag.
        for file in all_files[:int(0.8*n)]: ## Logic to move 80% to shapes/train directory
            shutil.copy(os.path.join(src,directory,file),os.path.join(dest,'train',directory))
        for file in all_files[int(0.8*n):int(0.9*n)]: ## Logic to move 10% to shapes/val directory
            shutil.copy(os.path.join(src,directory,file),os.path.join(dest,'val',directory))
        for file in all_files[int(0.9*n):]: ## Logic to move 10% to shapes/test directory
            shutil.copy(os.path.join(src,directory,file),os.path.join(dest,'test',directory))

create_ml_file_strcuture_and_move_files('image_folders','shapes')


# Listing of print statements below which verifies that whether above function properly created folder structure and moved files properly.

# In[ ]:


print('List of subdirs in Images folder : %d'%len(os.listdir('image_folders')))
print('List of subdirs in dogs/train folder : %d'%len(os.listdir('shapes/train')))
print('List of subdirs in dogs/val folder : %d'%len(os.listdir('shapes/val')))
print('List of subdirs in dogs/test folder : %d'%len(os.listdir('shapes/test')))
print('List of JPGs in original Images directory : %d'%len(glob.glob('image_folders/*/*.jpg')))
print('List of JPGs in dogs sub directories : %d'%len(glob.glob('shapes/*/*/*.jpg')))

print("List of JPGs in Heading & Image Tag of train set: %d"%len(os.listdir('shapes/train/heading_and_image_tag')))
print("List of JPGs in Heading & Image Tag of validation set : %d"%len(os.listdir('shapes/val/heading_and_image_tag')))
print("List of JPGs in Heading & Image Tag of test set : %d"%len(os.listdir('shapes/test/heading_and_image_tag')))

print("List of JPGs in Image Tag of train set: %d"%len(os.listdir('shapes/train/image_tag')))
print("List of JPGs in Image Tag of validation set: %d"%len(os.listdir('shapes/val/image_tag')))
print("List of JPGs in Image Tag of test set: %d"%len(os.listdir('shapes/test/image_tag')))


# ## Initializing Dataset And Creating Transformations On Images
# Below has defined `data_trainsform` which is transformation that will be applied to all images. As we'll be using VGG model for our purpose, it expects images to be of size (224 px,224 px). It also need each RGB channel of image to be normalized. Normalization refers to subtraction of mean and division by standard deviation. First resizing all images to 256x256 pixel , then will crop center part of image of size 224x224 pixel, then will convert image to tensor as needed by model as input and then will normalize tensor.
# 
# Has defined `dsets` dictionary which holds datasets for train, val and test parts. Using `ImageFolder` class provided by torchvision.datasets module to load images from particular folders and apply transformation at same time.
# 
# Then has defined `loaders` dictionary which is another wrapper around dataset which will group images into batches defined by `batch_size` and shuffle them based on `shuffle` flag. `num_workers` refers to number of parallel task for loading and also defined batch size for train dataset as 8 mean batch of 8 images will be trained together.

# In[ ]:


root_folder = "shapes"

data_transform = transforms.Compose([ transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dsets = {}
dsets['train'] = datasets.ImageFolder('shapes/train', transform=data_transform)
dsets['val'] = datasets.ImageFolder('shapes/val', transform=data_transform)
dsets['test'] = datasets.ImageFolder('shapes/test', transform=data_transform)
loaders = {}
loaders['train'] = torch.utils.data.DataLoader(dsets['train'], batch_size=8, shuffle=True,num_workers=4)
loaders['val'] = torch.utils.data.DataLoader(dsets['val'], batch_size=8, shuffle=True,num_workers=4)
loaders['test'] = torch.utils.data.DataLoader(dsets['test'], batch_size=1, shuffle=False,num_workers=4)


# Below have defined few dictionaries. shapes_to_idx refers to dictionary of shape names to their index used by model. idx_to_shape is revers of shapes_to_idx. Neural Network will be outputing index for tags which will convert to tag names using this dictionary.

# In[ ]:


shapes = dsets['train'].classes
shapes_to_idx = dsets['train'].class_to_idx
shapes, idx = zip(*dsets['train'].class_to_idx.items())
idx_to_shape = dict(zip(idx, shapes))
idx_to_shape, shapes_to_idx


# Lets check size of first few tensors.

# In[ ]:


for i,(images,labels) in enumerate(loaders['train']):
    if i == 3: break;
    print(images.size(),labels.size())


# Below are visualizing first few images.

# In[ ]:


images,labels = next(iter(loaders['train']))
inp = torchvision.utils.make_grid(images)
print('Type of Image : '+ str(type(inp)))
inp = inp.numpy().transpose(1,2,0)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image = std * inp + mean
iamge = image.clip(0,1)
plt.figure(figsize=(25,5))
plt.imshow(image)
plt.title(str([idx_to_shape[label] for label in labels.numpy()]))
plt.xticks([])
plt.yticks([])
None


# ## Initializing Model
# 
# Here using VGG model provided by `torchvision.models` module. VGG is neural network model which was 2nd runner up in imagenet competition of 2014. Authors of VGG has released weights of model and model structure afte competition which many deep learning practitioners uses for training their model. This practice of using existing pre-trained network is called transferred learning and have passed pretrained as True which will results in downloading weights from internet.
# 
# Also using model as it by modifying it's last layer. VGG model outputs 1000 probabilites as competition had 1000 categories to classify of images. For the purpose of having only 7 categories, so will be modfying last layer to output 7 probabilities instead of 1000. Also will set all other layers `requires_grad` parameters as `False`. Only last layer which will be modified, will have `requires_grad` as `True`. Reason behind doing this is that all other layers are well trained already and just need to train last layer which modified for the purpose. This way of resuing model architecture and its weights is called Transfer Learning.
# 
# Also will print architecture of VGG at end which is convolution neural network.

# In[ ]:


vgg = models.vgg16(pretrained=True)
for param in list(vgg.parameters())[:-1]:
    param.requires_grad = False
vgg.classifier[6] = nn.Linear(4096, len(shapes))
vgg = vgg.to(device)
vgg


# ## Defining Loss Function And Optimization Function
# 
# Here has been defined below loss function which will be used to calculate loss of training/validation and test phases. Loss generally refers to how far our prediction is from actual prediction. `CrossEntropyLoss` is multi-class loss function defined in PyTorch.
# 
# Also has been defined optimization function which will optimize out parameters so that the model outputs as accuracte label as possible and defined learning rate of 0.001 which will be used to update model weights.

# In[ ]:


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=vgg.parameters(), lr = 0.001)


# ## Training Model
# Below has been defined method which will be used to train model by providing epochs. 1 epoch refers to one pass through Train and Validation data. Training model number of epochs times and validate it against validation set. Here train model through training dataset and then validate it against validation set then printed Train and validation accuracy at each epoch.

# In[ ]:


def train(epochs):
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                vgg.train() ## We set model to train phase as it activates layers like Dropout and BatchNormalization.
            else:
                vgg.eval() ## We set model to evaluation phase as it de-activates layers like Dropout and BatchNormalization.

            total_loss = 0.0
            correct_preds = 0

            for i, (images, labels) in enumerate(loaders[phase]):
                images, labels = images.to(device), labels.to(device) ## Translate normal tensor to cuda tensors it GPU is available.
                optimizer.zero_grad() ## At start of each batch we set gradients of loss with respect to parameters to zero.
                with torch.set_grad_enabled(phase == 'train'): ## This enables gradients calculation based on phase.
                    results = vgg.forward(images) ## We do forward pass thorugh batch images.
                    _, predictions = torch.max(results,1) ## We get indexes of max probabilities for each image of batch.
                    loss = loss_function(results, labels) ## We calculation loss based on predicted probabilities and actual labels.

                    if phase == 'train':
                        loss.backward() # Backpropogation execution which calculates gradients for each weight parameter.
                        optimizer.step() ## This step updates weights based on gradients calculated above and learning rate set above.
                #print(i)
                total_loss += loss.item()
                correct_preds += torch.sum(predictions == labels) ## We find out correct predictions.

            print('Epoch : %d'%(epoch+1))
            print('Stage : %s'%phase)
            print('Loss : %f'%total_loss)
            #print(correct_preds.item())
            print('Accuracy : %f'% (int(correct_preds.item()) / len(dsets[phase])))
            print('-'*100)


# Initiallizing train model for 3 epochs (3 times pass through train and validation dataset).

# In[ ]:


get_ipython().run_line_magic('time', 'train(3)')


# In[ ]:


#%time train(2)


# Then lowers learning rate and then train model again for 2 epochs.

# In[ ]:


optimizer.lr = 0.0001
get_ipython().run_line_magic('time', 'train(2)')


# After that reduce learning rate again and then train model again for 1 epoch.

# In[ ]:


optimizer.lr = 0.00001
get_ipython().run_line_magic('time', 'train(1)')


# ## Evaluating Model On Test Dataset.
# 
# In her have been defined below test model which test the models accuracy on test dataset which have never seen before. Test data set is like production dataset which is never seen by the model.

# In[ ]:


def test():
    with torch.no_grad(): ## We are setting it to no grads as we don't need gradients during testing.
        correct = 0
        #loss = 0
        for images,labels in loaders['test']: ## We loop through test dataset images
            images,labels = images.to(device), labels.to(device)

            predictions = vgg(images) ## Our model predicts labels for images
            _, preds = torch.max(predictions, 1) ## We take out index of maximum probability.
            correct += torch.sum(preds == labels) ## Summing up correct labels.
        print('Test Set Accuracy : %f'%(correct.item() / len(dsets['test'])))

get_ipython().run_line_magic('time', 'test()')


# ## Visualizing First Few Test Image Predictions.
# 
# Author has defined below method which loops through images of test dataset and then prints images with their actual labels and predictions made by the model.

# In[ ]:


def visualizing_predictions_on_test_data():
    plt.figure(figsize=(20,28))
    with torch.no_grad():
        for i, (image,label) in enumerate(loaders['test']):
            if i == 80:
                break
            plt.subplot(10,8,i+1)
            image,label = image.to(device), label.to(device)

            prediction = vgg(image)
            _, pred = torch.max(prediction,1)
            img = image.to('cpu').numpy()[0].transpose(1,2,0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            plt.imshow(img.clip(0.0,1.0))
            plt.title('Actual : %s,\nPredicted : %s'%(idx_to_shape[int(label.item())], idx_to_shape[int(pred.item())]))
            plt.xticks([])
            plt.yticks([])

visualizing_predictions_on_test_data()


# ## Defining Method For Making Prediction
# 
# Here has been defined below method named `predict()` which takes as input path to image and then predicts top 5 probabilities and top 5 indexes. `process_image()` method performs same steps that performed by `data_transform` author has defined above. It resize image to 256x256, then does center crop of size 224x224, convert to tensor and then normalize image to make it ready for model input.

# In[ ]:


def process_image(image,normalize=True):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img = img.convert('RGB')
    img = np.array(img.resize((256,256)).crop((16,16,240,240)))
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    img = img.numpy()
    img = img.transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    if normalize:
        img = ((img - mean) / std)
    img = img.transpose((2,0,1))
    
    img = torch.tensor(img,dtype=torch.float32)
    return img

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        img = process_image(image_path)
        output = vgg.forward(img.unsqueeze(0).to(device) if len(img.size())==3 else img.to(device))
        top_5_probs,classes = output.topk(topk)
        return top_5_probs, classes


# In[ ]:


ls ../input/unseen-images/unseen_images/


# In[ ]:


get_ipython().run_line_magic('ls', 'shapes/test/')


# **Below author has set image name in `future_shape_path` variable and then use `predict()` method defined above to get top 5 probabilities and takes highest probability as out guess. Author then uses `idx_to_shape` dictionary to find out tag name correcponding to that index. Using tag name, author finds out html contents of that tag using `tag_to_html` dictionary.

# In[ ]:


#future_shape_path = 'shapes/test/heading_and_image_tag/20190315_133359.jpg'
unseen_image_root = '../input/unseen-images/unseen_images/'
future_shape_path = os.path.join(unseen_image_root,'IMG-20190315-WA0024.jpg')
prob, idx = predict(future_shape_path,vgg)
#print(idx)
predicted_name = [idx_to_shape[i.item()] for i in idx.data[0]][0]
print(predicted_name)
print(tag_to_html[predicted_name])


# ## Generating HTML Files From New Unseen Images.
# 
# Here author has kept all unseen images under `unseen-images` folder which have uploaded as different dataset. We can upload many images to this dataset. We then loop through each image of that unseen-images dataset, guess tag for each image, get html for that particular tag and save it as `.html` file.

# In[ ]:


plt.figure(figsize=(20,8))
total_image_count = len(os.listdir(unseen_image_root))
for i, image in enumerate(os.listdir(unseen_image_root),1):
    prob, idx = predict(os.path.join(unseen_image_root,image),vgg)
    predicted_name = [idx_to_shape[i.item()] for i in idx.data[0]][0]
    plt.subplot((total_image_count//8)+1, 8, i)
    plt.imshow(Image.open(os.path.join(unseen_image_root,image)))
    plt.xticks([])
    plt.yticks([])
    plt.title(image+ '\n'+predicted_name)
    with open(image.split('.')[0]+'.html', 'w') as f:
        f.write(tag_to_html[predicted_name])


# Below is logic which displays all html files generated from unseen-images folder and show them as hypterlink.

# In[ ]:


for file in os.listdir('.'):
    if '.html' in file:
        display(HTML("<a href=%s target='_blank'>%s</a>"%(file, file)))


# ## Visualizing Failed Test Predictions
# 
# We have defined below method which loops through all images of testset and display images which are predicted wrong by our model.

# In[ ]:


def visualizing_failed_predictions_on_test_data():
    plt.figure(figsize=(20,25))
    with torch.no_grad():
        for i, (image,label) in enumerate(loaders['test']):
            image,label = image.to(device), label.to(device)
            prediction = vgg(image)
            _, pred = torch.max(prediction,1)
            if label.item() != pred.item():
                plt.subplots()
                #print('Actual : %s,\nPredicted : %s'%(idx_to_shape[int(label.item())], idx_to_shape[int(pred.item())]))
                img = image.to('cpu').numpy()[0].transpose(1,2,0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                plt.imshow(img.clip(0.0,1.0))
                plt.title('Actual : %s,\nPredicted : %s'%(idx_to_shape[int(label.item())], idx_to_shape[int(pred.item())]))
                plt.xticks([])
                plt.yticks([]);

visualizing_failed_predictions_on_test_data();


# Below code displays image defined by `future_shape_path` variable and displays top 5 probabilities predicted by model.

# In[ ]:


probs2, idx2 = predict(future_shape_path,vgg)
names = [idx_to_shape[i.item()] for i in idx2.data[0]]
probs = probs2.data[0].cpu().numpy()
probs = np.exp(probs)

## Display an image along with the top 5 classes
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.imshow(Image.open(future_shape_path))
plt.xticks([])
plt.yticks([])
plt.title(future_shape_path)

plt.subplot(122)
plt.barh(names,probs);


# ## Saving model for future use
# We have below logic which stores current trained model into current directory. This model can be later loaded for future use. We can directly load model from this saved checkpoint and then do prediction directly without everytime training model.

# In[ ]:


checkpoint_dict = {
    'model_dict': vgg.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'loss' : loss_function.state_dict(),
    'class_to_idx' : dsets['train'].class_to_idx,
}
torch.save(checkpoint_dict, 'vgg_99.pth')


# ## Load Model From Saved Checkpoint And Make Prediction
# Below we have defined logic which loads model from saved checkpoint above. We are also making prediction in next line from this model loaded from checkpoint.

# In[ ]:


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    reloaded_model = models.vgg16(pretrained=False).to(device)
    for param in list(reloaded_model.parameters())[:-1]:
        param.requires_grad = False
    reloaded_model.classifier[6] = nn.Linear(4096, len(checkpoint['class_to_idx']))
    reloaded_model.load_state_dict(checkpoint['model_dict'])
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(reloaded_model.parameters(), lr = 0.00001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion.load_state_dict(checkpoint['loss'])
    #reloaded_model = reloaded_model.to(device)
    return reloaded_model,optimizer,criterion,checkpoint['class_to_idx']

model, optimizer, loss_function, class_to_idx = load_checkpoint('vgg_99.pth')


# In[ ]:


#future_shape_path = 'shapes/test/heading_and_image_tag/20190315_133359.jpg'
future_shape_path = os.path.join(unseen_image_root,'DSC_2454.JPG')
prob, idx = predict(future_shape_path,model)
print(idx)
predicted_name = [idx_to_shape[i.item()] for i in idx.data[0]][0]
print(predicted_name)
print(tag_to_html[predicted_name])


# In[ ]:


shutil.rmtree('shapes')
shutil.rmtree('image_folders')

