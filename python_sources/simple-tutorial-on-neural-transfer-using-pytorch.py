#!/usr/bin/env python
# coding: utf-8

# ## Neural Transfer Using Pytorch
# 
# In this notebook we are going to learn and explore in detail about neural style transfer.
# 

# **Table of Contents**
# 
# 1. [Introduction](#point1)
# 2. [Principle](#point2)
# 3. [Loading Images](#point3)
# 4. [Visualization](#point4)
# 5. [Loss Functions](#point5)  
# 6. [Loading Vgg19](#point6)
# 7. [Training the model](#point7)
# 8. [Transfer Model output](#point8)
# 9. [Conclusion](#point9)
# 10. [Resources](#point10)

# ### Introduction <a id='point1'></a>
# 
# The Neural-Style algorithm developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge. 
# Style transfer method that is outlined in the paper, [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) in PyTorch.
# 
# In this paper, style transfer uses the features found in the 19-layer VGG Network, which is comprised of a series of convolutional and pooling layers, and a few fully-connected layers. 

# ### Principle <a class="anchor" id="point2"></a>
# 
# The principle concept behind the neural transfer is that we define two distances, one for the content which measures how different the content is between two images and one for the style which measures how different the style is between two images.
# 
# Then, we take a third image, the input, and transform it to minimize both its content-distance with the content-image and its style-distance with the style-image.
# 
# **Importing Packages**
# 
# Below is a list of the packages needed to implement the neural transfer.
# 
# * torch, torch.nn, numpy (indispensables packages for neural networks with PyTorch)
# * torch.optim (efficient gradient descents)
# * PIL, PIL.Image, matplotlib.pyplot (load and display images)
# * torchvision.transforms (transform PIL images into tensors)
# * torchvision.models (train or load pre-trained models)
# * copy (to deep copy the models; system package)

# In[ ]:


from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


# In[ ]:


# move the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 
# ### Loading Images <a class="anchor" id="point3"></a>
# 
# Below, is a helper function for loading in any type and size of image. The load_image function also converts images to normalized Tensors.
# 
# Additionally, it will be easier to have smaller images and to squish the content and style images so that they are of the same size.

# In[ ]:


def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    
    image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image


# In[ ]:


# load in content and style image
content = load_image('/kaggle/input/images/lionKing.jpg').to(device)
# Resize style to match content, makes code easier
style = load_image('/kaggle/input/images/cartoon.jpg', shape=content.shape[-2:]).to(device)


# ### Loading Vgg19 <a class="anchor" id="point6"></a>
# VGG19 is split into two portions:
# 
# 1. vgg19.features, which are all the convolutional and pooling layers.
# 2. vgg19.classifier, which are the three linear, classifier layers at the end.
# 
# We only need the features portion, which we're going to load in and "freeze" the weights of, below.

# In[ ]:


# get the "features" portion of VGG19 (we will not need the "classifier" portion)
vgg = models.vgg19(pretrained=True).features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)


# In[ ]:


vgg.to(device)


# 
# ### Visualization <a class="anchor" id="point4"></a>

# In[ ]:


# helper function for un-normalizing an image 
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


# In[ ]:


# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# content and style ims side-by-side
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))


# ### Loss Functions <a class="anchor" id="point5"></a>
# Mapping of layer names to the content representation and the style representation.

# In[ ]:


def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)    """
    
   
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features


# **Content Loss**
# 
# The content loss is a function that represents a weighted version of the content distance for an individual layer.
# 
# **Style Loss**
# 
# It will act as a transparent layer in a network that computes the style loss of that layer.

# **Gram Matrix**
# 
# The output of every convolutional layer is a Tensor with dimensions associated with the batch_size, a depth, d and some height and width (h, w).
# 
# The Gram matrix of a convolutional layer can be calculated as follows:
# 
# Get the depth, height, and width of a tensor using batch_size, d, h, w = tensor.size
# Reshape that tensor so that the spatial dimensions are flattened
# Calculate the gram matrix by multiplying the reshaped tensor by it's transpose
# 
# 
# Note: You can multiply two matrices using torch.mm(matrix1, matrix2).

# In[ ]:



def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram


# Putting it all Together
# Now that we've written functions for extracting features and computing the gram matrix of a given convolutional layer; let's put all these pieces together! We'll extract our features from our images and calculate the gram matrices for each layer in our style representation.

# In[ ]:


# get content and style features only once before training
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# it is a good idea to start off with the target as a copy of our *content* image
# then iteratively change its style
target = content.clone().requires_grad_(True).to(device)


# #### Loss and Weights
# Individual Layer Style Weights
# Below, we have the option to weight the style representation at each relevant layer. It's suggested that you use a range between 0-1 to weight these layers. By weighting earlier layers (conv1_1 and conv2_1) more, you can expect to get larger style artifacts in your resulting, target image. Should you choose to weight later layers, you'll get more emphasis on smaller features. This is because each layer is a different size and together they create a multi-scale style representation!
# 
# Content and Style Weight
# Just like in the paper, we define an alpha (content_weight) and a beta (style_weight). This ratio will affect how stylized your final image is. It's recommended that you leave the content_weight = 1 and set the style_weight to achieve the ratio you want.

# In[ ]:


# weights for each style layer 
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1  # alpha
style_weight = 1e6  # beta


# ### Training the model <a class="anchor" id="point7"></a>
# Updating the Target & Calculating Losses we will decide on a number of steps for which to update the image, we are changing our target image and nothing else about VGG19 or any other image. Therefore, the number of steps is really up to us. We may want to start out with fewer steps if we are just testing out different weight values or experimenting with different images.
# Inside the iteration loop, we will calculate the content and style losses and update your target image, accordingly.
# 
# The content loss will be the mean squared difference between the target and content features at layer conv4_2. This can be calculated as follows:
# 
# **content_loss = torch.mean((target_features['conv4_2']- content_features['conv4_2'])**2)**
# 
# The style loss is calculated in a similar way, only we have to iterate through a number of layers, specified by name in our dictionary style_weights.
# 
# 
# we will calculate the gram matrix for the target image, target_gram and style image style_gram at each of these layers and compare those gram matrices, calculating the layer_style_loss. Later, we will see that this value is normalized by the size of the layer.
# Finally, we will create the total loss by adding up the style and content losses and weighting them with your specified alpha and beta!
# Intermittently, we'll print out this loss; don't be alarmed if the loss is very large. It takes some time for an image's style to change and you should focus on the appearance of your target image rather than any loss value. Still, we should see that this loss decreases over some number of iterations.

# In[ ]:


# for displaying the target image, intermittently
show_every = 400

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 2000  # decide how many iterations to update your image (5000)

for ii in range(1, steps+1):
    
    # get the features from your target image
    target_features = get_features(target, vgg)
    
    # the content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)
        
    # calculate the *total* loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # display intermediate images and print the loss
    if  ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()


# ### Display the Target Image

# In[ ]:


# display content and final, target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))


# 
# ### Saving the Transfer Model <a class="anchor" id="point8"></a>

# In[ ]:


torch.save(vgg, 'model.pt')


# ### Conclusion <a class="anchor" id="point9"></a>
# 
# We have built a pytorch vgg19 model which performs nerural transfer.

# ### Resources <a class="anchor" id="point10"></a>
# 
# I referred the code from the below links.
# 1. [Pytorch Official documentation](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
# 2. [Udacity Pytorch Github](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/style-transfer/Style_Transfer_Solution.ipynb)
# 
# 

# In[ ]:




