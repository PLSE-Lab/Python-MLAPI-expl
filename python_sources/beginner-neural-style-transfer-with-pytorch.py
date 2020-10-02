#!/usr/bin/env python
# coding: utf-8

# #### Im newbie in this new field and I practice pytorch here.
# 
# - *The main objective of this notebook is to understand and hands-on coding in Pytorch.*
# - *This notebook is inspired from serveral works and I listed them in the Reference at the end.*
# - *Key words:*
# > - pretrained VGG19
# > - hook function
# > - GPU
# > - content loss, style loss, gram matrix
# - *Feel free to discuss :)*

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#from torch.utils.data import Dataset
from torchvision.models import vgg19
import torchvision.transforms as transforms
#from collections import OrderedDict
get_ipython().run_line_magic('pylab', 'inline')
print('Pytorch version: {}'.format(torch.__version__))


from PIL import Image
import time
import os
print(os.listdir("../input"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Computational device: {}'.format(device))


# In[ ]:


StylePath = '/kaggle/input/styles/'
ContentPath = '/kaggle/input/content/'

name_imgC = 'WechatIMG1.jpeg'
name_imgS = 'picasso1.jpg'

cont_image_path = ContentPath + name_imgC
style_image_path = StylePath + name_imgS

imageC = Image.open(cont_image_path)
imageS = Image.open(style_image_path)


# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(18, 12))
ax[0].set_title('Content image', fontsize="20")
ax[0].imshow(imageC.resize((512,512)))  
ax[1].set_title('Style image', fontsize="20")
ax[1].imshow(imageS.resize((512,512)))


# # Preprocessing and postprocessing utility functions
# ### transforms:
# > - Resize
# > - convert PIL Image to np array ( [0,255] -> [0,1] )
# > - RGB < - > BGR
# > - Normalize (use imagenet mean and std)
# > - Varaible, Lambda, Normalize
# > - unsqueeze(0): add one dimension because the input needs one sample in the batch ( ex. (4,)->(1,4) )
# 

# In[ ]:


imsize = 512

prep = transforms.Compose([transforms.Resize((imsize,imsize)),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), 
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])


postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])


postpb = transforms.Compose([transforms.ToPILImage()])

# Preprocessing: convert image to array
def prep_img(img_path):
    image = Image.open(img_path)
    image = Variable(prep(image))
    # network's input need at least a batch size 1
    image = image.unsqueeze(0)
    return image.to(device,torch.float)




# Postprocessing: convert array to image
def postp(tensor): 
    t = postpa(tensor)
    # to clip results in the range [0,1]
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img


# # Load the pretrained model (VGG19)

# In[ ]:


# input style image and content inmage
style_img = prep_img(style_image_path)
content_img = prep_img(cont_image_path)
#assert style_img.size() == content_img.size(),"import style and content images are not in the same size"

# load model in eval mode (model uses BN or Dropout)
vgg = vgg19(pretrained=True).features.to(device).eval()

# set requires_grad as false, as a result no backprop of the gradients
for param in vgg.parameters():
    param.requires_grad = False
#    print(param.requires_grad)


# initialize the output image as same as the content image or a random noise. The image need to be modified. 
opt_img = Variable(content_img.data.clone(),requires_grad=True)

#input_img = torch.randn(content_img.data.size(), device=device)
#opt_img = Variable(input_img,requires_grad=True)


# In[ ]:


#style_img.shape
#content_img.shape
#opt_img
vgg


# In[ ]:


# choose layers for style
style_layers = [1,6,11,20,26,35]

# one layer for content
content_layers = [29]


# # Loss functions
# - forward propagate image S and obtain the activation  $a^{(S)}$  at some layers (style).
# - forward propagate image C and obtain the activation  $a^{(C)}$  at one layer (content).
# - forward propagate image G and obtain the activation  $a^{(G)}$  at some layer (style + content).
# 
# - content loss function  $J_{content}(C,G)$ :
# 
# $J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum _{ \text{all entries}} (a^{(C)} - a^{(G)})^2\tag{1} $
# 
# - style loss function $J_{style}(S,G)$ 
# > - Gram matix: measures how similar the activations of filter i are to the activations of filter j, $G_{ij}=A_iA^T_j$.
# > - $J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}
# (G^{(S)}_{(gram)i,j} - G^{(G)}_{(gram)i,j})^2\tag{2} $
# > - $J_{style}(S,G) = \sum_{l} \lambda^{[l]} J^{[l]}_{style}(S,G)\tag{3}  $ 
# 
# - total loss:  $J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,Gv) \tag{4} $

# In[ ]:


# use hook to extract activations during forward prop
class LayerActivations():
    features=[]
    
    def __init__(self,model,layer_nums):
        
        self.hooks = []
# register activation after forword at eatch layer 
        for layer_num in layer_nums:
            self.hooks.append(model[layer_num].register_forward_hook(self.hook_fn))
#     
    def hook_fn(self,module,input,output):
        self.features.append(output)

    
    def remove(self):
        for hook in self.hooks:
            hook.remove()


# In[ ]:


def extract_layers(layers,img,model=None):
    la = LayerActivations(model,layers)
    #Clearing the cache 
    la.features = []
    # forward prop img and hook registes automatically activations
    out = model(img)
    # remove hook but features are already extracted.
    la.remove()
    return la.features


# In[ ]:


class ContentLoss(nn.Module):    
    
    def forward(self,inputs,targets):
        assert inputs.size() == targets.size(),"need the same size"
        b,c,h,w = inputs.size()
        loss = nn.MSELoss()(inputs, targets)
        loss.div_(4*c*h*w)
        return (loss)


# In[ ]:


class GramMatrix(nn.Module):
    
    def forward(self,input):
# batch, channel, height, width        
        b,c,h,w = input.size()
        features = input.view(b,c,h*w)
# batch matrix product (b*n*m)        
        gram_matrix =  torch.bmm(features,features.transpose(1,2))
        return gram_matrix


# In[ ]:


class StyleLoss(nn.Module):
    def forward(self,inputs,targets):
        assert inputs.size() == targets.size(),"need the same size"
        b,c,h,w = inputs.size()
        loss = F.mse_loss(GramMatrix()(inputs), GramMatrix()(targets))
        loss.div_(4*(c*h*w)**2)
        return (loss)


# # Precompute activations of the content and style images

# In[ ]:


a_c = extract_layers(content_layers,content_img,model=vgg)
a_c = [t.detach() for t in a_c]

a_s = extract_layers(style_layers,style_img,model=vgg)
a_s = [t.detach() for t in a_s]

activations = a_s + a_c 
#activations


# In[ ]:


loss_fns = [StyleLoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)


# In[ ]:


# weight of layers (alpha and beta)
style_weights = [100000 for n in range(len(style_layers))]
content_weights = [1]
weights = style_weights + content_weights


# # Main loop

# In[ ]:


max_iter = 500
show_iter = 100

# parameters to optimize (tensors or dicts)
optimizer = optim.LBFGS([opt_img]);
n_iter=[0]

while n_iter[0] <= max_iter:
    
    # evaluate the model and return the loss for optimizer
    def closure():
        
        # clean cach
        optimizer.zero_grad()
        
        # extract acivations of the output image
        out_sty = extract_layers(style_layers,opt_img,model=vgg)
        out_cnt = extract_layers(content_layers,opt_img,model=vgg)
        out =  out_sty + out_cnt
        
        # compute losses
        layer_losses = [weights[a] * loss_fns[a](A, activations[a]) for a,A in enumerate(out)]
        #print(layer_losses[0])
        
        # .backward apply to a scaler
        loss = sum(layer_losses)
        
        # compute gradients
        loss.backward()
        n_iter[0]+=1
        
        if n_iter[0]%show_iter == (show_iter-1):
            print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))

        return loss
    # parameters update
    optimizer.step(closure)
    


# In[ ]:


#display result
out_img_hr = postp(opt_img.data[0].cpu().squeeze())

imshow(out_img_hr)
gcf().set_size_inches(10,10)


# # Reference
# - [PyTorch Manual](https://pytorch.org/)
# - Sylvin, Chateau, [Style transfer](https://www.kaggle.com/schateau/style-transfer)
# - [Deep Learning with PyTorch](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-pytorch?utm_source=github&utm_medium=repository&utm_campaign=9781788624336)
# - [NEURAL TRANSFER USING PYTORCH](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss)
# - Neural Networks and Deep Learning by deeplearning.ai

# In[ ]:




