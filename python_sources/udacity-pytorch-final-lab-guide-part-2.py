#!/usr/bin/env python
# coding: utf-8

# # **Udacity PyTorch Final Lab Challenge Guide - Part 2**

# By [Soumya Ranjan Behera](https://www.linkedin.com/in/soumya044)

# ## This is the continuation of my previous kernel [Udacity PyTorch Final Lab Guide - Part 1](https://www.kaggle.com/soumya044/udacity-pytorch-final-lab-guide-part-1/) 

# **Please Fork and Run this Notebook from Top-to-Bottom after running Part - 1's Notebook**

# ### In [Part - 1 (Build and Train our Model)](https://www.kaggle.com/soumya044/udacity-pytorch-final-lab-guide-part-1), 
# * We imported one pretrained model
# * Added our own Fully-Connected Layer at the end (to get 102 classes)
# * Trained the model
# * Evaluated it's performance
# * Prepared it for Export

# # **2. Submit in Udacity's Workspace to Evaluate**

# ### **Points To Remember:**  
# * **We trained our model in GPU environment** 
# * **But Udacity's Workspace doesn't provide GPU while evaluating**
# * **We'll deal with the conversion of GPU model to CPU model**

# ## **Step 1**  
# **Inside Udacity Workspace, Add a Code Cell Next to the Example Cell and write the following code**

# In[ ]:


get_ipython().system('wget "YOUR_COPIED_LINK_TO_MODEL_STATE_CHECKPOINT_FROM_PART_1"')
# You will get model.pt as the download result


# ## **Step 2**  
# **Import the Model and add necessary components**

# **ADD this Code Snippets in the "TEST" Cell located in a BLUE Outlined Cell** (REMOVE ALL PREVIOUS CODE SNIPPETS PRESENT INSIDE THE ' TEST ' CELL)

# In[ ]:


import torch
import torch.nn as nn
from torchvision import models

#Load your model to this variable
model = models.vgg16(pretrained = True) ## Change this if you don't use VGG16

#Add Last Linear Layer, n_inputs -> 102 flower classses
n_inputs = model.classifier[6].in_features ## ResNet and Inception Code may differ slightly, refer Part 1
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, 102)
model.classifier[6] = last_layer


#Add Loss Function (Categorical Cross Entropy)
criterion = nn.CrossEntropyLoss()

#Specify Optimizer: SGD with LR 0.01
import torch.optim as optim
optimizer = optim.SGD(model.classifier.parameters(), lr = 0.01) ## ResNet, Inception may differ Slightly
# USE model.parameters() in ResNet or Inception

#Load Model State Dictionary from Downloaded model.pt file
Model_State_Path = '/home/workspace/model.pt'
# Convert GPU Model State Dictionary to CPU based
model.load_state_dict(torch.load(Model_State_Path, map_location=lambda storage, loc: storage),strict=False)
model.eval() #Don't Forget to add this Line

#### DEFAULT THINGS FOR UDACITY WORKSPACE #######
# If you used something other than 224x224 cropped images, set the correct size here
image_size = 224 #For Inception v3 it is 299
# Values you used for normalizing the images. Default here are for 
# pretrained models from torchvision.
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


# **Don't ADD any other code cell below the above TEST  cell otherwise you may face some error while submitting **  
# 
# ### **Now Go On and Hit That Blue TEST Button**  
# ### ***ALL THE BEST !***

# ## **If you prefer to import the whole model file (NOT Recommended)**

# In[ ]:


get_ipython().system('wget "LINK_TO_CLASSIFIER_PICKLE_PATH"')
# You will get the whole model path in .pt or .pth format


# In[ ]:


#Load your model to this variable
model = torch.load("PATH_TO_MODEL_SAVE_FILE", map_location=lambda storage, loc: storage)
#Add Loss Function (Categorical Cross Entropy)
criterion = nn.CrossEntropyLoss()

#Specify Optimizer: SGD with LR 0.01
import torch.optim as optim
optimizer = optim.SGD(model.classifier.parameters(), lr = 0.01) ## ResNet, Inception may differ Slightly
# USE model.parameters() in ResNet or Inception

model.eval() #Don't Forget to add this Line

#### DEFAULT THINGS FOR UDACITY WORKSPACE #######
# If you used something other than 224x224 cropped images, set the correct size here
image_size = 224 #For Inception v3 it is 299
# Values you used for normalizing the images. Default here are for 
# pretrained models from torchvision.
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


# **This method is not at all recommended and also I have not tried it yet. PyTorch also don't recommend this method. So, you better go with FIRST approach**

# **References:** https://pytorch.org/tutorials/beginner/saving_loading_models.html

# # Thank You  
# 
# If you liked this kernel please **Upvote**. Don't forget to drop a comment or suggestion.  
# 
# ### *Soumya Ranjan Behera*
# Let's stay Connected! [LinkedIn](https://www.linkedin.com/in/soumya044)  
# 
# **Happy Coding !**
