#!/usr/bin/env python
# coding: utf-8

# # Install the fastai2 library and EfficientNet

# In[ ]:


get_ipython().system(' pip install fastai2 -q')
get_ipython().system(' pip install efficientnet-pytorch -q')


# In[ ]:


# Importing the libraies
from fastai2.vision.all import *
from efficientnet_pytorch import EfficientNet


# # Create the labels dictionary using the txt file

# In[ ]:


# Define the working directory
path = Path('/kaggle/input/10-monkey-species')
path.ls()


# In[ ]:


labels = pd.read_csv(path/'monkey_labels.txt', sep=' *, *') # Set sep=' *, *' to remove trailing whitespace
labels.head()


# Create a dictionary using column **"Label"** and column **"Common Name"** so that we can extract the labels by passing it into the Pipeline of **get_y** later using the DataBlock

# In[ ]:


lbl_dict = dict(zip(labels.iloc[:, 0], labels.iloc[:, 2])); lbl_dict


# # Load the dataset using the DataBlock API

# In[ ]:


# Define some hyperparameters (batch size and transformations)
bs = 64
item_tfms = Resize(300)
batch_tfms = [*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]


# In[ ]:


db = DataBlock(blocks=(ImageBlock, CategoryBlock),
               get_items=get_image_files,
               splitter=GrandparentSplitter(train_name='training', valid_name='validation'),
               get_y=Pipeline([parent_label, lbl_dict.__getitem__]),
               item_tfms=item_tfms,
               batch_tfms=batch_tfms)
db.summary(path) # This method is used to get an overview of the data preparation pipeline


# In[ ]:


dls = db.dataloaders(path, bs=bs)
dls.show_batch(figsize=(20,12)) # This method is used to show some samples of the data


# # Build the model (EfficientNet)

# In[ ]:


# Define some hyperparameters (optimizer, loss function, and metrics)
opt_func = partial(Adam)
loss_func = LabelSmoothingCrossEntropy()
metrics = [accuracy]


# Before using Efficientnet, let's try using **resnet18** as a baseline model

# In[ ]:


# Mixed precision training is used to speed up the process
learn = cnn_learner(dls, resnet18, opt_func=opt_func, loss_func=loss_func, metrics=metrics).to_fp16()
# learn.summary() # This method can be used to get an overview of the entire model architecture


# In[ ]:


# The fine_tune method automatically trains the freezed model first, then unfreezes it and retrain 
learn.fine_tune(4, 1e-3)


# Here I created a baseline model using **EfficientNetB0**. For better performance, we can use higher level model from B1 to B7

# In[ ]:


# Set up the model
model = EfficientNet.from_pretrained("efficientnet-b0")
num_ftrs = model._fc.in_features

# Replace the last fully connected layer with our own layers, I add in an additional Dropout layer for some regularizing effects
model._fc = nn.Sequential(nn.Linear(num_ftrs, 1000),
                              nn.ReLU(),
                              nn.Dropout(),
                              nn.Linear(1000, dls.c))


# In[ ]:


# Mixed precision training is used to speed up the process
learn = Learner(dls, model, opt_func=opt_func, loss_func=loss_func, metrics=metrics).to_fp16()
# learn.summary() # This method can be used to get an overview of the entire model architecture


# In[ ]:


# The fine_tune method automatically trains the freezed model first, then unfreezes it and retrain 
learn.fine_tune(4, 1e-3)


# # Evaluate the model

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


# See the confusion matrix
interp.plot_confusion_matrix(normalize=True, figsize=(6, 6))


# In[ ]:


# See the top misidentifications that our model made
interp.plot_top_losses(figsize=(20,12))


# In[ ]:


# We can also directly grab our most confused, and pass in a threshold
interp.most_confused(min_val=3)


# # That's all. Hope that you find this notebook useful =))
