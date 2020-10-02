#!/usr/bin/env python
# coding: utf-8

# # Mixup training data augmentation, Test time Augmentation

# ## Brief Background

# The goal of this challenge is to to decet Kannada digits. Kannada is a language spoken predominantly by people of Karnataka in southwestern India. We are given a .csv file containing pixel data and labels. After transforming the data, we train a network to label the numbers from omdu (1) to hattu (10).

# ## Let's Mix it up!

# This notebook implements two types of data augmentation: Mix Up Augmentation for training and Test Time Augmentation (TTA) for testing. 
# 
# With [mixup](https://arxiv.org/pdf/1710.09412.pdf) augmentation, two images are combined during testing. For example, imagine a toy example with the digits "1" and "2", Instead of training the model with an image containing the digit "1," we give the model an image containing a linear combination of digits "1" and "2". As you might expect the outcome variable needs to be changed, too. In contrast to a target of [1,0] for the digit "1", we would instead have a target of [.3,.7] for an image that is 30% digit "1" and 70% digit "2". The model is given a harder tasks, and is forced to generalize. Sounds tricky, but fastai lets us implement this augmentation easily.
# 
# In [Test Time Augmentation](https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification/), we use the same data augmenatation commonly used for training the model (like rotating, zooming, etc.) when obtaining the test set predictions. This gives us multiple potential digits per image, and we take the most commonly predicted digit as the outcome.
# 
# This kernel proceeds as follows:
# 
# 1. Loading the data into a DataBunch
# 1. Train the model using Mixup Augmentation
# 1. Get model predictions using Test time Augmentation

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.callbacks import *

DATAPATH = Path('/kaggle/input/Kannada-MNIST/')

import os
for dirname, _, filenames in os.walk(DATAPATH):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train = pd.read_csv(DATAPATH/'train.csv')
df_train['fn'] = df_train.index


# # Making a DataBunch

# In this notebook, I'm going to be using the Fastai APIs. Fastai uses the `DataBunch` as its container for the training and validation data. For this project we need to create a custom `ImageList` to handle our data because it is stored in a csv file. This custom ImageList for pixels was inspired by this very [helpful kernel](https://www.kaggle.com/heye0507/fastai-1-0-with-customized-itemlist). To open an image, the function is passed a 'filename' (fn), which is a string containing the index number with '../' in front. This code finds the correct row, and selects the needed colums of that row. The resulting array is reshaped into a 28x28 matrix, and returned as an image.

# In[ ]:


class PixelImageItemList(ImageList):
    def open(self,fn):
        img_pixel = self.inner_df.loc[self.inner_df['fn'] == int(fn[2:])].values[0,1:785]
        img_pixel = img_pixel.reshape(28,28)
        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))


# My data source is created from the `PixelImageList`. In this code, I randomly obtain a validation set, and label my training data from the `DataFrame`'

# In[ ]:


src = (PixelImageItemList.from_df(df_train,'.',cols='fn')
      .split_by_rand_pct()
      .label_from_df(cols='label'))


# Since I want to augment only my testing data, I first get the regular fastai transforms. Fastai transformations are a tuple with two values: the trainig transforms and the testing transforms. Copying the 'training' transformations to the 'test' transformations to allow for test time augmentation. 

# In[ ]:


tfms=get_transforms(do_flip=False)
with_tta = ([],tfms[0])


# The `DataBunch` holds all of my training and validation data, along with the transformations I want to use. 

# In[ ]:


bs = 1024
data = (src.transform(tfms=with_tta)
       .databunch(num_workers=2,bs=bs)
       .normalize())


# Let's take a peek at our data and make sure the digits look as we expect.

# In[ ]:


data.show_batch(rows=3,figsize=(4,4),cmap='bone')


# # Training a model 

# Before we can train a model, we need to define which ones we are using. I used a custom model, instead of a Resnet or Densenet, because of the size of the images. Typically, pretrained models are designed for images of size 224x224. Here, our images are 28x28. Larger models have more of a chance of overfitting. Below is an implementation of the 'best' original MNIST architecture found [here](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist). The tutorial was created using Keras, and here I've re-implemented it using a combination of Pytorch and Fastai. The Fastai `conv_layer` function returns a sequence of convolutional, ReLU and batchnorm layers. 

# In[ ]:


best_architecture = nn.Sequential(
    conv_layer(1,32,stride=1,ks=3),
    conv_layer(32,32,stride=1,ks=3),
    conv_layer(32,32,stride=2,ks=5),
    nn.Dropout(0.4),
    
    conv_layer(32,64,stride=1,ks=3),
    conv_layer(64,64,stride=1,ks=3),
    conv_layer(64,64,stride=2,ks=5),
    nn.Dropout(0.4),
    
    Flatten(),
    nn.Linear(3136, 128),
    relu(inplace=True),
    nn.BatchNorm1d(128),
    nn.Dropout(0.4),
    nn.Linear(128,10),
    nn.Softmax(-1)
)


# This 'learner' in Fastai holds the data, model, loss function, and metric of interest. It's at this point that we specify the mixup augmentation by adding `.mixup` when defining the learner.

# In[ ]:


learn = Learner(data, best_architecture, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy]).mixup()


# Implement the callbacks that save the best model, and implement early stopping. Honestly, I haven't played around with these very much, so I can't say if the training process is that much more efficient by using these. Any links to good articles with a way to systematically evaluate this would be appreciated.

# In[ ]:


callbacks = [
SaveModelCallback(learn, monitor='valid_loss', mode='min',name='bestweights'),
ShowGraph(learn),
EarlyStoppingCallback(learn, min_delta=1e-5, patience=3),
]
learn.callbacks = callbacks


# Visualize the loss at different learning rates.  

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# The model is trained using a [one cycle policy](https://docs.fast.ai/callbacks.one_cycle.html), which is easily implemented in Fastai. I set this model to train for an arbitrary 50 epochs, but may be stopped earlier.

# In[ ]:


learn.fit_one_cycle(50,1e-2)


# # Test Time Augmentation

# Now, we can get the predictions for the test set. But, since we are doing test time augmentation, we'll run the digits through an arbitrary 15 times, and take the 'winner' of all the runs.  
# 
# Load the data like the training set.

# In[ ]:


df_test = pd.read_csv(DATAPATH/'test.csv')
df_test.rename(columns={'id':'label'}, inplace=True)
df_test['fn'] = df_test.index
test_set = PixelImageItemList.from_df(df_test,path='.',cols='fn')


# Add the test set to the DataBunch.

# In[ ]:


learn.data.add_test(test_set)


# Get the predictions for the first run of the test set. After we get the predictions, 'unsqueeze' the tensor to give it an extra dimension. The extra dimension is needed because we will stack the predictions on top of one another.

# In[ ]:


preds, _ = learn.get_preds(DatasetType.Test)
preds.unsqueeze_(1);preds.shape


# Get 14 more predictions, for a total of 15 (potentially) different predictions on the test set. Unsqueeze, and stack on top of each other.

# In[ ]:


num_preds = 14
for x in range(num_preds):
    new_preds, _ = learn.get_preds(DatasetType.Test)
    new_preds.unsqueeze_(1);preds.shape
    preds = torch.cat((preds,new_preds),1)


# Here, the resulting predictions are DIGITS X NUMBER_PREDS X ACTUAL_PREDS. In other words, we have 5,000 digits in the test set, 15 different predictions (each using test time augmentation) from the model, and 10 outputs from the model.

# In[ ]:


preds.shape


# Get the most likely prediction from each of the 15 prediction runs. Now we have 500 digits, along with the 15 specific predictions.

# In[ ]:


indv_preds = torch.argmax(preds, dim=2);indv_preds.shape


# Now, we get the mode (most frequent value) of the predictions. This gives us our overall digit predictions. 

# In[ ]:


winner = torch.mode(indv_preds, dim=1).values;winner.shape


# Then, all that's left is to write the results to a .csv file to submit.

# In[ ]:


submission = pd.DataFrame({ 'id': np.arange(0,len(winner)),'label': winner })


# In[ ]:


submission.to_csv(path_or_buf ="submission.csv", index=False)


# fin.
