#!/usr/bin/env python
# coding: utf-8

# # Fastai/Pytorch - Implementing 'Best' Original MNIST Architecture

# This notebook contains an implementation of the 'best' original MNIST architecture found in [this awesome notebook](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist) (but no data augmentation). Here, I've implemented the 'best' bare bones model with Fastai/Pytorch (original experiments in Keras). 
# 
# This is a version 3 notebook. Changes from version 1 include normalizing the image pixel values from 0..225 to 0..1, and running the model for 20 epochs rather than 15 epochs, and adding some leaky-ness to the RELUs. Some text documentation cleaned up for more clarity.

# In[ ]:


from fastai import *
from fastai.vision import *
DATAPATH = Path('/kaggle/input/Kannada-MNIST/')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # The Goal

# The goal of this challenge is to to decet Kannada digits. Kannada is a language spoken predominantly by people of Karnataka in southwestern India. We are given a .csv file containing pixel data and labels. After transforming the data, we train a network to label the numbers from _omdu_ (1) to _hattu_ (10).

# # Data processing

# The data is transformed from a .csv format into two numpy arrays. In the .csv, the first column contains the image label/id, and the rest of the columns contain the image pixel values in grayscale (0...225). The first numpy array contains the image data in the following shape: num_images x num_channels X image_height X image_witdth, and the other contains the label/id in the following shape: num_images. This is accomplised by: 
# - reading the csv into a pandas dataframe
# - extracting the label/id
# - removing the label/id column, changing to a numpy float array
# - dividing pixel values by 255 and reshaping the image data into a 28x28 square
# - giving the resulting array an extra dimension to indicate the images are in grayscale

# In[ ]:


def get_images_and_labels(csv,label):
    fileraw = pd.read_csv(csv)
    labels = fileraw[label].to_numpy()
    data = fileraw.drop([label],axis=1).to_numpy(dtype=np.float32)
    data = np.true_divide(data,255.).reshape((fileraw.shape[0],28,28))
    data = np.expand_dims(data, axis=1)
    return data, labels


# Process the training, testing and 'other' datasets, and then check to ensure the arrays look reasonable.

# In[ ]:


train_data, train_labels = get_images_and_labels(DATAPATH/'train.csv','label')
test_data, test_labels = get_images_and_labels(DATAPATH/'test.csv','id')
other_data, other_labels = get_images_and_labels(DATAPATH/'Dig-MNIST.csv','label')

print(f' Train:\tdata shape {train_data.shape}\tlabel shape {train_labels.shape}\n Test:\tdata shape {test_data.shape}\tlabel shape {test_labels.shape}\n Other:\tdata shape {other_data.shape}\tlabel shape {other_labels.shape}')

The resulting data arrays look reasonable, and the size of the labels is the same. Let's display a labelled image:
# In[ ]:


plt.title(f'Training Label: {train_labels[4]}')
plt.imshow(train_data[4,0],cmap='gray');


# Before moving forward, I need to create a validation set from the training data. Here, I create an array with random number, and assign 'True' if the number is less then 0.1, which then allows me to separate my training and validation set.

# In[ ]:


np.random.seed(42)
ran_10_pct_idx = (np.random.random_sample(train_labels.shape)) < .1

train_90_labels = train_labels[np.invert(ran_10_pct_idx)]
train_90_data = train_data[np.invert(ran_10_pct_idx)]

valid_10_labels = train_labels[ran_10_pct_idx]
valid_10_data = train_data[ran_10_pct_idx]


# # Creating a Fastai Databunch

# Because Fastai does not have an API for directly adding numpy arrays into a databunch (_as far as I know, please leave a comment if you know a way!_), I created a bare-bones Torch Dataset class [based on this example](https://docs.fast.ai/basic_data.html) to allow me to create a `DataBunch`.

# In[ ]:


class ArrayDataset(Dataset):
    "Dataset for numpy arrays based on fastai example: "
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.c = len(np.unique(y))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]


# In[ ]:


train_ds = ArrayDataset(train_90_data,train_90_labels)
valid_ds = ArrayDataset(valid_10_data,valid_10_labels)
other_ds = ArrayDataset(other_data, other_labels)
test_ds = ArrayDataset(test_data, test_labels)


# Finally, I can create a Databunch, which contains, my training, validation and test sets, along with the batch size. I do not use the 'other' set, but it can be used for further model selection.

# In[ ]:


bs = 128
databunch = DataBunch.create(train_ds, valid_ds, test_ds=test_ds, bs=bs)


# # Model Architecture: use the 'best' 

# Below is an implementation of the 'best' original MNIST architecture found [here](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist). The tutorial was created using Keras, and here I've re-implemented it using a combination of Pytorch and Fastai. The Fastai `conv_layer` function returns a sequence of convolutional, ReLU and batchnorm layers. 

# In[ ]:


leak = 0.15

best_architecture = nn.Sequential(
    
    conv_layer(1,32,stride=1,ks=3,leaky=leak),
    conv_layer(32,32,stride=1,ks=3,leaky=leak),
    conv_layer(32,32,stride=2,ks=5,leaky=leak),
    nn.Dropout(0.4),
    
    conv_layer(32,64,stride=1,ks=3,leaky=leak),
    conv_layer(64,64,stride=1,ks=3,leaky=leak),
    conv_layer(64,64,stride=2,ks=5,leaky=leak),
    nn.Dropout(0.4),
    
    Flatten(),
    nn.Linear(3136, 128),
    relu(inplace=True),
    nn.BatchNorm1d(128),
    nn.Dropout(0.4),
    nn.Linear(128,10)
)


# This 'learner' in Fastai holds the data, model, loss function, and metric of interest. Note that a lot of the Fastai examples you will see use `cnn_learner`. I don't use that here because my model is not pretrained. 

# In[ ]:


learn = Learner(databunch, best_architecture, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy] )


# The model is trained using a [one cycle policy](https://docs.fast.ai/callbacks.one_cycle.html), which from what I understand is not implemented in [this notebook](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist). For best their best results, they train for 30 epochs. Here, I'm running for 20 epochs.

# In[ ]:


learn.fit_one_cycle(20)


# # Predictions for the Test data set

# Now, we can get the predictions for the test set. 

# In[ ]:


preds, ids = learn.get_preds(DatasetType.Test)
y = torch.argmax(preds, dim=1)


# In[ ]:


submission = pd.DataFrame({ 'id': ids,'label': y })
submission.to_csv(path_or_buf ="submission.csv", index=False)

