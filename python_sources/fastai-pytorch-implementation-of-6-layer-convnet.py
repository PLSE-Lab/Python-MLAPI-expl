#!/usr/bin/env python
# coding: utf-8

# # Fastai/Pytorch - Implementation of 6 Layer Convnet

# This notebook contains an architecture similar to [Anshuman Narayan's](https://www.kaggle.com/anshumandec94/6-layer-conv-nn-using-adam) and [Jinbao's](https://www.kaggle.com/jinbao/kannada-mnist-baseline) notebooks, but implemented in Fastai/Pytorch (original experiments in Keras). 

# In[ ]:


from fastai import *
from fastai.vision import *

DATAPATH = Path('/kaggle/input/Kannada-MNIST/')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Data processing

# The data is transformed from a .csv format. The first column contains the image label/id, and the rest of the columns contain the image pixel values in grayscale (0...225). They are processed into two numpy arrays: one containg the data in the following shape: num_images x num_channels X image_height X image_witdth, and the other containing the label/id in the following shape: num_images. The fucntion uses the following process: 
# - read the csv into a pandas dataframe
# - extract the label/id
# - remove the label/id column, change to a numpy float array
# - divide pixel values by 255 and reshape the image data into a 28x28 square
# - give the resulting array an extra dimension to indicate the images are in grayscale

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


# Display a labelled image:

# In[ ]:


plt.title(f'Training Label: {train_labels[4]}')
plt.imshow(train_data[4,0],cmap='gray');


# # Creating a Fastai Databunch

# Before creating a `Databunch`, I need to create a validation set from my training data.

# In[ ]:


np.random.seed(42)
ran_10_pct_idx = (np.random.random_sample(train_labels.shape)) < .1

train_90_labels = train_labels[np.invert(ran_10_pct_idx)]
train_90_data = train_data[np.invert(ran_10_pct_idx)]

valid_10_labels = train_labels[ran_10_pct_idx]
valid_10_data = train_data[ran_10_pct_idx]


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


# # Model Architecture

# Here I've created an architecture based on [Anshuman Narayan's](https://www.kaggle.com/anshumandec94/6-layer-conv-nn-using-adam) and [Jinbao's](https://www.kaggle.com/jinbao/kannada-mnist-baseline) notebooks. Changes that I made to those models:
# - used Fastai/Pytorch instead of Keras
# - used a different order to the components of a convolutional layer
#     - they used conv, batchnorm, relu
#     - I used conv, relu, batchnorm
# - used leaky relus
# - used AdaptiveConcatPooling instead of MaxPooling, which made my filters grow in this layer rather than a convolution

# In[ ]:


leak = 0.25
conv_drop = 0.35
lin_drop = 0.08

six_conv_architecture = nn.Sequential(
    
    conv_layer(1,16,stride=1,ks=3,leaky=leak),
    nn.Dropout(conv_drop),
    
    conv_layer(16,32,stride=1,ks=3,leaky=leak),
    nn.Dropout(conv_drop),
    AdaptiveConcatPool2d(14), 
    # return twice the number of filters 
    
    conv_layer(64,64,stride=1,ks=5,leaky=leak),
    nn.Dropout(conv_drop),
    AdaptiveConcatPool2d(sz=7),
    # return twice the number of filters 
    
    conv_layer(128,128,stride=1,ks=5,leaky=leak),
    nn.Dropout(conv_drop),
    
    conv_layer(128,64,stride=1,ks=3,leaky=leak),
    nn.Dropout(conv_drop),
    
    conv_layer(64,32,stride=1,ks=3,leaky=leak),
    nn.Dropout(conv_drop),
    
    Flatten(),
    nn.Linear(1568, 50),
    relu(inplace=True,leaky=leak),
    nn.Dropout(lin_drop),
    nn.Linear(50,25),
    relu(inplace=True,leaky=leak),
    nn.Dropout(lin_drop),
    nn.Linear(25,10)
)


# This 'learner' in Fastai holds the data, model, loss function, and metric of interest.

# In[ ]:


learn = Learner(databunch, six_conv_architecture, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy] )


# The model is trained using a [one cycle policy](https://docs.fast.ai/callbacks.one_cycle.html), for an arbitrary 15 epochs. 

# In[ ]:


learn.fit_one_cycle(15)


# # Test data set

# Now, we can get the predictions for the test set. 

# In[ ]:


preds, ids = learn.get_preds(DatasetType.Test)
y = torch.argmax(preds, dim=1)


# In[ ]:


submission = pd.DataFrame({ 'id': ids,'label': y })


# In[ ]:


submission.to_csv(path_or_buf ="submission4.csv", index=False)


# In[ ]:




