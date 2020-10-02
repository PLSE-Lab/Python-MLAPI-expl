#!/usr/bin/env python
# coding: utf-8

# # Simple Fastai/Pytorch Example

# Here is my first bare-bones attempt at the MNIST Kannada challenge. This 1st commit notebook contains a fully convolutional network implemented with Fastai/Pytorch. 

# In[ ]:


from fastai import *
from fastai.vision import *
DATAPATH = Path('/kaggle/input/Kannada-MNIST/')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Data processing

# The data given to us is in a .csv format. The first column contains the image label/id, and the rest of the columns contain the image pixel values in grayscale. To process the csv into something we can train a network with, I 
# - read the csv into a pandas dataframe
# - extract the label/id
# - remove the label/id column and reshape the image data into a 28x28 square
# - give the resulting array an extra dimension to indicate the images are in grayscale

# In[ ]:


def get_data_labels(csv,label):
    fileraw = pd.read_csv(csv)
    labels = fileraw[label].to_numpy()
    data = fileraw.drop([label],axis=1).to_numpy(dtype=np.float32).reshape((fileraw.shape[0],28,28))
    data = np.expand_dims(data, axis=1)
    return data, labels


# Each of the datasets is processed to obtain the data and label/id. 

# In[ ]:


train_data, train_labels = get_data_labels(DATAPATH/'train.csv','label')
test_data, test_labels = get_data_labels(DATAPATH/'test.csv','id')
other_data, other_labels = get_data_labels(DATAPATH/'Dig-MNIST.csv','label')
train_data.shape, train_labels.shape, test_data.shape, test_labels.shape, other_data.shape, other_labels.shape


# To ensure the the data read in properly, I display one. Seems ok.

# In[ ]:


plt.title(f'Training Label: {train_labels[2]}')
plt.imshow(train_data[2,0],cmap='gray');


# # Creating a Fastai Databunch

# I want to use Fastai's code for training the model. First, however, I created a validation dataset from 10% of the training data.

# In[ ]:


np.random.seed(42)
ran_10_pct_idx = (np.random.random_sample(train_labels.shape)) < .1

train_90_labels = train_labels[np.invert(ran_10_pct_idx)]
train_90_data = train_data[np.invert(ran_10_pct_idx)]

valid_10_labels = train_labels[ran_10_pct_idx]
valid_10_data = train_data[ran_10_pct_idx]


# The image data is currently in numpy arrays. Fastai does not have a built in dataset to handle images in this format, so I created a bare-bones Torch Dataset class. [Code from here](https://docs.fast.ai/basic_data.html)

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


# Finally, I can create a Databunch, which contains, my training, validation and test sets, along with the batch size.

# In[ ]:


bs = 128
databunch = DataBunch.create(train_ds, valid_ds, test_ds=test_ds, bs=bs)


# # Simple model: fully convolutional network

# The 28x28 images are tiny, so I did not want to use a large pretrained model, like ResNet. This is a fully convolutional network created in a blend of in pytorchand Fastai. The 'conv_layer' is a Fastai function that returns a convolutional layer, batchnorm, and RELU. 

# In[ ]:


def conv2(ni,nf,stride=2,ks=3): return conv_layer(ni,nf,stride=stride,ks=ks)


# In[ ]:


smallConvolutional = nn.Sequential(
    conv2(1,8,ks=5),
    conv2(8,16),
    conv2(16,32),
    conv2(32, 16),
    conv2(16, 10),
    Flatten()
)


# This 'learner' in Fastai holds the data, model, loss function, and metric of interest.

# In[ ]:


learn = Learner(databunch, smallConvolutional, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy] )


# This uses a [one cycle policy](https://docs.fast.ai/callbacks.one_cycle.html) for training. Eight cycles a an arbitrary number; the accuracy on the validation set typically reaches ~99.2%.

# In[ ]:


learn.fit_one_cycle(8)


# # Test data set

# Now, we can get the predictions for the test set. 

# In[ ]:


preds, ids = learn.get_preds(DatasetType.Test)
y = torch.argmax(preds, dim=1)


# In[ ]:


submission = pd.DataFrame({ 'id': ids,'label': y })


# In[ ]:


submission.to_csv(path_or_buf ="submission.csv", index=False)


# In[ ]:




