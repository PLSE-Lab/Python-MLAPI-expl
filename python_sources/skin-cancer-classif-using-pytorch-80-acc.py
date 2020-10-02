#!/usr/bin/env python
# coding: utf-8

# # Classification example using PyTorch

# This tutorial shows you a quick approach to the classification problem using PyTorch. It is kept rather simple by making use of pre-defined and pre-trained models. This allows to put more focus on the data-loading and data-augmentation process. Feel free to ask questions in the comments :)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from torch.utils import data
import torch
from PIL import Image
torch.manual_seed(42)
np.random.seed(42)


# # First of all: Import the data!

# For this we make use of some code from this kernel: https://www.kaggle.com/kmader/dermatology-mnist-loading-and-processing

# In[ ]:


base_skin_dir = os.path.join('..', 'input')

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# In[ ]:


tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes


# In[ ]:


tile_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()


# In the table above we have a nice overview of our groundtruth data. What we care for is only the column 'cell_type_idx', because these values are needed for the model training. But of course it is nice to know what these labels mean, that is what the column 'cell_type' is for. Lets quickly check how often the different tumors occur in our dataset:

# In[ ]:


tile_df['cell_type'].value_counts()


# As it seems the Melanocytic nevi occurs nearly 58 times as often as the Dermatofibroma. It can happen that the Melanocytic nevi is preferred in the prediction compared to the Dermatofibroma. One solution could be to show the less frequent classes more often in the training. However, for the puropose of showing a solution to the classification problem in PyTorch, this would complicate matters and is therefore not further pursued here.
# 
# Let's get a quick glance on the complete table:

# In[ ]:


tile_df.sample(3)


# The table above can be used to get the input data using the path. The corresponding ground truth label is already given in the same line by the column 'cell_type_idx'. Later on we will create an input batch X of several loaded images and a corresponding ground-truth value y given by the corresponding ground-truth labels. But another step is necessary first: Choosing the right model for training.

# # Loading and adjusting pretrained models in pytorch

# PyTorch has a nice feature that it offers well established models. These models can optionally allready be pretrained on the ImageNet dataset, causing the training time to be lower in general. Let us load a pretrained ResNet50 and adjust the last layer a little bit. 

# In[ ]:


import torchvision.models as models
model_conv = models.resnet50(pretrained=True)


# In[ ]:


print(model_conv)


# In[ ]:


print(model_conv.fc)


# We have to adjust the last layer (fc). We can see that the last layer is a linear layer, having 2048 input neurons and having 1000 output neurons. This is useful if you have 1000 different classes. However, we only have to deal with 7 different classes - the 7 different tumortypes - therefore we have to change the last layer:

# In[ ]:


num_ftrs = model_conv.fc.in_features
model_conv.fc = torch.nn.Linear(num_ftrs, 7)


# In[ ]:


print(model_conv.fc)


# Successfully adjusted. Let us put the model on the GPU, because that is the place where we want to train the model in the end:

# In[ ]:


# Define the device:
device = torch.device('cuda:0')

# Put the model on the device:
model = model_conv.to(device)


# # Back to the data

# ## Split the data into a train and a validation set

# In[ ]:


from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(tile_df, test_size=0.1)


# In[ ]:


# We can split the test set again in a validation set and a true test set:
validation_df, test_df = train_test_split(test_df, test_size=0.5)


# In[ ]:


train_df = train_df.reset_index()
validation_df = validation_df.reset_index()
test_df = test_df.reset_index()


# ## Create a Class 'Dataset'

# The dataset class will allow us to easily load and transform batches of data in the background on multiple CPUs. A very good article about how to create a Dataset class and a dataloader for PyTorch can be found here: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

# In[ ]:


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df, transform=None):
        'Initialization'
        self.df = df
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


# In[ ]:


# Define the parameters for the dataloader
params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 6}


# Another nice thing about using the dataset class defined above is that we can easily perform preprocessing of the data and/or data augmentation. In this example we only perform mirroring (RandomHorizontalFlip, RandomVerticalFlip), Crop the image to the image center, where the melanom is most often located (CenterCrop), randomly crop from the center of the image (RandomCrop) and normalize the image according to what the pretrained model needs (Normalize). We then transform the image to a tensor using, which is required to use it for learning with PyTorch, with the function ToTensor:

# In[ ]:


# define the transformation of the images.
import torchvision.transforms as trf
composed = trf.Compose([trf.RandomHorizontalFlip(), trf.RandomVerticalFlip(), trf.CenterCrop(256), trf.RandomCrop(224),  trf.ToTensor(),
                        trf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# In[ ]:


# Define the trainingsset using the table train_df and using our defined transitions (composed)
training_set = Dataset(train_df, transform=composed)
training_generator = data.DataLoader(training_set, **params)

# Same for the validation set:
validation_set = Dataset(validation_df, transform=composed)
validation_generator = data.DataLoader(validation_set, **params)


# Now we have to define the optimizer we want to use. In our case this will be an Adam optimizer with a learning rate of $1e-6$. The criterion or the loss function that we will use is the CrossEntropyLoss. This is a typical chosen loss function for multiclass classification problems.

# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
criterion = torch.nn.CrossEntropyLoss()


# We now have a dataloader for the data in the training set, a dataloader for the data in the validation set and we have the optimizer and the criterion defined. We can now come to the core business of machine learning: Training and testing the model:

# # Train the model

# In[ ]:


max_epochs = 20
trainings_error = []
validation_error = []
for epoch in range(max_epochs):
    print('epoch:', epoch)
    count_train = 0
    trainings_error_tmp = []
    model.train()
    for data_sample, y in training_generator:
        data_gpu = data_sample.to(device)
        y_gpu = y.to(device)
        output = model(data_gpu)
        err = criterion(output, y_gpu)
        err.backward()
        optimizer.step()
        trainings_error_tmp.append(err.item())
        count_train += 1
        if count_train >= 100:
            count_train = 0
            mean_trainings_error = np.mean(trainings_error_tmp)
            trainings_error.append(mean_trainings_error)
            print('trainings error:', mean_trainings_error)
            break
    with torch.set_grad_enabled(False):
        validation_error_tmp = []
        count_val = 0
        model.eval()
        for data_sample, y in validation_generator:
            data_gpu = data_sample.to(device)
            y_gpu = y.to(device)
            output = model(data_gpu)
            err = criterion(output, y_gpu)
            validation_error_tmp.append(err.item())
            count_val += 1
            if count_val >= 10:
                count_val = 0
                mean_val_error = np.mean(validation_error_tmp)
                validation_error.append(mean_val_error)
                print('validation error:', mean_val_error)
                break
            


# In[ ]:


plt.plot(trainings_error, label = 'training error')
plt.plot(validation_error, label = 'validation error')
plt.legend()
plt.show()


# # Test the actual classification ability of the model:

# In[ ]:


model.eval()
test_set = Dataset(validation_df, transform=composed)
test_generator = data.SequentialSampler(validation_set)


# In[ ]:


result_array = []
gt_array = []
for i in test_generator:
    data_sample, y = validation_set.__getitem__(i)
    data_gpu = data_sample.unsqueeze(0).to(device)
    output = model(data_gpu)
    result = torch.argmax(output)
    result_array.append(result.item())
    gt_array.append(y.item())
    


# In[ ]:


correct_results = np.array(result_array)==np.array(gt_array)


# In[ ]:


sum_correct = np.sum(correct_results)


# In[ ]:


accuracy = sum_correct/test_generator.__len__()


# In[ ]:


print(accuracy)


# So we have a nice and high accuracy. I hope that you enjoyed reading this short example and that you got to know a little bit more about PyTorch. It is very convenient to be able to use pretrained models and not having to define a model on your own. Feel free to forge the notebook or leave comments :)
