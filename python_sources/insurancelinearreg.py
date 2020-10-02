#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#It is a good practice to import all the modules that may be needed for the project.
import torch
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split


# ## Step 1: Download and explore the data
# 
# Let us begin by downloading the data. We'll use the `download_url` function from PyTorch to get the data as a CSV (comma-separated values) file. 

# In[ ]:


DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
DATA_FILENAME = "insurance.csv"
download_url(DATASET_URL, '.')


# To load the dataset into memory, we'll use the `read_csv` function from the `pandas` library. The data will be loaded as a Pandas dataframe. See this short tutorial to learn more: https://data36.com/pandas-tutorial-1-basics-reading-data-files-dataframes-data-selection/

# In[ ]:


dataframe_raw= pd.read_csv(DATA_FILENAME) #To read csv from a csv file using pandas dataframe where dataframe is like rows by columns.
dataframe_raw.head(5) #used to see first 5 data in the dataframe


# In[ ]:


your_name = "karthikayan" # at least 5 characters


# In[ ]:


def customize_dataset(dataframe_raw, rand_str):
    dataframe = dataframe_raw.copy(deep=True)
    # drop some rows
    dataframe = dataframe.sample(int(0.95*len(dataframe)), random_state=int(ord(rand_str[0])))
    # scale input
    dataframe.bmi = dataframe.bmi * ord(rand_str[1])/100.
    # scale target
    dataframe.charges = dataframe.charges * ord(rand_str[2])/100.
    # drop column
    if ord(rand_str[3]) % 2 == 1:
        dataframe = dataframe.drop(['region'], axis=1)
    return dataframe


# In[ ]:


dataframe = customize_dataset(dataframe_raw, your_name)
dataframe.head()


# In[ ]:


num_rows = len(dataframe)
print(num_rows)


# In[ ]:


num_cols = sum(1 for i in dataframe.columns)
print(num_cols)


# In[ ]:


input_cols = [i for i in dataframe.columns if(i!='charges')] #to get the input column titles
print(input_cols)


# In[ ]:


#to get the columns that contain string values instead of float values(will explain why we are doing this later)
categorical_cols = [i for i in input_cols if(type(dataframe[i][0]) is str)] 
print(categorical_cols)


# In[ ]:


output_cols = ['charges'] #the output column titles are obtained here.


# In[ ]:


mini=min(dataframe['charges'])
maxi=max(dataframe['charges'])
mean=sum(dataframe['charges'])/(len(dataframe['charges']))
print(mini,maxi,mean)
plt.plot([i for i in range(num_rows)],dataframe['charges'],'x-r')
plt.title("Output Distribution")
plt.xlabel('people')
plt.ylabel('charges')
plt.show()


# ## Step 2: Prepare the dataset for training
# 
# We need to convert the data from the Pandas dataframe into a PyTorch tensors for training. To do this, the first step is to convert it numpy arrays. If you've filled out `input_cols`, `categorial_cols` and `output_cols` correctly, this following function will perform the conversion to numpy arrays.

# In[ ]:


import numpy as np
def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe in order to make changes to the dataframe according to our needs
    dataframe1 = dataframe.copy(deep=True) 
    # Convert non-numeric categorical columns to numbers (i.e, string variables to numbers as we can communicate with the machine only with numbers)
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes 
        #used to convert string to numbers(i.e if a column contains 'yes' or 'no' predominantly,it will assign 1-'yes' and 0-'no' likewise)
        #print(dataframe1[col][:10],dataframe[col][:10])
    # Extract input & outupts as numpy arrays in datatype float32 as the tensor expects the data to be of float type.
    inputs_array = dataframe1[input_cols].to_numpy().astype(np.float32)
    targets_array = dataframe1[output_cols].to_numpy().astype(np.float32)
    return inputs_array, targets_array


# In[ ]:


inputs_array, targets_array = dataframe_to_arrays(dataframe)#you can understand better on looking into the outputs
inputs_array, targets_array 


# **Q: Convert the numpy arrays `inputs_array` and `targets_array` into PyTorch tensors. Make sure that the data type is `torch.float32`.**

# In[ ]:


#now the numpy arrays are converted to tensors
inputs = torch.from_numpy(inputs_array)
targets = torch.from_numpy(targets_array)


# In[ ]:


inputs.dtype, targets.dtype


# Next, we need to create PyTorch datasets & data loaders for training & validation. We'll start by creating a `TensorDataset`.

# In[ ]:


dataset = TensorDataset(inputs, targets) #now the input and outputs are combined
dataset[0] #you can see that the first part contains input feature values and the second part contains output values


# In[ ]:


#let's now split the train data into train and validation so that we can see the performance on the untouched data parallely
val_percent = 0.1 # may be between 0.1 and 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size


train_ds, val_ds = random_split(dataset,[train_size,val_size]) # Use the random_split function to split dataset into 2 parts of the desired length


# In[ ]:


batch_size = 128 # it is used to speed up the learning process


# In[ ]:


#we are shuffling the data in order to avoid overfitting as the input data is contiguous in aspect of categories(i.e, initial half of data represents a category and as it goes on other categories)
train_loader = DataLoader(train_ds, batch_size, shuffle=True) 
val_loader = DataLoader(val_ds, batch_size)


# Let's look at a batch of data to verify everything is working fine so far.

# In[ ]:


#let's have a look at the 1st batch 
for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break


# ## Step 3: Create a Linear Regression Model
# 
# Our model itself is a fairly straightforward linear regression . 
# 

# In[ ]:


input_size = len(input_cols)
output_size = len(output_cols)
print(input_size,output_size)


# In[ ]:


class InsuranceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)                  # fill this (hint: use input_size & output_size defined above)
        
    def forward(self, xb):
        out = self.linear(xb)                         # fill this
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out,targets)                          # fill this
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out,targets)                           # fill this    
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))


# Let us create a model using the `InsuranceModel` class. You may need to come back later and re-run the next cell to reinitialize the model, in case the loss becomes `nan` or `infinity`.

# In[ ]:


model = InsuranceModel()


# Let's check out the weights and biases of the model using `model.parameters`.

# In[ ]:


list(model.parameters())


# One final commit before we train the model.

# In[ ]:


jovian.commit(project=project_name, environment=None)


# ## Step 4: Train the model to fit the data
# 
# To train our model, we'll use the same `fit` function explained in the lecture. That's the benefit of defining a generic training loop - you can use it for any problem.

# In[ ]:


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history


# In[ ]:


result = evaluate(model,val_loader) # Use the the evaluate function
print(result)


# 
# We are now ready to train the model. You may need to run the training loop many times, for different number of epochs and with different learning rates, to get a good result. Also, if your loss becomes too large (or `nan`), you may have to re-initialize the model by running the cell `model = InsuranceModel()`. Experiment with this for a while, and try to get to as low a loss as possible.

# In[ ]:


epochs = 100
lr = 1e-4
history1 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


epochs = 100
lr = 1e-5
model = InsuranceModel()
history2 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


epochs = 100
lr = 1e-6
model = InsuranceModel()
history3 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


epochs = 100
lr = 1e-7
model = InsuranceModel()
history4 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


epochs = 40000
lr = 1e-4
model = InsuranceModel()
history5 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


val_loss = history5[-1]['val_loss']
print(val_loss)


# ## Step 5: Make predictions using the trained model
# 

# In[ ]:


def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)     
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)


# In[ ]:


input, target = val_ds[0]
predict_single(input, target, model)


# In[ ]:


input, target = val_ds[10]
predict_single(input, target, model)


# In[ ]:


input, target = val_ds[20]
predict_single(input, target, model)


# In[ ]:




