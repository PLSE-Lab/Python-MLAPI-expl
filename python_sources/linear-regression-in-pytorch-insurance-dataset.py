#!/usr/bin/env python
# coding: utf-8

# # Insurance cost prediction using linear regression
# 
# In this assignment we're going to use information like a person's age, sex, BMI, no. of children and smoking habit to predict the price of yearly medical bills. This kind of model is useful for insurance companies to determine the yearly insurance premium for a person. The dataset for this problem is taken from: https://www.kaggle.com/mirichoi0218/insurance
# 
# We will create a model with the following steps:
# 1. Download and explore the dataset
# 2. Prepare the dataset for training
# 3. Create a linear regression model
# 4. Train the model to fit the data
# 5. Make predictions using the trained model
# 
# Try to experiment with the hypeparameters to get the lowest loss.
# 

# In[ ]:


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


dataframe_raw = pd.read_csv(DATA_FILENAME)
dataframe_raw.head()


# We're going to do a slight customization of the data, so that you every participant receives a slightly different version of the dataset. We will fill a name below as a string (at least 5 characters)

# In[ ]:


your_name = "Vighnesh" # at least 5 characters


# The `customize_dataset` function will customize the dataset slightly using your name as a source of random numbers.

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


# Let us answer some basic questions about the dataset. 
# 
# 
# **Q: How many rows does the dataset have?**

# In[ ]:


num_rows = len(dataframe)
print(num_rows)


# **Q: How many columns doe the dataset have**

# In[ ]:


num_cols = len(dataframe.columns)
print(num_cols)


# **Q: What are the column titles of the input variables?**

# In[ ]:


input_cols = list(dataframe.drop('charges',axis=1).columns)
input_cols


# **Q: Which of the input columns are non-numeric or categorial variables ?**
# 
# Hint: `sex` is one of them. List the columns that are not numbers.

# In[ ]:


categorical_cols = list(dataframe.select_dtypes(include='object').columns)
categorical_cols


# **Q: What are the column titles of output/target variable(s)?**

# In[ ]:


output_cols = [dataframe.columns[-1]]
output_cols


# **Q: (Optional) What is the minimum, maximum and average value of the `charges` column? Can you show the distribution of values in a graph?**

# In[ ]:


# Write your answer here
import numpy as np
# min_charge = np.min(dataframe.charges)
min_charge = dataframe.charges.min()
print("Minimum charge = ",min_charge)
# max_charge = np.max(dataframe.charges)
max_charge = dataframe.charges.max()
print("Maximum charge = ",max_charge)
# avg_charge = np.mean(dataframe.charges)
avg_charge = dataframe.charges.mean()
print("Average charge = ",avg_charge)


# In[ ]:


# Plotting the distribution of 'charges' column
import seaborn as sns
fig, axs = plt.subplots(ncols=2)
sns.set_style("darkgrid")
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (9, 5)
#plt.title("Distribution of charges")
sns.distplot(dataframe.charges, ax=axs[0]) # Skewed data
sns.distplot(np.log(dataframe.charges),ax=axs[1]) # Trying to make data normal using log transformation


# ## Step 2: Prepare the dataset for training
# 
# We need to convert the data from the Pandas dataframe into a PyTorch tensors for training. To do this, the first step is to convert it numpy arrays. If you've filled out `input_cols`, `categorial_cols` and `output_cols` correctly, this following function will perform the conversion to numpy arrays.

# In[ ]:


def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    #inputs_array = np.array(dataframe1[input_cols])
    inputs_array = dataframe1.drop('charges',axis=1).values
    #targets_array = np.array(dataframe1[output_cols])
    targets_array = dataframe1[['charges']].values
    return inputs_array, targets_array


# Read through the [Pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html) to understand how we're converting categorical variables into numbers.

# In[ ]:


inputs_array, targets_array = dataframe_to_arrays(dataframe)
print(inputs_array.shape, targets_array.shape)
inputs_array, targets_array


# **Q: Convert the numpy arrays `inputs_array` and `targets_array` into PyTorch tensors. Make sure that the data type is `torch.float32`.**

# In[ ]:


inputs = torch.from_numpy(inputs_array).to(torch.float32)
targets = torch.from_numpy(targets_array).to(torch.float32)


# In[ ]:


inputs.dtype, targets.dtype


# In[ ]:


print(inputs,targets)


# Next, we need to create PyTorch datasets & data loaders for training & validation. We'll start by creating a `TensorDataset`.

# In[ ]:


dataset = TensorDataset(inputs, targets)


# ***Q: Pick a number between `0.1` and `0.2` to determine the fraction of data that will be used for creating the validation set. Then use `random_split` to create training & validation datasets. ***

# In[ ]:


val_percent = 0.1 # between 0.1 and 0.2
val_size = int(num_rows * val_percent)
print(val_size)
train_size = num_rows - val_size
print(train_size)

train_ds, val_ds = random_split(dataset,[train_size, val_size]) # Use the random_split function to split dataset into 2 parts of the desired length


# In[ ]:


print(len(train_ds))
print(len(val_ds))


# Finally, we can create data loaders for training & validation.
# 
# **Q: Pick a batch size for the data loader.**

# In[ ]:


batch_size = 64 # Try to experiment with different batch sizes


# In[ ]:


train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)


# Let's look at a batch of data to verify everything is working fine so far.

# In[ ]:


for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break


# ## Step 3: Create a Linear Regression Model
# 
# Our model itself is a fairly straightforward linear regression.

# In[ ]:


input_size = len(input_cols)
print(input_size)
output_size = len(output_cols)
print(output_size)


# In[ ]:


class InsuranceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size) 
        
    def forward(self, xb):
        out = self.linear(xb)                          
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out, targets)                
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out, targets)                    
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 500 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))


# Let us create a model using the `InsuranceModel` class. You may need to come back later and re-run the next cell to reinitialize the model, in case the loss becomes `nan` or `infinity`.

# In[ ]:


model = InsuranceModel()


# Let's check out the weights and biases of the model using `model.parameters`.

# In[ ]:


list(model.parameters())


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


result = evaluate(model, val_loader) # Use the the evaluate function
print(result)


# 
# We are now ready to train the model. You may need to run the training loop many times, for different number of epochs and with different learning rates, to get a good result. Also, if your loss becomes too large (or `nan`), you may have to re-initialize the model by running the cell `model = InsuranceModel()`. Experiment with this for a while, and try to get to as low a loss as possible.

# In[ ]:


# model = InsuranceModel() # In case of re-initialization


# **Q: Train the model 4-5 times with different learning rates & for different number of epochs.**
# 
# Hint: Vary learning rates by orders of 10 (e.g. `1e-2`, `1e-3`, `1e-4`, `1e-5`, `1e-6`) to figure out what works.

# In[ ]:


epochs = 1000
lr = 0.001
history1 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


epochs = 1500
lr = 0.05
history2 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


epochs = 2000
lr = 0.1
history3 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


epochs = 2500
lr = 0.4
history4 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


epochs = 3000
lr = 0.8
history5 = fit(epochs, lr, model, train_loader, val_loader)


# **Q: What is the final validation loss of your model?**

# In[ ]:


val_loss = history5[-1]
val_loss


# Now scroll back up, re-initialize the model, and try different set of values for batch size, number of epochs, learning rate etc.

# ## Step 5: Make predictions using the trained model

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


input, target = val_ds[13]
predict_single(input, target, model)


# In[ ]:


input, target = val_ds[54]
predict_single(input, target, model)


# In[ ]:


input, target = val_ds[87]
predict_single(input, target, model)


# Are you happy with your model's predictions? Try to improve them further.
