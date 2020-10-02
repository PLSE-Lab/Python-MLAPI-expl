#!/usr/bin/env python
# coding: utf-8

# # Assignment 2
# # CalCOFI: predict the water temperature based on salinity
# 
# CalCOFI: Over 60 years of oceanographic data: Is there a relationship between water salinity & water temperature? Can you predict the water temperature based on salinity?
# 
# I took the data from the Kaggle website: [https://www.kaggle.com/sohier/calcofi/kernels](https://www.kaggle.com/sohier/calcofi/kernels) and ran the notebook on Kaggle with the CalCOFI data files provided.
# 
# 

# In[ ]:


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Preparing the workspace

# In[ ]:


# check tha the last version of Jovian is installed
get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:



# Imports

import torch
import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split


# In[ ]:


# Other constants
DATASET_URL = "../input/calcofi/bottle.csv"
DATA_FILENAME = "bottle.csv"


# ## Download and explore my dataset

# The data will first be loaded as a Pandas dataframe.

# In[ ]:


dataframe = pd.read_csv(DATASET_URL)
# I take the first 700 data points to examine more in detail
df = dataframe[:][:700]
#df.iloc[0:3,8:25]
df.head()

# I get a warning because of the different datatypes on import. I will clean up my dataset in the next steps


# I can see that I have some columns which are not numeric which are ID's and some which have many missing values in it shown as 'NaN' in the dataset

# In[ ]:


df['Salnty'].head()


# In[ ]:


# I first take the rows where temperature and the salinity is not NaN:

df = df[df['Salnty'].notna()]
print("rows are now ", len(df))
df.head()


# In[ ]:



df = df[df['T_degC'].notna()]
print("rows are now ", len(df))
df.head()


# I had a few rows with NaNs in the temperature and salinity columns which I now have taken out, I get now 675 rows

# Next I remove the columns containing NaN values. I think for this dataset it will not impact the results. 

# In[ ]:


# will drop any columns (axis= 1) with a NaN value. I can do that, since we already made sure that the temperature and salinity columns have no NaNs values.
df = df.dropna(axis = 1, how = 'any') 
df.head()


# In[ ]:


# Also, I want to have only the columns with numerical data. I am not interested in Strings and ID's for this dataset.
df = df._get_numeric_data()

df.head()
# The three variables that interest me the most are 'Depthm, T_degC, Salnty', for water depth of the sample, the salinity and the temperature. 
# the temperature is my output


# In[ ]:


# I define my output column
output_cols = ['T_degC']

# The inputs will be all my columns except the temperature column:
input_cols = df.columns[df.columns!='T_degC']
# so finally I have my pandas inputs and outputs
input_cols,output_cols


# In[ ]:


# these are my input columns
len(input_cols)


# In[ ]:


df[['T_degC']].head()


# ## Inspect relationships between the data 
# 
# I am interested in the relationship of the data I extracted from the dataset. The inputs are the saltiness of the water and the depth. I want to create a model able to predict the temperature.

# In[ ]:


# I make a scatter plot to see any visual relationship between the data 
sns.lmplot(x="Salnty", y="T_degC", data=df,
           order=2, ci=None);


# In[ ]:


sns.lmplot(x="Depthm", y="T_degC", data=df,
           order=2, ci=None);


# In[ ]:


inputs_array = df[input_cols].to_numpy()
targets_array = df[output_cols].to_numpy()
inputs_array, targets_array


# I convert the numpy arrays inputs_array and targets_array into PyTorch tensors. I make sure that the data type is torch.float32.

# In[ ]:


dtype = torch.float32
inputs = torch.from_numpy(inputs_array).type(dtype)
targets = torch.from_numpy(targets_array).type(dtype)
inputs.shape, targets.shape


# In[ ]:


inputs.dtype, targets.dtype


# In[ ]:


jovian.commit(project='jovian-Assignment-2', environment=None)


# Next, we need to create PyTorch datasets & data loaders for training & validation. We'll start by creating a TensorDataset.

# In[ ]:


dataset = TensorDataset(inputs, targets)


# Pick a number between 0.1 and 0.2 to determine the fraction of data that will be used for creating the validation set. Then use random_split to create training & validation datasets.
# 
# 

# In[ ]:


val_percent = 0.1 # between 0.1 and 0.2
num_rows = len(df)
num_cols = len(df.columns)
val_size = int(num_rows * val_percent)

train_size = num_rows - val_size


train_ds, val_ds = random_split(df,(train_size, val_size)) # Use the random_split function to split dataset into 2 parts of the desired length


# Pick a batch size for the data loader.

# In[ ]:


batch_size = 32


# In[ ]:


train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)


# In[ ]:


train_loader


# Let's look at a batch of data to verify everything is working fine so far.

# In[ ]:


for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break


# Let's save our work by committing to Jovian.

# In[ ]:


jovian.commit(project='jovian-Assignment-2', environment=None) 


# ## Create a Linear Regression Model

# In[ ]:


input_size = len(input_cols)
print(input_size)
output_size = len(output_cols)
print(output_size)


# In[ ]:


class TempModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)                  # fill this (hint: use input_size & output_size defined above)
        
    def forward(self, xb):
        out = self.linear(xb)                          # fill this
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out,targets)                         # fill this
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out,targets)                     # fill this    
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))


# Let us create a model using the TempModel class. 

# In[ ]:


model = TempModel()


# Let's check out the weights and biases of the model using model.parameters.

# In[ ]:


list(model.parameters())


# In[ ]:


model.linear.weight.dtype,model.linear.bias.dtype


# In[ ]:


jovian.commit(project='jovian-Assignment-2', environment=None) 


# ## Train the model to fit the data

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


result = evaluate (model,val_loader)
print(result)


# We are now ready to train the model.

# In[ ]:


epochs = 100
lr = 1e-6
history1 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


epochs = 100
lr = 1e-6
history2 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


epochs = 500
lr = 1e-6
history3 = fit(epochs, lr, model, train_loader, val_loader)


# In[ ]:


epochs = 500
lr = 1e-8
history4 = fit(epochs, lr, model, train_loader, val_loader)


# What is the final validation loss of your model?

# In[ ]:


val_loss = 1.0058
jovian.log_metrics(val_loss=val_loss)


# In[ ]:


loss = []
for value in history1+history2+history3+history4:
    loss.append(value['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss vs. No. of epochs');
plt.plot(loss)


# ## Make predictions using the trained model

# In[ ]:


def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)                # fill this
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


input, target = val_ds[23]
predict_single(input, target, model)


# In[ ]:


jovian.commit(project='jovian-Assignment-2', environment=None) 
jovian.commit(project='jovian-Assignment-2', environment=None) 


# 
