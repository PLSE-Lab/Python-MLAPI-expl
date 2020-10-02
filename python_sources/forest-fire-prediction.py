#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Uncomment and run the commands below if imports fail
get_ipython().system('conda install numpy pytorch torchvision cpuonly -c pytorch -y')
get_ipython().system('pip install matplotlib --upgrade --quiet')
get_ipython().system('pip install pandas --upgrade --quiet')
get_ipython().system('pip install seaborn --upgrade --quiet')
get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import torch
import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


project_name='forest-fires-regression-prediction' # will be used by jovian.commit


# In[ ]:


df_raw = pd.read_csv('../input/forest-fires-data-set-portugal/forestfires.csv')
df_raw.head()


# In[ ]:


df = df_raw.drop(['X','Y','month','day','area','DMC','DC'], axis=1)
df.head()


# In[ ]:


# Compute the correlation matrix
corr_matrix = df.drop(['ISI'], axis=1).corr()
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 6))
cmap = sns.diverging_palette(200, 10, as_cmap=True)
sns.heatmap(corr_matrix, cmap=cmap, annot=True,linewidth = 0.1,  cbar_kws={"shrink": .5})


# In[ ]:


corr_with_isi = df.corr()['ISI'].sort_values(ascending=False)
plt.figure(figsize=(5,4))
corr_with_isi.drop('ISI').plot.bar()
plt.show();


# In[ ]:


sns.pairplot(df[['FFMC', 'temp', 'RH', 'wind', 'rain']])
plt.show()


# In[ ]:


num_rows = df.shape[0]
num_rows


# In[ ]:


print(df.columns)
#cols number
print(df.shape[1])


# In[ ]:


input_cols = ['FFMC','temp','RH','wind','rain']
output_cols = ['ISI']


# In[ ]:


jovian.commit(project=project_name, environment=None)


# In[ ]:


#using pytorch as framework, so we have to convert all to pytorch format
def dataframe_to_arrays(dataframe):
    #copying dataframe for later use
    dataframe1 = dataframe.copy(deep=True)
    #converting to numpy
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array


# In[ ]:


inputs_array.shape , targets_array.shape


# In[ ]:


inputs_array, targets_array = dataframe_to_arrays(df)


# In[ ]:


inputs_array.shape , targets_array.shape


# In[ ]:


#converting numpy arrays to pytorch format
inputs = torch.from_numpy(inputs_array).float()
targets = torch.from_numpy(targets_array).float()


# In[ ]:


inputs.shape, targets.shape


# In[ ]:


dataset = TensorDataset(inputs, targets)


# In[ ]:


val_percent = 0.19
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size

train_ds, val_ds = random_split(dataset, [train_size,val_size])


# In[ ]:


batch_size = 32


# In[ ]:


#creating training and validation loader
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)


# In[ ]:


input_size = len(input_cols)
output_size = len(output_cols)
input_size , output_size


# In[ ]:


class FFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
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
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 10th epoch
        if (epoch+1) % 10 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))


# In[ ]:


ffmodel = FFModel()


# In[ ]:


list(ffmodel.parameters())


# In[ ]:


jovian.commit(project=project_name, environment=None)


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
            #zero-grad for setting all parameters to zero , for new training phrase
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history


# In[ ]:


result = evaluate(ffmodel, val_loader)
print(result)


# In[ ]:


epochs = 1000
lr = 3e-4
opt_func = torch.optim.Adam
history1 = fit(epochs, lr, ffmodel, train_loader, val_loader , opt_func)


# In[ ]:


def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)


# In[ ]:


from random import sample

# showing all predicted , target given and input 
for i in sample(range(0,len(val_ds)),10):
    input, target = val_ds[i]
    predict_single(input, target, ffmodel)
    print()


# In[ ]:


jovian.commit(project=project_name, environment=None)


# In[ ]:




