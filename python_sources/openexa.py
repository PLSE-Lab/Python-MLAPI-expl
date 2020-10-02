#!/usr/bin/env python
# coding: utf-8

# # Exoplanet prediction using feedforward net
# 
# We will use NASA's Kepler open exoplanet archive dataset (cumulative)
# 
# [Source](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
# 
# This will be a binary classification task (i.e, planet or false positive aka not planet). The table contains useful properties of the planetary transit and other features like mass, orbital period etc. More details below.
# 

# # Importing the Dataset

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

koi_cumm_path = os.path.join('../input', 'koi-cummulative/koi_cummulative.csv')


# In[ ]:


dfc = pd.read_csv(koi_cumm_path)
dfc.shape


# The Cumulative data has 9564 data points, however, we will soon see that we wont be able to use all of them
# 
# For now, lets see what our Categorizations might be.
# 
# Each observation in the KOI dataset has a disposition value which tells us what that Object is confirmed to be.
# 
# > 1. CONFIRMED - confirmed planets, these are confirmed to be exoplanets. These are our positive examples.
# > 2. FALSE POSITIVE - as the name suggests, these were thought to be exoplanets but turned out to be false. These will serve us as negative examples.
# > 3. CANDIDATE - these are potential candidates for exoplanets. These will be used for prediction
# 
# The KOI dataset also has a `koi-pdisposition` value which tells us the most probable explanation. `koi-disposition` values are finalized after further observations and analyses. 

# In[ ]:


dfc = pd.read_csv(koi_cumm_path)
dfc['koi_disposition'].unique()


# In[ ]:


dfc['koi_pdisposition'].unique()


# In[ ]:


# 2418 candidates

(dfc['koi_disposition'] == "CANDIDATE").value_counts()


# Lets take a look at our dataset before deconstructing it.

# In[ ]:


dfc.head(10)    # first 20 samples


# In[ ]:


# the columns

dfc.columns


# We use pandas.DataFrame.info() method to get more info on the dataset. As we see, so many columns have null values in them. Also so many columns which we do not need, like the kepler_name would not help in our classification at all.

# In[ ]:


dfc.info()


# ## Basic preprocessing

# We filter out the non numeric columns as they would serve us no purpose. Except `koi_disposition` since that is for labelling purpose.
# 
# Now, we need to encode the labels, ie, the koi_disposition values. We replace 'CONFIRMED' with 1 and 'FALSE POSITIVE' with 0 because we will use these label values for categorization. The other two values are arbitrary and serve only to filter out CANDIDATES and NOT DISPOSITIONED samples.
# 
# Also, we do the same with `koi-pdisposition` column as well.

# In[ ]:


# all the non-numeric columns

df_numeric = dfc.copy()

koi_disposition_labels = {
    "koi_disposition": {
        "CONFIRMED": 1,
        "FALSE POSITIVE": 0,
        "CANDIDATE": 2,
        "NOT DISPOSITIONED": 3
    },
    "koi_pdisposition": {
        "CONFIRMED": 1,
        "FALSE POSITIVE": 0,
        "CANDIDATE": 2,
        "NOT DISPOSITIONED": 3
    }
}

df_numeric.replace(koi_disposition_labels, inplace=True)
df_numeric


# We now want to filter out some more columns.
# 
# koi_score is not needed. It is the probability values for the categorization.
# However. these probability values will help us in the test phase on hunting new exoplanets among the candidates. So we will need it later.
# 
# SO, it is ideal to make a copy of the dataframe at its current state because we will need to come back to some columns again.
# 
# 
# Finally, koi_time0bk and koi_time0bk_err1 and 2 are removed because they are the time of the first detected transit minus some offset which is not a useful feature.
# 
# ### for more info on columns check out https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html

# Let us create two copies of the dataframe now, we will use one containing `koi-pdisposition` and `koi-score` for test phase, and the dataframe containing none of these in train phase.

# In[ ]:


# this is train data

# first we remove all string type columns from the dataframe

df_numeric = df_numeric.select_dtypes(exclude=['object']).copy()
df_test = df_numeric.copy()    # test data

# second, we manually remove some columns which are not needed as mentioned above. 
# additionally, 'koi_teq_err1' and 'koi_teq_err2' have all null values so they too need to be removed

rem_cols = ['kepid', 'koi_pdisposition', 'koi_score', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_teq_err1', 'koi_teq_err2']
df_numeric.drop(rem_cols, axis=1, inplace=True)

# this is test data
rem_cols_test = [col for col in rem_cols if col not in ['koi_pdisposition', 'koi_score']]
df_test.drop(rem_cols_test, axis=1, inplace=True)



df_numeric.head()


# In[ ]:


df_test.head()


# Now we have a somewhat decent dataset, however, this dataset still has a lot of missing values. 
# 
# We will simply discard the rows that have atleast one null entry and only consider the non-null dataset for our training.

# In[ ]:


df_numeric = df_numeric[df_numeric.isnull().sum(axis=1) == 0]
df_numeric.describe()


# As we see, `koi_fpflag_nt` has an outlier max value of `465.0` which is improbable since it is a flag and all the other flags are 0 or 1 valued.

# In[ ]:


index = df_numeric[df_numeric.koi_fpflag_nt == df_numeric.koi_fpflag_nt.max()].index
df_numeric.drop(index, inplace=True)


# In[ ]:


df_numeric.info()


# In[ ]:


df_test = df_test[df_test.isnull().sum(axis=1) == 0]
df_test.info()


# Test dataset should only contain `koi_disposition` value 2, since these are all candidate data. We will come back to this dataset later to predict

# In[ ]:


df_test = df_test[df_test.koi_disposition == 2]
df_test


# Now is a good time to save the `df_test` dataframe to csv for future use.

# In[ ]:


df_test.to_csv('koi_test.csv')


# We create a copy of this dataframe now to use it for our first neural network training, but we will come back to this dataframe again later.
# For now we are saving the `df_numeric` into a csv which can be later found in `input/koinumeric/koi_numeric.csv` file.

# In[ ]:


df_numeric.to_csv('koi_numeric.csv')


# In[ ]:


df_numeric1 = df_numeric.copy()


# In[ ]:


df_numeric1.info()


# We plot a heatmap of the correlation matrix for our dataframe and we see that overall the data has a lot of uncertainties and very few columns are sufficiently correlated to the target `koi_disposition`

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(30, 30))
sns.heatmap(df_numeric1.corr(), annot=True, cmap="RdYlGn", ax=ax)


# We try to standardize our dataframe because a lot of columns have huge values while others have very small values.

# In[ ]:


from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

# need to exclude the `koi_disposition` column from being standardized


df_numeric1.iloc[:, 5:] = std_scaler.fit_transform(df_numeric1.iloc[:, 5:])


# df_numeric.iloc[:, 0].to_numpy().reshape(-1, 1).shape
# df_standardized_w_labels = np.c_[df_standardized, df_numeric.iloc[:, 0].to_numpy().reshape(-1, 1)]
# df_standardized_w_labels[:3]

df_numeric1.values


# Congratulations. Now we have a complete dataframe with no null values, and also standardized for easier processing.

# ### Now that the preprocessing part is over, we want to create a PyTorch dataset to handle this data and create DataLoader batches
# 

# In[ ]:


import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


# ### We create a custom PyTorch dataset called `KeplerDataset` class by inheriting the `torch.utils.data.Dataset` class and overriding the `init` , `len` and `getitem` methods.
# ### We have included a flag called `test` which, if set to True, will generate the dataset with `koi_disposition` value `2` or `CANDIDATE`, which we will use for testing

# In[ ]:


class KeplerDataset(Dataset):
    def __init__(self, test=False):
        self.dataframe_orig = pd.read_csv(koi_cumm_path)

        if (test == False):
            self.data = df_numeric1[( df_numeric1.koi_disposition == 1 ) | ( df_numeric1.koi_disposition == 0 )].values
        else:
            self.data = df_numeric1[~(( df_numeric1.koi_disposition == 1 ) | ( df_numeric1.koi_disposition == 0 ))].values
            
        self.X_data = torch.FloatTensor(self.data[:, 1:])
        self.y_data = torch.FloatTensor(self.data[:, 0])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    
    def get_col_len(self):
        return self.X_data.shape[1]
    
kepler_df = KeplerDataset()


# In[ ]:


feature, target = kepler_df[1]
target, feature


# In[ ]:


kepler_df.get_col_len()


# Now, we want to split our data into training and validation set and also transfer the data to a cuda-enabled device before performing computations

# In[ ]:


# splitting into training and validation set

torch.manual_seed(42)

split_ratio = .7 # 70 / 30 split

train_size = int(len(kepler_df) * split_ratio)
val_size = len(kepler_df) - train_size
train_ds, val_ds = random_split(kepler_df, [train_size, val_size])

len(train_ds), len(val_ds)


# In[ ]:


batch_size = 32

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)


# In[ ]:


for features, target in train_loader:
    print(features.size(), target.size())
    break


# # First feedforward network model

# This is a rather simple feedforward architecture with just linear combinations and sigmoid activation.
# The model architecture is as followed:
# 
# 1. Input-layer (fully connected) (37 x 32)
# 2. Sigmoid activation (32)
# 3. 1st Hidden-layer (fully connected) (32 x 16)
# 4. Sigmoid activation (16)
# 5. 2nd Hidden-layer (fully connected) (16 x 8)
# 6. Sigmoid activation (8)
# 7. Output-layer (fully connected) (8 x 1)
# 8. Sigmoid activation (output) (1)
# 
# 
# **Note that this model incorporates sigmoid at the output layer, so BCELoss() is used.**

# In[ ]:


class KOIClassifier(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(KOIClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)    
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, out_dim)
        
        
        
    def forward(self, xb):
        out = self.linear1(xb)
        out = torch.sigmoid(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        out = self.linear3(out)
        out = torch.sigmoid(out)
        out = self.linear4(out)
        out = torch.sigmoid(out)
        out = self.linear5(out)
        out = torch.sigmoid(out)

    
        return out
    
    
    def predict(self, x):
        pred = self.forward(x)
        return pred
    
        
    def print_params(self):
        for params in self.parameters():
            print(params)


# In[ ]:


input_dim = kepler_df.get_col_len()
out_dim = 1
model = KOIClassifier(input_dim, out_dim)



# ### I have already trained this model using the same hyperparameters, the stats are located in `\input\first-nn-stats`
# 
# ### If you want to use the previous stats then uncomment the following cell and run

# In[ ]:


"""

model_prev = KOIClassifier(input_dim, out_dim)
construct = torch.load('../input/first-nn-stats/checkpoint.pth')
model_prev.load_state_dict(construct['state_dict'])

import seaborn as sns
%matplotlib inline

cf_mat_train = pred_confusion_matrix(model_prev, train_loader)
cf_mat_val = pred_confusion_matrix(model_prev, val_loader)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

ax1, ax2 = axes
sns.heatmap(cf_mat_train, fmt='g', annot=True, ax=ax1)
ax1.set_title('Training Data')

sns.heatmap(cf_mat_val, fmt='g', annot=True, ax=ax2)
ax2.set_title('Validation Data')

"""


# This is where the training happens. 
# * optimiser = SGD
# * number of epochs = 1000
# * learning-rate = 0.01
# * device of computation = CPU

# In[ ]:


# training phase
criterion = nn.BCELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)
n_epochs = 1000

def train_model():
    for X, y in train_loader:
        for epoch in range(n_epochs):
            optim.zero_grad()
            y_pred = model.forward(X).flatten()
            loss = criterion(y_pred, y)
            loss.backward()
            optim.step()

train_model()


# In[ ]:


# testing the predictions
for X, y in train_loader:
    y_pred = model.forward(X)
    y_pred = y_pred > 0.5
    y_pred = torch.tensor(y_pred, dtype=torch.int32)
    print(y_pred)
    break


# In[ ]:


from sklearn.metrics import confusion_matrix
def pred_confusion_matrix(model, loader):
    with torch.no_grad():
        all_preds = torch.tensor([])
        all_true = torch.tensor([])
        for X, y in loader:
            y_pred = model(X)
            y_pred = torch.tensor(y_pred > 0.5, dtype=torch.float32).flatten()
            all_preds = torch.cat([all_preds, y_pred])

            all_true = torch.cat([all_true, y])
            
    
    return confusion_matrix(all_true.numpy(), all_preds.numpy())


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

cf_mat_train = pred_confusion_matrix(model, train_loader)
cf_mat_val = pred_confusion_matrix(model, val_loader)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

ax1, ax2 = axes
sns.heatmap(cf_mat_train, fmt='g', annot=True, ax=ax1)
ax1.set_title('Training Data')

sns.heatmap(cf_mat_val, fmt='g', annot=True, ax=ax2)
ax2.set_title('Validation Data')


# In[ ]:


checkpoint = {
    'state_dict': model.state_dict(),
    'optimizer': optim.state_dict()
}

torch.save(checkpoint, 'checkpoint.pth')


# # More preprocessing and feature selection

# Even though the model seems to perform exceptionally well, we have made some fatal mistakes. 
# Firstly, we standardized the whole dataset, as a result, the informations about the test data got mixed up with the train data. The test data and train data should be separate.
# Secondly, we added so many columns which are not much needed, for example the columns which have very high correlation coefficient with some others.
# 
# Finally, we need a more organized model with fewer parameters, otherwise we will risk overfitting.
# 
# 
# Let us try to reduce dimensions first by removing columns which have correlation coeffs higher than 0.80

# In[ ]:


# this is where we return back to the point from where we branched, we take the numeric dataframe again and apply some feature selection
df_new = pd.read_csv('koi_numeric.csv', index_col=0)
df_new.head()


# In[ ]:


# a function to remove high correlation columns by selecting the upper triangle of the correlation matrix
# and dropping all columns which have corr value > threshold at any row

def remove_high_corr(df, threshold):
    corr_mat = df.corr()
    trimask = corr_mat.abs().mask(~np.triu(np.ones(corr_mat.shape, dtype=bool), k=1))
    blocklist = [col for col in trimask.columns if (trimask[col] > threshold).any()]
    df.drop(columns=blocklist, axis=1,inplace=True)
    return blocklist


# In[ ]:


remove_high_corr(df_new, 0.80)


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df_new.corr(), cmap="Blues", ax=ax)


# Great! Now we can save this csv again and move on the the next parts

# In[ ]:


df_new.head()


# So now we have a reduced dataset with 29 columns.

# Let us save this reduced dataset also, so that we can use it in our pytorch dataset

# In[ ]:


df_new.to_csv('koi_numeric_reduced.csv')


# # Trying GPU accelaration, new Dataset and model architecture

# We want to change the way our dataset class performs. Earlier we stiched together a bunch of modification but this time we want to maintain consistency.
# We previously standardized the entire dataset, including the test set (which included `koi_disposition` value 2) which was a bad practice.
# We will now only do standardization on the training data and validation data. 
# When using test data we will do the standardization separately.

# In[ ]:


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
    
device = get_default_device()
device


# The standardization process in the previous model was flawed because it standardized the entire dataset, introducing test data statistics into training and validation data.
# This is bad, because then our model will be influenced by test data and will never truly learn anything. 
# 
# I therefore, used `sklearn.model_selection.train_test_split` to split the training data into training and validation data, and created a separate test data by filtering out based on `koi-disposition` values.
# 
# I then used `StandardScaler` separately on each dataset to produce independently standardized samples.
# 
# The rest of it are similar as before.

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

std_scaler = StandardScaler()

dataframe = pd.read_csv('koi_numeric_reduced.csv', index_col=0)

train_data = dataframe.query('not koi_disposition == 2').values

X = train_data[:, 1:]
y = train_data[:, 0]

val_size = .3
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=val_size, shuffle=True)

train_X[:, 4:] = std_scaler.fit_transform(train_X[:, 4:])
val_X[:, 4:] = std_scaler.fit_transform(val_X[:, 4:])


# print(f'train_X = {train_X.shape}\n\nval_X = {val_X.shape}\n')

class KOIDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.FloatTensor(X_data)
        self.y_data = torch.FloatTensor(y_data)
        
    
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    


    

train_ds = KOIDataset(train_X, train_y)
val_ds = KOIDataset(val_X, val_y)

for feature, target in train_ds:
    print(feature, target)
    break
    


# This time, I used a batch size of 64

# In[ ]:


batch_size = 64
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)


# Ported all the dataloaders to GPU for faster processing.

# In[ ]:


train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)


# In[ ]:


for features, target in train_loader:
    print(target, features)
    break


# ## New feedforward network
# 
# > The architecture is as followed.
# 1. Input Layer (fully connected) (28 x 24)
# 2. Sigmoid (Activation) (24)
# 3. Batch Normalization Layer (1D) (24)
# 4. Hidden Layer (1st) (24 x 16)
# 5. Sigmoid (Activation) (16)
# 6. Batch Normalization Layer (1D) (16)
# 7. Dropout Layer with probability 0.1 (16)
# 1. 8. Output Layer (fully connected) (16 x 1)
# 
# 

# In[ ]:


# a function to measure prediction accuracy 

def accuracy(outputs, labels):
    output_labels = torch.round(torch.sigmoid(outputs))    # manually have to activate sigmoid since the nn does not incorporate sigmoid at final layer
    
    return torch.tensor(torch.sum(output_labels == labels.unsqueeze(1)).item() / len(output_labels))
    


# In[ ]:


from collections import OrderedDict

input_dim = train_X.shape[1]

class KOIClassifierSeq(nn.Module):
    def __init__(self):
        super(KOIClassifierSeq, self).__init__()
        self.model = nn.Sequential(OrderedDict([
              ('fc1', nn.Linear(input_dim, 24)),
              ('sigmoid1', nn.Sigmoid()),
              ('batchnorm1', nn.BatchNorm1d(24)),
              ('fc2', nn.Linear(24, 16)),
              ('sigmoid2', nn.Sigmoid()),
              ('batchnorm2', nn.BatchNorm1d(16)),
              ('dropout', nn.Dropout(p=0.1)),
              ('fc3', nn.Linear(16, 1))
            ]))
    
    def forward(self, xb):
        return self.model(xb)
    
    def training_step(self, batch):
        features, label = batch 
        out = self(features)
        loss = F.binary_cross_entropy_with_logits(out, label.unsqueeze(1)) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        features, label = batch 
        out = self(features)                    
        loss = F.binary_cross_entropy_with_logits(out, label.unsqueeze(1))   # Calculate loss
        acc = accuracy(out, label)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# In[ ]:


@torch.no_grad()
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# Finally, I have my model ready now. I ported the model to GPU again. The layers can be seen from the following output.

# In[ ]:


model1 = to_device(KOIClassifierSeq(), device)
model1


# ### Let us fit our model using Adam optimiser and a small learning rate `1e-5`

# In[ ]:


num_epochs = 10
lr = 1e-4
history = fit(num_epochs, lr, model1, train_loader, val_loader, opt_func=torch.optim.Adam)


# this seems to perform really well. It got a steep jump in terms of accuracy. Let us keep training.

# In[ ]:


num_epochs = 5
lr = 1e-4
history = fit(num_epochs, lr, model1, train_loader, val_loader, opt_func=torch.optim.Adam)


# In[ ]:


# a function to calculate training accuracy

def train_accuracy(model):
    train_acc = []
    for X, y in train_loader:
        out = model(X)
        train_acc.append(accuracy(out, y))

    return torch.stack(train_acc).mean().item()


# In[ ]:


train_accuracy(model1)


# So, at the end of training which was relatively fast, We have 97.7% training accuracy and 97.2% validation accuracy.
# Let us calculate confusion matrix and visualize our predictions.

# In[ ]:


from sklearn.metrics import confusion_matrix
def pred_confusion_matrix(model, loader):
    with torch.no_grad():
        all_preds = to_device(torch.tensor([]), device)
        all_true = to_device(torch.tensor([]), device)
        for X, y in loader:
            y_pred = model(X)
            y_pred = torch.round(torch.sigmoid(y_pred))
            all_preds = torch.cat([all_preds, y_pred])

            all_true = torch.cat([all_true, y.unsqueeze(1)])
            
    
    return confusion_matrix(all_true.cpu().numpy(), all_preds.cpu().numpy())


# In[ ]:


cf_mat_train = pred_confusion_matrix(model1, train_loader)
cf_mat_val = pred_confusion_matrix(model1, val_loader)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

ax1, ax2 = axes
sns.heatmap(cf_mat_train, fmt='g', annot=True, ax=ax1)
ax1.set_title('Training Data')

sns.heatmap(cf_mat_val, fmt='g', annot=True, ax=ax2)
ax2.set_title('Validation Data')


# As we can see, both training and validation data are predicted super accurately. 
# We are not going to train any further. This is more than enough stats for a feedforward neural network classification.
# 
# Let us plot the accuracies and predictions.

# In[ ]:


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')


# In[ ]:


plot_accuracies(history)


# In[ ]:


plot_losses(history)


# Saving our model state similar as before. 

# In[ ]:


second_model = {
    'state_dict': model1.state_dict()
}

torch.save(second_model, 'second_model.pth')

# I have uploaded the pth file to the /input directory. If needed, you can load it from there and load it into a model instance of KOIClassifierSeq


# # Testing on the Test data

# In[ ]:


test_df = pd.read_csv('koi_test.csv', index_col=0)
test_df


# We need to apply the same preprocessing steps on the test dataset as well.
# 
# We can remove `koi_disposition` column as only has value 2 for `CANDIDATE`,
# We can remove `koi_pdisposition` aswell since it contains same data as `koi_disposition`.
# We will use the `koi_score` to see our prediction accuracy.
# We also have to remove the columns we had removed previously from the train and validation data, otherwise there will be a dimensionality mismatch.

# In[ ]:


cols = [
 'koi_disposition',
 'koi_pdisposition',
 'koi_period_err2',
 'koi_impact_err2',
 'koi_duration_err2',
 'koi_depth_err2',
 'koi_prad_err2',
 'koi_insol_err1',
 'koi_insol_err2',
 'koi_steff_err2',
 'koi_srad_err2']

test_df.drop(cols, axis=1, inplace=True)


# In[ ]:


test_df.head()


# We perform standardization same as before.

# In[ ]:


test_X = test_df.iloc[:, 1:].values
test_probs = test_df.iloc[:, 0].values

test_X[:, 4:] = std_scaler.fit_transform(test_X[:, 4:])

  

KOI_test = KOIDataset(test_X, test_probs)


# In[ ]:


batch_size = 64
test_loader = DataLoader(KOI_test, batch_size, num_workers=4, pin_memory=True)
test_loader = DeviceDataLoader(test_loader, device)

for X, y in test_loader:
    print(X.size(), y.size())
    break


# In[ ]:


def predict_probs(model, X):
    probs = torch.sigmoid(model(X))
    return probs


# As we can see, the predictions are not as accurate. 

# In[ ]:


torch.set_printoptions(precision=5, threshold=5000)
with torch.no_grad():
    for X, y in test_loader:
        #print(X, y)
        preds = torch.sigmoid(model1(X))
        for pred, true in zip(preds, y.unsqueeze(1)):
            print(f'model prediction: {pred.item()}\tKOI prediction: {true.item()}')
        break


# In[ ]:


def accuracy_test(outputs, label_prob):
    output_labels = torch.round(torch.sigmoid(outputs))    
    labels = torch.round(label_prob)
    return torch.tensor(torch.sum(output_labels == labels.unsqueeze(1)).item() / len(output_labels))
    
    
def test_accuracy(model):
    test_acc = []
    with torch.no_grad():
        for X, y in test_loader:
            out = model(X)
            test_acc.append(accuracy_test(out, y))

    return torch.stack(test_acc).mean().item()


# In[ ]:


test_accuracy(model1)


# We got a test accuracy which is, unfortunately, not as good. But its a start. We will save the model at its current state.
# 
# 

# In[ ]:


torch.save(model1.state_dict(), 'final_model_53_percent.pth')


# # Using a simpler model

# We will now try a simpler model with only one hidden layer with no batchnorm or dropout layer and only sigmoid as activation. 

# In[ ]:


class KOIClassifierSimple(nn.Module):
    def __init__(self):
        super(KOIClassifierSimple, self).__init__()
        self.model = nn.Sequential(OrderedDict([
              ('fc1', nn.Linear(input_dim, 24)),
              ('sigmoid1', nn.Sigmoid()),
              ('fc2', nn.Linear(24, 16)),
              ('sigmoid2', nn.Sigmoid()),
              ('fc3', nn.Linear(16, 1))
            ]))
    
    def forward(self, xb):
        return self.model(xb)
    
    def training_step(self, batch):
        features, label = batch 
        out = self(features)
        loss = F.binary_cross_entropy_with_logits(out, label.unsqueeze(1)) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        features, label = batch 
        out = self(features)                    
        loss = F.binary_cross_entropy_with_logits(out, label.unsqueeze(1))   # Calculate loss
        acc = accuracy(out, label)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# In[ ]:


model2 = to_device(KOIClassifierSimple(), device)
model2


# In[ ]:


num_epochs = 10
lr = 1e-3
history2 = fit(num_epochs, lr, model2, train_loader, val_loader, opt_func=torch.optim.Adam)


# In[ ]:


train_accuracy(model2)


# In[ ]:


plot_accuracies(history2)


# In[ ]:


plot_losses(history2)


# In[ ]:


test_accuracy(model2)


# As we see, a simpler model was able to give us a test accuracy of 67.9% which is a lot better than our previous model. This goes to show that a more complex model might not always be the go-to solution for every task. We could even use other machine learning algorithms like SVM or Decision Trees to come into agreeable accuracy. 

# # Conclusion

# There might be a number of reasons why our model failed to perform accurately in the test set. The test set is predominated by positive probabilities, with an uneven distribution of positive and negative candidates. For this reason our model might underperform. Another reason might be the case of overfitting. 
# Simpler model is always better. Maybe by changing and tinkering with the network architecture a bit, we can come up to a decent enough prediction accuracy. 
# 
# And as evidently shown, a simpler model might be able to generalize better and give better estimations.
# 
# Although, this is nowhere close to being an actual prediction modelling for exoplanet search. A much better analysis would be on time-series data or transit curve images using CNN architectures. 
# 
# 
# Having basically no idea about the deeper intricacies of Astronomy, and only relying on the column descriptions from the official website, it was pretty much a wild guess but the fact that a seemingly random prediction model was able to perform with 53% accuracy and then being able to get a 68% accuracy on the test data with an even simpler model was pretty nice! 
# 
# If anyone is interested in tinkering with this notebook even more and has domain-specific knowledge as to which columns are more imoprtant in planetary predictions, feel free to fork this and modify it. Thanks!
