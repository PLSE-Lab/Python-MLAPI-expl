#!/usr/bin/env python
# coding: utf-8

# # Acknowledgement
# It is my first time to apply Pytorch for building Deep Learning models, especially for LSTM. After looking at the following notebook, I have some problems with its process of automatic batching training. Therefore, I just explore by experiment based on following documents.
# 
# **The [official document](https://pytorch.org/docs/stable/data.html) gives thorough explanation for this.**
# 
# https://www.kaggle.com/gopidurgaprasad/m5-forecasting-eda-lstm-pytorch-modeling/
# 
# https://www.youtube.com/watch?v=zN49HdDxHi8
# 

# # Basic knowledge of Batch training for newies
# 
# For somebody who is not familiar with concepts like iteration, batch, epochs, here is the **basic background knowledge of Batch Training**.
# 
# 
# ### Iteration for a Batch/Mini-batch of training sample
# Backward and forward pass makes together one "iteration". 
# 
# * **<span style="color:red">Forward pass</span>**: refers to calculation process, values of the output layers from the inputs data. It's traversing through all neurons from first to last layer. A **<span style="color:blue">loss function</span>** is calculated from the output values.
# 
# * **<span style="color:red">Backward pass</span>**: refers to process of computation from the last layer backward to the first layer to count changes in weights (de facto learning), using **<span style="color:blue">gradient descent</span>** algorithm (or similar) and update weights.
# 
# Each iteration train <b>one-batch-size</b> samples. During one iteration, you usually pass a subset of the data set, which is called ***mini-batch*** (if you pass all data at once, it is called ***batch***).
# 
# 
# ### Epoch for All training samples
# For newies like me, sometimes it would be confused with "Epoch" which means passing the entire data set. One epoch contains ***number_of_items/batch_size*** iterations.
# 

# # Normal implementation of Batch Training (for complete and clear understanding, Epoch loop is also added)
# 
# ```
# for epoch in range(EPOCH):
#     # Iteration: loop over all the batches
#     for i in range(BATCH_NUM):
#         x_batch, y_batch = ..
# ```

# # Understand: Pytorch built-in support for batch iteration
# Different to normal implementation of Batch Training as below, in Pytorch, we could use built-in `DataSet` and `DataLoader` in Pytorch to load the training and validation data which would do the batch iteration for us. 

# Now, let's experiment.
# 
# 
# 
# First, prepare the data for training and validation
# * one dataset is from my [previous notebook](https://www.kaggle.com/sergioli212/m5-all-feature-engineering-ready-to-use) for feature engineering of M5 forecasting. 
# 
# * Another is from [this notebook](https://www.kaggle.com/gopidurgaprasad/m5-forecasting-eda-lstm-pytorch-modeling/). Thanks for contribution this notebook. For convenience, I donot need to construct data for each item by using my whole dataset (Lazy me~~ ). It adds date, price and sales lags as features. I think it may not be rational to add lag features for sequential models like LSTM. BUT anyway, I am lazy to make new ones. And RATIONALITY does not mean GOOD PERFORMANCE.
# 
# 
# Just clarify again, the first dataset is only kept to identify the item_id and days which can give index to take the specific series for one item which is loaded from the second dataset.
# 
# Get data for Store: CA_1 
# * **1800 ~ 1857** for training
# * **1858 ~ 1885** for validation

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn import preprocessing


import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import math


# In[ ]:


INPUT_DIR1 = '../input/m5fedata/'
INPUT_DIR2 = '../input/m5-forecasting-eda-lstm-pytorch-modeling/'
# load train/val data
grid_df = pd.read_pickle(f'{INPUT_DIR1}grid_df_evaluation.pkl')
grid_df = grid_df[grid_df.d <= 1913]



# select the data for one store(CA_1) with specific columns
CA_1 = grid_df[grid_df.store_id == 'CA_1']
select_cols = ['item_id', 'd', 'sales', 'sell_price']
CA_1 = CA_1[select_cols]

# create dataframe for loading npy files and  train/valid split
data_info = CA_1[["item_id", "d"]]

# total number of days: 1913
# for training we are taking data in 57 days between 1800 < train < 1913-28-28 = 1857
train_df = data_info[(1800 < data_info.d) &( data_info.d <= 1857)]

# valid data: 1859 -> 1885 (next 28days)
valid_df = data_info[data_info.d == 1885]


# ### torch.utils.data.Dataset
# 
# DataLoading which inherits `Dataset` has to implement 3 methods:
# * `__init__`: Constructor
# * `__getitem__`: with this, you can give index to the instance and get returned with both the feature and label of the indexed sample. In our case, it takes data of the last 28 days as input and sales of the next 28 day as output. 
# 
#     ps: You can better understand by doing testing after defining and instantiating such class. 
#     
#     
# * `__len__`: length of dataset

# In[ ]:


# item_id = "HOBBIES_1_005"
# item_npy = joblib.load(INPUT_DIR2+f'something_spl/{item_id}.npy')
# len(item_npy[:,0])


# In[ ]:


import joblib
# encode item id
label = preprocessing.LabelEncoder()
label.fit(train_df.item_id)
# Example
# label.transform(["FOODS_3_8dd27"])



class DataLoading(torch.utils.data.Dataset):
    def __init__(self, df, train_window = 28, predicting_window=28):
        self.df = df.values
        
        # Here define the length of series for training and length of series for prediction
        self.train_window = train_window
        self.predicting_window = predicting_window

    def __len__(self):
        return len(self.df)
    
    
    
    def __getitem__(self, index):
       
        # Select one series of an item 
        item_id = self.df[index][0]
        day = self.df[index][1]
        print('You are getting item now!!! for day: '+str(day)+', item: '+item_id)
        item_npy = joblib.load(INPUT_DIR2+f'something_spl/{item_id}.npy')  # item_id = "HOBBIES_1_005"
        
        # INPUT: get previous 28-day data including sales as input
        features = item_npy[day-self.train_window:day] 
        # sales of the upcoming 28 days, the model also predict such OUTPUT
        sales = item_npy[:,0][day: day+self.predicting_window] 
        
        
        # represent item_id by onehot-code vector and add it to feature
        item_label = label.transform([item_id])
        item_onehot = [0] * 3049
        item_onehot[item_label[0]] = 1
        final_features = []
        for f in features: ## each row of feature would be added item_onehot
            one_f = []
            one_f.extend(item_onehot)
            one_f.extend(f)
            final_features.append(one_f)

        
        return {
            "features" : torch.Tensor(final_features),
            "label" : torch.Tensor(sales)
        }
        # the return data structure do not have to be dictionary,  you can return anything.
        


# Let's test `Dataset` Class.

# In[ ]:


dataset = DataLoading(train_df)
first_data = dataset[0]

# features is for 28 days before 1801, label is for 28 day after(include) 1801
print('Feature Shape:', first_data['features'].shape, 'Label Shape: ', first_data['label'].shape)


# ### torch.utils.data.DataLoader
# Normally, `DataLoader` could be directly instantiated using `Dataset` instance. Furthermore, `batch_size` could be specified for batch training so that you do not need to explicitly loop each batch. More details about how to draw samples for each batch refer to: 
# https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

# In[ ]:


train_batch_size = 512
test_batch_size = 128

train_dataset = DataLoading(train_df)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size= train_batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

valid_dataset = DataLoading(valid_df)

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size= test_batch_size,
    shuffle=False,
    num_workers=4, # fast to use multiple processing
    drop_last=True
)


# In[ ]:


The `Dataloader` should have the length as the iteration(the total number of samples / batch size). Let's check that.(it has 1 difference. I think it is because `Dataloader` at default will discard the left few samples which cannot make it the batch size).


# In[ ]:


n_epochs = 2
n_total_samples = len(dataset)
n_iterations = math.ceil(n_total_samples/train_batch_size)
print('Iteration is: ', n_iterations)
print('Length of train_loader is: ', len(train_loader))


# In[ ]:


# dataiter = iter(train_loader)
# dataiter.next()


# In[ ]:


# Training loop like this
# NOTICE: For demonstration purpose, do not run this cell.
for epoch in range(n_epochs):   
    for i, data in enumerate(train_loader): # each time takes one batch
        features, label = d["features"], d["label"]
        # For demonstration of how batching works with Dataset and Dataloader, here ignore the training process which would be done below


# ### Training and validation

# In[ ]:


# Define something for modelling
DEVICE = "cuda"
model = LSTM()
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.7, verbose=True, min_lr=1e-5)

class LSTM(nn.Module):
    def __init__(self, input_size=3062, hidden_layer_size=100, output_size=28):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True) # 3062 -> 100
        self.linear = nn.Linear(hidden_layer_size, output_size)              # 100  -> 28

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
        
    def forward(self, input_seq):

        lstm_out, self.hidden_cell = self.lstm(input_seq)

        lstm_out = lstm_out[:, -1]

        predictions = self.linear(lstm_out)

        return predictions
    
    
# loss function
def criterion1(pred1, targets):
    l1 = nn.MSELoss()(pred1, targets)
    return l1


        

def evaluate_model(model, val_loader, epoch, scheduler=None, history=None):
    model.eval()
    loss = 0
    pred_list = []
    real_list = []
    RMSE_list = []
    with torch.no_grad():
        for i,d in enumerate(tqdm(val_loader)):
            item = d["features"].cuda().float()
            y_batch = d["label"].cuda().float()

            o1 = model(item)
            l1 = criterion1(o1, y_batch)
            loss += l1
            
            o1 = o1.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            
            for pred, real in zip(o1, y_batch):
                rmse = np.sqrt(sklearn.metrics.mean_squared_error(real, pred))
                RMSE_list.append(rmse)
                pred_list.append(pred)
                real_list.append(real)

    loss /= len(val_loader)
    
    if scheduler is not None:
        scheduler.step(loss)

    print(f'\n Dev loss: %.4f RMSE : %.4f'%(loss, np.mean(RMSE_list)))
    
    
def train_model(model,train_loader, epoch, optimizer, scheduler=None, history=None):
        model.train()
        total_loss = 0

        t = tqdm(train_loader)

        for i, d in enumerate(t):

            item = d["features"].cuda().float()
            y_batch = d["label"].cuda().float()

            optimizer.zero_grad()

            out = model(item)
            loss = criterion1(out, y_batch)

            total_loss += loss

            t.set_description(f'Epoch {epoch+1} : , LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(i+1)))

            if history is not None:
                history.loc[epoch + i / len(X), 'train_loss'] = loss.data.cpu().numpy()
                history.loc[epoch + i / len(X), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']

            loss.backward()
            optimizer.step()


# In[ ]:


for epoch in range(EPOCHS):   
    
    train_model(model, train_loader, epoch, optimizer, scheduler=scheduler, history=None)
    evaluate_model(model, valid_loader, epoch, scheduler=scheduler, history=None)

