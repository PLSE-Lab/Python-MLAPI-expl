#!/usr/bin/env python
# coding: utf-8

# ### "The Earth is what we all have in common."
# -Wendell Berry

# Disclamer: I know some pieces of code are messy yet, for now my purpose is to test approaches, then to clean up the code

# Humanity develops civilizations quickly, discover new technologies rapidly, but relatively recently we stopped and started to think what we do to our Earth. Air pollutions, oceans depletions, erroded soils is just a few terms humans are accounted for. Days when we will happily live on Mars do not seem close (and who knows how we might disrupt this planet too) so the best way is to care about our Earth.

# My purpose with this project is to explore trends of air pollution, not to tell what we should do. But if at least someone will stop and think about our impact on the world around might work will not be in wain.

# I was switching back and forth between data, the main reason for it being my insentive. I wanted a model that would predict pollutant gases for particular region but was not able to find any good dataset for it. Finally, EPA provides tones of data for USA. Here I will use a subset of the data. Full database can be found at https://aqs.epa.gov/aqsweb/airdata/download_files.html. Lookup for county codes you can at https://aqs.epa.gov/aqsweb/documents/codetables/states_and_counties.html Precisely the one I am using I uploaded to Kaggle and it is attached to the notebook.

# The subset I will consider is daily measures of SO2 concentration in the air in Philadelphia county, PA. Generaly with minor tweaks you can run the notebook for any other county. And yet I will exlore the data throught USA too.

# ### Imports & preconfigurations

# In[ ]:


# Uncomment if you do not have folium installed
# !pip3 install folium


# In[ ]:


get_ipython().system('pip3 install jovian --quiet')


# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from torch.optim import Adam
import jovian
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from tqdm import tqdm
#from category_encoders import Label
# jovian.commit(project=project_name)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 5);


# In[ ]:


project_name = 'air-pollution-case'
author = 'Valentyna Fihurska'


# ### Basic EDA

# In[ ]:


pol_df = pd.read_csv(
    '../input/so2-emissions-daily-summary-data/daily_42401_2017/daily_42401_2017.csv')
pol_df.head(5)


# Let us see locations from where the pollution comes from, or at least where we have 'latitude' and 'longitude' values. Or probably something is missing? P.S. Be patient, we have quite a lot of data to map.

# In[ ]:


pol_df
samp = pol_df.loc[pol_df['Latitude'].notnull()]
map_sample = folium.Map(location=[50,-85], zoom_start=3)
HeatMap(data=samp[['Latitude', 'Longitude']], radius=10).add_to(map_sample)
map_sample


# Now I will narrow it down to the site of interest and add data for 2018 and 2019.

# In[ ]:


p2018 = pd.read_csv('../input/so2-emissions-daily-summary-data/daily_42401_2018.csv')
p2019 = pd.read_csv('../input/so2-emissions-daily-summary-data/daily_42401_2019.csv')
pol_df = pol_df.loc[(pol_df['County Code']==101) & (pol_df['State Code']==42)]
p2018 = p2018.loc[(p2018['County Code']==101) & (p2018['State Code']==42)]
p2019 = p2019.loc[(p2019['County Code']==101) & (p2019['State Code']==42)]
pol_df = pd.concat([pol_df, p2018, p2019], axis=0)
del p2018, p2019, samp
pol_df.head()


# In[ ]:


samp = pol_df.loc[pol_df['Latitude'].notnull()]
map_sample = folium.Map(location=[40,-75], zoom_start=9)
HeatMap(data=samp[['Latitude', 'Longitude']], radius=10).add_to(map_sample)
map_sample


# In[ ]:


pol_df.isnull().sum()


# Many values are intuitive as for what that means but some are not. Let's explain them:
# > POC is a code used to distinguish between different monitors at one site that are measuring a parameter. For example, the first monitor established to measure CO at a site would have POC = "1". If an additional monitor were established at the same site to measure CO, that monitor would have POC = "2". However, if a new instrument were installed to replace the original instrument used as the first monitor, that would be the same monitor and it would have POC = "1". For criteria pollutants, data from different sampling methods should only be stored under the same POC if the sampling intervals are the same and the methods or references are equivalent. For sites where duplicate sampling is being conducted by multiple agencies or one agency with multiple samplers, multiple POC's must be utilized to store all samples. For non-criteria pollutants, data from multiple sampling methods can be stored under the same POC if the sampling intervals are the same. <br><br>
# > The AQI is an index for reporting daily air quality. It tells you how clean or polluted your air is, and what associated health effects might be a concern for you. The AQI focuses on health effects you may experience within a few hours or days after breathing polluted air. \[...\] An AQI value of 100 generally corresponds to the national air quality standard for the pollutant, which is the level EPA has set to protect public health. AQI values below 100 are generally thought of as satisfactory. When AQI values are above 100, air quality is considered to be unhealthy-at first for certain sensitive groups of people, then for everyone as AQI values get higher. 
# <br>https://cfpub.epa.gov/airnow/index.cfm?action=aqibasics.aqi

# To put it simply, POC could be thought of as 'id' of the measurement device. AQI tells us how polluted the air is, the lower AQI the better for human health and the health of the ecosystem.[](http://)

# In[ ]:


pol_df.shape


# Let us get rid of columns we will not need (for now at least)

# In[ ]:


to_drop = ['State Code', 'County Code', 'Site Num', 
           'Parameter Code', 'Latitude', 'Longitude',
          'Parameter Name', 'Sample Duration', 
           'Pollutant Standard']
df = pol_df.drop(to_drop, axis=1)
df.tail()


# In[ ]:


def get_uniques(df):
    
    for col in df.columns:
        if df[col].dtype=='object' and col != 'Date Local':
            print(f'Unique values for {col}: {df[col].unique()}')

get_uniques(df)


# In[ ]:


df.drop(['Datum', 'Units of Measure', 'Event Type', 
         'State Name', 'County Name', 'City Name',
        'CBSA Name'], axis=1, inplace=True)
df.loc[:, 'Date Local'] = pd.to_datetime(df['Date Local'])
df.set_index('Date Local', inplace=True)
df


# In[ ]:


t11 = df.loc[(df['Address']==df['Address'].unique()[0]) & (
    df['Method Name']==df['Method Name'].unique()[0])]
t12 = df.loc[(df['Address']==df['Address'].unique()[0]) & (
    df['Method Name']==df['Method Name'].unique()[1])]
t21 = df.loc[(df['Address']==df['Address'].unique()[1]) & (
    df['Method Name']==df['Method Name'].unique()[0])]
t22 = df.loc[(df['Address']==df['Address'].unique()[1]) & (
    df['Method Name']==df['Method Name'].unique()[1])]
plt.plot(t11['Arithmetic Mean'], label='parts per billion {}'.format(t11['Method Name'][0]))
plt.plot(t12['Arithmetic Mean'], label='parts per billion {}'.format(t12['Method Name'][0]))
plt.title('S02 particles concentraction at {}'.format(t11['Address'][0]))
plt.legend()
#plt.plot(t21['Observation Count'])
#plt.plot(t22['Observation Count'])


# In[ ]:


plt.plot(t21['Arithmetic Mean'], label='parts per billion {}'.format(t21['Method Name'][0]))
plt.plot(t22['Arithmetic Mean'], label='parts per billion {}'.format(t22['Method Name'][0]))
plt.title('S02 particles concentraction at {}'.format(t21['Address'][0]))
plt.legend()


# In[ ]:


plt.plot(t11['AQI'], label='method {}'.format(t11['Method Name'][0]))
plt.plot(t12['AQI'], label='method {}'.format(t12['Method Name'][0]))
plt.title('AQI at {}'.format(t11['Address'][0]))
plt.legend()


# In[ ]:


plt.plot(t21['AQI'], label='method: {}'.format(t21['Method Name'][0]))
plt.plot(t22['AQI'], label='method: {}'.format(t22['Method Name'][0]))
plt.title('AQI at {}'.format(t21['Address'][0]))
plt.legend()


# It is clear for the first station there is no difference in method of measurement. First attempt I tired it I have had missing AQI columns (which is not the case now and seems to be the case for some stations). If you will play around this dataset I have developed a simple model to impute those values since there is a strong link between maximum value and AQI. You might use it for your case.

# In[ ]:


# in case we will come back later
df = df.loc[(df['Address']==df['Address'].unique()[0]) & (
    df['Method Name']==df['Method Name'].unique()[0])]
df


# ### Preprocessing

# In[ ]:


df.drop(['Address', 'Method Name', 'Method Code', 'Local Site Name', 
         'Date of Last Change', 'POC', 'Observation Count', 
         'Observation Percent', '1st Max Hour'], axis=1, inplace=True)
df


# In[ ]:


df.isnull().sum()


# In[ ]:


# inputer_data = df.loc[df['AQI'].notnull()]
plt.plot(df.dropna()['Arithmetic Mean'][-100:], label='parts per billion')
plt.plot(df.dropna()['1st Max Value'][-100:], label='max value')
plt.plot(df.dropna()['AQI'][-100:], label='AQI')
plt.legend(loc = 'upper right')


# In[ ]:


# jovian.commit(project=project_name)
df #.loc[df.index > np.datetime64('2019-01-01')] 


# Nice! We can see a strong relationship between maximum observed particulates in a day and air quility index. This relationship seems to hold true to the mean parts per billion although this is not as appearent. Anyway, now we can impute our missing AQI much smarter. Let's build a tiny slightly better than linear model for that. Use it as needed (here we do not need this step)

# In[ ]:


def inpute_aqi(aqi, max_values, mv_pred, epochs=50):
    linreg = nn.Linear(1, 1)
    loss_f = F.mse_loss #.l1_loss
    opt = torch.optim.Adam(linreg.parameters(), lr=0.0001)
    if isinstance(aqi, np.ndarray):
        aqi_max = np.max(aqi)
        aqi = torch.from_numpy(aqi/aqi_max)
    if isinstance(max_values, np.ndarray):
        max_values_max = np.max(max_values)
        max_values = torch.from_numpy(max_values/max_values_max)
    if isinstance(mv_pred, np.ndarray):
        mv_pred_max = np.max(mv_pred)
        mv_pred = torch.from_numpy(mv_pred/max_values_max)
    for epoch in range(0, epochs):
        
        # Train with batches of data
        for x, y in zip(aqi, max_values):
            #print(x, y)
            out = F.relu(linreg(x))
            loss = loss_f(out, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(
                epoch+1, epochs, loss.item()))
    l = [round(linreg(x).item()*max_values_max) for x in mv_pred]
    return l

# if you have NaN values for AQI use similar procedure for 
# value imputation
'''
aqi = np.array([[x] for x in df.dropna()['AQI'].values], dtype=np.float32)
max_values = np.array([[x] for x in df.dropna()['1st Max Value'].values], dtype=np.float32)
mv_pred = np.array([[x] for x in df.loc[df['AQI'].isnull()]['1st Max Value'].values], dtype=np.float32)
#print(aqi[0], max_values[0], mv_pred[0])
l = inpute_aqi(aqi, max_values, mv_pred)
df.loc[df['AQI'].isnull(), 'AQI'] = l
df
'''


# In[ ]:


def rescale(df):
    for col in df.columns:
        scalers = []
        if np.max(df[col].values)>1:
            scaler = MinMaxScaler()
            scaler.fit(df[col].values.reshape(-1, 1))
            df.loc[:, col] = scaler.transform(df[col].values.reshape(-1, 1))
            scalers.append(scaler)
    return df, scalers

df, scalers = rescale(df)
df


# ## Dataset and DataLoader

# The key point here to keep in mind is that we should not shuffle our data since our goal is to predict values based on date. Still, for starters let us not overcomplicate the matter and build not too sophisticated dataset and model.

# In[ ]:


temp_df = df.copy()
temp_df = temp_df.reset_index().drop('Date Local', axis=1)
temp_df


# In[ ]:


class PollutionDataset(Dataset):
    
    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        
    def __len__(self):
        return self.frame.shape[0]
    
    def __getitem__(self, ind):
        x = torch.tensor(ind).float()
        y = self.frame.loc[ind, :].values.astype(np.float32)
        return x, y


# In[ ]:


test_date = temp_df.index[int(len(temp_df)*0.85)]
batch_size = 16
train_df = PollutionDataset(temp_df[:test_date])
val_df = PollutionDataset(temp_df[test_date:])
train_loader = DataLoader(train_df,
                         shuffle=False)
val_loader = DataLoader(val_df,
                         shuffle=False)


# In[ ]:


temp_df[:test_date]


# In[ ]:


class PollutionModel(nn.Module):
    
    def __init__(self, out_features=3):
        super().__init__()
        self.input = nn.Linear(1, 32)
        self.lin1 = nn.Linear(32, 64)
        self.norm1 = nn.BatchNorm1d(64)
        self.lin2 = nn.Linear(64, 128)
        self.norm2 = nn.BatchNorm1d(128)
        self.lin3 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 3)
    
    def forward(self, x):
        x = F.relu(self.input(x)) #.view(1, -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return F.relu(self.out(x))


# In[ ]:


model = PollutionModel()


# In[ ]:


epochs = 50
lr = 0.001

loss_f = F.mse_loss


# In[ ]:


train_loader.dataset[0]


# In[ ]:


def fit(model, epochs=50, lr=0.001, loss_f=F.mse_loss):

    optimizer = Adam(model.parameters(), lr=lr)
    total_loss = []
    for epoch in tqdm(range(epochs)):
        model.train()
        for sample in train_loader:
            # print(sample)
            x, y = sample
            model.zero_grad()
            out = model(x)
            loss = loss_f(out, y)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            optimizer.zero_grad()
    print(np.array(total_loss).mean())
    return model

model = fit(model)


# In[ ]:


def predict(model, val_loader):
    
    preds = []    
    model.eval()
    for sample in train_loader:
        x, y = sample
        model.zero_grad()
        out = model(x)
        preds.append(out.detach().numpy())
    return np.array(preds)

preds = predict(model, val_loader)
temp_df[test_date:]['AQI'], [x[-1] for x in preds]


# That did not quite work. Why? First, our model does not know it is a sequencial data, these integers index were our sequence. Second, the integers themselves were treated as pure numerical values. Lastly, we would have some predictions if we feed the model the data it has seen before, i.e. numbers up to 910, other values our model not just has not seen, it does not know what to do with them. We need much smarter approach.

# We need our model to consume sequence of elements (input) and make it predict the next item in the sequence (output). So here we will define some utils to help us achieve what we want. For now let's restrict our task to predict 'AQI' column only. Later we can expend the approach to other values too. Check out <a href='https://www.curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/'>this</a> blog post to see where I've insiration from.

# In[ ]:


def get_sequences(df, seq_len):
    xs = []
    ys = []
    for i in range(len(df)-seq_len-1):
        x = df[i:(i+seq_len)]
        y = df[i+seq_len]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

s_len = 7
test_size = int(len(df)*0.85)
train_seq = get_sequences(df.loc[:df.index[test_size], 'AQI'], s_len)
val_seq = get_sequences(df.loc[df.index[test_size]:, 'AQI'], s_len)
train_seq[0].shape, train_seq[1].shape


# In[ ]:


class StatelessModel(nn.Module):
    
    def __init__(self, n_features, n_hidden, seq_len=7, 
                 n_layers=2, out_features=1):
        super().__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(n_features, 
                            n_hidden, 
                            n_layers,
                            dropout=0.5)
        self.linear = nn.Linear(n_hidden, out_features)
    
    def reset_hidden_state(self):
        self.hidden = (torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
                       torch.zeros(self.n_layers, self.seq_len, self.n_hidden))
    
    def forward(self, x):
        #print(type(x))
        x = torch.tensor(x).view(1, self.seq_len, -1)
        lstm_out, _ = self.lstm(x.float(), self.hidden)
        last_time_step = lstm_out.view(
            self.seq_len, len(x), self.n_hidden)[-1]
        x = self.linear(last_time_step)
        return torch.flatten(x)


# In[ ]:


def fit_lstm(model, data, epochs=50, 
             lr=0.0001, loss_f=F.mse_loss):

    optimizer = Adam(model.parameters(), lr=lr)
    x, y = data
    model.train()
    for epoch in range(epochs):
        total_loss = []
        for x_seq, y_seq in tqdm(zip(x, y)):
            model.zero_grad()
            model.reset_hidden_state()
            out = model(np.array([[i] for i in x_seq]))
            #print(out, y_seq)
            loss = loss_f(out, torch.tensor([y_seq]).float())#.float()
            #print(loss)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            optimizer.zero_grad()
        if epoch+1 % 10 ==0:
            print(np.array(total_loss).mean())
    return model


# In[ ]:


model = StatelessModel(1, 256)
# batch_size = 14
# train_loader = DataLoader(train_seq, shuffle=False, batch_size=batch_size)
# val_loader = DataLoader(val_seq, shuffle=False, batch_size=batch_size)
model = fit_lstm(model, train_seq)


# In[ ]:


def evaluate(model, val_set):
    x, y = val_set
    diviance = []
    model.eval()
    total_loss = []
    for x_seq, y_seq in tqdm(zip(x, y)):
        #model.zero_grad()
        model.reset_hidden_state()
        out = model(np.array([[i] for i in x_seq]))
        #print(out, y_seq)
        loss = loss_f(out, torch.tensor([y_seq]).float())
        diviance.append(np.abs(out.detach().item()-y_seq))
        total_loss.append(loss.item())
        #optimizer.zero_grad()
    mean_loss = np.array(total_loss).mean()
    mean_div = np.array(diviance).mean()
    return mean_loss, mean_div

evaluate(model, val_seq)


# In[ ]:


import jovian
jovian.commit(project=project_name)


# In[ ]:




