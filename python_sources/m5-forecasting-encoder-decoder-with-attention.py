#!/usr/bin/env python
# coding: utf-8

# # M5 Forecasting Encoder Decoder Model with Attention using Pytorch
# * ## Teaching an AI to Forecast
# * ## It has achieved a score of 0.603 in just 16 epochs( It has figured out exponential smoothening i guess)
# * ## It took 9 hours to complete 16 epochs though XD ( Slow learner)
# * ## It is performing well but is so scared of overfitting XD
# * ## Help my network only 16 days left!!
# * ## Run the last cell to visualize the predictions of the network and give your input
# 
# Upcoming Developments stay tuned :
# 1. Implementing multiple layer bidirectional LSTM
# 2. Implementing custom loss based on negative binomial distributions ( Lets see if we can get uncertainity)
# 3. Visualizing attention mechanism ( going to be difficult)
# 4. Trying dropout layers
# 5. Refining the input features
# 6. Any of your suggestions ( Top priority!!! :D)
# 
# Disclaimer : I am a simple guy trying complex things forgive me if I get some of the technicalities wrong.
# This is my first kaggle notebook :). And I am a mechanical engineer with and MBA in analytics tryin to get a grip on coding

# # **A Gentle Introduction**
# Time Series forecasting has always been a challenge it has evolved from simple statistical techniques like exponential smoothening, ARIMA , SARIMA, SARIMAX to using machine learning regression models based on algorithms like Xgboost , LGBM , linear regression, RandomForest etc.
# 
# The earlier statistical techniques like exponential smoothening,ARIMA and SARIMA had a limitation as it was designed on a single feature that is the time series itself as the input. To put it simply we could not feed it explicit features like the weather , holidays, or for example in our M5 forecasting dataset the selling price of the item. This limitation was overcome by machine learning models which could also include these explicit features to train the model.
# But as the features started increasing the Machine learning models started overfitting also they faced a problem of autocorrelation causing them to act funnily in some cases.
# 
# Neural networks trie to address these limitations of machine learning solutions but at the cost of the difficulty of training such networks.
# 
# In the below solution we adopted a novel approach by using an encoder decoder model to solve the M5 Forecasting problem
# The solution below assumes that in the rawest form times series is just a pattern and when we do forecasting we look at the input pattern which is a sequence of numbers and accordingly predict the future pattern/sequence.
# While making the predictions we also use additional inputs such as date features ( is it a holiday? which day is  it? which month is it) we also keep in mind the amount to variation in the time series ( is it fluctuating a lot? is there seasonality? is it stagnant?) and we also ask questions about the underlying item ( Which store is it kept? what type of item is it etc)
# 
# **Static Features** : If you give it a thought properties of the underlying item like the store in which its kept , the type of the item, are static features that dont change with each timestep.( we will call them static inputs)
# 
# **Dynamic features** :The other inputs like selling price and date features change with each time step. ( We will call them dynamic inputs)
# 
# The ability of a encoder decoder model to capture both these static and dynamic features and treat them accordingly gives it the edge over machine learning solutions

#  ## The Encoder Decoder Design
# *People not familiar with it can think of it as models that take in a sequence/pattern and based on it generates a sequence/pattern and hence they are also know as seq2seq models*
# 
# In our approach we will use a simple neural network to capture these static features and feed them in the hidden state of the encoder. The encoder will then go across the timeseries (1913 time steps) taking the demand concatenated with the selling price and date features as the input. Finally the decoder will make predictions step by step the input will be the ouput from previous state ( or ground truth in case of teacher enforcing) , concatenated with the date features and sale prices. We will also concat attention to the hidden state before we make the output predictions ( 28 time steps).

# # The Code
# 
# Components:
# * The Model Class : M5_Encoder_Decoder
# * Transformer Class : MinMaxTransformer
# * Dataloader Class : M5dataloader ( Given a batch of ids loads the required tensors for the ANN , encoder and decoder respectively)
# (*Yup it's a batched implementation!! Vectorization for the win!! I also wanted to include padding but couldnt due to Pytorch not supporting forward padding.
# I could manually do it though. Lets see*)
# * Batch trainer function : train_batch
# * Training helper function : train_setup
# * Plotting function : get_plots
# * Prediction function : Helps in postprocessing of decoder outputs to a usable dataframe format
# * Validation score fucntion
# 
# I'll add more details but thats a brief on the content of the code
# People interested can through the code I would also be happy to provide any clarifications
# 
# *Lets jump down to the interesting part for now!!!*
# **Go to model training those who dont want to see copious amounts of code!!! XD**
# 
# 
# 

# In[ ]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import random


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
date_features = pd.read_csv('/kaggle/input/m5features/date_features.csv')
sales_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
ann_features = pd.read_csv('/kaggle/input/m5features/ann_features.csv')
df['id'] = df['id'].str[:-11]
sales_prices['sell_price'] = MinMaxScaler().fit_transform(sales_prices['sell_price'].to_numpy().reshape(-1,1))
sales_prices['id'] = sales_prices['item_id'] + "_"  + sales_prices['store_id'] 
sales_prices.drop(['item_id','store_id'], axis =1, inplace = True)


# # A look at the Neural Network(ANN) features and date features

# In[ ]:


date_features.head()


# In[ ]:


ann_features.head()
#std_x , mean_x ....skew_x are at the item level
#std_y , mean_y and skew_y are at the category level we can add more such encodings if needed at the store level and state level
#Help : Suggest me more aggregate properties of the time series


# # Data Transformer :
# The purpose of the below transformer is to scale **each** of the time series based on the max and min values from d_1 to d_1913 we also later use this transformer to reverse transform the output values. We could have used a minmaxtransformer from sklearn but then I would have to individually apply it on each of the time series and also store the objects in a list fo future use.
# 
# Note : Though a min max transformer scales from 0-1 there can be greater than 1 values between dates d_1914 and d_1941 as the demand may increase with time.
# This is important to note as we cannot apply sigmoidal function to the output.

# In[ ]:


class MinMaxtransformer():
    ''' A class to scale the time series data for each item_id'''
    def __init__(self,d_x,d_y, info = None):
        self.d_x = d_x
        self.d_y = d_y
        if info is None :
            self.info = pd.DataFrame({'id': [],'min':[],'max':[]})
        else :
            self.info = info
    
    def fit(self, df):
        '''Will store in min and max values of the rows in a info dataframe'''
        self.info['id'] = df['id']
        self.info['max']= df.loc[:,self.d_x:self.d_y].max(axis=1)
        self.info['min']= df.loc[:,self.d_x:self.d_y].min(axis=1)
        self.info['maxdiffmin'] = self.info['max'] - self.info['min']
    
    def transform(self , df, d_x = None ,d_y = None):
        if d_x == None or d_y == None :
            d_x = self.d_x
            d_y = self.d_y
        df = pd.merge(df,self.info, on ='id', how ='left')
        for col in df.loc[:,d_x:d_y].columns:
            df[col] = (df[col] - df['min'])/(df['maxdiffmin'])
        df.drop(['min','max', 'maxdiffmin'],axis =1, inplace = True)
        return df
    
    def reverse_transform(self, df, d_x =None,d_y = None, round_ = False):
        df = pd.merge(df,self.info, on ='id', how ='left')
        if d_x == None or d_y == None :
            d_x = self.d_x
            d_y = self.d_y
        for col in df.loc[:,d_x:d_y].columns:
            df[col] = df[col] * df['maxdiffmin'] + df['min']
            if round_ :
                df[col] = round(df[col])
        df.drop(['min','max', 'maxdiffmin'],axis =1, inplace = True)
        return df
    


# # **Fitting the transformer we will use it in the data loader**

# In[ ]:


mmt  = MinMaxtransformer('d_1','d_1913')
mmt.fit(df)


# # Train Validation Split
# 
# Its important to understand for our model our model is predicting an output sequence given an input sequence hence the below split

# In[ ]:


from sklearn.model_selection import train_test_split
trainids, testids = train_test_split(df.loc[:,['id']], train_size = 0.8, random_state = 1234)
trainids = trainids['id'].to_list()
testids = testids['id'].to_list()


# In[ ]:


len(trainids)


# In[ ]:


len(testids)


# # The Dataloader :
# I think showing the working will be better than explaining.
# 

# In[ ]:


class M5dataloader():
    def __init__(self, ids, batch_size):
        '''IDs to be passed in list format'''
        self.ids = ids
        self.iteration = 0
        self.batch_size = batch_size
        
        
    def get_data(self, df,date_features,sales_prices, ann_features):
        start = (self.iteration * self.batch_size)% len(self.ids)
        end = start + self.batch_size
        self.iteration +=1
        if( end < len(self.ids)):
            batchidlist = self.ids[start:end]
        else :
            end = end%len(self.ids)
            batchidlist = [id for id in self.ids[start:]] + [ id for id in self.ids[:end]]
        filt = df['id'].isin(batchidlist)
        batch_data = df.loc[filt,:].drop(['item_id','dept_id','cat_id', 'store_id','state_id'], axis = 1 )
        batch_data =  mmt.transform(batch_data,'d_1', 'd_1941')
        encoder_data = batch_data.loc[:,'id':'d_1913']
        decoder_data = batch_data.loc[:,'d_1914':'d_1941']
        decoder_data['id'] = encoder_data['id']
        decoder_data = pd.concat([batch_data.loc[:,['id']], batch_data.loc[:,'d_1914':'d_1941']], axis=1 )
        encoder_data = encoder_data.melt(id_vars =['id'], value_vars = encoder_data.columns.to_list()[1:],var_name ='d', value_name ='count')
        decoder_data = decoder_data.melt(id_vars =['id'], value_vars = decoder_data.columns.to_list()[1:],var_name ='d', value_name ='count')
        encoder_data = pd.merge(encoder_data,date_features,how = 'left',on ='d').drop(['Unnamed: 0'], axis =1)
        decoder_data = pd.merge(decoder_data, date_features, how = 'left' , on = 'd').drop(['Unnamed: 0'], axis =1)
        encoder_data = pd.merge(encoder_data,sales_prices,how = 'left',on =['id','wm_yr_wk']).drop(['date','wm_yr_wk'], axis =1)
        decoder_data = pd.merge(decoder_data,sales_prices,how = 'left',on =['id','wm_yr_wk']).drop(['date','wm_yr_wk'], axis =1)
        encoder_data['id'] = encoder_data['id'] + encoder_data['d']
        decoder_data['id'] = decoder_data['id'] + decoder_data['d']
        encoder_data.set_index('id', inplace = True)
        decoder_data.set_index('id', inplace = True)
        encoder_data.drop('d', axis = 1, inplace = True)
        ground_truth = decoder_data.loc[:,['count']]
        decoder_data.drop(['d', 'count'], axis = 1, inplace = True)
        encoder_data = encoder_data.fillna(value = 0)
        encoder_data = torch.tensor(encoder_data.to_numpy().reshape(int(encoder_data.shape[0]/self.batch_size),self.batch_size,encoder_data.shape[1]),dtype=torch.float32)
        decoder_data = torch.tensor(decoder_data.to_numpy().reshape(int(decoder_data.shape[0]/self.batch_size),self.batch_size,decoder_data.shape[1]),dtype=torch.float32)
        ground_truth = torch.tensor(ground_truth.to_numpy().reshape(int(ground_truth.shape[0]/self.batch_size),self.batch_size,ground_truth.shape[1]),dtype=torch.float32)
        filt2 = ann_features['id'].isin(batchidlist)
        ann_data = ann_features.loc[filt2,:]
        ann_data.set_index('id', inplace = True)
        ann_data = torch.tensor(ann_data.to_numpy().reshape(self.batch_size, ann_data.shape[1]),dtype=torch.float32)
        return (encoder_data, decoder_data,ground_truth,ann_data)
       
    
        #return decoder_input, encoder_input , ANN_input , ground_truth


# Given a set of ids it will load the tensors needed for the neural network training or evaluation
# Everytime the get_data function is called it loads a different batch from the givenids till it goes through the entire dataset

# In[ ]:


dataloader = M5dataloader(trainids,10) # initialize it


# In[ ]:


encoder_data, decoder_data, ground_truth,ann_data = dataloader.get_data(df,date_features,sales_prices,ann_features) 
# getting input dimensions of encoder , decoder and ann and testing the data loader
ann_input_size = ann_data.shape[1] 
enc_input_size= encoder_data.shape[2]
dec_input_size= decoder_data.shape[2]+1 # we add one to include output from previous state as input


# In[ ]:


#Lets look at the shapes of the tensor
print('\n ANN tensor shape :', ann_data.shape)
print('\n encoder tensor shape :', encoder_data.shape)
print('\n decoder tensor shape :', decoder_data.shape)
print('\n ground_truth tensor shape :', ground_truth.shape)


# # **The Model**

# In[ ]:


class M5_EncoderDecoder(nn.Module):
    
    def __init__(self, ann_input_size,enc_input_size, dec_input_size, hidden_size, output_size = 1, verbose=False):
        super(M5_EncoderDecoder, self).__init__()
        self.ann_input_size = ann_input_size
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size + hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.s2h = nn.Sequential(nn.Linear(ann_input_size,96), 
                                 nn.ReLU(),
                                 nn.Linear(96,hidden_size))
        
        self.encoder_rnn_cell = nn.GRU(enc_input_size, hidden_size)
        self.decoder_rnn_cell = nn.GRU(dec_input_size+hidden_size, hidden_size)
        
        self.h2o = nn.Linear(hidden_size, output_size)
        self.verbose = verbose
        self.U = nn.Linear(self.hidden_size, self.hidden_size)
        self.W = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size,1)
    
        
    def forward(self, ann_data, encoder_data, decoder_data, ground_truth = None, steps = 28 , device = device):
        
        # encoder
        ann_data =ann_data.to(device)
        encoder_data =encoder_data.to(device)
        decoder_data = decoder_data.to(device)
        if ground_truth is not None:
            ground_truth = ground_truth.to(device)
        batch_size = encoder_data.shape[1]
        hidden = self.s2h(ann_data.float())
        encoder_outputs,hidden = self.encoder_rnn_cell(encoder_data.float(),hidden.view(1,batch_size,self.hidden_size).float())
        
        U = self.U(encoder_outputs)
        initial_ground = encoder_data[encoder_data.shape[0]-1,:,0].view(1,encoder_data.shape[1],1).to(device)
        # getting the last known decoder count from encoder data (in most cases it is initialized randomly but in this case we know what the firs output to the decoder should be)
        flag = 1 # some stupid coding XD 
        if ground_truth is None:
            ground_truth = initial_ground
            flag = 0
        else :
            ground_truth = torch.cat((initial_ground,ground_truth),0)
        
        outputs = []
        for i in range(steps) :
            W  = net.W(hidden.repeat(encoder_data.shape[0],1,1))
            V= net.V(torch.tanh(U+W))
            alpha = F.softmax(V, dim=0)
            attn_applied = torch.bmm(alpha.T.transpose(0,1),encoder_outputs.transpose(0,1))
            if (i == 0):
                decoder_input = torch.cat((ground_truth[i,:,:].float(),decoder_data[i,:,:].float(),attn_applied.transpose(0,1)[0,:,:].float()),1)
            
            if(i > 0):
                if flag != 0 :
                    decoder_input = torch.cat((ground_truth[i,:,:].float(),decoder_data[i,:,:].float(),attn_applied.transpose(0,1)[0,:,:].float()),1)
                    
                else:
                    decoder_input = torch.cat((out.float(),decoder_data[i,:,:].float(),attn_applied.transpose(0,1)[0,:,:].float()),1) 
                    # no need to use i-1 as we have added a timestep inthe form of initial ground
            
            _ ,hidden = self.decoder_rnn_cell(decoder_input.view(1,decoder_input.shape[0],decoder_input.shape[1]).float(), hidden)
            out = self.h2o(hidden) 
            out = out.view(out.shape[1],1)
            outputs.append(out) # verify dimensions
        return outputs


# In[ ]:


def train_batch(net, batch,batch_size,opt,criterion,device, teacher_force = False):
    net.train().to(device)
    opt.zero_grad()
    encoder_data, decoder_data, ground_truth,ann_data = batch
    ground_truth = ground_truth.to(device)
    if teacher_force :
        outputs = net.forward(ann_data, encoder_data, decoder_data, ground_truth)
    else :
        outputs = net.forward(ann_data,encoder_data,decoder_data)
    
    loss = torch.zeros(1,1).float().to(device)
    for i, output in enumerate(outputs):
        loss += criterion(outputs[i], ground_truth[i,:,:]).float()
    loss.backward()
    opt.step()
    
    return loss/batch_size


# In[ ]:


def prediction(testids, net, df, sales_prices, date_features, ann_features,round_ = False, steps = 28,idstart = 1914):
    testloader = M5dataloader(testids,len(testids))
    encoder_data, decoder_data, ground_truth,ann_data = testloader.get_data(df,date_features,sales_prices,ann_features)
    outputs = net.forward(ann_data,encoder_data,decoder_data)
    pred= pd.DataFrame({'id' : testids})
    for i, output in enumerate(outputs):
        pred['d_' + str(idstart + i)] = output.cpu().data.numpy()
    
    start = 'd_' + str(idstart)
    end = 'd_' + str(idstart + 27) 
    pred = mmt.reverse_transform(pred,start,end, round_ = round_)
    pred.set_index('id', inplace = True)
    return pred


# In[ ]:


def actual_values(testids,df,steps = 28, idstart = 1914):
    df.set_index('id',inplace = True)
    start = 'd_' + str(idstart)
    end = 'd_' + str(idstart + steps -1) 
    act = df.loc[testids,start:end]
    df.reset_index(inplace = True)
    return act


# In[ ]:


def validation_error(pred,act):
    return np.square(pred.to_numpy() - act.to_numpy()).sum()/act.to_numpy().size


# In[ ]:


def get_plots(ids, net):
    if len(ids) > 25:
        return "the number of ids in the list exceeds the limit of 25"
    
    trainhead = actual_values(ids, df,steps = 100 , idstart = 1814).T
    testvalues = actual_values(ids, df,steps = 28 , idstart = 1914).T
    predictions = prediction(ids, net, df, sales_prices, date_features, ann_features, round_ = False).T
    b = ['_encoder','_actual', '_pred']
    for i , x in enumerate([trainhead, testvalues, predictions]):
        x.columns = [x for x in map(lambda x: x + b[i],x.columns.to_list())]
        x.reset_index(inplace = True)
        x.rename(columns={'index' : 'Days'}, inplace = True)
        x['Days'] = x['Days'].str[2:].apply(int)
        
    fig, ax = plt.subplots(nrows = len(ids), ncols = 1,figsize=(25,4 * len(ids)))
    for i in range(len(ids)):
        if len(ids) == 1:
            trainhead.plot(x = 'Days',y=[i+1],ax=ax);
            testvalues.plot(x = 'Days',y=[i+1],ax=ax);
            predictions.plot(x = 'Days',y=[i+1],ax=ax);
        else :
            trainhead.plot(x = 'Days',y=[i+1],ax=ax[i]);
            testvalues.plot(x = 'Days',y=[i+1],ax=ax[i]);
            predictions.plot(x = 'Days',y=[i+1],ax=ax[i]);


# In[ ]:


def get_submission(idlist,net, batch = 200):
    
    submission = []
    for i in range(len(idlist)//batch + int(len(idlist)%batch !=0)):
        print("Iteration ", i, "/", len(idlist)//batch + int(len(idlist)%batch !=0))
        start = i * batch
        end = start + batch
        if(end > len(idlist)): 
            end = len(idlist)
        batchidlist = idlist[start:end] 
        pred = prediction(batchidlist, net, df, sales_prices, date_features, ann_features, round_ = False)
        submission.append(pred) 
    return pd.concat(submission, axis =0)


# In[ ]:


def train_setup(net,trainids,testids,validation = False,plots = False, lr = 0.01, n_batches = 1000, batch_size = 200, momentum = 0.9, display_freq=5, device = device,test_batch_size = 100):
    
    net = net.to(device)
    criterion = nn.MSELoss()
    opt = optim.Adam(net.parameters(), lr=lr)
    teacher_force_upto = n_batches//3
    trainloader = M5dataloader(trainids,batch_size)
    loss_arr = np.zeros(n_batches + 1)
    if validation :
        valid_error = []
        valid_xaxis =[]
    for i in range(n_batches):
        batch = trainloader.get_data(df, date_features, sales_prices, ann_features) 
        loss_arr[i+1] = (loss_arr[i]*i + train_batch(net,batch,batch_size, opt, criterion, device = device, teacher_force = False ))/(i + 1)
        
        if i%display_freq == display_freq-1:
            if plots :
                clear_output(wait=True)
            
            if validation :
                ids = random.sample(testids,test_batch_size)
                pred = prediction(ids, net, df, sales_prices, date_features, ann_features, round_ = False)
                act = actual_values(ids, df,steps = 28 , idstart = 1914)
                valid_xaxis.append(i)
                v = validation_error(pred,act)
                valid_error.append(v)
                print('Validation error  ', v, " per observation as tested on ", test_batch_size, " random samples from test set")
            print('Epoch ',(i*batch_size)/len(trainids),' Iteration', i, 'Loss ', loss_arr[i])
            
            if plots:
                fig, axes = plt.subplots(nrows =1, ncols=2 , figsize=(20,6))
                axes[0].set_title('Train Error')
                axes[0].plot(loss_arr[2:i], '-*')
                axes[0].set_xlabel('Iteration')
                axes[0].set_ylabel('Loss')

                if validation :
                    axes[1].set_title('Validation error')
                    axes[1].plot(valid_xaxis, valid_error,'-*')
                    axes[1].set_xlabel('Iteration')
                    axes[1].set_ylabel('validation error')
                plt.show()
                print('\n\n')
        if(i%100 ==0):    
            torch.save(net.state_dict(), 'model.pt')
        if(i%500 ==0):
            filename = str(i) + 'model.pt'
            torch.save(net.state_dict(), filename)
    filename = str(i) + 'model_4K_iter.pt'
    torch.save(net.state_dict(), filename)    
    return (loss_arr,net)


# # Model Training :

# In[ ]:


#Training cell

#net = M5_EncoderDecoder(ann_input_size = ann_data.shape[1] ,enc_input_size= encoder_data.shape[2], dec_input_size= decoder_data.shape[2]+1, hidden_size=126)
#losses,net = train_setup(net,trainids,testids, lr = 0.01)


# I have already trained the model batch_size = 200 , iterations = 4000, learning_rate = 0.01 , teacher enforced upto 2k iterations
# the train_setup function supports plots to monitor loss and also validation feel free to play with it


# In[ ]:


# We will load my pretrained model
net = M5_EncoderDecoder(ann_input_size = ann_data.shape[1] ,enc_input_size= encoder_data.shape[2], dec_input_size= decoder_data.shape[2]+1, hidden_size=126)
net.load_state_dict(torch.load('../input/m5models-2/model_4K_iter.pt', map_location = device))
net.eval()
net.to(device)


# # Model Performance on Test Data
# Lets look at 20 random plots of the 4000 iteration model
# The plot is of 128 days
# This is where I need your help !!
# Please suggest techniques to improve the performance :
# * Should i decrease encoder timesteps ? Is 1913 steps huge?
# * Should I replace GRU cell with LSTM cell?
# * Should I add more layers to the LSTM cell?
# * Should I decrease size of input features?
# 
# Kaggle Army help me in this time of peril so that we can revolutionize forecasting !!!!!
# 
# 15 Days to go...!!!!
# 
# 

# In[ ]:


get_plots(random.sample(trainids,20),net)


# In[ ]:


idlist = df['id'].to_list()
submission_validation = get_submission(idlist,net)
submission_validation.to_csv("submission.csv")
#cant submit this submission file coz its not in submission format I appended evaluation with 0's to it and submitted to get an WRMSE of 0.603


# In[ ]:





# In[ ]:





# In[ ]:




