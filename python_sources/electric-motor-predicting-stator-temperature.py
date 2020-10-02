#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this notebook we will try to predict the stator temperature of a permanent magnet synchronous motor (PMSM) deployed on a test bench.
# As suggested in the explanatory notes of the dataset, the metrics "rotor temperature", "stator temperature" and "torque" are hard to measure in practical applications. A sufficiently accurate prediction model would therefore eliminate the need for actual measurements to determine the stator temperature.
# 
# Please feel free to comment on any possible additions or improvements.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bottleneck as bn # library used for moving average

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense, Dropout,BatchNormalization
from keras.regularizers import l2
from keras.layers import LSTM
from keras.layers import Dropout


# # Exploring the data

# In[ ]:


# load the dataset into a pandas dataframe
dataset = pd.read_csv('../input/electric-motor-temperature/pmsm_temperature_data.csv')
dataset.head()


# In[ ]:


# check if the dataset contains NaN values
dataset.isnull().values.any()


# In[ ]:


dataset.describe()


# In[ ]:


# plot the boxplots of all features
plt.tight_layout(pad=0.9)
plt.figure(figsize=(20,15)) 
plt.subplots_adjust(wspace = 0.2)
nbr_columns = 4 
nbr_graphs = len(dataset.columns)
nbr_rows = int(np.ceil(nbr_graphs/nbr_columns)) 
columns = list(dataset.columns.values) 
with sns.axes_style("whitegrid"):
    for i in range(0,len(columns)-1): 
        plt.subplot(nbr_rows,nbr_columns,i+1) 
        ax1=sns.boxplot(x= columns[i], data= dataset, orient="h",color=sns.color_palette("Blues")[3]) 
    plt.show() 


# In[ ]:


# plotting the Correlation matrix
fig = plt.figure(figsize=(12,9))
sns.heatmap(dataset.corr(),annot=True)
plt.show()


# In[ ]:


# print the list of all testruns
profile_id_list = dataset.profile_id.unique()
print(profile_id_list)
print("amount of test runs: {0}".format(profile_id_list.size))


# After exploring the dataset, following observations can be made: 
# * The various testruns (52 in total) are labelled by the profile_id (as mentioned in the dataset description). The indexes for these testruns however, are not incremental.
# * The dataset description provides us with no references to the units used for each of the samples, making it harder to interpret the values measured. 
# * When we look at the statistical overview of the dataset and the histograms, it seems the dataset already has had some kind of normalisation. 
# * Features like torque, motor speed, rotor temperature (pm), stator yoke, winding and tooth temperature (resp. stator_yoke, stator_winding, stator_tooth), coolwater temperature (coolant) are all reasonably self-explanatory. The active and reactive current and voltage (resp. i_d,i_q,u_d,u_q) of the PMSM however, require some background knowledge of how a synchronous motor works.
# * The ambient temperature is measured by a thermal sensor located closely to the stator (as stated in the explanatory notes of the dataset). We can therefore assume that this will have an impact on the selfcooling capacity of the motor. A higher ambient temperature will probably result in a higher temperature for both the motor's stator and rotor.
# * The correlation matrix shows that there is a significant correlation between the three different stator temperatures.
# 

# # Predicting the stator temperature
# We already discovered a significant correlation between the stator winding, yoke and tooth temperature. This is of course due to the fact that the stator winding is wounded round the stator tooth which in his turn is connected to the stator yoke. 
# To get a better insight into the relationship between the three features we will plot the feature values for various randomly selected testruns.

# In[ ]:


# plotting 'stator_yoke','stator_tooth','stator_winding' for a random set of testruns
columns = ['stator_yoke','stator_tooth','stator_winding']
profile_id_list = np.random.choice(profile_id_list, size=8, replace=False)    
nbr_column = 2 
nbr_graph= len(profile_id_list) 
nbr_row = int(np.ceil(nbr_graph/nbr_column)) 
kolomlijst = list(dataset.columns.values) 
plt.figure(figsize=(30,nbr_row*5)) 
    
with sns.axes_style("whitegrid"):    
    for i in range(0,nbr_graph): 
        plt.subplot(nbr_row,nbr_column,i+1) 
        temp = dataset.loc[dataset['profile_id'] == profile_id_list[i]]
        temp = temp.loc[:,columns]
        temp = temp.iloc[::100, :]
        ax1=sns.lineplot(data=temp.loc[:,columns], 
                        dashes = False,
                        palette=sns.color_palette('Dark2',n_colors=len(columns)))
        ax1.set_title("profile id: {0}".format(profile_id_list[i]))
    plt.show    


# The lineplots confirm that all three temperatures follow the same trend. The stator winding temperatures shows the biggest variation followed by the stator tooth and stator yoke temperature. This is especially noticeable when there is a lot of variation in the stator winding temperature. If this is the case, the stator tooth and yoke temperatures follow a smoother path than the temperature recorded on the stator winding. In other words, the heat dissipated by the stator windings takes some time to heat up the stator tooth and yoke due to the thermal inertia of both stator parts.
# 
# A second observation that can be made on the various lineplots is that sometimes the stator yoke temperature has a higher value than the stator winding. Because of the presumed normalisation mentioned earlier, we can not determine whether this is due to the normalisation method used or whether these values actually represent higher temperatures measured on the stator yoke. 

# In[ ]:


# plotting the stator winding temperature in comparison to torque and motorspeed
profile_id = 6
feat_plot_1 = ['stator_winding']
feat_plot_2 = ['torque','motor_speed']
temp = dataset.loc[dataset['profile_id'] == profile_id]
temp = temp.iloc[::10, :]

with sns.axes_style("whitegrid"):
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)
    ax1 = sns.lineplot(data=temp.loc[:,feat_plot_1], dashes = False,
                       palette=sns.color_palette('Dark2',n_colors=len(feat_plot_1)),linewidth=0.8)
    ax2 = fig.add_subplot(212)
    ax2 = sns.lineplot(data=temp.loc[:,feat_plot_2], dashes = False,
                       palette=sns.color_palette('Dark2',n_colors=len(feat_plot_2)),linewidth=0.8)
    plt.show()


# In the second part of the above testrun there seems to be a relation between the stator temperatures, torque and motorspeed. When the torque and/or motorspeed is increased the temperature rises and vice versa. The temperatures follow the change in torque and/or motorspeed in what seems a  $1-e^{t}$ type of equation. However, looking at the first part of the plot, this is not always the case. Even with constant torque and motor speed the stator winding temperature shows several sudden changes in temperature. In other words, there seem to be one or more other variables which have an impact on the stator temperature.

# # Predicting the Stator winding temperature
# Because measuring the torque, rotor and stator temperatures of the electromotor is not reliable nor economically feasible in commercial applications (as stated in the dataset description), we will try to predict the stator temperature by using the other available features in the dataset.  We will start with removing the torque, rotor and stator temperatures from our dataset and use the stator winding temperature as our target value. 
# 
# Once this is done, we will train multiple models to predict the correct stator winding temperature as an output value for the input variables given.

# ##  Random Forest Regressor
# 
# We start by training a Random Forest Regressor model.  

# In[ ]:


# Seperating input and output variables
X = dataset.drop('torque', axis=1).loc[:,'ambient':'i_q'].values 
y = dataset.loc[:,'stator_winding'].values 

# split up in training and test data
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


X.shape


# In[ ]:


# training the Random Forest Regressor on the dataset
from sklearn.ensemble import RandomForestRegressor
RFR_model = RandomForestRegressor(n_estimators = 10, random_state = 0)
RFR_model.fit(X_train, y_train)


# In[ ]:


# Calculate MSE and MAE for the entire testset
y_pred = RFR_model.predict(X_test)
RFR_MSE = mean_squared_error(y_test, y_pred)
RFR_MAE = mean_absolute_error(y_test, y_pred)
print("MSE: {0}".format(RFR_MSE))
print("MAE: {0}".format(RFR_MAE))


# In[ ]:


# plot the true vs predicted values for multiple testruns
test_run_list = np.array([27,45,60,74])
#test_run_list = np.random.choice(profile_id_list, size=4, replace=False)]
output_value = 'stator_winding'
model = RFR_model

with sns.axes_style("whitegrid"):    
    fig, axs = plt.subplots(len(test_run_list),1,figsize=(20,len(test_run_list)*5),squeeze=False)
    
    for i in range(0,len(test_run_list)):
        X_plot = dataset.drop('torque', axis=1).loc[dataset['profile_id'] == test_run_list[i],'ambient':'i_q'].values 
        y_plot = dataset.loc[dataset['profile_id'] == test_run_list[i],output_value].values 
        y_pred_plot = model.predict(X_plot)

        time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])
        axs[i,0].plot(time,y_pred_plot,label='predict',color='red',alpha=0.4,linewidth=0.8)
        axs[i,0].plot(time,y_plot,label='True',color='black',linewidth=1)
        axs[i,0].legend(loc='best')
        axs[i,0].set_title("profile id: {0}".format(test_run_list[i]))
    plt.show()   


# The graphs depicted above show the true stator winding temperature in comparison to the values predicted by the Random Forest Regressor. As we can see, the Random Forest Regressor seems to be able to capture the global trend in the stator winding value but still there is a lot of noise in the predicted values, especially when the true signal shows a lot of variation. 
# To filter out the noise generated by the model, we will apply a moving average function to the output signal.

# In[ ]:


# This is the testrun we will use as an example for the evaluation of the models
# change this value to use a different testrun
choosen_example_testrun = 76


# In[ ]:


# plot the true vs predicted values for a choosen testrun without and with moving average smoothing:
profile_id = choosen_example_testrun
output_value = 'stator_winding'
model = RFR_model
moving_average_window = 100

X_plot = dataset.drop('torque', axis=1).loc[dataset['profile_id'] == profile_id,'ambient':'i_q'].values 
y_plot = dataset.loc[dataset['profile_id'] == profile_id,output_value].values 
y_pred_plot = model.predict(X_plot)
y_pred_plot_smooth = bn.move_mean(y_pred_plot,moving_average_window,1)
time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])

with sns.axes_style("whitegrid"):
    fig = plt.figure(figsize=(20, 10))
    
    ax1 = fig.add_subplot(211)
    ax1.plot(time,y_pred_plot,label='predict without smoothing',color='red',alpha=0.4,linewidth=0.8)
    ax1.plot(time,y_plot,label='True',color='black',linewidth=1)
    ax1.legend(loc='best')
    ax1.set_title("profile id: {0} without smoothing".format(profile_id))
    
    ax2 = fig.add_subplot(212)
    ax2.plot(time,y_pred_plot_smooth,label='predict with smoothing',color='red',alpha=0.8,linewidth=0.8)
    ax2.plot(time,y_plot,label='True',color='black',linewidth=1)
    ax2.legend(loc='best')
    ax2.set_title("profile id: {0} with smoothing".format(profile_id))
    
    plt.show()

# Calculate MSE and MAE for the choosen testrun without and with moving average smoothing:
MSE_RFR_model = mean_squared_error(y_plot, y_pred_plot)
MAE_RFR_model = mean_absolute_error(y_plot, y_pred_plot)
print("metrics without moving average smoothing:")
print("MSE: {0}".format(MSE_RFR_model))
print("MAE: {0}".format(MAE_RFR_model))
MSE_RFR_model = mean_squared_error(y_plot, y_pred_plot_smooth)
MAE_RFR_model = mean_absolute_error(y_plot, y_pred_plot_smooth)
print("metrics with moving average smoothing:")
print("MSE: {0}".format(MSE_RFR_model))
print("MAE: {0}".format(MAE_RFR_model))


# Combined with the moving average function, the Random Forest Regressor gives us a fairly accurate prediction of the stator winding temperature. The moving average has reduced the noise in the predicted output values and has had a positive impact on the MSE for this specific testrun.

# ## Feed Forward Neural network
# Let us now check whether a Feed forward neural network would do better in predicting the stator winding temperature. Just like with the Random Forest model, we will only use the actual values of the input variables to predict the stator winding temperature. 

# In[ ]:


# constructing and training the neural network
nr_epochs=200
b_size=1000

NN_reg_model = Sequential()
NN_reg_model.add(Dense(11, input_dim=X_train.shape[1], activation='relu'))
NN_reg_model.add(Dense(9, activation='relu'))
NN_reg_model.add(Dense(7, activation='relu'))
NN_reg_model.add(Dense(5, activation='relu'))
NN_reg_model.add(Dense(1))
NN_reg_model.compile(loss='mean_squared_error', optimizer='adam',metrics = ["mean_squared_error"])
history = NN_reg_model.fit(X_train, y_train, validation_split=0.33,epochs=nr_epochs, batch_size=b_size, verbose=0)


# In[ ]:


#plot the history of the model accuracy during training
plt.figure(figsize=(18,6))
ax1=plt.subplot(1, 2, 1)
ax1=plt.plot(history.history['mean_squared_error'],color='blue')
ax1=plt.plot(history.history['val_mean_squared_error'],color='red',alpha=0.5)
ax1=plt.title('model accuracy')
ax1=plt.ylabel('accuracy')
ax1=plt.xlabel('epoch')
ax1=plt.legend(['train', 'test'], loc='upper left')

# plot the history of the model loss during training
ax2=plt.subplot(1, 2, 2)
ax2=plt.plot(history.history['loss'],color='blue')
ax2=plt.plot(history.history['val_loss'],color='red',alpha=0.5)
ax2=plt.title('model loss')
ax2=plt.ylabel('loss')
ax2=plt.xlabel('epoch')
ax2=plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# Calculate MSE and MAE of the entire testset
y_pred = NN_reg_model.predict(X_test)
NN_MSE = mean_squared_error(y_test, y_pred)
NN_MAE = mean_absolute_error(y_test, y_pred)
print("MSE: {0}".format(NN_MSE))
print("MAE: {0}".format(NN_MAE))


# In[ ]:


# plot the true vs predicted values for multiple testruns
test_run_list = np.array([27,45,60,74])
#test_run_list = np.random.choice(profile_id_list, size=4, replace=False)]
output_value = 'stator_winding'
model = NN_reg_model

with sns.axes_style("whitegrid"):    
    fig, axs = plt.subplots(len(test_run_list),1,figsize=(20,len(test_run_list)*5),squeeze=False)
    
    for i in range(0,len(test_run_list)):
        X_plot = dataset.drop('torque', axis=1).loc[dataset['profile_id'] == test_run_list[i],'ambient':'i_q'].values 
        y_plot = dataset.loc[dataset['profile_id'] == test_run_list[i],output_value].values 
        y_pred_plot = model.predict(X_plot)

        time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])
        axs[i,0].plot(time,y_pred_plot,label='predict',color='red',alpha=0.4,linewidth=0.8)
        axs[i,0].plot(time,y_plot,label='True',color='black',linewidth=1)
        axs[i,0].legend(loc='best')
        axs[i,0].set_title("profile id: {0}".format(test_run_list[i]))
    plt.show()   


# In[ ]:


# plot the true vs predicted values for a choosen testrun without and with moving average smoothing:
profile_id = choosen_example_testrun
output_value = 'stator_winding'
model = NN_reg_model
moving_average_window = 100

X_plot = dataset.drop('torque', axis=1).loc[dataset['profile_id'] == profile_id,'ambient':'i_q'].values 
y_plot = dataset.loc[dataset['profile_id'] == profile_id,output_value].values 
y_pred_plot = model.predict(X_plot).flatten()
y_pred_plot_smooth = bn.move_mean(y_pred_plot,moving_average_window,1)
time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])

with sns.axes_style("whitegrid"):
    fig = plt.figure(figsize=(20, 10))
    
    ax1 = fig.add_subplot(211)
    ax1.plot(time,y_pred_plot,label='predict without smoothing',color='red',alpha=0.4,linewidth=0.8)
    ax1.plot(time,y_plot,label='True',color='black',linewidth=1)
    ax1.legend(loc='best')
    ax1.set_title("profile id: {0} without smoothing".format(profile_id))
    
    ax2 = fig.add_subplot(212)
    ax2.plot(time,y_pred_plot_smooth,label='predict with smoothing',color='red',alpha=0.8,linewidth=0.8)
    ax2.plot(time,y_plot,label='True',color='black',linewidth=1)
    ax2.legend(loc='best')
    ax2.set_title("profile id: {0} with smoothing".format(profile_id))
    
    plt.show()

# Calculate MSE and MAE for the choosen testrun without and with moving average smoothing:
MSE_RFR_model = mean_squared_error(y_plot, y_pred_plot)
MAE_RFR_model = mean_absolute_error(y_plot, y_pred_plot)
print("metrics without moving average smoothing:")
print("MSE: {0}".format(MSE_RFR_model))
print("MAE: {0}".format(MAE_RFR_model))
MSE_RFR_model = mean_squared_error(y_plot, y_pred_plot_smooth)
MAE_RFR_model = mean_absolute_error(y_plot, y_pred_plot_smooth)
print("metrics with moving average smoothing:")
print("MSE: {0}".format(MSE_RFR_model))
print("MAE: {0}".format(MAE_RFR_model))


# When we compare the Feed forward neural network with the Random Forest Regressor we can conclude that the neural network does not outperform the Random Forest in correctly predicting the stator temperature. (We could probably increase the performance by tweaking the neural network, but it doesn't seem very feasable to use a more complex solution like a Feed forward neural network when we achieve a descent and comparable performance using the Random Forest Regressor.)

# ## LSTM
# 
# We already discovered that the stator temperature is not only influenced by the current input variables but also by the previous states. We will therefore try to train an LSTM and check if it outperforms our earlier trained Random Forest.
# 
# In order to be able to train the LSTM we will need to transform our dataset to a 3D input matrix which includes a time window for every inputvalue per sample. 
# 

# In[ ]:


# plot the amount of samples per testrun
plt.figure(figsize=(18,5))
sns.countplot(x="profile_id", data=dataset,color=sns.color_palette("Blues")[2])
plt.show()


# In[ ]:


# find the testrun with the least amount of samples
max_batch_size = dataset.profile_id.value_counts().min()
profile_id_list = dataset.profile_id.unique()
print("list of testruns:")
print(profile_id_list)
print("smallest amount of samples in one testrun: {0}".format(max_batch_size))
print("testrun with smallest amount of samples: {0}".format(dataset.profile_id.value_counts().idxmin()))


# As shown in the plot above, the length of the various testruns varies. This is important to keep in mind when composing our time windows because we can not allow the time windows to overlap 2 or more testruns. The function below will therefore only take time windows from one testrun as long as enough samples are present in that specific run, otherwise it will need to proceed to the next testrun until we have the required amount of samples with it's corresponding time windows.
# Furthermore the function will allow us to define a sample rate for taking the samples in the testruns.

# In[ ]:


# function to create time-step windows for LSTM

def sliding_window(profile_id_list,max_sample_count,sample_rate=1,window_size=100):
    # profile_id_list: list of testruns we want to use to extract our samples
    # max_sample_count: the total amount of samples we want in our trainingset
    # sample rate: amount of samples to skip between the previous and next sample
    # window_size: amount of time steps (samples in the past) the window contains
    
    nr_of_features = 7 #number of columns minus 'stator_winding','profile_id'
    sample_count = 0

    i = 0
    X = np.zeros((max_sample_count,window_size,nr_of_features))
    y = np.zeros((max_sample_count))

    for profile_id in profile_id_list:
        temp=(dataset[dataset['profile_id']==profile_id]).iloc[lambda x: x.index % sample_rate==0]     
        temp_y = temp['stator_winding']
        temp_X = temp.drop('torque', axis=1).loc[:,'ambient':'i_q']
    
        i=0
        while i < len(temp_X)-window_size and sample_count < max_sample_count:
            X[sample_count] = temp_X.iloc[i:i+window_size]
            y[sample_count] = temp_y.iloc[i+window_size]
            sample_count +=1
            i +=1
    return (X,y) 
        


# In[ ]:


# split the testruns in a training and testset
profile_id_list_train ,profile_id_list_test = train_test_split(profile_id_list,test_size=0.3)
print("the list of testruns used for extracting the training sample windows:")
print(profile_id_list_train)
print("the list of testruns used for extracting the testing sample windows:")
print(profile_id_list_test)


# Now we have prepared the trainingsdata, we can start training our LSTM.

# In[ ]:


# constructing and training the LSTM
window_Size = 100
sample_amount = 5000
sample_rate = 10
epoch= 200
b_size = 500

X_train, y_train = sliding_window(profile_id_list_train,sample_amount,sample_rate,window_Size)
X_test, y_test = sliding_window(profile_id_list_test,sample_amount,sample_rate,window_Size)

LSTM_model = Sequential()
LSTM_model.add(LSTM(128, input_shape = (window_Size,7),return_sequences=True))
LSTM_model.add(LSTM(64, return_sequences=False))
LSTM_model.add(Dense(32, activation='relu'))
LSTM_model.add(Dropout(0.2))
LSTM_model.add(Dense(16, activation='relu'))
LSTM_model.add(Dense(8, activation='relu'))
LSTM_model.add(Dense(1))
LSTM_model.compile(loss='mean_squared_error', optimizer='adam',metrics = ["mean_squared_error"])
history = LSTM_model.fit(X_train,y_train,validation_split=0.33, epochs = epoch, batch_size = b_size, verbose = 0)


# In[ ]:


#plot the history of the model accuracy during training
plt.figure(figsize=(18,6))
ax1=plt.subplot(1, 2, 1)
ax1=plt.plot(history.history['mean_squared_error'],color='blue')
ax1=plt.plot(history.history['val_mean_squared_error'],color='red',alpha=0.5)
ax1=plt.title('model accuracy')
ax1=plt.ylabel('accuracy')
ax1=plt.xlabel('epoch')
ax1=plt.legend(['train', 'test'], loc='upper left')

# plot the history of the model loss during training
ax2=plt.subplot(1, 2, 2)
ax2=plt.plot(history.history['loss'],color='blue')
ax2=plt.plot(history.history['val_loss'],color='red',alpha=0.5)
ax2=plt.title('model loss')
ax2=plt.ylabel('loss')
ax2=plt.xlabel('epoch')
ax2=plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# Calculate MSE and MAE of the entire testset
y_pred_LSTM = LSTM_model.predict(X_test)
LSTM_MSE = mean_squared_error(y_test, y_pred_LSTM)
LSTM_MAE = mean_absolute_error(y_test, y_pred_LSTM)
print("MSE: {0}".format(LSTM_MSE))
print("MAE: {0}".format(LSTM_MAE))


# In[ ]:


# plot the true vs predicted values for multiple testruns
test_run_list = np.array([27,45,60,74])
#test_run_list = np.random.choice(profile_id_list, size=4, replace=False)]
output_value = 'stator_winding'
model = LSTM_model

window_Size = 100
sample_rate = 10

with sns.axes_style("whitegrid"):    
    fig, axs = plt.subplots(len(test_run_list),1,figsize=(20,len(test_run_list)*5),squeeze=False)
    
    for i in range(0,len(test_run_list)):
        sample_amount = (len(dataset[dataset['profile_id']==test_run_list[i]])-window_Size)//sample_rate
        X_plot, y_plot = sliding_window([test_run_list[i]],sample_amount,sample_rate,window_Size)
        y_pred_plot = model.predict(X_plot)

        time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])
        axs[i,0].plot(time,y_pred_plot,label='predict',color='red',alpha=0.4,linewidth=0.8)
        axs[i,0].plot(time,y_plot,label='True',color='black',linewidth=1)
        axs[i,0].legend(loc='best')
        axs[i,0].set_title("profile id: {0}".format(test_run_list[i]))
    plt.show()   


# In[ ]:


# plot the true vs predicted values for one run
test_run_list = np.array([choosen_example_testrun])
#test_run_list = np.random.choice(profile_id_list, size=4, replace=False)]
output_value = 'stator_winding'
model = LSTM_model

window_Size = 100
#sample_amount = 10000
sample_rate = 10

with sns.axes_style("whitegrid"):    
    fig, axs = plt.subplots(len(test_run_list),1,figsize=(20,len(test_run_list)*5),squeeze=False)
    
    for i in range(0,len(test_run_list)):
        sample_amount = (len(dataset[dataset['profile_id']==test_run_list[i]])-window_Size)//sample_rate
        X_plot, y_plot = sliding_window([test_run_list[i]],sample_amount,sample_rate,window_Size)
        y_pred_plot = model.predict(X_plot)

        time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])
        axs[i,0].plot(time,y_pred_plot,label='predict',color='red',alpha=0.4,linewidth=0.8)
        axs[i,0].plot(time,y_plot,label='True',color='black',linewidth=1)
        axs[i,0].legend(loc='best')
        axs[i,0].set_title("profile id: {0}".format(test_run_list[i]))
    plt.show()
    
print("MSE: {0}".format(mean_squared_error(y_plot, y_pred_plot)))
print("MAE: {0}".format(mean_absolute_error(y_plot, y_pred_plot)))


# # Conclusion
# When we compare the performance of the three models, we can conclude that in this case the Random Forest Regressor with moving average smoothing outperforms both a classic Feedforward Neural Network and an LSTM. The underperforming of the two deep learning models can probably be reduced by changing the models structure and hyperparameters. 
# However, because the Random Forest Regressor with moving average smoothing performs more than adequate in predicting the stator temperature (as the various examples below show) using a relative modest model like this is preferred above more computational and memory intensive alternatives like an NN or LSTM.  

# In[ ]:


# plot the true vs predicted values for multiple testruns for the Random Forest Regressor
#test_run_list = np.array([27,45,60,74])
test_run_list = np.random.choice(profile_id_list, size=4, replace=False)
output_value = 'stator_winding'
model = RFR_model
moving_average = 100

with sns.axes_style("whitegrid"):    
    fig, axs = plt.subplots(len(test_run_list),1,figsize=(20,len(test_run_list)*5),squeeze=False)
    
    for i in range(0,len(test_run_list)):
        X_plot = dataset.drop('torque', axis=1).loc[dataset['profile_id'] == test_run_list[i],'ambient':'i_q'].values 
        y_plot = dataset.loc[dataset['profile_id'] == test_run_list[i],output_value].values 
        y_pred_plot = bn.move_mean(model.predict(X_plot),moving_average_window,1)

        time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])
        axs[i,0].plot(time,y_pred_plot,label='predict',color='red',alpha=0.4,linewidth=0.8)
        axs[i,0].plot(time,y_plot,label='True',color='black',linewidth=1)
        axs[i,0].legend(loc='best')
        axs[i,0].set_title("profile id: {0}".format(test_run_list[i]))
    plt.show()   

