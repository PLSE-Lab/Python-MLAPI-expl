#!/usr/bin/env python
# coding: utf-8

# ## IMPORT LIBRARIES
# 
# We begin by importing the libraries needed to perform the incoming tasks.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models, layers, regularizers
import seaborn as sns
from scipy import stats


# ## DEFINE AUXILIARY FUNCTIONS
# > 
# Next, we define the functions that we need to perform the task

# In[ ]:


def standardize_0_1(ts):
    sts = (ts - np.min(ts))/(np.max(ts) - np.min(ts))
    return sts

def standardize_m_s(ts):
    sts = (ts - np.mean(ts))/(np.std(ts))
    return sts

def one_hot_encoding(data, exclusions):
    
    columns      = data.columns
    n_samples    = data.shape[0]
    one_hot_data = pd.DataFrame()
    for col in columns:
        if col not in exclusions:
                unique_vals = np.unique(data.loc[:,col])
                for val in unique_vals:
                    new_col_name = col + "_" + str(val)
                    zeros_array  = np.zeros(n_samples)
                    zeros_array[(data[col] == val)] = 1
                    one_hot_data[new_col_name] = zeros_array
    for col in exclusions:
        one_hot_data[col] = data[col]
    return one_hot_data
    
def build_model(regression_problem, hidden_layers_neurons, hidden_activation_function, L1_coeffs, L2_coeffs, hidden_layers_dropout, final_layer_neurons, final_activation_function, shape, model_optimizer, loss_function, metrics):
    
    model = models.Sequential()
    
    for i in range(len(hidden_activation_function)):
        
        if (i == 0):
            model.add(layers.Dense(hidden_layers_neurons[i], 
                                   kernel_regularizer = regularizers.l1_l2(l1 = L1_coeffs[i], l2 = L2_coeffs[i]),                             
                                   activation=hidden_activation_function[i], 
                                   input_shape=(shape,)))
        else:
            model.add(layers.Dense(hidden_layers_neurons[i], 
                                   kernel_regularizer = regularizers.l1_l2(l1 = L1_coeffs[i], l2 =  L2_coeffs[i]),  
                                   activation=hidden_activation_function[i]))
        if (hidden_layers_dropout[i] > 0.0):
            model.add(layers.Dropout(hidden_layers_dropout[i]))
    if regression_problem:
            model.add(layers.Dense(final_layer_neurons))
    else:
            model.add(layers.Dense(final_layer_neurons,activation = final_activation_function))
            
    model.compile(optimizer = model_optimizer, loss = loss_function, metrics = metrics)
    
    return model

            
    


# ## INPUT DASHBOARD
# 
# Then, we let the user set several variables that affect the estimation of the model:
# 
# datapath: set the path (including the target file name) where the code finds the input data
# 
# regression_problem: indicates whether we are facing a regression problem. If = True, the final layer of a neural network won't have any specified activation function.         
#                               
# training_set_size: the number of samples forming the training data set.
# 
# K_fold_shuffles: given a number of folds K, this sets the number of k-fold validations j performed on j different                shuffled versions of the same training data set.
# 
# K_fold_range: the number of folds the training data set must be split into. The training loop can consider                        multiple values of this variable.
# 
# hidden_activation_function: list containing the name of the activation functions (available in Keras) used in the hidden layers of the neural network.
# 
# hidden_layers_neurons: list containing the number of neurons forming each hidden layer of the neural network
# 
# hidden_layers_L1_coeffs: scalars multiplying the L1-penalty terms for each hidden layer weights during the training phase. Set it to 0 to avoid L1-regularization.
# 
# hidden_layers_L2_coeffs: scalars multiplying the L2-penalty terms for each hidden layer weights during the training phase. Set it to 0 to avoid L2-regularization.
# 
# hidden_layers_dropouts:  fractions of weights that are randomly set to zero for each hidden layer during the training phase. Set it to 0 to avoid dropout regularization.
# 
# final_activation_function: name of the activation function (available in Keras) used in the terminal layer of the neural network.
# 
# final_layer_neurons: number of neurons forming the terminal layer of the neural network.
# 
# model_optimizer: name of the method (available in Keras) to iteratively update the search of the set of parameters that minimize the loss function.
# 
# loss_function: name the loss function (available in Keras) that measures how well model predictions match the data during the training phase.
# 
# metrics: list containing the name of the metrics (available in Keras) that we use to measure how well the model using the current set of parameters predicts the validation data set.
# 
# n_epochs: the times the optimization algorithm goes through the entire training data set.
# 
# batch_size: the number of samples included in a single batch.
#        

# In[ ]:


datapath                   = "/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv"
regression_problem         = True
training_set_size          = 400
k_fold_shuffles            = 15
k_folds_range              = [2,3,4,5,6,7]
hidden_activation_function = ['sigmoid','sigmoid']
hidden_layers_neurons      = [16,8,8]
hidden_layers_L1_coeffs    = [0.00,0.00,0.00]
hidden_layers_L2_coeffs    = [0.00,0.00,0.00]
hidden_layers_dropout      = [0.00,0.00,0.00]
final_activation_function  = ''
final_layer_neurons        = 1
model_optimizer            = 'Adam'
loss_function              = 'mse'
metrics                    = ["mae"]
n_epochs                   = 75
batch_size                 = 20


# ## DATA IMPORT
# 
# We load the full dataset first and then we retrieve the features and the targer vector.

# In[ ]:


df                      = pd.read_csv(datapath,sep=',')
df                      = df.iloc[:,1:]


# We add a column to link each sample with its subset of data (train vs. test)
# 

# In[ ]:


df.loc[:(training_set_size-1),'Dataset'] = 1
df.loc[training_set_size:,'Dataset'] = 2


# ## DESCRIPTIVE ANALYSIS OF THE DATASET
# 
# Next, we study the histograms of each feature

# In[ ]:


fig, ax = plt.subplots(2,4, figsize=(30, 15))
row_id  = 0
col_id  = 0
for var, subplot in zip(['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research','Chance of Admit '], ax.flatten()):
    sns.distplot(df.loc[df['Dataset'] == 1,var], color = 'red', label = 'Train', ax = ax[row_id,col_id])
    sns.distplot(df.loc[df['Dataset'] == 2,var], color='green', label='Test', ax=ax[row_id, col_id])
    ax[row_id, col_id].set(title = "Histogram of " + var, ylabel= 'frequency')
    ax[row_id, col_id].legend()
    col_id += 1
    if col_id > 3:
       col_id  = 0
       row_id += 1
plt.show()


# Then, we study how each discrete variable relates with the target (Chance of Admit)

# In[ ]:


fig, axes = plt.subplots(1,4, figsize=(40, 10))
col_id  = 0
cols    = ['SOP','LOR ','University Rating','Research']
for col in cols:
        sns.boxplot(x = col, y= "Chance of Admit ", data=df, ax= axes[col_id])
        axes[col_id].set(ylabel='Chance of admission ',title = "Chance of admission within the " + col + " class.")
        axes[col_id].legend(loc='upper right')
        col_id += 1
plt.show()


# Next, we study how each continuous variable relates with the target (Chance of Admit) 

# In[ ]:


fig, axes = plt.subplots(1,3, figsize=(30, 10))
col_id  = 0
cols    = ['GRE Score','TOEFL Score','CGPA']
colors  = ['red','blue','green']
for col in cols:
        sns.scatterplot(x = col, y= "Chance of Admit ", color= colors[col_id], data=df, ax= axes[col_id])
        axes[col_id].set(ylabel='Chance of admission ',title = "Chance of admission within the " + col + " class.")
        axes[col_id].legend(loc='upper right')
        col_id += 1
plt.show()


# ## ONE-HOT ENCODING OF "DISCRETE" FEATURES

# In[ ]:


df = one_hot_encoding(df, ['Chance of Admit ','GRE Score','TOEFL Score','CGPA','Dataset'])


# ## CORRELATION MATRIX

# In[ ]:


fig, axes = plt.subplots(1,1, figsize=(30, 20))
sns.heatmap(df.loc[df['Dataset']==1,:].corr(), annot = True, ax = axes)
plt.show()


# ## SIMPLE FEATURE REDUCTION
# 
# then, we retain those features that exhibit a significant correlation (or anti-correlation) with the target labels

# In[ ]:


abs_corr                  = abs(df.loc[df['Dataset']==1,:].corr())
selected_features         = (abs_corr.loc['Chance of Admit ',:] > 0.35)
df = df.filter(items = list(abs_corr.loc[selected_features,:].index) + ['Dataset'])


# ## "CONTINUOUS" FEATURES STANDARDISATION
# 
# Next, we "normalize" the continuous features
# 

# In[ ]:


df.loc[df['Dataset']==1,'GRE Score']           = standardize_m_s(df.loc[df['Dataset']==1,'GRE Score'])
df.loc[df['Dataset']==1,'TOEFL Score']         = standardize_m_s(df.loc[df['Dataset']==1,'TOEFL Score'])
df.loc[df['Dataset']==1,'CGPA']                = standardize_m_s(df.loc[df['Dataset']==1,'CGPA'])
df.loc[df['Dataset']==2,'GRE Score']           = standardize_m_s(df.loc[df['Dataset']==2,'GRE Score'])
df.loc[df['Dataset']==2,'TOEFL Score']         = standardize_m_s(df.loc[df['Dataset']==2,'TOEFL Score'])
df.loc[df['Dataset']==2,'CGPA']                = standardize_m_s(df.loc[df['Dataset']==2,'CGPA'])


# ## FINAL REFINEMENTS
# 
# The dataset is converted into a numpy array 

# In[ ]:


train_data = df.loc[df['Dataset'] == 1, :]
test_data  = df.loc[df['Dataset'] == 2, :]
train_targets          = train_data['Chance of Admit ']
test_targets           = test_data['Chance of Admit ']
train_data.drop(columns = ['Chance of Admit ','Dataset'],inplace = True)
test_data.drop(columns = ['Chance of Admit ','Dataset'],inplace = True)

train_data                      =  train_data.to_numpy()
test_data                       =  test_data.to_numpy()
n_samples                       =  train_data.shape[0]
n_features                      =  train_data.shape[1]
total_experiments               =  np.sum(k_folds_range)*k_fold_shuffles


# ## ITERATED K-FOLD + SHUFFLING VALIDATION
# 
# We finally set-up the training phase. To train our neural network, we adopt a shuffling k-fold approach 
# 
# 1) Randomly shuffle the training dataset
# 
# 2) Split it into K folds, i.e., K adjacent partitions containing the same number of samples.
# 
# 3) K - 1 folds are used to train the model, i.e., parameter estimation, and 1 fold is used as a validation sample, i.e., the predictive ability of the newly trained model is assessed on this subset.
# 
# 4) Collect the vector containing the value of the mean absolute error (MAE) over the validation dataset in each epoch.
# 
# 5) Test the trained network on previously unseen data, i.e., the test data set, and collect the root mean squared error (RMSE).
# 
# 6) Go back to (3) and ensure that another fold will serve as a validation set. This can be done in a sliding fashion. For instance, for K = 3 we have:
# 
# 6.1) |Validation fold|Training fold 1|Training fold 2| -> |Validation data|       Training data       |  -> perform step (4)-(5)
# 
# 6.2) |Training fold 2|Validation fold|Training fold 1| -> |Training data|Validation data|Training data|  -> perform step (4)-(5)
# 
# 6.3) |Training fold 1|Training fold 2|Validation fold| -> |       Training data       |Validation data|  -> perform step (4)-(5)
# 
# 7) Go back to (1) and perform steps (2)-(6) j times, i.e., the user defined k_fold shuffles.
# 
# 8) Consider the other values of K, i.e., the number of folds specified by the user in k_folds_range, and repeat steps (1)-(7).
# 
# During this cycle, we have k_fold shuffles*(sum_{i = 0}^{L - 1}k_folds_range[i]), where L is len(k_folds_range), experiments. An experiment includes a training phase over K - 1 folds, a validation over the remaining fold and a test over the unseen test set. For each experiment, we are interested in three things:
# 
# a) The value of the MAE in each epoch to assess whether adding more epochs to the training improves our validation.
# b) The value of the RMSE to assess how well the current trained model performs on the test data set.
# c) Whether the current trained model achieves the running minimum RMSE. If this the case, the object "model" is kept in memory.

# In[ ]:


k_fold_sample_sizes             = [n_samples//k_folds for k_folds in k_folds_range]
MAE_matrix                      = pd.DataFrame(data = np.zeros((len(k_folds_range)*k_fold_shuffles, 2 + n_epochs)), columns = ['K','Shuffle #'] + [str(ij + 1) for ij in range(n_epochs)])
RMSE_matrix                     = pd.DataFrame(data = np.zeros((len(k_folds_range)*k_fold_shuffles, 3)), columns = ['K','Shuffle #','<RMSE>'])
row_idx                         = 0
experiment_counter              = 0
for j in range(0,len(k_folds_range),1):
        for p in range(k_fold_shuffles):

                mean_MAE_matrix_list   = []
                mean_RMSE_matrix_list  = []
                
                shuffled_indexes       = list(range(n_samples))
                np.random.shuffle(shuffled_indexes)
                shuffled_train_data    = train_data[shuffled_indexes]
                shuffled_train_targets = train_targets[shuffled_indexes]

                k_fold_sample_size     = k_fold_sample_sizes[j]
                current_K              = k_folds_range[j]
                
                for i in range(0,current_K,1):


                            validation_data         = shuffled_train_data[(i*k_fold_sample_size):((i+1)*k_fold_sample_size)]
                            validation_targets      = shuffled_train_targets[(i*k_fold_sample_size):((i+1)*k_fold_sample_size)]

                            current_train_data_1    = shuffled_train_data[:(i*k_fold_sample_size)]
                            current_train_targets_1 = shuffled_train_targets[:(i*k_fold_sample_size)]

                            current_train_data_2    = shuffled_train_data[((i+1) * k_fold_sample_size):]
                            current_train_targets_2 = shuffled_train_targets[((i+1) * k_fold_sample_size):]

                            current_train_data      = np.concatenate([current_train_data_1,current_train_data_2],axis=0)
                            current_train_targets   = np.concatenate([current_train_targets_1, current_train_targets_2], axis=0)

                            model                   = build_model(regression_problem, hidden_layers_neurons, hidden_activation_function, 
                                                                      hidden_layers_L1_coeffs, hidden_layers_L2_coeffs, hidden_layers_dropout,
                                                                      final_layer_neurons, final_activation_function, current_train_data.shape[1], 
                                                                      model_optimizer, loss_function, metrics)

                            history                 = model.fit(current_train_data, current_train_targets, 
                                                                validation_data = (validation_data,validation_targets), 
                                                                epochs=n_epochs, 
                                                                batch_size=batch_size,
                                                                verbose=0)
                            
                            mean_MAE_matrix_list.append(history.history['val_mae'])#mean_absolute_error'])
                            
                            forecasts               = np.array([pred[0] for pred in model.predict(test_data)])
                            RMSE                    = np.sqrt(np.average((forecasts - test_targets)**2))
                            mean_RMSE_matrix_list.append(RMSE)
                            

                            if (row_idx == 0):
                                best_model = model
                                best_RMSE  = RMSE
                            else:
                                current_MAE= np.min(history.history['val_mae'])
                                if (RMSE < best_RMSE):
                                    best_model = model
                                    best_RMSE  = RMSE
                            
                            experiment_counter += 1
                            print('Processed window #: ' + str(i+1) + " out of " +str(current_K) +  
                                  ', current shuffle #: ' + str(p) +
                                  ', RMSE: ' + str(np.round(RMSE,4)) +
                                  ', Best RMSE: ' + str(np.round(best_RMSE,4)) +
                                  ', % experiment completed: ' + str(np.round(100*(experiment_counter/float(total_experiments)),2)))
                            

                MAE_matrix.iloc[row_idx]  = np.concatenate((np.array([current_K, p + 1]), np.mean(np.array(mean_MAE_matrix_list),axis=0)))
                RMSE_matrix.iloc[row_idx] = np.array([current_K, p + 1,np.mean(mean_RMSE_matrix_list)])
                row_idx += 1


# ## OUTCOMES ANALYSIS
# 
# Finally, we propose various plots to analyze the outcomes of the training process.
# First, we examine how the average mean absolute error <MAE> changes with the training epoch. In this case, the statistics of interest is obtained by averaging the MAE values on a given epoch regardless the value of K or the id of the shuffle.

# In[ ]:


validation_df                   = pd.DataFrame()
validation_df ['Epoch']         = range(1,n_epochs + 1)
validation_df ['<MAE>']         = np.average(np.array(MAE_matrix.iloc[:,2:].to_numpy()),axis=0)
f, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x="Epoch",y="<MAE>", data=validation_df, palette="Blues_d") .set_title('Average mean absolute error (<MAE>) as a function of the training epoch.')
plt.show()

f, ax = plt.subplots(figsize=(18, 18*((len(k_folds_range)*k_fold_shuffles)/float(n_epochs))))
yticklabels = "(K: " + MAE_matrix['K'].apply(int).apply(str) + ", #: " + MAE_matrix['Shuffle #'].apply(int).apply(str) + ")"
sns.heatmap(MAE_matrix.iloc[:,2:], yticklabels=yticklabels, annot=False,linewidths=.5, ax=ax).set_title('Mean absolute error (<MAE>) in each training fold (x:epoch, y:(K, # shuffle))')
plt.show()


# Then, we look at how the average rooted mean squared error <RMSE> changes across different shuffles and values of K. In this case, the statistics of interest is the average of the RMSE collected across the k fold slides within a single shuffle of the training dataset.

# In[ ]:


f, ax = plt.subplots(figsize=(18, 18*((len(k_folds_range)*k_fold_shuffles)/float(n_epochs))))
yticklabels = [str(k) for k in k_folds_range]
xticklabels = [str(ij + 1) for ij in range(k_fold_shuffles)]
data = (RMSE_matrix.iloc[:,2:]).to_numpy()
sns.heatmap(data.reshape((len(k_folds_range),k_fold_shuffles)), xticklabels=xticklabels, yticklabels=yticklabels, 
            annot=True,linewidths=.5, ax=ax)
ax.set_title('Average root mean squared erros (<RMSE>) in each k-fold shuffle (x:# shuffle, y:K)')
ax.set_xlabel("Shuffle #")
ax.set_ylabel("K")
plt.show()


# We examine the best performing model, i.e., the lowest recorded RMSE, by displaying a scatter plot between the test data targets and the predictions of the model.

# In[ ]:


forecasts                      = np.array([pred[0] for pred in best_model.predict(test_data)])
RMSE                           = np.sqrt(np.average((forecasts - test_targets)**2))
prediction_df                  = pd.DataFrame()
prediction_df['Target']        = test_targets
prediction_df['Forecast']      = forecasts
prediction_df['Squared error'] = (forecasts - test_targets)**2
sns.jointplot(x="Target", y="Forecast", data=prediction_df);
plt.show
slope, intercept, r_value, p_value, std_err = stats.linregress(test_targets, forecasts)
print("The lowest recorded RMSE is: " + str(np.round(RMSE,4)))
print("The R^2 of the linear model test data vs. prediction is: " + str(np.round(r_value**2,4)))


# We conclude by returning our target value, i.e., the lowest RMSE achieved in this session.

# In[ ]:


best_RMSE

