# %% [code]
# Boltzmann Machines
# https://skymind.ai/wiki/restricted-boltzmann-machine

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

# Any results you write to the current directory are saved as output.

# METHOD 1: Preparing the training set and the test set
training_set = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u1.base', sep = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u1.test', sep = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
# Assume that movie ids and user ids are sequential numbers starting from 1 
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))


# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users] # data[:,1] - get all movies. [data[:,0] == id_users] - returns Boolean list with True if it's user_id and False otherwise
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings # movie ids start from 1. List indexes start from 0
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)


# METHOD 2: Preparing the training set and the test set
# # Read data
# training_set = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u1.base', sep='\t').as_matrix()
# test_set = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u1.test', sep='\t').as_matrix()
 
# # Compute counts
# nb_users = len(set(training_set[:, 0]) | set(test_set[:, 0]))
# nb_movies = len(set(training_set[:, 1]) | set(test_set[:, 1]))
 
# # Reshape training set
# temp = np.zeros((nb_users, nb_movies))
# temp[training_set[:, 0] - 1, training_set[:, 1] - 1] = training_set[:, 2]

# # example with simple nd-array:
# # temp = np.zeros((3, 4))
# # temp[[0,1],[2,3]] = [111,333] # set matrix element a[0][2] = 111; a[1][3] = 333

# training_set = temp.tolist()
 
# # Reshape test set
# temp = np.zeros((nb_users, nb_movies))
# temp[test_set[:, 0] - 1, test_set[:, 1] - 1] = test_set[:, 2]
# test_set = temp.tolist()

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Initially 0 means no rating. Recode it to -1 to distinguish from disliked movies
training_set[training_set == 0] = -1 
test_set[test_set == 0] = -1

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1



class RBM():
    def __init__(self, visible_node_count, hidden_node_count):        
        # initialise weights (synapses/connections) and individual node biases with normal distributed rands
        self.weights = torch.randn(hidden_node_count, visible_node_count)
        
        # Hidden biases: when propagating up from the visible layer to the hidden layer we have the rbm weights plus
        # the bias weights, which in this upwards positive stage are called hidden biases .
        self.hidden_node_bias   = torch.randn(1, hidden_node_count)

        # Visible biases: for the downward propagation from the hidden layer, after sampling the hidden layer,
        # to the reconstruction of the visible layer we use the same rbm weights but a different set of bias weights. 
        # This second set of bias weights are visible biases.         
        self.visible_node_bias  = torch.randn(1, visible_node_count)
 
    def sample_hidden_nodes(self, x_visible_node_input_values):
        # get product of [input visible nodes]•[Weights.transpose()]
        weighted_input = torch.mm(x_visible_node_input_values, self.weights.t())
        
        # create activation value by adding the [hidden node] bias to the weighted input 
        # where hidden node bias (expanded to the shape of the the weighted inputs) 
        values_for_activation = weighted_input + self.hidden_node_bias.expand_as(weighted_input)
 
        # calculate probability of hidden node activation for given visible node weight
        # i.e. it is a vector containing the probabilities the hidden nodes are activated, given the values of the visible note (i.e. ratings of the users)
        activation_probabilities = torch.sigmoid(values_for_activation)
 
        # Bernoulli sampling involves generating a random number between 0 and 1 
        # to activate the neuron (100*prob)% of the time)
        # torch.bernoulli(0.8) means 80% of time value returned would be 1 and 20% it would be 0 
        # the values that are going to be input to network will be learnt during training
        return activation_probabilities, torch.bernoulli(activation_probabilities)
    
    
    
    def sample_visible_nodes(self, y_hidden_node_input_values):
        weighted_input = torch.mm(y_hidden_node_input_values, self.weights) # no transpose
        values_for_activation = weighted_input + self.visible_node_bias.expand_as(weighted_input)
        activation_probabilities = torch.sigmoid(values_for_activation) # probability of visible node activation for given hidden node weight
        return activation_probabilities, torch.bernoulli(activation_probabilities)
    
    # v_0 - input vector of observations ( the ratings of movies)
    # v_k - visible nodes obtained after K iterations (contrastive divergence) - corresponding to movie ratings  after K iterations.
    # p_h_0 - vector of probabilities that at the first iteration the hidden nodes equal 1 given the values of the input vector v_0
    # p_h_k - probabilities of the hidden nodes after K sampling equal 1 given the values of the visible nodes v_k
    def train (self, v_0, v_k, p_h_0, p_h_k):
        self.weights += (torch.mm(v_0.t(), p_h_0) - torch.mm(v_k.t(), p_h_k)).t()
        self.visible_node_bias += torch.sum((v_0 - v_k), 0) # torch.sum to keep the format of hidden_node_bias as a tensor of two dimensions 
        self.hidden_node_bias += torch.sum((p_h_0 - p_h_k), 0)
        
    def predict(self, x): # x: visible nodes
        _,h = self.sample_hidden_nodes(x)
        _,v = self.sample_visible_nodes(h)
        return v    
        
visible_node_count = len(training_set[0]) # number of movies, i.e visible nodes that are the ratings of all the movies by a specific user
hidden_node_count = 100 # number of hidden nodes = number of features we want to detect. Can be tuned
batch_size = 100 # number of users to process as 1 batch. Can be tuned
rbm = RBM(visible_node_count, hidden_node_count)

number_of_epochs = 10 # too few observations (only 943 users)
for epoch in range(1, number_of_epochs + 1):
    train_loss = 0    #Loss function to measure the error between the predictions and the real ratings
    s = 0. # counter (type float) to normalize loss
    for user_id in range(0, nb_users - batch_size, batch_size): # user numeration starts from 0
        v_k = training_set[user_id:user_id+batch_size]
        v_0 = training_set[user_id:user_id+batch_size]
        p_h_0,_ = rbm.sample_hidden_nodes(v_0) 
        for k in range(10): # Gibbs sampling
            _,h_k = rbm.sample_hidden_nodes(v_k) 
            _,v_k = rbm.sample_visible_nodes(h_k) 
            v_k[v_0 < 0] = v_0[v_0 < 0] # we don't wanna learn where there is no rating,so freeze visible nodes that contain the minus one ratings.

        p_h_k,_ = rbm.sample_hidden_nodes(v_k) 
        rbm.train (v_0, v_k, p_h_0, p_h_k)
        train_loss += torch.mean(torch.abs(v_0[v_0 >= 0] - v_k[v_0 >= 0]))  # use absolute distance as loss function
        # train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE 
        
        s += 1.
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))  # print normalised loss
    
    
# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users): # make predictions for each of the users one by one
    v = training_set[id_user:id_user+1] # input on which we are making predictions: 
    # We keep the training set, because actually the training set is 
    # the input that will be used to activate the hidden neurons to get the output.
    
    vt = test_set[id_user:id_user+1] # target = original ratings of the test set, this is what we will compare to our predictions in the end
    if len(vt[vt>=0]) > 0: # condition to allow us to make a prediction: for all ratings in test set
        _,h = rbm.sample_hidden_nodes(v)
        _,v = rbm.sample_visible_nodes(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))    

# Take a test user and the whole list of movies of that user and convert to PyTorch
userd_id = 23
user_input = Variable(test_set[user_id-1]).unsqueeze(0)

# Make the prediction using this input
ouput = rbm.predict(user_input)
#output = output.data.numpy()


# Stack the inputs and outputs as one numpy array (the first row is the input and the second row is the predicted.)
#input_output = np.vstack([user_input, output])
#input_output
#Check User-Movie match in a single instance
# target_user_id = 1
# target_movie_id = 50
# z = Variable(training_set[target_user_id-1]).unsqueeze(0)
# output = rbm.predict(z)
# output_numpy = output.data.numpy()
# print (''+ str(output_numpy[0,target_movie_id-1]))

# #Create copy of input files
# ratings2=ratings.copy()
# users2=users.copy()
# movies2=movies.copy()
# ratings2.columns = ['User','MovieId','Rating','Timestamp']
# users2.columns = ['UserId','Gender','X','Y','Z']
# movies2.columns = ['Reference','Name','Genre']

# #create recommendation set for each user (movies sorted in no particular order)
# reco = pd.DataFrame(columns=['UserId','MovieId'])
# result = pd.DataFrame(columns=['UserId','MovieId','Reco'])
# for user in range(1, nb_users + 1):
#     seen = ratings2[ratings2['User']==user]
#     seenlist = seen['MovieId'].tolist()        
#     z = Variable(training_set[user-1]).unsqueeze(0)
#     output = rbm.predict(z)
#     output_numpy = output.data.numpy()
#     result = pd.DataFrame(output_numpy)
#     result=result.melt()
#     result.columns=['MovieId','Reco']
#     result['UserId']=user
#     result = result.loc[~result['MovieId'].isin(seenlist)]
#     result=result[result['Reco']==1]
#     result=result[['UserId','MovieId']]
#     reco=reco.append(result)

# %% [code]



# %% [code]
