# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Import PyTorch elements
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
# special character in movie title that UTF-8 cannot read, therefore latin-1 was specified at encoding
movies = pd.read_csv('/kaggle/input/dataml1m/movies.dat', sep='::', header = None, engine = 'python', encoding='latin-1')
users = pd.read_csv('/kaggle/input/dataml1m/users.dat', sep='::', header = None, engine = 'python', encoding='latin-1')
ratings = pd.read_csv('/kaggle/input/dataml1m/ratings.dat', sep='::', header = None, engine = 'python', encoding='latin-1')

# Preparing the training set and the test set
# data are seperated by 'tab', so use delimiter '\t'
training_set = pd.read_csv('/kaggle/input/dataml100k/u1.base', delimiter='\t')
# Convert dataframe to array with integers - dtype = 'int'
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('/kaggle/input/dataml100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
# Structure of data expected by the neura network - features in column, and observations in line
def convert(data):
    # Create a list of list
    # Initialize a list
    new_data = []
    # max number is excl in for loop, therefore +1 is required
    for id_users in range(1, nb_users + 1):
        # get the movies id & rating id for the id_user, last [] is the condition
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        # create a zero table of rating to the unrated movies are marked as zero, 
        # then insert rating into the data to complete the set
        ratings = np.zeros(nb_movies)
        # Replace zero with the existing rating, and move item 1 to match with 0, therefore -1
        ratings[id_movies - 1] = id_ratings
        # Add small list to the big list
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating an architecture of the Stacked AutoEncoder
class SAE(nn.Module):
    def __init__(self, ): # define architecture of the autoencoder
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20) # 1st visible layer is number of movies, 20 node of hidden layer
        self.fc2 = nn.Linear(20, 10) # 2nd hidden layer
        self.fc3 = nn.Linear(10, 20) # 3rd hidden layer - 1st part of the decoding
        self.fc4 = nn.Linear(20, nb_movies) # decoding layer
        # Define Activation function
        self.activation = nn.Sigmoid()
    # Create encoding function    
    def forward(self, x):
        x = self.activation(self.fc1(x)) # this return the 1st encoded vector
        x = self.activation(self.fc2(x)) # this return the 2nd encoded vector
        x = self.activation(self.fc3(x))
        x = self.fc4(x) # no activation function required for last decoding layer
        return x

sae = SAE()
criterion = nn.MSELoss() # criterion for the loss function = mean square error
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # Optimizer - apply stocastic gradient descent


# Training the SAE
nb_epoch = 200 # define no. of epoch
for epoch in range(1, nb_epoch + 1): # make a for loop
    # initialize loss error
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        # Take care of the target - same as input vector
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False # make sure we don't compute the gradien with respect the target - save computational resources
            output[target == 0] = 0 # save some memory by setting them to zero
            loss = criterion(output, target) # Compute the loss
            mean_corrector = nb_movies/float(torch.sum(target.data > 0)+ 1e-10) # add 1e-10 in demoniator to make it non-zero value
            loss.backward() # back propagation - decide the direction of the update
            train_loss += np.sqrt(loss.data.item() * mean_corrector) # I used .item() to convert tensor object to float
            s += 1.
            optimizer.step() # back propagation - step the intensity of the update (amount)
    print('epoch: ' +  str(epoch) + ' loss: ' + str(train_loss/s))

# Testing the SAE
# initialize loss error
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    # Take care of the target - same as input vector
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False # make sure we don't compute the gradien with respect the target - save computational resources
        output[target == 0] = 0 # save some memory by setting them to zero
        loss = criterion(output, target) # Compute the loss
        mean_corrector = nb_movies/float(torch.sum(target.data > 0)+ 1e-10) # add 1e-10 in demoniator to make it non-zero value
        test_loss += np.sqrt(loss.data.item() * mean_corrector) # I used .item() to convert tensor object to float
        s += 1.
print('test loss: ' + str(test_loss/s))

        
    
