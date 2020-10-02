#!/usr/bin/env python
# coding: utf-8

# OK, folks! How simple can you have a neural network?

# In[ ]:


import numpy as np
import pandas as pd
# Look! No scikit learn!


# ### Read the data

# In[ ]:


df_train = pd.read_json(open("../input/train.json", "r"))
df_train.set_index("listing_id", inplace=True)
df_test  = pd.read_json(open("../input/test.json", "r"))
df_test.set_index("listing_id", inplace=True)
# We will work with a concatenation of the two, then split after the scaling.
df = pd.concat([df_train, df_test])


# ### Simple feature engineering
# Let's do the same feature engineering as Li Li. However w/o the year. Some years are missing, I believe. Edit: No! all samples are 2016. Then we can safely ignore this.

# In[ ]:


df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
#df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day


# Since the distribution of prices are so skewed and also probably an important indicator, we must transform the price in some way. The first thing that pops into my head is logarithm transform. Let's try.

# In[ ]:


df["logprice"] = np.log(df.price)


# One of the samples is listed with 112(!) bathrooms. That must be wrong. I'm adjusting that to 1 bathroom. (This gets a more realistic scaling so it improves the overall result)

# In[ ]:


df.loc[df.bathrooms == 112, "bathrooms"] = 1


# Since we will use a neural network we need to scale the features. How simple can we do that?

# In[ ]:


numeric_feat = ["bathrooms", "bedrooms", "latitude", "longitude", "logprice",
             "num_photos", "num_features", "num_description_words",
              "created_month", "created_day"]
for col in numeric_feat:
    df[col] -= df[col].min()
    df[col] /= df[col].max()


# In[ ]:


X_train = df.loc[df.interest_level.notnull(), numeric_feat]
y_train = pd.get_dummies(df_train[["interest_level"]], prefix="")
y_train = y_train[["_high", "_medium", "_low"]]  # Set the order according to submission
X_test  = df.loc[df.interest_level.isnull(), numeric_feat]


# ### Clean up
# Clean up the unused data frames.

# In[ ]:


del df
del df_train
del df_test


# ### A really simple neural network
# Here is an implementation of a really simple neural network. This is the kind of neural network you would expect in the late 1990's. There is no weight decay regularisation or dropout or anything fancy, so the only way to prevent overfitting is early stopping, and limiting the capacity by setting the number of hidden units.
# 
# Also note that there is only three layers: input, hidden and output. The output has softmax outputs and the hidden layer has sigmoid activation function. Please try other configurations if you like.
# 
# (This is the very same implementation I user in [Ghouls, Goblins, and Ghosts... Boo!][1], no modifications what so ever!)
# 
#   [1]: https://www.kaggle.com/oysteijo/ghouls-goblins-and-ghosts-boo/ghosts-n-goblins-n-neural-networks-lb-0-74858

# In[ ]:


## A dead simple neural network class in Python+Numpy. Plain SGD, and no regularization.
def sigmoid(X):
    return 1.0 / ( 1.0 + np.exp(-X) )

def softmax(X):
    _sum = np.exp(X).sum()
    return np.exp(X) / _sum

class neuralnet(object):
    def __init__(self, num_input, num_hidden, num_output):
        self._W1 = (np.random.random_sample((num_input, num_hidden)) - 0.5).astype(np.float32)
        self._b1 = np.zeros((1, num_hidden)).astype(np.float32)
        self._W2 = (np.random.random_sample((num_hidden, num_output)) - 0.5).astype(np.float32)
        self._b2 = np.zeros((1, num_output)).astype(np.float32)

    def forward(self,X):
        net1 = np.matmul( X, self._W1 ) + self._b1
        y = sigmoid(net1)
        net2 = np.matmul( y, self._W2 ) + self._b2
        z = softmax(net2)
        return z,y

    def backpropagation(self, X, target, eta):
        z, y = self.forward(X)
        d2 = (z - target)
        d1 = y*(1.0-y) * np.matmul(d2, self._W2.T)
        # The updates are done within this method. This more or less implies
        # utpdates with Stochastic Gradient Decent. Let's fix that later.
        # TODO: Support for full batch and mini-batches etc.
        self._W2 -= eta * np.matmul(y.T,d2)
        self._W1 -= eta * np.matmul(X.reshape((-1,1)),d1)
        self._b2 -= eta * d2
        self._b1 -= eta * d1


# In[ ]:


# Some hyper-parameters to tune.
num_hidden = 17    # I think I get about 1 epoch/sec with this size on the docker instance
n_epochs   = 100
eta        = 0.01


# Create the neural network.

# In[ ]:


nn = neuralnet( X_train.shape[1], num_hidden, y_train.shape[1])


# **New!** Let's have a logloss calculation, such that we're not totally in blindness.

# In[ ]:


def logloss( nn, X, Y ):
    err = 0
    for apartment, target in zip( X, Y ):
        probs = nn.forward( np.array(apartment, dtype=np.float32))[0][0]
        err += sum(target*np.log(probs))
    return -err/X.shape[0]


# Do the training!

# In[ ]:


# It's much faster to convert the dataframes to numpy arrays and then iterate.
X = np.array(X_train, dtype=np.float32)
Y = np.array(y_train, dtype=np.float32)
for epoch in range(n_epochs):
    print("Epoch: {:3d} train-error: {}".format(epoch, logloss(nn, X, Y)), end='\r')    
    for apartment, target in zip(X,Y):
        nn.backpropagation( apartment, target, eta)


# In[ ]:


with open('submission-{}-hidden.csv'.format(num_hidden), 'w') as f:
    f.write("listing_id,high,medium,low\n")
    for index, apartment in X_test.iterrows():
        probs = nn.forward( np.array(apartment, dtype=np.float32))[0][0]
        f.write("{},{},{},{}\n".format(index, *probs))


# ### TODO
# 
#  - Local CV!! (Update: we now have training error, but that does not give us the whole truth.)
#  - Parameter tuning
#  - Feature engineering
#  - Add batch/mini-batch training
