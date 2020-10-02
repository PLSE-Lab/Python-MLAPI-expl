#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls ../input')


# # Sentiment Analysis with an RNN
# 
# In this notebook, you'll implement a recurrent neural network that performs sentiment analysis. 
# >Using an RNN rather than a strictly feedforward network is more accurate since we can include information about the *sequence* of words. 
# 
# Here we'll use a dataset of movie reviews, accompanied by sentiment labels: positive or negative.
# 
# <img src="assets/reviews_ex.png" width=40%>
# 
# ### Network Architecture
# 
# The architecture for this network is shown below.
# 
# <img src="assets/network_diagram.png" width=40%>
# 
# >**First, we'll pass in words to an embedding layer.** We need an embedding layer because we have tens of thousands of words, so we'll need a more efficient representation for our input data than one-hot encoded vectors. You should have seen this before from the Word2Vec lesson. You can actually train an embedding with the Skip-gram Word2Vec model and use those embeddings as input, here. However, it's good enough to just have an embedding layer and let the network learn a different embedding table on its own. *In this case, the embedding layer is for dimensionality reduction, rather than for learning semantic representations.*
# 
# >**After input words are passed to an embedding layer, the new embeddings will be passed to LSTM cells.** The LSTM cells will add *recurrent* connections to the network and give us the ability to include information about the *sequence* of words in the movie review data. 
# 
# >**Finally, the LSTM outputs will go to a sigmoid output layer.** We're using a sigmoid function because positive and negative = 1 and 0, respectively, and a sigmoid will output predicted, sentiment values between 0-1. 
# 
# We don't care about the sigmoid outputs except for the **very last one**; we can ignore the rest. We'll calculate the loss by comparing the output at the last time step and the training label (pos or neg).

# ---
# ### Load in and visualize the data

# In[ ]:


import numpy as np

# read data from text files
with open('../input/reviews.txt', 'r') as f:
    reviews = f.read()
with open('../input/labels.txt', 'r') as f:
    labels = f.read()


# In[ ]:


print(reviews[:2000])
print()
print(labels[:20])


# ## Data pre-processing
# 
# The first step when building a neural network model is getting your data into the proper form to feed into the network. Since we're using embedding layers, we'll need to encode each word with an integer. We'll also want to clean it up a bit.
# 
# You can see an example of the reviews data above. Here are the processing steps, we'll want to take:
# >* We'll want to get rid of periods and extraneous punctuation.
# * Also, you might notice that the reviews are delimited with newline characters `\n`. To deal with those, I'm going to split the text into each review using `\n` as the delimiter. 
# * Then I can combined all the reviews back together into one big string.
# 
# First, let's remove all punctuation. Then get all the text without the newlines and split it into individual words.

# In[ ]:


from string import punctuation

print(punctuation)

# get rid of punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])


# In[ ]:


# split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()


# In[ ]:


words[:30]


# ### Encoding the words
# 
# The embedding lookup requires that we pass in integers to our network. The easiest way to do this is to create dictionaries that map the words in the vocabulary to integers. Then we can convert each of our reviews into integers so they can be passed into the network.
# 
# > **Exercise:** Now you're going to encode the words with integers. Build a dictionary that maps words to integers. Later we're going to pad our input vectors with zeros, so make sure the integers **start at 1, not 0**.
# > Also, convert the reviews to integers and store the reviews in a new list called `reviews_ints`. 

# In[ ]:


len(words)


# In[ ]:


# feel free to use this import 
from collections import Counter

temp = Counter(words)
temp = temp.most_common()

## Build a dictionary that maps words to integers
vocab_to_int = {}
i = 1
for pair in temp:
    vocab_to_int.update({pair[0]:i})
    i+=1


## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
reviews_ints = []
for review in reviews_split:
    word_list = review.split()
    num_list = []
    for word in word_list:
        num_list.append(vocab_to_int[word])
    reviews_ints.append(num_list)


# **Test your code**
# 
# As a text that you've implemented the dictionary correctly, print out the number of unique words in your vocabulary and the contents of the first, tokenized review.

# In[ ]:


# stats about vocabulary
print('Unique words: ', len((vocab_to_int)))  # should ~ 74000+
print()

# print tokens in first review
print('Tokenized review: \n', reviews_ints[:1])


# ### Encoding the labels
# 
# Our labels are "positive" or "negative". To use these labels in our network, we need to convert them to 0 and 1.
# 
# > **Exercise:** Convert labels from `positive` and `negative` to 1 and 0, respectively, and place those in a new list, `encoded_labels`.

# In[ ]:


# 1=positive, 0=negative label conversion
labels = labels.split("\n")
encoded_labels = [1 if i == "positive" else 0 for i in labels]


# ### Removing Outliers
# 
# As an additional pre-processing step, we want to make sure that our reviews are in good shape for standard processing. That is, our network will expect a standard input text size, and so, we'll want to shape our reviews into a specific length. We'll approach this task in two main steps:
# 
# 1. Getting rid of extremely long or short reviews; the outliers
# 2. Padding/truncating the remaining data so that we have reviews of the same length.
# 
# <img src="assets/outliers_padding_ex.png" width=40%>
# 
# Before we pad our review text, we should check for reviews of extremely short or long lengths; outliers that may mess with our training.

# In[ ]:


# outlier review stats
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# Okay, a couple issues here. We seem to have one review with zero length. And, the maximum review length is way too many steps for our RNN. We'll have to remove any super short reviews and truncate super long reviews. This removes outliers and should allow our model to train more efficiently.
# 
# > **Exercise:** First, remove *any* reviews with zero length from the `reviews_ints` list and their corresponding label in `encoded_labels`.

# In[ ]:


print('Number of reviews before removing outliers: ', len(reviews_ints))

idx = [i for i,review in enumerate(reviews_ints) if len(reviews_ints[i])!=0]
## remove any reviews/labels with zero length from the reviews_ints list.

reviews_ints = [reviews_ints[i] for i in idx] 
encoded_labels = [encoded_labels[i] for i in idx]

print('Number of reviews after removing outliers: ', len(reviews_ints))


# ---
# ## Padding sequences
# 
# To deal with both short and very long reviews, we'll pad or truncate all our reviews to a specific length. For reviews shorter than some `seq_length`, we'll pad with 0s. For reviews longer than `seq_length`, we can truncate them to the first `seq_length` words. A good `seq_length`, in this case, is 200.
# 
# > **Exercise:** Define a function that returns an array `features` that contains the padded data, of a standard size, that we'll pass to the network. 
# * The data should come from `review_ints`, since we want to feed integers to the network. 
# * Each row should be `seq_length` elements long. 
# * For reviews shorter than `seq_length` words, **left pad** with 0s. That is, if the review is `['best', 'movie', 'ever']`, `[117, 18, 128]` as integers, the row will look like `[0, 0, 0, ..., 0, 117, 18, 128]`. 
# * For reviews longer than `seq_length`, use only the first `seq_length` words as the feature vector.
# 
# As a small example, if the `seq_length=10` and an input review is: 
# ```
# [117, 18, 128]
# ```
# The resultant, padded sequence should be: 
# 
# ```
# [0, 0, 0, 0, 0, 0, 0, 117, 18, 128]
# ```
# 
# **Your final `features` array should be a 2D array, with as many rows as there are reviews, and as many columns as the specified `seq_length`.**
# 
# This isn't trivial and there are a bunch of ways to do this. But, if you're going to be building your own deep learning networks, you're going to have to get used to preparing your data.

# In[ ]:


def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    features = []
    
    ## implement function
    for review in reviews_ints:
        if len(review)<seq_length:
            features.append(list(np.zeros(seq_length-len(review)))+review)
        elif len(review)>seq_length:
            features.append(review[:seq_length])
        else:
            features.append(review)
    
    features = np.asarray(features, dtype=int)
    return features


# In[ ]:


# Test your implementation!

seq_length = 200

features = pad_features(reviews_ints, seq_length=seq_length)

## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches 
print(features[:30,:10])


# ## Training, Validation, Test
# 
# With our data in nice shape, we'll split it into training, validation, and test sets.
# 
# > **Exercise:** Create the training, validation, and test sets. 
# * You'll need to create sets for the features and the labels, `train_x` and `train_y`, for example. 
# * Define a split fraction, `split_frac` as the fraction of data to **keep** in the training set. Usually this is set to 0.8 or 0.9. 
# * Whatever data is left will be split in half to create the validation and *testing* data.

# In[ ]:


from sklearn.model_selection import train_test_split

split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)
train_x,remaining_x,train_y,remaining_y = train_test_split(features, encoded_labels, test_size = 0.2)
test_x,valid_x,test_y,valid_y = train_test_split(remaining_x,remaining_y, test_size = 0.5)

#as the labels returned by train_test_split is list, we will convert them to ndarray
train_y = np.asarray(train_y)
test_y = np.asarray(test_y)
valid_y = np.asarray(valid_y)
## print out the shapes of your resultant feature data
print((train_x.shape), (test_x.shape), (valid_x.shape))


# **Check your work**
# 
# With train, validation, and test fractions equal to 0.8, 0.1, 0.1, respectively, the final, feature data shapes should look like:
# ```
#                     Feature Shapes:
# Train set: 		 (20000, 200) 
# Validation set: 	(2500, 200) 
# Test set: 		  (2500, 200)
# ```

# ---
# ## DataLoaders and Batching
# 
# After creating training, test, and validation data, we can create DataLoaders for this data by following two steps:
# 1. Create a known format for accessing our data, using [TensorDataset](https://pytorch.org/docs/stable/data.html#) which takes in an input set of data and a target set of data with the same first dimension, and creates a dataset.
# 2. Create DataLoaders and batch our training, validation, and test Tensor datasets.
# 
# ```
# train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
# train_loader = DataLoader(train_data, batch_size=batch_size)
# ```
# 
# This is an alternative to creating a generator function for batching our data into full batches.

# In[ ]:


type(train_y)


# In[ ]:


import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 50

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


# In[ ]:


# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)


# ---
# # Sentiment Network with PyTorch
# 
# Below is where you'll define the network.
# 
# <img src="assets/network_diagram.png" width=40%>
# 
# The layers are as follows:
# 1. An [embedding layer](https://pytorch.org/docs/stable/nn.html#embedding) that converts our word tokens (integers) into embeddings of a specific size.
# 2. An [LSTM layer](https://pytorch.org/docs/stable/nn.html#lstm) defined by a hidden_state size and number of layers
# 3. A fully-connected output layer that maps the LSTM layer outputs to a desired output_size
# 4. A sigmoid activation layer which turns all outputs into a value 0-1; return **only the last sigmoid output** as the output of this network.
# 
# ### The Embedding Layer
# 
# We need to add an [embedding layer](https://pytorch.org/docs/stable/nn.html#embedding) because there are 74000+ words in our vocabulary. It is massively inefficient to one-hot encode that many classes. So, instead of one-hot encoding, we can have an embedding layer and use that layer as a lookup table. You could train an embedding layer using Word2Vec, then load it here. But, it's fine to just make a new layer, using it for only dimensionality reduction, and let the network learn the weights.
# 
# 
# ### The LSTM Layer(s)
# 
# We'll create an [LSTM](https://pytorch.org/docs/stable/nn.html#lstm) to use in our recurrent network, which takes in an input_size, a hidden_dim, a number of layers, a dropout probability (for dropout between multiple layers), and a batch_first parameter.
# 
# Most of the time, you're network will have better performance with more layers; between 2-3. Adding more layers allows the network to learn really complex relationships. 
# 
# > **Exercise:** Complete the `__init__`, `forward`, and `init_hidden` functions for the SentimentRNN model class.
# 
# Note: `init_hidden` should initialize the hidden and cell state of an lstm layer to all zeros, and move those state to GPU, if available.

# In[ ]:


# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')


# In[ ]:


import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # define all layers
        #embedding
        #LSTM
        #fully_connected
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,n_layers,
                            dropout=drop_prob, batch_first = True)
        self.FC = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        
        
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        #stack_up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.FC(lstm_out)
        
        sig_out = self.sig(out)
        
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers,batch_size,self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers,batch_size,self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers,batch_size,self.hidden_dim).zero_(),
                      weight.new(self.n_layers,batch_size,self.hidden_dim).zero_())

        return hidden
        


# ## Instantiate the network
# 
# Here, we'll instantiate the network. First up, defining the hyperparameters.
# 
# * `vocab_size`: Size of our vocabulary or the range of values for our input, word tokens.
# * `output_size`: Size of our desired output; the number of class scores we want to output (pos/neg).
# * `embedding_dim`: Number of columns in the embedding lookup table; size of our embeddings.
# * `hidden_dim`: Number of units in the hidden layers of our LSTM cells. Usually larger is better performance wise. Common values are 128, 256, 512, etc.
# * `n_layers`: Number of LSTM layers in the network. Typically between 1-3
# 
# > **Exercise:** Define the model  hyperparameters.
# 

# In[ ]:


# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int) + 1
output_size = 1
embedding_dim = 200 
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)


# ---
# ## Training
# 
# Below is the typical training code. If you want to do this yourself, feel free to delete all this code and implement it yourself. You can also add code to save a model by name.
# 
# >We'll also be using a new kind of cross entropy loss, which is designed to work with a single Sigmoid output. [BCELoss](https://pytorch.org/docs/stable/nn.html#bceloss), or **Binary Cross Entropy Loss**, applies cross entropy loss to a single value between 0 and 1.
# 
# We also have some data and training hyparameters:
# 
# * `lr`: Learning rate for our optimizer.
# * `epochs`: Number of times to iterate through the training dataset.
# * `clip`: The maximum gradient value to clip at (to prevent exploding gradients).

# In[ ]:


# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


# In[ ]:


# training params

epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1
        #print(counter)

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))


# ---
# ## Testing
# 
# There are a few ways to test your network.
# 
# * **Test data performance:** First, we'll see how our trained model performs on all of our defined test_data, above. We'll calculate the average loss and accuracy over the test data.
# 
# * **Inference on user-generated data:** Second, we'll see if we can input just one example review at a time (without a label), and see what the trained model predicts. Looking at new, user input data like this, and predicting an output label, is called **inference**.

# In[ ]:


# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    # get predicted outputs
    output, h = net(inputs, h)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))


# ### Inference on a test review
# 
# You can change this test_review to any text that you want. Read it and think: is it pos or neg? Then see if your model predicts correctly!
#     
# > **Exercise:** Write a `predict` function that takes in a trained net, a plain text_review, and a sequence length, and prints out a custom statement for a positive or negative review!
# * You can use any functions that you've already defined or define any helper functions you want to complete `predict`, but it should just take in a trained net, a text review, and a sequence length.
# 

# In[ ]:


# negative test review
test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'


# In[ ]:





# In[ ]:


def preprocess(review, vocab_to_int):
    review = review.lower()
    word_list = review.split()
    num_list = []
    #list of reviews
    #though it contains only one review as of now
    reviews_int = []
    for word in word_list:
        if word in vocab_to_int.keys():
            num_list.append(vocab_to_int[word])
    reviews_int.append(num_list)
    return reviews_int
    


# In[ ]:


def predict(net, test_review, sequence_length=200):
    ''' Prints out whether a give review is predicted to be 
        positive or negative in sentiment, using a trained model.
        
        params:
        net - A trained net 
        test_review - a review made of normal text and punctuation
        sequence_length - the padded length of a review
        '''
    #change the reviews to sequence of integers
    int_rev = preprocess(test_review, vocab_to_int)
    #pad the reviews as per the sequence length of the feature
    features = pad_features(int_rev, seq_length=seq_length)
    
    #changing the features to PyTorch tensor
    features = torch.from_numpy(features)
    
    #pass the features to the model to get prediction
    net.eval()
    val_h = net.init_hidden(1)
    val_h = tuple([each.data for each in val_h])

    if(train_on_gpu):
        features = features.cuda()

    output, val_h = net(features, val_h)
    
    #rounding the output to nearest 0 or 1
    pred = torch.round(output)
    
    #mapping the numeric values to postive or negative
    output = ["Positive" if pred.item() == 1 else "Negative"]
    
    # print custom response based on whether test_review is pos/neg
    print(output)
        


# In[ ]:


# positive test review
test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'


# In[ ]:


# call function
# try negative and positive reviews!
seq_length=200
predict(net, test_review_pos, seq_length)


# ### Try out test_reviews of your own!
# 
# Now that you have a trained model and a predict function, you can pass in _any_ kind of text and this model will predict whether the text has a positive or negative sentiment. Push this model to its limits and try to find what words it associates with positive or negative.
# 
# Later, you'll learn how to deploy a model like this to a production environment so that it can respond to any kind of user data put into a web app!

# In[ ]:




