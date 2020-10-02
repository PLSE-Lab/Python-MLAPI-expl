#!/usr/bin/env python
# coding: utf-8

# # Tutorial for training Word Embeddings and Pytorch

# For this tutorial I'll be using the getting started twitter data challenge.

# In[ ]:


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


# # Word Embeddings

# This section is designed to teach you how to train your own word embedding vectors and assumes you know the concept of [word embedding vectors](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa). (if not click the link)  
# 
# Premise: Pretrained vectors are awesome since majority of the work has been done for you; however, not all pretrained vectors are appropiate for all tasks.
# e.g. using twitter embeddings to predict newspaper articles. 

# A useful library to train word embeddings is the gensim library. This library was constructed to process and create word vectors with ease. So first step is to load the data (everybody loves PANDAS!!) and import the library. 

# In[ ]:


training_data = pd.read_csv("../input/nlp-getting-started/train.csv")
training_data.head()


# In[ ]:


import gensim


# Now the library has been loaded. So the first step is to [read the documentation](https://radimrehurek.com/gensim/auto_examples/index.html). Just kidding. The first step to training ones own word embeddings is to pick the model they want to use. Gensim has a few models implemented in their library such as vanilla [Word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), [Doc2vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) and Facebook's [fastText](https://arxiv.org/abs/1607.04606). For the sake of this introudction I'll be using Word2Vec, but try out others yourself.

# Gensim's word2vec model takes sentences as a parameter. This variable assumes the data being passed in are list of list of strings (example of input provided in the code). This means that each "document" is contained within one entry of the list. Luckily, using this dataset each tweet is consider their own document which makes this process very simple.

# In[ ]:


(
    training_data
    .text
    .apply(lambda x: x.split(" "))
    .head()
    .values
)


# Note that running the line below will begin the training process and model is the word embeddings network that we just trained. For this tutorial I'll be using default parameters, but the [documentation](https://radimrehurek.com/gensim/models/word2vec.html) explains each parameter (scroll down to the end to get the idea). 

# In[ ]:


training_corpus = training_data.text.apply(lambda x: x.split(" "))
model = gensim.models.Word2Vec(sentences=training_corpus)


# Wow that was fast. Well now we have our model and now we can do interesting things such as observe words in our vocabulary, look at the vectors themselves and observe word similarity. Note that anytime you want to use wordvectors from the model object you have to call: "model.wv". Older packages of this library allowed you to just do "model['word']", but has since be depreciated

# In[ ]:


model.wv.vocab


# In[ ]:


model.wv['Forest']


# Note that twitter data is messy and it would be wise to do some preprocessing of the data to make sure you dont associate the word "Forest" with unhelpful words like 4 and 2.

# In[ ]:


model.wv.most_similar("Forest")


# To get the word vectors themselves you have to call: "model.wv.vectors"

# In[ ]:


model.wv.vectors


# Now that you have ran through this quick tutorial you should now be able to train your own models and get your own word vectors.
# Note that I glossed over a lot more this library can do, so it would do you wonders to take a look at the documentation of the library and familiarize yourself with everything.

# # Pytorch

# ## Object-Oriented Programming

# Pytorch is a deep learning library (like Tensorflow) that allows you to create and run deep learning models. This section is designed to teach you how pytorch works.

# Before we dive into pytorch one needs to understand the concept of object oriented programming. This central idea is based on the idea of an object. Think of an object as a simple entity like a Cat, Car or a backpack. These objects are able to perform tasks such as meow, move foward or carry something. (Obviously a car can't meow, but you get the idea) So why does any of this matter? Well in programming languages like Java and Python you can construct your own objects also referred to as classes. For example:

# In[ ]:


class MyBackpack():
    pass


# This line of code constructs the mybackpack class (object) and we have yet to add methods for it. So let's add a method to the class.

# In[ ]:


class MyBackPack():
    def __init__(self):
        self.container = []
        
    def hold(self, obj):
        self.container.append(obj)


# The init function is called the constructor that instantiates the object, which is a fancy term that means constructs the object in memory. Note that all functions within the class has to have the keyword self which refers to itself. 

# So now we created our class that represents a backpack. Lets put something in the backpack.

# In[ ]:


backpack = MyBackPack()
backpack.hold("candy")


# Now the backpack holds "candy". Confirmed below:

# In[ ]:


print(backpack.container)


# Now the one important concept to learn is object inheritance. This works almost like genetic inheritance, where offspring "inherits" attributes from their parents. In this case objects "inherits" methods and attributes from its *parent* class.

# To perform inheritance in python you have to do the following:

# In[ ]:


class Container():
    def __init__(self):
        self.message = "I am a container whoo."


# In[ ]:


class MyBackPack(Container):
    def __init__(self):
        super().__init__()
        self.container = []
        
    def hold(self, obj):
        self.container.append(obj)


# In the class declaration the contain class is in parenthesis from the mybackpack class. This means the container class is the parent of mybackpack class. The super keyword says access the parent class and instantiate it before creating the child class. So going back to holding the candy string example lets see inheritance at work:

# In[ ]:


backpack = MyBackPack()
backpack.hold("candy")
print(backpack.message)


# As you may note the mybackpack object had a message variable that was not declared in its class. This is important to know as pytorch uses this concept to allow you to construct your own models.
# 
# Awesome hopefully you get the concept of classes, how to creating methods within classes and how to allow classes "inherit" other classes

# ## Pytorch

# Now is the time to explain pytorch (Yay!!). Main reason I like pytorch is that it uses object oriented programming to build deep learning networks and makes the code SOOO MUCH easier to read and understand.

# First thing to do is import the torch package.

# In[ ]:


import torch 
import torch.nn as nn


# You may be wondering why the double import and the reason for this is that pytorch wraps all their deep learning layers under the nn package. The first import contains functions for objects called Tensors, which is a fancy way of describing a free form matrix ([more info here](https://en.wikipedia.org/wiki/Tensor)). This tutorial describes the syntax and creates a simple network, but there are a TON of resources out there that implements complex networks etc. [Pytorch's documentation here](https://pytorch.org/docs/stable/nn.html). [Introduction to Pytorch Here](https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html)

# So lets create our neural network. This network is a simple two layer neural network that takes features in the first layer and outputs predictions within the second layer.

# In[ ]:


class FirstNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        pass


# Within this class description you will start to see some basic object oriented concepts. The nerual network is the sibiling of the nn.Module class. Every model you create within pytorch will inherit some form of the nn.Module class, which is one reason why pytorch is quite intuitive.

# Every model created using pytorch must define a function called forward, which takes in the data you want to pass into your network and spits out the output of the network. This function is important to implement, because it is how your network learns to function. So lets fill in the foward function now:

# In[ ]:


class FirstNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, data_input):
        output_layer1 = self.layer1(data_input)
        output_layer1 = self.relu(output_layer1)
        output_layer2 = self.layer2(output_layer1)
        return output_layer2


# Within this forward function the data is passed into the first layer and then the data is passed into an activation function which is then passed into the second layer which then gets outputed for downstream operations (discussed below).

# Now lets instantiate the model and take a look at what pytorch does:

# In[ ]:


first_neural_network = FirstNeuralNetwork(300, 100, 2)
print(first_neural_network)


# The second beauty of pytorch is that you can take a look at your network by printing it out. This network says that we have two layers and an activation function. The first takes 300 features and maps them onto a 100 feature space. The second takes the 100 features passes it into an activation function and then maps them onto a 2 feature space.

# ### Train a model

# Ok that was simple. Now we need to get our data together to train a network.

# Machines cannot process words only numbers. Given that problem we need to define a way to map words to numbers. An easy way to do this is to use word embeddings, which is what was described above. Pytorch lets you incorporate your own word embeddings through a layer called the embedding layer. This layer is designed to map words directly with the emedding vectors as input. The only catch is that we have to individually map the words to indicies first. I'll visually show you in a second.**

# In[ ]:


# Gather the sentences from the training data
sentence_length = training_data.text.apply(lambda x: x.split(" ")).apply(len).max()
sentence_length


# In[ ]:


# Get a dictionary mapping of the vocab
word_map = {
    word:idx
    for idx,word in enumerate(model.wv.vocab, start=2)
}
word_map


# So now we have words associated with numbers. This is important the nerual network model can know which embedding index to use. The next step is that we need to map words in our corpus onto the individual numbers themselves. Code below:

# In[ ]:


(
   training_data
   .text
   .apply(lambda x: x.split(" "))
)[0]


# In[ ]:


training_sentence_data = (
    training_data
    .text
    .apply(lambda x: list(map(lambda word: word_map[word] if word in word_map else 1, x.split(" "))))
)
print(training_sentence_data[0])


# Now the last step is to 0 pad each sentence so every sentence is the same length for the network to learn.

# In[ ]:


training_sentence_data = (
    list(
        map(
            lambda x: pd.np.pad(x, (0, sentence_length-len(x))), 
            training_sentence_data
        )
    )
)
training_sentence_data[0]


# Awesome now every word in each sentence is associated with number that our network can use. One problem is that every sentence is just a numpy array but we need tensors. Question is how can we convert these numbers to tensors? Answer is that pytorch has forseen this issue and makes it super easy to convert from numpy arrays to tensors. Just call the tensor class.

# In[ ]:


torch.LongTensor(training_sentence_data[0])


# In[ ]:


training_sentence_data = (
    list(
        map(
            lambda x: torch.LongTensor(x), 
            training_sentence_data
        )
    )
)
training_sentence_data[0]


# For neural networks in NLP we often find words that are out of our vocabulary (words that may not have appeared in our training set). When this happens usually an unknown token is used to represent this problem. Conveniently, for this tutorial the number 1 represent unknown token for the nerual network to process.

# In[ ]:


pd.np.random.seed(100)


# In[ ]:


model.wv.vectors.shape


# In[ ]:


word_vectors_for_training = pd.np.insert(
    model.wv.vectors,   
    0, 
    pd.np.random.uniform(model.wv.vectors.min(),model.wv.vectors.max(),100),
    axis=0
)

word_vectors_for_training = pd.np.insert(
    word_vectors_for_training,   
    0, 
    pd.np.zeros(100),
    axis=0
)
word_vectors_for_training = torch.FloatTensor(word_vectors_for_training)
word_vectors_for_training.shape


# Now we have everything we need to set up the network. So now it is time to update the nerual network to contain the embedding layer and process the data

# In[ ]:


class FirstNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.embedding_layer = nn.EmbeddingBag.from_pretrained(word_vectors_for_training, mode="mean")
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, data_input):
        embedded_data_input = self.embedding_layer(data_input)
        output_layer1 = self.layer1(embedded_data_input)
        output_layer1 = self.relu(output_layer1)
        output_layer2 = self.layer2(output_layer1)
        # return the predictions but drop the axis
        return output_layer2.squeeze()


# Note that the only difference is that the embedding layer was added in both the init and the forward method.

# In[ ]:


# finalize the training data
training_sentence_data = torch.stack(training_sentence_data)

# Define the network
first_neural_network = FirstNeuralNetwork(100, 50, 1)


# Before we can train the model we now have to set up the data loading section. This involves constructing a dataloader object and a dataset object that pytorch conveniently provides for you. To get these objects we will have to import torch.utils.data.

# In[ ]:


import torch.utils.data as data


# Now lets construct the dataloader and a dataset so our model can learn.

# In[ ]:


dataset = data.TensorDataset(training_sentence_data, torch.FloatTensor(training_data.target.values))
dataloader = data.DataLoader(dataset, batch_size=256)


# Now we have our data loader and we are almost ready to train the model. Next step is to figure out the loss function and optmizer. In our case we are going to use the BinaryCrossEntropy loss function and the Adam optmimizer. To get the optimizer we will need to import the torch.optim library.

# In[ ]:


import torch.optim as optim


# In[ ]:


optimizer = optim.Adam(first_neural_network.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()


# In[ ]:


loss = nn.BCEWithLogitsLoss()
in_mat = torch.randn(3, requires_grad=True)
tar_mat = torch.empty(3).random_(2)
output = loss(in_mat, tar_mat)
print(in_mat)
print(tar_mat)
print(output)


# One thing to note is that pytorch handles backpropagation automatically which makes updating the weights of the network as simple as function calling. (loss.backwards()) With everything in place we can now set up the training loop to train the network:

# In[ ]:


#progress bar
import tqdm

# number of epochs
for n in tqdm.tqdm(range(100)):
    avg_loss = []

    for batch in dataloader:
        # batch [0] - the sentence data or X
        # batch [1] - the label for each sentence or Y
        
        # for every back pass you need to zero out the optimizer
        # less you get residual gradients
        optimizer.zero_grad()
        
        # pass the model into the batch
        # this line is the same as calling first_neural_network.foward(batch)
        # yay shortcuts
        output = first_neural_network(batch[0])
        
        # Calculate the loss function
        loss = loss_fn(output, batch[1])
        
        # Save the loss for each epoch
        avg_loss.append(loss.item())
        
        # Tell pytorch to calculate the gradient
        loss.backward()
        
        # tell pytorch to pass the gradients back into the model
        optimizer.step()
    print(pd.np.mean(avg_loss))


# Voila we have successfully trained a neural network!! Now let's run it on our testing data.
# 

# In[ ]:


# load testing data
testing_data = pd.read_csv("../input/nlp-getting-started/test.csv")
testing_data.head()


# convert each text to a testing matrix. Label each word in the word embedding matrix by its index in the word embedding matrix otherwise label it 1.

# In[ ]:


testing_sentence_data = (
    testing_data
    .text
    .apply(lambda x: list(map(lambda word: word_map[word] if word in word_map else 1, x.split(" "))))
)
print(testing_sentence_data[0])


# Now the last step is to 0 pad each sentence so every sentence is the same length for the network to test.

# In[ ]:


testing_sentence_data = (
    list(
        map(
            lambda x: pd.np.pad(x, (0, sentence_length-len(x))), 
            testing_sentence_data
        )
    )
)
testing_sentence_data[0]


# convert our numpy matrix to a series of tensors

# In[ ]:


testing_sentence_data = (
    list(
        map(
            lambda x: torch.LongTensor(x), 
            testing_sentence_data
        )
    )
)

# finalize the testing data
testing_sentence_data = torch.stack(testing_sentence_data)


# load the testing data

# In[ ]:


test_dataset = data.TensorDataset(testing_sentence_data)
test_dataloader = data.DataLoader(test_dataset, batch_size=256)


# run the testing tweets through the model

# In[ ]:


prediction_data = testing_data.loc[:,["id","text"]].copy()
lowest_idx=0
for t_batch in test_dataloader:
        # batch [0] - the sentence data or X
        
        
        # pass the model into the batch
        # this line is the same as calling first_neural_network.foward(batch)
        # yay shortcuts
        output = nn.functional.softmax(first_neural_network(t_batch[0]))
        bsize = len(output)
        prediction_data.loc[range(lowest_idx,lowest_idx+bsize),"prob_target"] = (
            output.detach().numpy()
        )
        lowest_idx += bsize
prediction_data.head()


# In[ ]:


prediction_data.loc[prediction_data["log_target"]<0.5,"target"]=0
prediction_data.loc[prediction_data["log_target"]>0.5,"target"]=1


# In[ ]:


prediction_data.to_csv(
    "submission.csv",
    sep=",",
    columns=["id","target"],
    header=True,
    index=False
)


# In[ ]:


pd.read_csv("submission.csv").head()


# There are a vast majority of options for neural networks and this is a VERY simple example. The world is your oyster, so feel free to being shipping out your own forms of network architecture.
