#!/usr/bin/env python
# coding: utf-8

# # Introduction
# All of us, irrespective of our current level of expertise have come across LSTMs. They are intriguing in my opinion, some kind of a magical boxes which can take in sequences and spit out very accurate results.
# 
# However, although LSTMs are very good at what it does, deep learning newbies (read wannabees) like me cannot help think "they are too complicated", or "coding LSTM based models isnt my cup of tea". I have been there,.... hmmm... to be honest, some parts of me thinks that I am still there, inspite of working in Deep Learning architectures for over a year. But although LSTMs are actually very difficult concepts to grasp mathematically (remember "backpropagation through time" - sounds like some sci fi movie IMHO), coding them is not as difficult, thanks to PyTorch's simple coding style. Like always the simplicity with which we can develop complicated LSTM based RNNs in PyTorch is another example why it just brushes aside the competition (read. Tensorflow and its sidekick Keras).
# 
# In this notebook, I will try and explain how to code up a Part of Speech tagger. One simple one and a somewhat complicated one for POS Tagging. So lets get started.
# 
# ##### P.S.: I do not work for PyTorch or Facebook (although I really wish I did). My lame attempt at promoting PyTorch is just because I am a fan of its simplicity and interactivity.
# 
# ##### P.S. 2: In case my insignificant post comes across someone who develops Tensorflow/Keras or anyone in Google in general, please know that I am a huge fan of all the things you've done from scratch in order to promote deep learning and playing a major part in making it what it is today. 

# ### Importing all the necesarry packages

# In[ ]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ### Part 1: Simple small LSTM tagger
# The below code is a very lame attempt at developing a LSTM for part of speech tagging. One glance at the code and you'll probably realize its too simplistic. But for beginners it will be useful IMHO.

# In[ ]:


def prepare_sequence(seq, to_ix):
    """Input: takes in a list of words, and a dictionary containing the index of the words
    Output: a tensor containing the indexes of the word"""
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# This is the example training data
training_data = [
    ("the dog happily ate the big apple".split(), ["DET", "NN", "ADV", "V", "DET", "ADJ", "NN"]),
    ("everybody read that good book quietly in the hall".split(), ["NN", "V", "DET", "ADJ", "NN", "ADV", "PRP", "DET", "NN"]),
    ("the old head master sternly scolded the naughty children for \
     being very loud".split(), ["DET", "ADJ", "ADJ", "NN", "ADV", "V", "DET", "ADJ",  "NN", "PRP", "V", "ADJ", "NN"]),
    ("i love you loads".split(), ["PRN", "V", "PRN", "ADV"])
]
#  These are other words which we would like to predict (within sentences) using the model
other_words = ["area", "book", "business", "case", "child", "company", "country", 
               "day", "eye", "fact", "family", "government", "group", "hand", "home", 
               "job", "life", "lot", "man", "money", "month", "mother", "food", "night", 
               "number", "part", "people", "place", "point", "problem", "program", 
               "question", "right", "room", "school", "state", "story", "student", 
               "study", "system", "thing", "time", "water", "way", "week", "woman", 
               "word", "work", "world", "year", "ask", "be", "become", "begin", "can", 
               "come", "do", "find", "get", "go", "have", "hear", "keep", "know", "let", 
               "like", "look", "make", "may", "mean", "might", "move", "play", "put", 
               "run", "say", "see", "seem", "should", "start", "think", "try", "turn", 
               "use", "want", "will", "work", "would", "asked", "was", "became", "began", 
               "can", "come", "do", "did", "found", "got", "went", "had", "heard", "kept", 
               "knew", "let", "liked", "looked", "made", "might", "meant", "might", "moved", 
               "played", "put", "ran", "said", "saw", "seemed", "should", "started", 
               "thought", "tried", "turned", "used", "wanted" "worked", "would", "able", 
               "bad", "best", "better", "big", "black", "certain", "clear", "different", 
               "early", "easy", "economic", "federal", "free", "full", "good", "great", 
               "hard", "high", "human", "important", "international", "large", "late", 
               "little", "local", "long", "low", "major", "military", "national", "new", 
               "old", "only", "other", "political", "possible", "public", "real", "recent", 
               "right", "small", "social", "special", "strong", "sure", "true", "white", 
               "whole", "young", "he", "she", "it", "they", "i", "my", "mine", "your", "his", 
               "her", "father", "mother", "dog", "cat", "cow", "tiger", "a", "about", "all", 
               "also", "and", "as", "at", "be", "because", "but", "by", "can", "come", "could", 
               "day", "do", "even", "find", "first", "for", "from", "get", "give", "go", 
               "have", "he", "her", "here", "him", "his", "how", "I", "if", "in", "into", 
               "it", "its", "just", "know", "like", "look", "make", "man", "many", "me", 
               "more", "my", "new", "no", "not", "now", "of", "on", "one", "only", "or", 
               "other", "our", "out", "people", "say", "see", "she", "so", "some", "take", 
               "tell", "than", "that", "the", "their", "them", "then", "there", "these", 
               "they", "thing", "think", "this", "those", "time", "to", "two", "up", "use", 
               "very", "want", "way", "we", "well", "what", "when", "which", "who", "will", 
               "with", "would", "year", "you", "your"]

word_to_ix = {} # This is the word dictionary which will contain the index to each word

for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix.keys():
            word_to_ix[word] = len(word_to_ix)
for word in other_words:
    if word not in word_to_ix.keys():
            word_to_ix[word] = len(word_to_ix)

# print(word_to_ix) # Just have a look at what it contains

tag_to_ix = {"DET": 0, "NN": 1, "V": 2, "ADJ": 3, "ADV": 4, "PRP": 5, "PRN": 6} # This dictionary contains the indices of the tags

EMBEDDING_DIM = 64
HIDDEN_DIM = 64


# #### Model
# The below code is for making a very simple LSTM Model in PyTorch.
# The class LSTMTagger must subclass the nn.Module class of PyTorch so that it can get all the functionalities of the nn.Module class. Very simple.
# If this is confusing to you, dont worry just now just try and remember it as a convention.

# In[ ]:


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# In[ ]:



# Here I initialize the model with all the necesarry parameters
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix.keys()), len(tag_to_ix.keys()))

# Define the loss function as the Negative Log Likelihood loss (NLLLoss)
loss_function = nn.NLLLoss()

# We will be using a simple SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# The test sentence
seq1 = "everybody read the book and ate the food".split()
seq2 = "she like my dog".split()
print("Running a check on the model before training.\nSentences:\n{}\n{}".format(" ".join(seq1), " ".join(seq2)))
with torch.no_grad():
    for seq in [seq1, seq2]:
        inputs = prepare_sequence(seq, word_to_ix)
        tag_scores = model(inputs)
        _, indices = torch.max(tag_scores, 1)
        ret = []
        for i in range(len(indices)):
            for key, value in tag_to_ix.items():
                if indices[i] == value:
                    ret.append((seq[i], key))
        print(ret)
    
print("Training Started")
for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        
        tag_scores = model(sentence_in)
        
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
print("Training Finished!!!\nAgain testing on unknown data")
with torch.no_grad():
    for seq in [seq1, seq2]:
        inputs = prepare_sequence(seq, word_to_ix)
        tag_scores = model(inputs)
        _, indices = torch.max(tag_scores, 1)
        ret = []
        for i in range(len(indices)):
            for key, value in tag_to_ix.items():
                if indices[i] == value:
                    ret.append((seq[i], key))
        print(ret)


# ### Performance:
# So you might have guessed that this model is not that great in itself. Actually it is not supposed to be good. The words it can predict accurately are mostly words which the model had encountered during training, and remembers them during test time. Whenever it encounters a new word, it fails badly. 

# ## Part 2: Somewhat complicated LSTM Model
# In this part, we (including me!! This is my first attempt) will learn how to make a slightly more complicated LSTM Model which does the same thing: takes in a sentence and predicts all the POS Tags of the words.
# 
# The model will comprise of two LSTM based models, one for character wise and the other for word wise tag classification. So without further adieu lets get started.

# ### Downloading Data
# First we will download data from nltk, a very popular python natural language processing toolkit, incase you haven't heard of it.
# 
# Kindly refer to their site for more information: https://www.nltk.org/

# In[ ]:


import nltk

tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')
print("Number of Tagged Sentences ",len(tagged_sentence))


# In[ ]:


# Let's have a look at some sentences

print(tagged_sentence[:3][:10])


# Now we will construct a dictionary for all these:
# 1. A word/tag dictionary,
# 2. A letter/character dictionary
# 3. A POS tag dictionary

# In[ ]:


def word_to_ix(word, ix):
    return torch.tensor(ix[word], dtype = torch.long)

def char_to_ix(char, ix):
    return torch.tensor(ix[char], dtype= torch.long)

def tag_to_ix(tag, ix):
    return torch.tensor(ix[tag], dtype= torch.long)

def sequence_to_idx(sequence, ix):
    return torch.tensor([ix[s] for s in sequence], dtype=torch.long)


word_to_idx = {}
tag_to_idx = {}
char_to_idx = {}
for sentence in tagged_sentence:
    for word, pos_tag in sentence:
        if word not in word_to_idx.keys():
            word_to_idx[word] = len(word_to_idx)
        if pos_tag not in tag_to_idx.keys():
            tag_to_idx[pos_tag] = len(tag_to_idx)
        for char in word:
            if char not in char_to_idx.keys():
                char_to_idx[char] = len(char_to_idx)


# In[ ]:


word_vocab_size = len(word_to_idx)
tag_vocab_size = len(tag_to_idx)
char_vocab_size = len(char_to_idx)

print("Unique words: {}".format(len(word_to_idx)))
print("Unique tags: {}".format(len(tag_to_idx)))
print("Unique characters: {}".format(len(char_to_idx)))


# ## Train/Test Splitting the data

# In[30]:


import random

tr_random = random.sample(list(range(len(tagged_sentence))), int(0.95 * len(tagged_sentence)))

train = [tagged_sentence[i] for i in tr_random]
test = [tagged_sentence[i] for i in range(len(tagged_sentence)) if i not in tr_random]


# ### Defining the model parameters here

# In[31]:


WORD_EMBEDDING_DIM = 1024
CHAR_EMBEDDING_DIM = 128
WORD_HIDDEN_DIM = 1024
CHAR_HIDDEN_DIM = 1024
EPOCHS = 70


# ## The model classes
# Here I will construct the model classes, the Word LSTM tagger and the Character LSTM Tagger.
# The LSTM in both cases takes in a sequence (words or characters), embeds the sequence into an embedding space (dimension of the space is a hyperparameter), and runs the LSTM model.
# 
# In our case, first the word level LSTM will take in a sequence of words and convert them into the word embedding space.
# Similarly, it will take the words sequentially  and run the character level LSTM model, which will first take in the sequence of characters in each word and project it in the character embedding space and then run the LSTM model and take its hidden state and feed it back to the word LSTM model.
# 
# Using the character level hidden representation for every word as well as the word embedding, the word level model then runs LSTM on the sequence of words, and outputs the predictions for every tag. This prediction is later processed to find the corresponding tags.

# In[32]:


class DualLSTMTagger(nn.Module):
    def __init__(self, word_embedding_dim, word_hidden_dim, char_embedding_dim, char_hidden_dim, word_vocab_size, char_vocab_size, tag_vocab_size):
        super(DualLSTMTagger, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)
        
        self.lstm = nn.LSTM(word_embedding_dim + char_hidden_dim, word_hidden_dim)
        self.hidden2tag = nn.Linear(word_hidden_dim, tag_vocab_size)
        
    def forward(self, sentence, words):
        embeds = self.word_embedding(sentence)
        char_hidden_final = []
        for word in words:
            char_embeds = self.char_embedding(word)
            _, (char_hidden, char_cell_state) = self.char_lstm(char_embeds.view(len(word), 1, -1))
            word_char_hidden_state = char_hidden.view(-1)
            char_hidden_final.append(word_char_hidden_state)
        char_hidden_final = torch.stack(tuple(char_hidden_final))
        
        combined = torch.cat((embeds, char_hidden_final), 1)

        lstm_out, _ = self.lstm(combined.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# In[ ]:


model = DualLSTMTagger(WORD_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, word_vocab_size, char_vocab_size, tag_vocab_size)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if use_cuda:
    model.cuda()

# Define the loss function as the Negative Log Likelihood loss (NLLLoss)
loss_function = nn.NLLLoss()

# We will be using a simple SGD optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# The test sentence
seq = "everybody eat the food . I kept looking out the window , trying to find the one I was waiting for .".split()
print("Running a check on the model before training.\nSentences:\n{}".format(" ".join(seq)))
with torch.no_grad():
    words = [torch.tensor(sequence_to_idx(s[0], char_to_idx), dtype=torch.long).to(device) for s in seq]
    sentence = torch.tensor(sequence_to_idx(seq, word_to_idx), dtype=torch.long).to(device)
        
    tag_scores = model(sentence, words)
    _, indices = torch.max(tag_scores, 1)
    ret = []
    for i in range(len(indices)):
        for key, value in tag_to_idx.items():
            if indices[i] == value:
                ret.append((seq[i], key))
    print(ret)
# Training start
print("Training Started")
accuracy_list = []
loss_list = []
interval = round(len(train) / 100.)
epochs = EPOCHS
e_interval = round(epochs / 10.)
for epoch in range(epochs):
    acc = 0 #to keep track of accuracy
    loss = 0 # To keep track of the loss value
    i = 0
    for sentence_tag in train:
        i += 1
        words = [torch.tensor(sequence_to_idx(s[0], char_to_idx), dtype=torch.long).to(device) for s in sentence_tag]
        sentence = [s[0] for s in sentence_tag]
        sentence = torch.tensor(sequence_to_idx(sentence, word_to_idx), dtype=torch.long).to(device)
        targets = [s[1] for s in sentence_tag]
        targets = torch.tensor(sequence_to_idx(targets, tag_to_idx), dtype=torch.long).to(device)
        
        model.zero_grad()
        
        tag_scores = model(sentence, words)
        
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        loss += loss.item()
        _, indices = torch.max(tag_scores, 1)
#         print(indices == targets)
        acc += torch.mean(torch.tensor(targets == indices, dtype=torch.float))
        if i % interval == 0:
            print("Epoch {} Running;\t{}% Complete".format(epoch + 1, i / interval), end = "\r", flush = True)
    loss = loss / len(train)
    acc = acc / len(train)
    loss_list.append(float(loss))
    accuracy_list.append(float(acc))
    if (epoch + 1) % e_interval == 0:
        print("Epoch {} Completed,\tLoss {}\tAccuracy: {}".format(epoch + 1, np.mean(loss_list[-e_interval:]), np.mean(accuracy_list[-e_interval:])))


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(accuracy_list, c="red", label ="Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.plot(loss_list, c="blue", label ="Loss")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.show()


# ### Singe Test after Training

# In[ ]:


with torch.no_grad():
    words = [torch.tensor(sequence_to_idx(s[0], char_to_idx), dtype=torch.long).to(device) for s in seq]
    sentence = torch.tensor(sequence_to_idx(seq, word_to_idx), dtype=torch.long).to(device)
        
    tag_scores = model(sentence, words)
    _, indices = torch.max(tag_scores, 1)
    ret = []
    for i in range(len(indices)):
        for key, value in tag_to_idx.items():
            if indices[i] == value:
                ret.append((seq[i], key))
    print(ret)


# ### Result:
# It seems our model is able to learn the words well even in the short training.
# 
# In the next kernel, we will explore deeper LSTM models based on similar lines.
# 
# The deeper model can be found here.
# https://www.kaggle.com/krishanudb/deep-pos-tagger/
