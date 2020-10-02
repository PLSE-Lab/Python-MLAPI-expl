#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch # Deep learning framework
import torch.nn.functional as F

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

#Init random seed to get reproducible results
seed = 1111
random.seed(seed)
np.random.RandomState(seed)
torch.manual_seed(seed)

print(seed)

# Any results you write to the current directory are saved as output.
x_train_full = open("../input/x_train.txt").read().splitlines()
y_train_full = open("../input/y_train.txt").read().splitlines()
print('Example:')
print('LANG =', y_train_full[0])
print('TEXT =', x_train_full[0])


# In[ ]:


class Dictionary(object):
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

    def add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def __len__(self):
        return len(self.idx2token)


# The **Dictionary** class is used to map tokens (characters, words, subwords) into consecutive integer indexes.  
# The index **0** is reserved for padding sequences up to a fixed lenght, and the index **1** for any 'unknown' character

# In[ ]:


char_vocab = Dictionary()
pad_token = '<pad>' # reserve index 0 for padding
unk_token = '<unk>' # reserve index 1 for unknown token
pad_index = char_vocab.add_token(pad_token)
unk_index = char_vocab.add_token(unk_token)

# join all the training sentences in a single string
# and obtain the list of different characters with set
chars = set(''.join(x_train_full))
for char in sorted(chars):
    char_vocab.add_token(char)
print("Vocabulary:", len(char_vocab), "UTF characters")

lang_vocab = Dictionary()
# use python set to obtain the list of languages without repetitions
languages = set(y_train_full)
for lang in sorted(languages):
    lang_vocab.add_token(lang)
print("Labels:", len(lang_vocab), "languages")


# In[ ]:


#From token or label to index
print('a ->', char_vocab.token2idx['a'])
print('cat ->', lang_vocab.token2idx['cat'])
print(y_train_full[0], x_train_full[0][:10])
x_train_idx = [np.array([char_vocab.token2idx[c] for c in line]) for line in x_train_full]
y_train_idx = np.array([lang_vocab.token2idx[lang] for lang in y_train_full])
print(y_train_idx[0], x_train_idx[0][:10])


# Radomly select 15% of the database for validation  
# Create lists of (input, target) tuples for training and validation

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_idx, y_train_idx, test_size=0.15, random_state=seed)
train_data = [(x, y) for x, y in zip(x_train, y_train)]
val_data = [(x, y) for x, y in zip(x_val, y_val)]
print(len(train_data), "training samples")
print(len(val_data), "validation samples")


# In[ ]:


def batch_generator(data, batch_size, token_size):
    """Yield elements from data in chunks with a maximum of batch_size sequences and token_size tokens."""
    minibatch, sequences_so_far, tokens_so_far = [], 0, 0
    for ex in data:
        minibatch.append(ex)
        seq_len = len(ex[0])
        if seq_len > token_size:
            ex = (ex[0][:token_size], ex[1])
            seq_len = token_size
        sequences_so_far += 1
        tokens_so_far += seq_len
        if sequences_so_far == batch_size or tokens_so_far == token_size:
            yield minibatch
            minibatch, sequences_so_far, tokens_so_far = [], 0, 0
        elif sequences_so_far > batch_size or tokens_so_far > token_size:
            yield minibatch[:-1]
            minibatch, sequences_so_far, tokens_so_far = minibatch[-1:], 1, len(minibatch[-1][0])
    if minibatch:
        yield minibatch


# In[ ]:


def pool_generator(data, batch_size, token_size, shuffle=False):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*token_size, sorts examples within
    each chunk, then batch these examples and shuffle the batches.
    """
    for p in batch_generator(data, batch_size * 100, token_size * 100):
        p_batch = batch_generator(sorted(p, key=lambda t: len(t[0]), reverse=True), batch_size, token_size)
        p_list = list(p_batch)
        if shuffle:
            for b in random.sample(p_list, len(p_list)):
                yield b
        else:
            for b in p_list:
                yield b


# **DNN Model**  
# Includes Python comments with the dimension of the input  matrix:  
# T = Max number of tokens in a sequence  
# B = Number of sequences (batch size)  
# E = Embedding size  
# H = Hidden size  
# O = Output size (number of languages)

# In[ ]:


class CharRNNClassifier(torch.nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, model="lstm", num_layers=1, bidirectional=False, pad_idx=0):
        super().__init__()
        self.model = model.lower()
        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(input_size, embedding_size, padding_idx=pad_idx)
        if self.model == "gru":
            self.rnn = torch.nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=bidirectional)
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=bidirectional)
            
        self.linear1 = torch.nn.Linear(hidden_size, output_size)
        self.linear2 = torch.nn.Linear(output_size, input_size)
        
    def forward(self, input, input_lengths):
        # T x B
        encoded = self.embed(input)
        # T x B x E
        packed = torch.nn.utils.rnn.pack_padded_sequence(encoded, input_lengths)
        # Packed T x B x E
        output, _ = self.rnn(packed)
        # Packed T x B x H
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=float('-inf'))
        # T x B x H
        padded = padded.permute(1,2,0)
        # B x H x T
        output = F.adaptive_max_pool1d(padded, 1).view(-1, self.hidden_size)
                
        # B x H
        output = self.linear1(output)
        # B x O
        
        output = self.linear2(output)
        
        return output


# The 2-gram modelling was implemented as recommended in the mytorch tutorials.

# In[ ]:


if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Select 'GPU On' on kernel settings")
device = torch.device("cuda")
torch.cuda.manual_seed(seed)


# The **nn.CrossEntropyLoss()** criterion combines **nn.LogSoftmax()** and **nn.NLLLoss()** in one single class.  
# It is useful when training a classification problem.

# In[ ]:


criterion = torch.nn.CrossEntropyLoss(reduction='sum')


# In[ ]:


def train(model, optimizer, data, batch_size, token_size, log=False):
    model.train()
    total_loss = 0
    ncorrect = 0
    nsentences = 0
    ntokens = 0
    niterations = 0
    for batch in pool_generator(data, batch_size, token_size, shuffle=True):
        # Get input and target sequences from batch
        X = [torch.from_numpy(d[0]) for d in batch]
        X_lengths = [x.numel() for x in X]
        ntokens += sum(X_lengths)
        X_lengths = torch.tensor(X_lengths, dtype=torch.long, device=device)
        y = torch.tensor([d[1] for d in batch], dtype=torch.long, device=device)
        # Pad the input sequences to create a matrix
        X = torch.nn.utils.rnn.pad_sequence(X).to(device)
        model.zero_grad()
        output = model(X, X_lengths)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        # Training statistics
        total_loss += loss.item()
        ncorrect += (torch.max(output, 1)[1] == y).sum().item()
        nsentences += y.numel()
        niterations += 1
    
    total_loss = total_loss / nsentences
    accuracy = 100 * ncorrect / nsentences
    if log:
        print(f'Train: wpb={ntokens//niterations}, bsz={nsentences//niterations}, num_updates={niterations}')
    return accuracy


# In[ ]:


def validate(model, data, batch_size, token_size):
    model.eval()
    # calculate accuracy on validation set
    ncorrect = 0
    nsentences = 0
    with torch.no_grad():
        for batch in pool_generator(data, batch_size, token_size):
            # Get input and target sequences from batch
            X = [torch.from_numpy(d[0]) for d in batch]
            X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long, device=device)
            y = torch.tensor([d[1] for d in batch], dtype=torch.long, device=device)
            # Pad the input sequences to create a matrix
            X = torch.nn.utils.rnn.pad_sequence(X).to(device)
            answer = model(X, X_lengths)
            ncorrect += (torch.max(answer, 1)[1] == y).sum().item()
            nsentences += y.numel()
        dev_acc = 100 * ncorrect / nsentences
    return dev_acc


# In[ ]:


hidden_size = 256
embedding_size = 64
bidirectional = False
ntokens = len(char_vocab)
nlabels = len(lang_vocab)


# 

# In[ ]:


def testParameters(model, optimizer, epochs, batch_size, token_size):    
    train_accuracy = []
    valid_accuracy = []
    for epoch in range(1, epochs + 1):
        acc = train(model, optimizer, train_data, batch_size, token_size, log=epoch==1)
        train_accuracy.append(acc)
        acc = validate(model, val_data, batch_size, token_size)
        valid_accuracy.append(acc)
    
    return train_accuracy, valid_accuracy


# In[ ]:


Hyperparameter optimisation:
    
    Trains and vlidates the model, then returns the accuracy to persist over iterations. 


# Model for cross-validation

# In[ ]:


model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, bidirectional=bidirectional, pad_idx=pad_index).to(device)
optimizer = torch.optim.Adam(model.parameters())


# In[ ]:


import matplotlib.pyplot as plt

batch_sizes = [64, 128, 256, 512, 1024]
token_sizes = np.arange(50000, 350000, 50000)
hidden_sizes = np.arange(128, 768, 64)
embedding_sizes = np.arange(64, 256*4, 64)

embedding_size = 128 #Final
hidden_size = 512 #Final

hidden_models = []
hidden_optimizers = []
    
figure = 0

batch_size = 256 #Final
token_size = 150000 #?
epochs = 20 #Final
max_acc = 0

std_epochs = 10

bidirectional = False #Final

#model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, bidirectional=bidirectional, pad_idx=pad_index).to(device)
#optimizer = torch.optim.Adam(model.parameters())

results = testParameters(model,optimizer, epochs, batch_size, token_size) 
    
plt.figure(figure)
plt.plot(range(1, len(results[0])+1), results[0])
plt.plot(range(1, len(results[1])+1), results[1])
plt.title('Acuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy');
    


# Here are some of the hyperparameters that gave best results.
# 
# Some were tested by trial and error, but most of them using the code below

# In[ ]:


'''

max_acc = 0
figure = 0
accuracies = []
hidden_models = []
hidden_optimizers = []

for count, size in enumerate(hidden_sizes):
    size = int(size)
    hidden_models.append(CharRNNClassifier(ntokens, embedding_size, size, nlabels, bidirectional=bidirectional, pad_idx=pad_index).to(device))
    hidden_optimizers.append(torch.optim.Adam(hidden_models[count].parameters()))
    
for count, size in enumerate(hidden_sizes):
    size = int(size)
    print('Testing hidden_size = ' + str(size))

    results = testParameters(hidden_models[count], hidden_optimizers[count], std_epochs, batch_size, token_size) 
    
    plt.figure(figure)
    plt.plot(range(1, len(results[0])+1), results[0])
    plt.plot(range(1, len(results[1])+1), results[1])
    plt.title('hidden_size = ' + str(size))
    plt.xlabel('epoch')
    plt.ylabel('Accuracy'); 
    
    figure += 1
    
    accuracies.append(results[1][std_epochs-1])
    if results[1][std_epochs-1] > max_acc:
        max_acc = results[1][std_epochs-1]
        best_size = size

hidden_size = best_size        

print('Best hidden_size: ' + str(best_size))
        
plt.figure(figure)
plt.plot(range(1, len(accuracies)+1), accuracies)
plt.title('Acc by hidden_size')
plt.xlabel('hidden_size')
plt.ylabel('Accuracy')


# Iterate over an array of values for the hidden size, create a new model for each of them and find the best one.

# In[ ]:


'''

max_acc = 0
accuracies = []
figure = 0

embedding_models = []
embedding_optimizers = []

for count, size in enumerate(embedding_sizes):
    size = int(size)
    embedding_models.append(CharRNNClassifier(ntokens, size, hidden_size, nlabels, bidirectional=bidirectional, pad_idx=pad_index).to(device))
    embedding_optimizers.append(torch.optim.Adam(hidden_models[count].parameters()))

for count, size in enumerate(embedding_sizes):
    size = int(size)
    print('Testing embedding_size = ' + str(size))
    results = testParameters(embedding_models[count], embedding_optimizers[count], std_epochs, batch_size, token_size) 
    
    plt.figure(figure)
    plt.plot(range(1, len(results[0])+1), results[0])
    plt.plot(range(1, len(results[1])+1), results[1])
    plt.title('embedding_size = ' + str(size))
    plt.xlabel('epoch')
    plt.ylabel('Accuracy'); 
    
    figure += 1
    
    accuracies.append(results[1][std_epochs-1])
    if results[1][std_epochs-1] > max_acc:
        max_acc = results[1][std_epochs-1]
        best_size = size

embedding_size = best_size        

print('Best embedding_size: ' + str(best_size))
        
plt.figure(figure)
plt.plot(range(1, len(accuracies)+1), accuracies)
plt.title('Acc by embedding_size')
plt.xlabel('embedding_size')
plt.ylabel('Accuracy')


# Iterate over an array of values for the hidden size, create a new model for each of them and find the best one.

# In[ ]:


model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, bidirectional=bidirectional, pad_idx=pad_index).to(device)
optimizer = torch.optim.Adam(model.parameters())


# In[ ]:


'''

max_acc = 0
accuracies = []
figure = 0
for batch_size in batch_sizes:
    print('Testing batch_size: ' + str(batch_size))
    results = testParameters(model, optimizer, std_epochs, batch_size, token_size) 
    
    plt.figure(figure)
    plt.plot(range(1, len(results[0])+1), results[0])
    plt.plot(range(1, len(results[1])+1), results[1])
    plt.title('Batch Size ' + str(batch_size))
    plt.xlabel('epoch')
    plt.ylabel('Accuracy'); 
    
    accuracies.append(results[1][std_epochs-1])
    if results[1][std_epochs-1] > max_acc:
        best_batch_size = batch_size
        max_acc = results[1][std_epochs-1]
    figure += 1
    
batch_size = best_batch_size  

print('Best batch_size: ' + str(best_batch_size))
        
plt.figure(figure)
plt.plot(range(1, len(accuracies)+1), accuracies)
plt.title('Acc by batch_size')
plt.xlabel('batch_size')
plt.ylabel('Accuracy')


# In[ ]:


'''

max_acc = 0
accuracies = []
figure = 0
for token_size in token_sizes:
    print('Testing token_size: ' + str(token_size))
    results = testParameters(model, optimizer, std_epochs, batch_size, token_size) 
    
    plt.figure(figure)
    plt.plot(range(1, len(results[0])+1), results[0])
    plt.plot(range(1, len(results[1])+1), results[1])
    plt.title('Token Size ' + str(batch_size))
    plt.xlabel('epoch')
    plt.ylabel('Accuracy'); 
    
    accuracies.append(results[1][std_epochs-1])
    if results[1][std_epochs-1] > max_acc:
        best_token_size = token_size
        max_acc = results[1][std_epochs-1]
    figure += 1
    
token_size = best_token_size   

print('Best token_size: ' + str(best_token_size))
        
plt.figure(figure)
plt.plot(range(1, len(accuracies)+1), accuracies)
plt.title('Acc by token_size')
plt.xlabel('token_size')
plt.ylabel('Accuracy')


# In[ ]:


'''

max_acc = 0
accuracies = []
figure = 0
for epoch in epochs:
    results = testParameters(model, optimizer, epoch, batch_size, token_size) 
    
    plt.figure(figure)
    plt.plot(range(1, len(results[0])+1), results[0])
    plt.plot(range(1, len(results[1])+1), results[1])
    plt.title('Epochs ' + str(epoch))
    plt.xlabel('epoch')
    plt.ylabel('Accuracy'); 
    
    accuracies.append(results[1][epoch-1])
    if results[1][epoch-1] > max_acc:
        best_epochs = epoch
        max_acc = results[1][epoch-1]
    figure += 1
    
epochs = best_epochs   

print('Best epochs: ' + str(best_epochs))
        
plt.figure(figure)
plt.plot(range(1, len(accuracies)+1), accuracies)
plt.title('Acc by epochs')
plt.xlabel('epochs')
plt.ylabel('Accuracy')


# The iteration for epochs, batch sizes and token sizes was performed in the same way as the other ones, but the data persisted and therefore, most of the time, the best value was the highest one, as the model had already been trained.
# In the end due to timing constraints i ran several instances of kaggle to try and find the best value for them

# 

# **Final model**  
# Finally, we create a model using all the training data and we generate the submission with the predicted test labels

# In[ ]:


model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, bidirectional=bidirectional).to(device)
optimizer = torch.optim.Adam(model.parameters())


# In[ ]:


print(f'Training final model for {epochs} epochs')
for epoch in range(1, epochs + 1):
    print(f'| epoch {epoch:03d} | train accuracy={train(model, optimizer, train_data + val_data, batch_size, token_size, log=epoch==1):.3f}')


# In[ ]:


def test(model, data, batch_size, token_size):
    model.eval()
    sindex = []
    labels = []
    with torch.no_grad():
        for batch in pool_generator(data, batch_size, token_size):
            # Get input sequences from batch
            X = [torch.from_numpy(d[0]) for d in batch]
            X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long, device=device)
            # Pad the input sequences to create a matrix
            X = torch.nn.utils.rnn.pad_sequence(X).to(device)
            answer = model(X, X_lengths)
            label = torch.max(answer, 1)[1].cpu().numpy()
            # Save labels and sentences index
            labels.append(label)
            sindex += [d[1] for d in batch]
    return np.array(sindex), np.concatenate(labels)


# In the test database we replace the label (language) with a sentence index.  

# In[ ]:


x_test_txt = open("../input/x_test.txt").read().splitlines()
x_test_idx = [np.array([char_vocab.token2idx[c] if c in char_vocab.token2idx else unk_index for c in line]) for line in x_test_txt]
test_data = [(x, idx) for idx, x in enumerate(x_test_idx)]


# The sentence index is used to rearrange the labels in the original sentence order

# In[ ]:


index, labels = test(model, test_data, batch_size, token_size)
order = np.argsort(index)
labels = labels[order]


# In[ ]:


with open('submission.csv', 'w') as f:
    print('Id,Language', file=f)
    for sentence_id, lang_id in enumerate(labels):
        language = lang_vocab.idx2token[lang_id]
        if sentence_id < 10:
            print(f'{sentence_id},{language}')
        print(f'{sentence_id},{language}', file=f)

