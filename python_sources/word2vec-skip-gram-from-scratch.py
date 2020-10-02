#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import re


# ![skip-gram.png](attachment:skip-gram.png)

# ### Generate the data

# In[ ]:


def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


# In[ ]:


tokens = tokenize("Hello from the other side")


# In[ ]:


tokens


# In[ ]:


id_to_word = {i:x for (i, x) in enumerate(tokens)}
word_to_id = {x:i for (i, x) in enumerate(tokens)}


# In[ ]:


print(word_to_id)
print(id_to_word)


# In[ ]:


def generate_training_data(tokens, word_to_id, window_size):
    X, Y = [], []

    for i in range(len(tokens)):
        nbr_inds = list(range(max(0, i - window_size), i)) +                    list(range(i + 1, min(len(tokens), i + window_size + 1)))
        for j in nbr_inds:
            X.append(word_to_id[tokens[i]])
            Y.append(word_to_id[tokens[j]])
            
    return np.array(X), np.array(Y)


# In[ ]:


def expand_dims(x, y):
    x = np.expand_dims(x, axis=0)
    y = np.expand_dims(y, axis=0)
    return x, y


# In[ ]:


x, y = generate_training_data(tokens, word_to_id, 3)
x, y = expand_dims(x, y)


# In[ ]:


# generated training data
x, y


# ### Forward Propagation

# In[ ]:


def init_parameters(vocab_size, emb_size):
    wrd_emb = np.random.randn(vocab_size, emb_size) * 0.01
    w = np.random.randn(vocab_size, emb_size) * 0.01
    
    return wrd_emb, w


# In[ ]:


def softmax(z):
    return np.divide(np.exp(z), np.sum(np.exp(z), axis=0, keepdims=True) + 0.001)


# In[ ]:


def forward(inds, params):
    wrd_emb, w = params
    word_vec = wrd_emb[inds.flatten(), :].T
    z = np.dot(w, word_vec)
    out = softmax(z)
    
    cache = inds, word_vec, w, z
    
    return out, cache


# ### Cost Function

# In[ ]:


def cross_entropy(y, y_hat):
    m = y.shape[1]
    cost = -(1 / m) * np.sum(np.sum(y_hat * np.log(y + 0.001), axis=0, keepdims=True), axis=1)
    return cost


# ### Backward Propagation

# In[ ]:


def dsoftmax(y, out):
    dl_dz = out - y
    
    return dl_dz


# In[ ]:


def backward(y, out, cache):
    inds, word_vec, w, z = cache
    wrd_emb, w = params
    
    dl_dz = dsoftmax(y, out)
    # deviding by the word_vec length to find the average
    dl_dw = (1/word_vec.shape[1]) * np.dot(dl_dz, word_vec.T)
    dl_dword_vec = np.dot(w.T, dl_dz)
    
    return dl_dz, dl_dw, dl_dword_vec


# In[ ]:


def update(params, cache, grads, lr=0.03):
    inds, word_vec, w, z = cache
    wrd_emb, w = params
    dl_dz, dl_dw, dl_dword_vec = grads
    
    wrd_emb[inds.flatten(), :] -= dl_dword_vec.T * lr
    w -= dl_dw * lr
    
    return wrd_emb, w


# ### Training

# In[ ]:


vocab_size = len(id_to_word)

m = y.shape[1]
y_one_hot = np.zeros((vocab_size, m))
y_one_hot[y.flatten(), np.arange(m)] = 1

y = y_one_hot


# In[ ]:


batch_size=256
embed_size = 50

params = init_parameters(vocab_size, 50)

costs = []

for epoch in range(5000):
    epoch_cost = 0
    
    batch_inds = list(range(0, x.shape[1], batch_size))
    np.random.shuffle(batch_inds)
    
    for i in batch_inds:
        x_batch = x[:, i:i+batch_size]
        y_batch = y[:, i:i+batch_size]
        
        pred, cache = forward(x_batch, params)
        grads = backward(y_batch, pred, cache)
        params = update(params, cache, grads, 0.03)
        cost = cross_entropy(pred, y_batch)
        
        epoch_cost += np.squeeze(cost)
        
    costs.append(epoch_cost)
    
    if(epoch % 250 == 0):
        print("Cost after epoch {}: {}".format(epoch, epoch_cost))


# ### Test

# In[ ]:


x_test = np.arange(vocab_size)
x_test = np.expand_dims(x_test, axis=0)
softmax_test, _ = forward(x_test, params)
top_sorted_inds = np.argsort(softmax_test, axis=0)[-4:,:]


# In[ ]:


for input_ind in range(vocab_size):
    input_word = id_to_word[input_ind]
    output_words = [id_to_word[output_ind] for output_ind in top_sorted_inds[::-1, input_ind]]
    print("{}'s skip-grams: {}".format(input_word, output_words))

