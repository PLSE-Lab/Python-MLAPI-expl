#!/usr/bin/env python
# coding: utf-8

# # Serializing objects (Q-Table, NN Weights) with base64, pickle and zlib
# In this very short notebook I will show a simple method to serialize a python's object and write it directly into a .py file as a string. This can be very useful when you want to store a Q-table or a neural network after an offline training. 

# In[ ]:


import pickle
import zlib
import base64 as b64
import numpy as np

def serializeAndCompress(value, verbose=True):
  serializedValue = pickle.dumps(value)
  if verbose:
    print('Lenght of serialized object:', len(serializedValue))
  c_data =  zlib.compress(serializedValue, 9)
  if verbose:
    print('Lenght of compressed and serialized object:', len(c_data))
  return b64.b64encode(c_data)

def decompressAndDeserialize(compresseData):
  d_data_byte = b64.b64decode(compresseData)
  data_byte = zlib.decompress(d_data_byte)
  value = pickle.loads(data_byte)
  return value


# Pickle can serialize all type of objects like ndarray, dictionaries, classes, lists, etc...
# This is an example of usage:

# In[ ]:


m = 12
n = 12
obs_space_n = m*n
action_space_n = 2
q_table = np.zeros([obs_space_n, action_space_n])
for n_sim in range(10000):
    # Train your agent
    # ...
    # Store a fake Qtable value
    q_table[np.random.randint(obs_space_n), np.random.randint(action_space_n)] = np.random.random()
serialized_q_table = serializeAndCompress(q_table)


# In[ ]:


print(serialized_q_table)


# You can now copy and paste this string directly into the .py submission file.

# In[ ]:


deserialized_q_table = decompressAndDeserialize(serialized_q_table)
print(deserialized_q_table[:10,:])


# In[ ]:




