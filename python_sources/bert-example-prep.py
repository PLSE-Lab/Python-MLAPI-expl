#!/usr/bin/env python
# coding: utf-8

# # Preparing GAP for BERT: Input Layer

# This kernel only shows how character offsets are converted to BERT token offsets for explanatory purposes. A full repository of the software to fine tune BERT and do predictions for this dataset is available [here.](https://github.com/kenkrige/BERT-Fine-tune-for-GAP)

# For a short working example, I've chosen something I might hear from one of my school students:
# > "I saw Oratile slap Rorisang on her left shoulder."

# In[ ]:


example = "example-1	I saw Oratile slap Rorisang on her left shoulder.	her	31	Oratile	6	False	Rorisang	19	True"
line = example.split('\t')


# In[ ]:


max_seq_length = 12 #For the cometition, I used 64 and 128, but keeping it very short for clarity.


# In[ ]:


text = line[1]
P_offset = int(line[3])
A_offset = int(line[5])
B_offset = int(line[8])


# Make an array of the char offsets, including an index column which will be used to remember positions of P, A, B after sorting by offset. Then sort by the offset column.

# In[ ]:


char_off = sorted([
  [P_offset, 0],
  [A_offset, 1],
  [B_offset, 2]
], key=lambda x: x[0])
char_off


# Use the offsets in the first column to split the text into 4 segments. This technique assists the accurate conversion of character offsets to token offsets without the need for any whitespace tokenization.

# In[ ]:


text_segments = [text[:char_off[0][0]], 
text[char_off[0][0]:char_off[1][0]], 
text[char_off[1][0]:char_off[2][0]], 
text[char_off[2][0]:]]
text_segments


# Use BERT Wordpiece to tokenize each segment.

# In[ ]:


get_ipython().system('cp ../input/bertfiles/tokenization.py .')
import tokenization
tokenizer = tokenization.FullTokenizer(vocab_file='../input/bertfiles/vocab.txt', do_lower_case=True)
token_segments = []
num_tokens = []
for segment in text_segments:
    token_segment = tokenizer.tokenize(segment)
    token_segments.append(token_segment)
    num_tokens.append(len(token_segment))
token_segments


# Truncate by removing one token at a time until the number of tokens is two less than the maximum sequence length. The extra two allow for BERT's start and end tokens. Each time remove the furthest token from any offset point.

# In[ ]:


import numpy as np
while np.sum(num_tokens) > (max_seq_length - 2):
    index = np.argmax([num_tokens[0] * 2, num_tokens[1], num_tokens[2], num_tokens[3] * 2])
    if index == 0:
        token_segments[index] = token_segments[index][1:]
    elif index == 3:
        token_segments[index] = token_segments[index][:-1]
    else: #middle segments
        middle = num_tokens[index] // 2
        token_segments[index] = token_segments[index][:middle] + token_segments[index][middle + 1:]
    num_tokens[index] -= 1
token_segments


# Concatenate the segments back together.

# In[ ]:


tokens = []
tokens.append("[CLS]")
for segment in token_segments:
    temp = ''
    for token in segment:
        tokens.append(token)
tokens.append("[SEP]")


# Replace the char offsets with token offsets, using the lengths of token segments cumulatively. Then sort offsets on column 2 back to the original order of P, A, B.

# In[ ]:


offset = 1 #to account for "[CLS]"
for i, row in enumerate(char_off):
    offset += num_tokens[i]
    row[0] = offset

token_off = sorted(char_off, key=lambda x: x[1])
token_off


# Makle the position masks for P, A, B

# In[ ]:


P_mask = [0] * max_seq_length
A_mask = [0] * max_seq_length
B_mask = [0] * max_seq_length

P_mask[token_off[0][0]] = 1
A_mask[token_off[1][0]] = 1
B_mask[token_off[2][0]] = 1

print(P_mask)
print(A_mask)
print(B_mask)
print(tokens)


# Those are the inputs to the model. The masks are forwarded to the output layer and the sentence of tokens is processed by the BERT hidden layers.
