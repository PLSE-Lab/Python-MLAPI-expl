#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import math


# http://www.cim.mcgill.ca/~latorres/Viterbi/va_alg.htm

# **Instructions**

# Having data
# 1. Assume that characters are dependent only on corresponding features. On others they are independent.
# 2. Compute probabilities that a chareacter c occurs. P(c) = n_c/length_of_word
# 3. Compute conditional probabilities P(c.i|c.i-1). Again, take frequency.
# 4. Compute conditional probabilities p(z.i|c.i). Again, take frequency.
# 
# Answering
# 1. Take Z = z1, z2, ..., zn inputs, which are features.
# 2. Want to find a sequence C, such that P(C|Z) is maximum. P(C|Z) = p(Z|C)*P(C)/P(Z), so enough to find C such that p(Z|C)*P(C) is maximum.
# 3. Can logarithm the latter. From independence of c.i and z.j (i != j) have log p(z.1|c.1) + log p(z.2|c.2) + ... + p(z.n|c.n) + logP(c.0) + logP(c.1|c.0) + ... + logP(c.n|c.n-1) and want to find C that maximizes it.
# c.0 is an abstract 0 character. P(c.1|c.0) = P(c.1).
# 4. Can find C, such that minimizes sum( -log p(z.i|c.i) - logP(c.i|c.i-1) )
# 5. Create a graph with layers L0, L1, ..., Ln. For i>0, Li = {char.1, char.2, ..., char.M}, and edge connecting char.i with char.j from L.k to L.k+1 has weight ( -log p(z.j | char.j) - logP(char.j | char.j+1) ).
# 6. For each layer and for each node in it, find the shortest path leading to it.
# 

# **Preparing data**

# In[ ]:


C = "abcdegsacbdcefabafgaaa"
feature_size = 10
Z = [random.randint(0,feature_size-1) for z in range(len(C))]
n = len(C)
Z


# In[ ]:


char_to_ix = {}
for char in C:
    if char not in char_to_ix:
        char_to_ix[char] = len(char_to_ix)
len(char_to_ix)
print(char_to_ix)


# **Computing the probabilities**

# In[ ]:


#P(C) and P(C.i|C.i-1)
PC = np.zeros( len(char_to_ix))
cond_PC = np.zeros(( len(char_to_ix), len(char_to_ix)) )

for e,c in enumerate(C):
    ix = char_to_ix[c]
    PC[ix] += 1 
    if( e > 0 ):
        cond_PC[char_to_ix[ C[e-1] ], char_to_ix[ C[e] ] ] += 1
        
for row in range( len(char_to_ix) ):
    cond_PC[row] = cond_PC[row]/np.sum(cond_PC[row])

print(cond_PC)


# In[ ]:


#p(z|c)
pz = np.zeros( ( feature_size, len(char_to_ix) ) )
for e,c in enumerate(C):
    pz[ Z[e], char_to_ix[c] ] += 1

for row in range( feature_size ):
    pz[row] = pz[row]/( 0.0001 + np.sum(pz[row]) )

print(pz)


# **QUERY**

# In[ ]:


n = 10
Z_new = [ random.randint(0, feature_size-1) for z in range(n)]
print(Z_new)


# In[ ]:


#Initialize structures
length = 0
words = [ [] for i in range( len(char_to_ix) ) ]
len_path = np.zeros( len(char_to_ix) )


# In[ ]:


#First layer
for ix,c in enumerate(char_to_ix):
    len_path[ix] = -math.log( PC[ix] + 0.001 ) - math.log( pz[ Z_new[0]][ix] + 0.001 )
    words[ ix ].append(c)
    print( words[ix], len_path[ ix] )
print(words)


# In[ ]:


for i in range(1, n):
    new_len_path = np.zeros( len(char_to_ix) )
    for ch in char_to_ix:
        min_len_path = 100000000
        best = 0
        for c in char_to_ix:
            cand =  len_path[ char_to_ix[c] ] - math.log( cond_PC[char_to_ix[c]][char_to_ix[ch]] + 0.01) -math.log( pz[i][ char_to_ix[c] ] + 0.01 )
            if cand < min_len_path:
                min_len_path = cand
                best = c
        words[char_to_ix[ch]].append(best)
        new_len_path[ char_to_ix[ch] ] = min_len_path
    len_path = new_len_path.copy()


# In[ ]:


print( len_path)


# In[ ]:


print(words)


# In[ ]:




