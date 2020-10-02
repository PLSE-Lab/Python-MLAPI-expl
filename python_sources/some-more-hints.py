#!/usr/bin/env python
# coding: utf-8

# # Some More Hints for Difficulty 1
# This Notebook continiues on apporach by Paul Dnt in his kernel [Something to begin with: a first hint](https://www.kaggle.com/pednt9/something-to-begin-with-a-first-hint)
# I will update this as new ideas come

# In[ ]:


#Importing Stuff
import numpy as np 
import pandas as pd 
from math import *
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Since we know lengths are padded we add length and padded length feilds to the dataframe

# In[ ]:


train['len']=train.apply(lambda x: len(x['text']),axis=1)
train['plen']=train.apply(lambda x: (ceil(x['len']/100)*100),axis=1)
train.head()


# In[ ]:


test['len']=test.apply(lambda x: len(x['ciphertext']),axis=1)
test.head()


# # We seprate test data according to difficulty

# In[ ]:


df1 = test[test.difficulty==1].copy()
df2 = test[test.difficulty==2].copy()
df3 = test[test.difficulty==3].copy()
df4 = test[test.difficulty==4].copy()


# # Let us First Focus on Difficulty 1
# We find out how many unique lengths are there

# In[ ]:


df1.groupby('len').nunique()


# Now we know that length of plaintext is not edited in this difficulty. So a plaintext of padded length 500 will be there are length 500 in our test case.
# <br/>
# Also since there is only one element in 500 and 400 category each, it is a good starting point for us.

# In[ ]:


df1[df1.len==500]


# In[ ]:


train[train.plen==500]


# In[ ]:


#We extract the 
cipherText500 = df1.loc[45272].ciphertext
cipherText500


# In[ ]:


for _,i in train[train.plen==500].iterrows():
    print(i.text)
    print(cipherText500)
    print("\n\n\n")


# Credit : The approach so far has been from [Something to begin with: a first hint
# ](https://www.kaggle.com/pednt9/something-to-begin-with-a-first-hint) from now on its entirely my own

# If you observe carefully, all punctuations marks line up with some padding in the first text
# ![](https://i.imgur.com/MlFKoJg.png?2)

# In[ ]:


text500 = train.loc[13862].text
text500


# ### We realise that padding is centered. 
# So `left_padding = (padded length-length)/2` and similarly `right_padding = padded length - (padded length-length)/2`
# <br/>We make a function that converts all a-x to x, A-Z to X and leaves punctuation as is

# In[ ]:



def permchk(s):
    a = ""
    for i in list(s):
        if i>='a' and i<='z':
            a+='x'
        elif i>='A' and i<='Z':
            a+='X'
        else:
            a+=i
    return a

p = len(cipherText500)
l = len(text500)
x = (p-l)//2
print(permchk(text500))
print(permchk(cipherText500[x:l+x]))


# ## As this is exactly same we can loop through all values in train to find corresponding values in df1

# In[ ]:


for _,i in df1[df1.len==500].iterrows():
    print(i.ciphertext_id,end=" : ")
    for _,j in train.iterrows():
        if(i.len==j.plen):
            p = j.plen
            l = j.len
            x = (p-l)//2
            pcj = permchk(j.text)
            pci = permchk(i.ciphertext[x:x+l])
            if pci==pcj:
                print(j.plaintext_id,end=",")
    print("")


# In[ ]:


for _,i in df1[df1.len==400].iterrows():
    print(i.ciphertext_id,end=" : ")
    for _,j in train.iterrows():
        if(i.len==j.plen):
            p = j.plen
            l = j.len
            x = (p-l)//2
            pcj = permchk(j.text)
            pci = permchk(i.ciphertext[x:x+l])
            if pci==pcj:
                print(j.plaintext_id,end=",")
    print("")


# In[ ]:


for _,i in df1[df1.len==300].iterrows():
    print(i.ciphertext_id,end=" : ")
    for _,j in train.iterrows():
        if(i.len==j.plen):
            p = j.plen
            l = j.len
            x = (p-l)//2
            pcj = permchk(j.text)
            pci = permchk(i.ciphertext[x:x+l])
            #print(j.text)
            #print(pci)
            #print(pcj)
            if pci==pcj:
                print(j.plaintext_id,end=",")
    print("")


# In[ ]:


for _,i in df1[df1.len==200].iterrows():
    print(i.ciphertext_id,end=" : ")
    for _,j in train.iterrows():
        if(i.len==j.plen):
            p = j.plen
            l = j.len
            x = (p-l)//2
            pcj = permchk(j.text)
            pci = permchk(i.ciphertext[x:x+l])
            #print(j.text)
            #print(pci)
            #print(pcj)
            if pci==pcj:
                print(j.plaintext_id,end=",")
    print("")


# # But this does not work that well with length 100. But we have enough pairs to decipher difficulty 1
