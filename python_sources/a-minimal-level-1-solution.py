#!/usr/bin/env python
# coding: utf-8

# A minimal level 1 solution with the help of wonderful notebook by @mithrillion
# 
# https://www.kaggle.com/mithrillion/enigma-was-gimped-by-weather-reports/notebook
# 

# In[ ]:


import pandas as pd
df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


level1 = str.maketrans(
    '*#^-G1>c\x03X\x1b_t%dO\x02ah8?]\'\x08s{0oWfTAvV\x18\x7fE:g\x1e2|y[9we@!&x;\x10Fz"/Ql\x1aLn<\tq,H~b\\5u J+B\x06}64UiKP`\x1cr)3Z(.\x0cYmIkSp$D=',
    'From: SubjectOganizl/fp.\nThywk,dsQVNEIHRC()>PKMUX?D\'YLvGA\tWq1895-42+@063<Bx$7J"|[]Z!#*&_;=^~%`\\}{\x08\x1e\x02\x0c\x10'
)

df.ciphertext = df.ciphertext.str.translate(level1)
df.head()


# I also see one line in 'difficulty == 1' set, which seems to be ciphered more than once. If you spot more, please comment below.

# In[ ]:


df[df.ciphertext.str.startswith('%Z*[U[H-+?5O0I81)@')]


# In[ ]:




