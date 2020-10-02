#!/usr/bin/env python
# coding: utf-8

# ##Using LSTM Cells in a recurrent neural network, this will generate chatbot profiles for each primary southpark character

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/All-seasons.csv")


# In[ ]:


lines = df["Line"]
characters = df["Character"]
episodes = df["Episode"]
charlines = "("+characters+") " + lines
text = ""
for line in charlines:
    text += line


# In[ ]:


token_dict = { 
    '!': '||EXCLAIMATIONMARK||',
    '?': '||QUESTIONMARK||',
    '--': '||DOUBLEDASH||',
    '"': '||DOUBLEQUOTE||',
    ',': '||COMMA||',
    '.': '||PERIOD||',
    ';': '||SEMICOLON||',
    '\n': '||NEWLINE||',
    '(': '||OPENPAREN||',
    ')': '||CLOSEPAREN||',
    #'+': '||PLUS||',
    #'&': '||AMPERSAND||',
    #':': '||COLON||',
    #'\'': '||APOSTROPHE||',
    #'-': '||DASH||',
}

for key, token in token_dict.items():
    lines[0] = lines[0].replace(key, ' {} '.format(token))

print(lines[0])


# In[ ]:


unique_lines = [set(line.lower().split()) for line in lines]
unique_lines[0]
#vocab_to_int = [w:i for i,w in enumerate(vocab)]


# In[ ]:


character_lines = {w: [] for w in characters}
for i in range(len(lines)):
    character_lines[str(characters[i])].append(lines[i])
#character_lines['Stan']

