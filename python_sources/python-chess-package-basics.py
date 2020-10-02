#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install python-chess')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import chess

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


first = df.head(1)


# In[ ]:


first['fen']


# In[ ]:


first['best']


# In[ ]:


board = chess.Board(first['fen'][0])
board


# In[ ]:


move = chess.Move.from_uci(first['best'][0])
board.san(move)


# In[ ]:


board.push(move)
board


# In[ ]:




