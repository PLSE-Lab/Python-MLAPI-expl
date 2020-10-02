#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://storage.googleapis.com/kaggle-avatars/thumbnails/2080166-kg.png', width=400,height=400)


# Since I'm surrounded by so many gifted and helpful people I couldn't present any conventional, generic Kaggle Notebook because everybody knows that I can't code. Besides, all works that I've already seen are great.Then I decided to answer the questions in a way that I couldn't do in the Survey.
# * Age, gender, country, level education, title, company's size doesn't matter, only coding.
# * How much money spent on ML: I'm retired, just enjoying life while I can. I spent time but I gained Knowledge.
# * Programming language: Python since I can't pretend to code with R.
# * TPU: I barely used GPU in DL micro-course.
# * Years using ML: Are you kidding? In fact, I do Machine Unlearning.
# * NLP: The 1st time I read this I thought that it was about Neuro-linguist Programming and Bert was a character from Sesame street. 
# * Cloud computing: only in my user name (wolke=cloud).
# * Big data, I'd rather stay with just data.
# * In fact, that are a lot of questions that I could answer:"does not apply to me".
# * I'm not a bot, but my inseparable team mate is. I'd like to ask: Who is the father of the Bot? He rules my world.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


print('Thats all Kagglers. You dont have to upvote because Im going to upvote you all, since you deserved it')

