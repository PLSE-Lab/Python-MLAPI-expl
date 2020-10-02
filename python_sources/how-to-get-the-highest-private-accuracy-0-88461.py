#!/usr/bin/env python
# coding: utf-8

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


# Here, I will tell you **why**? Cause the dataset is so **flawed** that **71%** of it consists of **1's** which means if you submitted with just **1's** you would get **71%**. Now you have to know that we have to submit **86 cases, 30%** of which were considered for the private score. Which means **26** cases. So now I just switched one value for a **2** and I got that accuracy, lets analyze this.
# 
# ## Machine Learning Time:
# 
# say number of **1's** in the private scoring data was X and number of **2's** in the private scoring data was Y. Then, we know **X + Y = 26**
# We also know that we switched only one value to **2** **(Id=7)**. Basically we got all the 1's correct which means all the X's correct and (Y-1) wrong. So my accuracy being **0.88461 * 26** gives us 23. So basically I got **23 correct**. Which means **3 incorrect**. So,
# 
# Y-1 = 3
# So, Y=4. That means only **4 values** were labelled **2**. Which is *dumb*, what are you trying to evaluate **dude**. You see the rise in ranks some of these people have gotten on the top shelves. Luck is the answer? I mean see the difference between the public and private ranks. That's not rise you see everyday unless its *Corona*. Plz don't be unfair and judge by this private score list. Look at both of them.
# 
# # Thanks for coming to my TED Talk~

# In[ ]:





# In[ ]:




