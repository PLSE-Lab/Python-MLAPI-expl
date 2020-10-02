#!/usr/bin/env python
# coding: utf-8

# In[28]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <h1>How do we evaluate our performance on DonorsChoose?</h1>
# <h2>What's our objective?</h2>
# We know from the problem statement that the mission is to:
# <p>**build targeted email campaigns recommending specific classroom requests to prior donors**
# <p>Let's assume that the output of this will be an email with a list of suggested projects. Clearly, we want to:
# <p>**1. show the potential donor the projects she is most likely to send money to**
# <p>And, maybe, we want to:
# <p>**2. highlight projects from schools or teachers that historically don't get a lot of attention**
# <p>Lastly, we want to help teachers pitch their projects more effectively, so we should
# <p>**3. evaluate the impact of project elements  on the likelihood of a project being funded**

# <h2>Part 1: Showing donors projects that interest them</h2>

# In[5]:


project_data = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,parse_dates=True,engine='python')
donation_data = pd.read_csv('../input/Donations.csv', error_bad_lines=False)


# In[19]:


num_donations_per_donor = donation_data.groupby('Donor ID')['Donor Cart Sequence'].max()


# Before we do anything else, let's set our success criteria. We want to be able to accurately predict the next donation a donor will make, based on the previous donation(s) they've made. Sounds like we need to solve an ML problem where the output is a list of (project, probability) for all the projects a donor can donate to (have start dates after their prior prjects). We'll score our algorithm based on how highly ranked their repeat donation is in our model. 
# <p>If we're training a model to predict likely repeat donors, we need a training and testing set of donors who donate more than once. That way, we can evaluate our performance.
# <p>Let's see how many repeat donors there are.

# In[27]:


print("{:10.2f}% of donors gave only once".format((num_donations_per_donor == 1).mean() * 100))
print("{:10.2f}% of donors gave 5 or fewer times".format((num_donations_per_donor <= 5).mean() * 100))


# Okay, we can clearly see the need for DonorsChoose. Nearly 70% of users are giving only one time.
# <p> Our training and testing is for now going to focus on those repeat donors, but we'll come back to the ones who only donate once. Is there a reason why?

# <h3>Thanks for reading. I'm working on the best way to set up these training/testing datasets and will update with my solution. Would appreciate any input.</h3>

# In[ ]:




