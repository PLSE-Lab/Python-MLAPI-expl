#!/usr/bin/env python
# coding: utf-8

# # What kind of person will making multiple donation?
# 
# ## Purpose
# I think analysis by people of making multiple donation is can helping recommended system.
# 
# ## Result
# I found 3 factor.
# 
# 1. People of making multiple donation many have teachers.
# 2. Lowa have many people of making multiple donation.
# 
# ## Load csv
# 

# In[ ]:


import numpy as np
import pandas as pd

resources = pd.read_csv("../input/Resources.csv", sep=",").dropna()
schools = pd.read_csv("../input/Schools.csv", sep=",").dropna()
donors = pd.read_csv("../input/Donors.csv", sep=",").dropna()
donations = pd.read_csv("../input/Donations.csv", sep=",").dropna()
teachers = pd.read_csv("../input/Teachers.csv", sep=",").dropna()
projects = pd.read_csv("../input/Projects.csv", sep=",").dropna()


# ## Split to two type -- making once donation user or making multiple donation users

# In[ ]:


donor_value = donations["Donor ID"].value_counts()

once_donated_users = donor_value[donor_value == 1].index
multiple_donated_users = donor_value[donor_value > 1].index


# ## Analysis two users -- Is teacher and states

# In[ ]:


once_donors_list = donors[donors["Donor ID"].isin(once_donated_users)]
multiple_donors_list = donors[donors["Donor ID"].isin(multiple_donated_users)]


# In[ ]:


import matplotlib.pyplot as plt

once_donors_list["Donor Is Teacher"].value_counts().plot(kind='bar')
plt.title("once_donated_users and is teacher")


# In[ ]:


multiple_donors_list["Donor Is Teacher"].value_counts().plot(kind='bar')
plt.title("multiple_donated_users and is teacher")


# people of making multiple donation exist many teacher.

# In[ ]:


ov = once_donors_list["Donor State"].value_counts()
mv = multiple_donors_list["Donor State"].value_counts()


# In[ ]:


ov[0:30].plot(kind='bar')
plt.title("once_donated_users and state")


# In[ ]:


mv[0:30].plot(kind='bar')
plt.title("multiple_donated_users and state")


# Get to relationship between state and rate by people of making multiple donation users.

# In[ ]:


(ov / mv)[0:30].sort_values()[::-1].plot(kind='bar')
plt.title("Donation rate per states")


# Lowa have many people of making multiple donation.
