#!/usr/bin/env python
# coding: utf-8

# **Numpy is Beautiful**
# 
# As I was exploring some of the kernels for this challenge, I realized how much time I spent understanding the way some of the cost matrices were coded. These cost matrices specified the penalty for a specific family (i) being assigned to a certain day (j). As I was reading through the code of lookups, loops, etc. I thought there has to be a cleaner, faster way.
# 
# And numpy came to my rescue. Hopefully this gives people some ideas for how to improve their coding skills!

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


N_DAYS = 100
N_FAMILIES = 5000
N_CHOICES = 10

data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')


# First create an array that is 5000 x 11. Each row is a family, and each column represents the cost of choosing choice 0 (column 0), choice 1, etc.

# In[ ]:


# 7 family sizes x 11 choice penalties
size_penalties = np.asarray([
    [
        0,
        50,
        50 + 9 * n,
        100 + 9 * n,
        200 + 9 * n,
        200 + 18 * n,
        300 + 18 * n,
        300 + 36 * n,
        400 + 36 * n,
        500 + 36 * n + 199 * n,
        500 + 36 * n + 398 * n
    ] for n in data["n_people"].sort_values().unique()
])
family_sizes = pd.get_dummies(data["n_people"]).values
# Matrix of 5000 x 11 choice penalties
family_day_choice_penalty = np.matmul(family_sizes, size_penalties)


# Now, create an array A (note it's an array, not a matrix) that is 5000 x 100 x 11. The first index is the family. The second index is the day counting backwards from Christmas Eve, and the third index is the level of choice. For instance A[0, 0, 0] = 1 specifies that family 0, has the first choice on Christmas Eve. A[50, 10, 3] = 1 means that family 50 has their 4th choice (remember, starts at 0) on the eleventh day before Christmas.

# In[ ]:


# Create a 5000 x 100 x 11 array for choices
family_choices = np.zeros((N_FAMILIES, N_DAYS, N_CHOICES+1))
choices = data.drop(columns="n_people")
for i in range(N_CHOICES):
    choice_col = "choice_" + str(i)
    dummies = pd.get_dummies(choices[choice_col], columns=[choice_col])
    family_choices[:, :, i] = dummies.values
# Now if not in any choices, then it is choice 10
family_choices[:, :, N_CHOICES] = np.logical_not(family_choices.sum(axis=2))


# Last, do multiply these arrays and collapse to get a matrix again

# In[ ]:


penalty = np.multiply(family_choices, family_day_choice_penalty.reshape((5000, 1, 11)))
penalty.sum(axis=2)


# Hope that elucidates a different way to code! Thanks to https://www.kaggle.com/vipito/santa-ip for the initial code I was looking at.

# In[ ]:




