#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

dt = pd.read_csv("../input/Pokemon.csv")
dt.describe()


# In[ ]:


first_gen_attack = dt["Attack"][dt["Generation"] == 1]
second_gen_attack = dt["Attack"][dt["Generation"] == 2]


# In[ ]:


ttest_ind(first_gen_attack, second_gen_attack, equal_var=False)


# In[ ]:


plt.title("Pokemons histogram for attack in first and secod generation")
plt.hist(first_gen_attack, alpha=.5, color="red")
plt.hist(second_gen_attack, alpha=.5, color="blue")
plt.show()

