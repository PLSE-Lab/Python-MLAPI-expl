#!/usr/bin/env python
# coding: utf-8

# # What effect
# 
# I stumbled across this data set and was curious to explore what the effects of the parents were on total winnings. Prior to this exploration I'd heard of incredible sums of money paid for studding fees in the throughbred racing world. While I suspect there is a passion for the hobby that slightly inflates these prices I would expect to see offspring of some sires and dames to win more.
# 
# I expect this will be skewed by more popular studs or other confounding factors. But to dive in let's look at the data set.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib

horses = pd.read_csv("../input/horses.csv")
print(horses.dtypes)

# Descriptive
# Total horses
print("####")
print("Head")
print("####")

print(horses.head())


# In[ ]:


# Count the number of each sire, I'm selecting the 'age' column to return
# counts but any column would return counts, there is probably a better
# way to accomplish this in pandas. Let me know if you know.
sire_group = horses.groupby('sire_id')
dam_group = horses.groupby('dam_id')


counts = dam_group.count()


# In[ ]:





# In[ ]:




