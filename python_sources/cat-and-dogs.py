#!/usr/bin/env python
# coding: utf-8

# Looking at differential outcomes for cats and dogs

# In[ ]:


import pandas as pd
#load data, seperate into cats andd dogs

df = pd.read_csv("../input/train.csv")
cats = df[df["AnimalType"] == "Cat"]
dogs = df[df["AnimalType"] == "Dog"]


# In[ ]:


cat_outcomes = cats["OutcomeType"].value_counts()
cat_count = cats["OutcomeType"].count()
print("Total Cat Outcomes")
print(cat_outcomes)
print("Total Cats")
print(cat_count)

print("\nTotal Dog Outcomes")
dog_outcomes = dogs["OutcomeType"].value_counts()
dog_count = dogs["OutcomeType"].count()
print(dog_outcomes)
print("Total Dogs")
print(dog_count)


# The data-set contains more dog entries than cats. Due to this we can see that a higher proportion of cats were transferred or died.

# In[ ]:


pcent_cat_outcomes = cat_outcomes / cat_count * 100
print("Percentage Cat Outcomes")
print(pcent_cat_outcomes)
print("\nPercentage Dog Outcomes")
pcent_dog_outcomes = dog_outcomes / dog_count * 100
print(pcent_dog_outcomes)


# In[ ]:


import matplotlib.pyplot as plt, numpy as np

cat_data = pcent_cat_outcomes.sort_index()
dog_data = pcent_dog_outcomes.sort_index()

ind = np.arange(len(cat_data))

fig, ax = plt.subplots()
bar_width = 0.35
cat_g = plt.bar(ind,cat_data.values,bar_width,color='b',label='Cat')
dog_g = plt.bar(ind+bar_width,dog_data.values,bar_width,color='r',label='Dog')
plt.xticks(ind + bar_width,cat_data.index,rotation = 90)
plt.legend()
plt.show()


# The rates of cats and dogs who were adopted or euthanized are approximately the same. However, dogs are more likely to be returned to their owners than cats are. Cats were also more likely to be transferred than dogs.
# 
# It is possible that the differences between number of animals returned to owners dogs reflect the fact that dogs are often taken from their owners whereas cats are often found as strays. The difference in transfer rates may be related to this; if less cats are returned to owners, then more will need to be housed. The facility may not have enough room for these extra cats and so transfer them to other locations

# In[ ]:


euthanasia = df[df["OutcomeType"] == "Euthanasia"]
subtype = euthanasia["OutcomeSubtype"].value_counts()
print(subtype)


# There are six different categories for why animals were euthanized. However this list can be condensed. For example, aggressive and behaviour are very similar, as are rabies risk and medical. There were also very few animals euthanized due to court/investigation reasons, this can be dropped.

# In[ ]:


suffering = subtype["Suffering"]
behaviour = subtype["Aggressive"] + subtype["Behavior"]
medical = subtype["Rabies Risk"] + subtype["Medical"]


subtype_list = [suffering,behaviour,medical]
list_names = ["Suffering","Behaviour","Medical"]
ind2 = np.arange(len(subtype_list))

plt.bar(ind2,subtype_list)
plt.xticks(ind2,list_names, rotation = 90)
plt.show()


# It appears that more animals were euthanized to prevent suffering.
