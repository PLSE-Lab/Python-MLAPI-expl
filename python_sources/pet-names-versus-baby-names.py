#!/usr/bin/env python
# coding: utf-8

# ![](http://i.imgur.com/MAvtmZ6.jpg)
# 
# 
# # Pet Names versus Baby Names
# 
# What makes for a good pet name? Sometimes people name their pets with human-sounding names like "Jacob", "Abbie", and "Tyler". However, the most popular pet names in the popular consciousness --- names like "Spot", "Rex", and "Pluto" --- are decidingly *not* human-sounding.
# 
# Can we figure out which of these two "kinds" of names is more common in reality?

# First, we need to import our data and coprocess it to find the overlapping dog/human names. We will combine two datasets: the present dataset of dog names in New York City, and a part of the US Baby Names by State dataset corresponding with recent newborns in New York State.

# In[ ]:


import pandas as pd

dog_names = pd.read_csv("../input/nyc-dog-names/dogNames2.csv")
dog_names = dog_names                      .set_index(dog_names.Row_Labels.str.title())                      .drop('Row_Labels', axis='columns')
people_names = pd.read_csv("../input/us-baby-names/StateNames.csv")                    .query("Year == '2014' and State == 'NY'")                    .rename(columns={'Count': 'Count_PersonName'})                    .drop(['State', 'Id', 'Year'], axis=1)                    .groupby('Name')                    .sum()                    .reset_index()                    .set_index('Name')


# In[ ]:


overlapping_names = dog_names.join(people_names, how='inner')
overlapping_names = (overlapping_names
                         .rename(columns={'Count_AnimalName': 'Dogs',
                                          'Count_PersonName': 'People'})
                    )

n_dogs = dog_names['Count_AnimalName'].sum()
n_people = people_names['Count_PersonName'].sum()

overlapping_names['People:Dogs'] = overlapping_names['People'] / overlapping_names['Dogs']
overlapping_names['% People'] = overlapping_names['People'] / n_people
overlapping_names['% Dogs'] = overlapping_names['Dogs'] / n_dogs

overlapping_names.head()


# The most popular human also-dog names are hardly distinguishable from simply very popular human names.

# In[ ]:


overlapping_names.sort_values(by='People', ascending=False).head()


# And here are some very popular pet names which are also occassionally baby names:

# In[ ]:


overlapping_names.sort_values(by='People:Dogs', ascending=True).head()


# So: how much overlap is there between baby names and pet names?
# 
# To answer, let's slightly rephrase this question: assuming an equal number of babies and pets, given a name, how often can correctly guess whether it belongs to a baby or a pet?

# In[ ]:


nonpet_names = set(people_names.index).difference(set(overlapping_names.index))
nonbaby_names = set(dog_names.index).difference(set(overlapping_names.index))

majority_pet_names = set(overlapping_names[overlapping_names["People:Dogs"] < 1].index)
majority_person_names = set(overlapping_names[overlapping_names["People:Dogs"] >= 1].index)

baby_predicted_actual = people_names.loc[list(nonpet_names.union(majority_person_names))]['Count_PersonName'].sum()
pet_predicted_actual = dog_names.loc[list(nonbaby_names.union(majority_pet_names))]['Count_AnimalName'].sum()

baby_predicted_nonactual = overlapping_names[overlapping_names["People:Dogs"] < 1]['People'].sum()
pet_predicted_nonactual = overlapping_names[overlapping_names["People:Dogs"] >= 1]['Dogs'].sum()


# We can examine the results of our heuristical classification system in a **confusion matrix**. A confusion matrix pitches the classes that we predict against the classes that are actually correct.

# In[ ]:


pd.DataFrame([
    [baby_predicted_actual, baby_predicted_nonactual],
    [pet_predicted_actual, pet_predicted_nonactual]
], columns=['People (Actual)', 'Dogs (Actual)'], index=['People (Predicted)', 'Dogs (Predicted)'])


# Converting this to percentages:

# In[ ]:


pd.DataFrame([
    [baby_predicted_actual / n_people, baby_predicted_nonactual / n_people],
    [pet_predicted_actual / n_dogs, pet_predicted_nonactual / n_dogs]
], columns=['People (Actual)', 'Dogs (Actual)'], index=['People (Predicted)', 'Dogs (Predicted)'])


# **80% of pet names are naively human sounding.**
# 
# We can see this intuitively! Let's draw sample pet names five at a time. Does it seem right to you that just one of these five names sounds doggy to you?

# In[ ]:


import numpy as np
sample_space = dog_names.cumsum().values

print(
    dog_names.iloc[[np.argmax(sample_space > n) for n in np.random.choice(n_dogs, 5)]].index.values
)


# "Spot" sounds like an obviously "doggy" name to me, while the rest of them sound quite human.
# 
# For your consideration, here's ten more lists of names:

# In[ ]:


for _ in range(10):
    print(
        dog_names.iloc[[np.argmax(sample_space > n) for n in np.random.choice(n_dogs, 5)]].index.values
    )

