#!/usr/bin/env python
# coding: utf-8

# # Gender Ambiguity in American Names
# 
# Some names are very gendered, while others are unisex. How we think of these names changes over time though, along with how open parents are to giving their children gender-neutral names in the first place. We're going to dig into the numbers here and see if anything interesting comes up.
# 
# I'm defining 'gender ambiguity' here as in the first script below: a number between 0 and 1 that describes how hard it is to guess a person's gender from the name alone. 0 means everybody with that name has the same gender, 1 means exactly 50% of people with that name have each gender. It's just double the probability you'd guess gender wrong from the name alone (which of course only goes up to 0.5).
# 
# Note that the dataset is very definitive and binary about genders, so I'm following those simplifying assumptions here (sorry).
# 
# **Also, I'm not super familiar with Pandas, so I'd love any suggestions or pointers to improve this!**

# In[ ]:


import math
import pandas as pd

names_data = pd.read_csv("../input/NationalNames.csv")

frequent_names = names_data[names_data['Count'] > 20]
indexed_names = frequent_names.set_index(['Year', 'Name'])['Count']

# Number between 0 and 1 representing ambiguity, from certain to totally ambiguous
# 0 = all the same gender, 1 = exactly 50% of each gender. Assumes only two options.
def ambiguity_measure(grouped_frame):
        return (2 * (1 - (grouped_frame.max() / grouped_frame.sum())))

# Various useful formattings of gender ambiguity data:
ambiguity_data = ambiguity_measure(indexed_names.groupby(level=['Year', 'Name'])).rename("Ambiguity")
yearly_ambiguity = ambiguity_data.groupby(level='Year')

ambiguity_with_counts = ambiguity_data.to_frame().join(indexed_names.groupby(level=['Year', 'Name']).sum())
data_vs_years = ambiguity_with_counts.unstack(level='Year')
data_vs_years["Total"] = data_vs_years['Count'].sum(axis=1)


# # The most gender-ambiguous name in America, by year of birth.

# In[ ]:


yearly_ambiguity.idxmax().apply(lambda x: x[1]).to_frame() 


# # How has the ambiguity of names changed over the years?
# 
# Filtered for an interesting selection here, as it's hard to look at all the names together. so we're only looking at names which at some point where reasonably ambiguous (more than a 5% chance you'd guess the gender wrong based on the name), and then the overall most popular 7 names from within those.
# 
# I've love to include more names, but it's hard to match the colours with more lines sadly. If anybody wants to send me (here or on twitter: @pimterry) a good way to add labels to lines, or a different way to portray this, that would be great!
# 
# Interesting points though: 'Willie' has been around for a long time, was once a very unisex name, and became steadily more male for 100 years (but has now basically disappeared). 'Ashley'5 appeared suddenly as a unisex option in 60s, and then settled towards being predominantly female over 25 years or so.
# 
# General pattern in a lot of these is names starting off less gender specific, and becoming more gendered over time. 'Ryan' is the one obvious counter-example in the names shown here: it appeared in the 70s, all male, and is slowly but steadily moving in the other direction.

# In[ ]:


ambiguous_names = data_vs_years[(data_vs_years['Ambiguity'] > 0.1).any(axis=1)]
popular_ambiguous_names = ambiguous_names.sort_values(by='Total', ascending=False).head(7).drop("Total", axis=1)
popular_ambiguous_names['Ambiguity'].transpose().plot(figsize=(10, 10))


# # How gender-ambiguous are the names used each year?
# 
# This is the simplistic measure: what's the average gender ambiguity, just of the names used each year?
# 
# Note that this doesn't weight by how much those names are used - that's the next chart. This is essentially a measure of the ambiguity of the names *available* and in common-ish use (i.e. at least 20 births in a year)

# In[ ]:


# Gender ambiguity by name (not *person*! See the next chart)
yearly_ambiguity.mean().transpose().plot(figsize=(10, 10))


# # How gender-ambiguous is the average person's name, by birth year?
# 
# The names available have become more ambiguous over time (with a dip in the 70s). Are we picking more gender ambiguous names though, given those options?
# 
# Spoilers: yes! Not only are the total set of names in use fitting less neatly into one gender or another, people are picking more gender-ambiguous names overall. There's approximately a 2.5% chance you'll guess the gender of somebody in the USA born in 2014 wrong, more than double the odds a century ago. Slow changes, but what looks like a clear steady shift in the numbers.

# In[ ]:


# = SUM(probability of name in given year * ambiguity of name)
total_people_per_year = ambiguity_with_counts['Count'].groupby(level='Year').sum()
ambiguity_by_year = ambiguity_with_counts.unstack('Name')
ambiguity_by_year["total_people"] = total_people_per_year
weighted_ambiguity = ambiguity_by_year.apply(lambda x : x['Ambiguity'] * (x['Count']/x['total_people'][0]), axis=1)
weighted_ambiguity.sum(axis=1).plot(figsize=(10, 10))

