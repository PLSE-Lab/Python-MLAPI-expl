#!/usr/bin/env python
# coding: utf-8

# # A close look at Nobel Laureates
# Nobel Laureates are some of the smartest and respectable people wwho have graced our world. Their contributions to the sciences within the last century have been tremendous. Let's take a look at what we can glean from data that is commonly available on them. <br/> Let's first load the data before we can do anything with it. Then, we shall take a look at the columns that make up this dataset.

# In[ ]:


import pandas  as pd

#load the nobel laureates data
laureates_data = pd.read_csv('/kaggle/input/nobel-laureates/archive.csv')

# get the columns of the data
print(laureates_data.columns)

# look for missing values
print(laureates_data[laureates_data.isnull().any(axis=1)])


# In[ ]:


# I want more insights into what sort of data to expect
print(laureates_data.head())
print(laureates_data.dtypes)


# In[ ]:


# Let's see what values other than 'Individual' the Laureate type can take
print(laureates_data[laureates_data['Laureate Type'] != 'Individual']['Laureate Type'])


# By looking at the data, we can see that the columns are,
# 1. Year: The year of award
# 2. Category: The field in which the award is given
# 3. Prize: The full name of the prize. Seems to be drivable from other column values.
# 4. Motivation: The reason for awarding the prize.
# 5. Prize Share: How many people share the prize
# 6. Laureate ID: An integer ID 
# 7. Laureate Type: Whether it is an individual or an organization
# 8. Full Name: Full name of the awardee 
# 9. Birth Date: Birth date of the awardee
# 10. Birth City: Birth city of the awardee 
# 11. Birth Country: Birth country of the awardee 
# 12. Sex: Gender of the awardee
# 13. Organization Name: (Only applicable if it's an organization)
# 14. Organization City: (Only applicable if it's an organization)
# 15. Organization Country: (Only applicable if it's an organization)
# 16. Death Date: Date of death of the awardee
# 17. Death City: City of death of the awardee
# 18. Death Country: Country of death of the awardee

# <em>Immediate questions I have based on the overview of data</em>
# 1. How many  organizations were awarded prizes and in what categories? 
# 2. What is the ratio of people to organizations winning prizes?

# # get the organization names of the organizations that received the Nobel prize
# print(laureates_data[laureates_data['Laureate Type'] != 'Individual']['Full Name'])
# print(len(laureates_data[laureates_data['Laureate Type'] != 'Individual']))

# In[ ]:


# Let's see what are the fields where organizations have been awarded a Nobel prize
print(laureates_data[laureates_data['Laureate Type'] != 'Individual']['Category'].unique())


# The organizations have only got a Nobel prize for Peace. I do not know whether this speaks for the biases of the Nobel committee or if any Scientific Organization has not done anythin warranting of the nobel committee's recognition.
# Growing up, I heard that Marie Curie was the only person to receive the Nobel prize twice. Let's see if we can verify this claim.

# In[ ]:


print('The number of entries: %d \n\n' % laureates_data[laureates_data['Full Name'].str.contains('Marie Curie')].shape[0])
print(laureates_data[laureates_data['Full Name'].str.contains('Marie Curie')])


# It is true! There are two entries for Marie Curie. But are there any others? Let's try to find out! 

# In[ ]:


name_counts = laureates_data['Full Name'].value_counts()
multi_name = list(name_counts[name_counts > 1].index)

for name in multi_name:
    temp = laureates_data[laureates_data['Full Name']==name].Year.unique()
    if len(temp) > 1:
        print(name, ' ', temp, '\n')


# Looks like someone gave me some wrong information. Apart from Marie Curie, there are 3 other individuals who have received the Nobel prize more than once.
