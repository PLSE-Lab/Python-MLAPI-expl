#!/usr/bin/env python
# coding: utf-8

# ## Hello!
# In this kernel I will study the animals names, namelly if the Name section from the dataframe can contain strings that say that the pet doesn't have a name. Confusing? I think so...

# In[ ]:



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/train/train.csv')
#color_data = pd.read_csv('../input/train/color_labels.csv')
data.head(10)
# data=data[data['Gender'] ]
# type(data['Name'][8])


# From the head we can see that some of the animals without names are some times anounced as NaN (wich is considered a float and not a string), and some times they have a string mentioning that they don't have a name. Because a string mentioning that the pet doesn't have a name should have the sequence of caracters 'name' in it we will try to isolate the possible nameless pets by looking for those names.
# 
# The reason for removing the other strings in this cell are explained futher bellow.
# 

# In[ ]:


names = data.Name.unique()
no_names=[]
for name in names:
    if type(name) is float:
        continue
    if 'name' in name.lower() or 'kitt' in name.lower() or 'pupp' in name.lower() or 'cats' in name.lower() or 'dogs' in name.lower():
        no_names+=[name]
# no_names


# From this list of names with "name" on it we can see that the vast majority doesn't have a name or  can be rename, so we will consider them all nameless.
#  Another detail we can see is that there are names that make it sound like they belonge to a group of pets category so we will considerer them as unnamed.

# In[ ]:


data['Is_Nameless'] = (pd.isnull(data['Name'])).astype(int)

for index in range(len(data['Name'])):
    if type(data['Name'].iloc[index]) == float:
        continue
    if data['Name'].iloc[index] in no_names:
        data.Is_Nameless[index] = 1


# Now we will visualize the influence of having a name on a pet adoption speed.

# In[ ]:


Named = data[data['Is_Nameless'] == 0]
NotNamed = data[data['Is_Nameless'] == 1]
NamedTotal =[]
NotNamedTotal = []
for i in range(5):
    curr_=sum(sum([Named.AdoptionSpeed == i]))
    NamedTotal+=[curr_/len(Named)*100]
    curr_=sum(sum([NotNamed.AdoptionSpeed == i]))
    NotNamedTotal+=[curr_/len(NotNamed)*100]
    
# Finding the probability of a pet getting adopted up to a period of time
cont_named=[NamedTotal[0]]
cont_unnamed=[NotNamedTotal[0]]

for i in range(1,5):
    cont_named+=[cont_named[i-1]+NamedTotal[i]]
    cont_unnamed+=[cont_unnamed[i-1]+NotNamedTotal[i]]


# In[ ]:


fig, ax = plt.subplots(figsize = (18, 6))
plt.subplot(1, 2, 1)
plt.plot(range(5),NamedTotal,range(5),NotNamedTotal)
plt.legend(['Named','NotNamed'])
plt.xlabel('Adoption Speed')
plt.ylabel('Probability of adoption in %')
plt.title('Probability that a pet gets adopted on a given period of time')

plt.subplot(1, 2, 2)
plt.plot(range(5),cont_named,range(5),cont_unnamed)
plt.legend(['Named','NotNamed'])
plt.xlabel('Adoption Speed')
plt.ylabel('Probability of adoption in %')
plt.title('Probability that a pet gets adopted up to a given period of time')
plt.show()


# In[ ]:


# lets try to get the percentage of group animals and singular pets with our without a name

num_named = len(Named['Name'])
num_named_group = len(Named[Named['Gender']==3])
print("\t\t# animals\t # groups\t % of groups")

print("named animals:    %d\t\t%d\t\t%.2f"%(num_named,num_named_group,num_named_group/num_named*100))

num_not_named = len(NotNamed['Name'])
num_not_named_group = len(NotNamed[NotNamed['Gender']==3])
print("unnamed animals:   %d\t\t%d\t\t%.2f"%(num_not_named,num_not_named_group,num_not_named_group/num_not_named*100))


# 32.85% (after changing the filters it increases to 39.31%) is an highish number, but not huge, lets invetigate if on the remaining group of animals we can find a term that can be used to describe them as being not named of if the remaining animals are actually named.

# In[ ]:


Names_Named_group=Named[Named['Gender']==3]
Names_NotNamed_group=NotNamed[NotNamed['Gender']==3]
#Names_Named_group['Name']


# On a first iteraction:
#     We can see 2 common terms on their names that mean that they don't have name are "puppies" and "kittens", so lets go back to  where we define the filter and add them and see how it works.
# On a second iteraction:
#     After this inteneraction we can see that we could change those for "kitt" and "pupp" to include mispellings, and add "cats" and "dogs"

# In[ ]:


fig, ax = plt.subplots(figsize = (16, 12))
plt.subplot(1, 2, 1)
text_cat = ' '.join(Named['Name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_cat)
plt.imshow(wordcloud)
plt.title('Pets Actual Names')
plt.axis("off")

plt.subplot(1, 2, 2)
text_dog = ' '.join(NotNamed['Name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_dog)
plt.imshow(wordcloud)
plt.title('Words of Unnamed Pets')
plt.axis("off")

plt.show()


# Interesting, 'Girl', 'Boy' and 'Baby' seem like they could be from not named pets, but I'm afraid that by removing them I get too many false positives... Lets study to see if that's the case: 

# In[ ]:


i=0
for Name in Named.Name.unique():
    print(Name)
    if i == 10: # during proper study this value is increased
        break
    i+=1


# We can se some names like "2 Hansome Boys" or "Little Yellow Cat" that look like not real names, but then we have names like ""Baraka" The Black Cat" or "Supergirl" that sound more legitime names. Now lets see the "Not real names" and see if we find false positives:

# In[ ]:


i=0
for Name in NotNamed.Name.unique():
    print(Name)
    if i == 10: # during proper study this value is increased
        break
    i+=1


# In regards to the names listed for those without a name (this is weird to say...) we get some false positive, like "Bella 4 Months Puppy!", but in general seems to get more true positives then false positives.
# 
# In conclusion to this part, I believe that by fillering the way I did I get good enough results, It would be posible to use a LSTM system to try to figure out the names by itself, and by doing it well I would get better results, but it would have to get the expected outputs, which would mean that I would have to read all names and write them has being real or not.
# 
# Thank you for reading!

# In[ ]:




