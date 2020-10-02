#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


hero_data1 = pd.read_csv('../input/character-deaths.csv')


# # Which Year had the highest number of deaths?
# 
# ###### Plotting a basic bar plot shows that the highest number of deaths occured in the year 299.  A total of 156 people died in 299. The year after that , 300, shows a decline in the number of deaths, which is at 100.

# In[ ]:


hero_data1.hist(column = 'Death Year', figsize= (15,5), alpha = 0.4)  
filter_year = hero_data1['Death Year'].dropna()
count = Counter(filter_year)

print("Year Death Count")
for k,v in count.items():
    print(k,v)


# # Which House suffered the most number of deaths
# 
# ###### Visually we can see that Houses's Stark and Lannister have the most number of deaths. House Stark at 107 and House Lannister at 102. This is from data collected over the last 5 books. The 3rd house with the most number of deaths is from House GreyJoy standing at 75. One could say that these 3 houses play a big part in the story line. Following House Greyjoy is House Baratheon at 64.

# In[ ]:


house_count = Counter(hero_data1['Allegiances'])

#Summing u all the values into one house. People who belonged to stark have been added to house stark.
house_count['House Lannister'] = house_count['Lannister'] + house_count['House Lannister']
house_count['House Stark'] = house_count['Stark'] + house_count['House Stark']
house_count['House Tyrell'] = house_count['Tyrell'] + house_count['House Tyrell']
house_count['House Tully'] = house_count['Tully'] + house_count['House Tully']
house_count['House Targaryen'] = house_count['Targaryen'] + house_count['House Targaryen']
house_count['House Baratheon'] = house_count['Baratheon'] + house_count['House Baratheon']
house_count['House Martell'] = house_count['Martell'] + house_count['House Martell']
house_count['House Arryn'] = house_count['Arryn'] + house_count['House Arryn']
house_count['House Greyjoy'] = house_count['Greyjoy'] + house_count['House Greyjoy']

del(house_count['Greyjoy'], house_count['Arryn'], house_count['Martell'], house_count['Baratheon'], house_count['Targaryen'], house_count['None'], house_count["Night's Watch"], house_count['Wildling'], house_count['Lannister'], house_count['Stark'], house_count['Tyrell'], house_count['Tully'])
df = pd.DataFrame.from_dict(house_count, orient = 'index')
df.plot(kind = 'barh',color='r', title='House Population', alpha = 0.4, grid=True, legend= False, figsize= (17,7), fontsize= 16)
print(df)


# # Which book has the highest number of deaths? 
#  
# ###### After analyzing the data for the last 5 books, it looks like the book "A Storm of Swords" has the highest number of deaths standing at 97. There may have been a lot of battles going on resulting the death of a lot of characters or maybe the author grew tired of writing about them and decided to end their story. You'll notice that the book "A Feast for Crows", which came after "A Storm of Swords" has the least number of deaths. The author may have had a change in mood and focussed his attention on another part of the story. "A Dance with Dragons" that came out after "A Feast for Crows" has a higher number of deaths, more than double that of the previous book. The author could have done this to make the book more interesting to the readers. The sixth book in the series, "The Winds of WInter", could have a higher count in deaths.

# In[ ]:


book_data = hero_data1['Book of Death']
filter_books = book_data.dropna()
count_books = Counter(filter_books)

plt.figure(figsize= (15,5))
show = plt.bar(range(len(count_books)), count_books.values(), align = 'center', alpha = 0.4)

show[0].set_color('brown')
show[1].set_color('c')
show[2].set_color('grey')
show[3].set_color('g')
show[4].set_color('gold')
Numbers = [49, 73, 97, 27, 61]
for i, v in enumerate(Numbers):
    plt.text(i, v+1, str(v), color = 'black', fontweight = 'bold')
plt.title('Total deaths in each book')
plt.xlabel('Books')
plt.ylabel('Number of characters dead', horizontalalignment='center')
plt.xticks([0, 1, 2, 3, 4], ['A Game of Thrones','A Clash Of Kings','A Storm of Swords','A Feast for Crows','A Dance with Dragons'], rotation = 30)
plt.yticks([0, 20, 40, 60, 80, 100, 120])
plt.grid()
plt.show()


# # Analyzing Battles.csv dataset.

# In[ ]:


battles = pd.read_csv('../input/battles.csv')


# # Is there a relationship between attacker size and defender size?

# In[ ]:


attack_defend_df = battles[['attacker_size','defender_size']].dropna()
print(attack_defend_df)


# #### The scatterplot shows that there is an outlier measuring at 100,000. On closer examination of this data point we can see that it comes from the 'Battle of Castle Black'. This is an extreme outlier and should be removed to get a better spread of the data.

# In[ ]:


plt.figure(figsize = (10,5))
plt.scatter(attack_defend_df['attacker_size'], attack_defend_df['defender_size'], color = 'r', marker = "*", s = 65)
plt.title("Attacker size vs Defender size")
plt.ylabel("Defender size")
plt.xlabel("Attacket size")


# ### Removing the Outlier

# In[ ]:


attack_defend_df = attack_defend_df[attack_defend_df.attacker_size != 100000]
print(attack_defend_df)


# ### After plotting the data we see a better spread of the data. There appears to be a positive relationship here, i.e, an increase in the size of the attacking army, the bigger is the size of the defending army. 

# In[ ]:


plt.figure(figsize = (15,6))
plt.scatter(attack_defend_df['attacker_size'], attack_defend_df['defender_size'], color = 'c', s = 65)
plt.title("Attacker size vs Defender size")
plt.xlabel('Attacker size')
plt.ylabel('Defender size')


# # Sumary Statistics
# ##### After generating the summary statistics, you can see that average of the attacking army(8122.466) is greater than the defending army (7638.33) . The largest attacking army in this dataset is 21,000 and the largest defending army is 20,000. The standard deviations are large for both armies, which means the data is spread. Not all the values lie close to the mean.
# 
# ## Comparing mean and median values of the attacking army.
# 
# ##### The mean(8122.46) is greater than the median(5000), hence we cannot expect the distribution to be normal. More righttly its right skewed. This can be observed in the boxplot titled 'Attacker size'.
# 
# ## Comparing mean and median values of the defending army.
# 
# ##### The mean(7638.33) and median(7250) are alost close to each other. Hence, the distribution of this data will be lightly skewed. This can beobserved in the boxplot titled 'Defender size'.

# In[ ]:


print(attack_defend_df.describe())

fig = plt.figure(figsize = (15, 15))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

ax1.boxplot(attack_defend_df.attacker_size)
ax1.set_title("Attacker size")
ax1.set_ylabel("Army size")

ax2.boxplot(attack_defend_df.defender_size)
ax2.set_title("Defender_size")
ax2.set_ylabel("Army size")


# ##### The covariance(2.0945) is positive which in this case means the larger the attacking army, the defending army also increases in size. I guess the author would do this to make the book more interesting to readers. 

# In[ ]:


print(attack_defend_df.cov())


# ##### The correlation(0.438731) is also positive which shows there is some kind of a relationship but not a very strong one. The correlation value has to lean towards 1 to exhibit a stronger relationship. The author does carefully balance out the proportion of men in both sides of the army. He may do it to lead the readers into beliving that the underdongs have a fighting chance. Overcoming all obstables and emerging victorious. 

# In[ ]:


print(attack_defend_df.corr())

