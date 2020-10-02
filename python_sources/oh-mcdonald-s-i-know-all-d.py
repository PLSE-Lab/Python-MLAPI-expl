#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np 
import pandas as pd 
import os
data = pd.read_csv('../input/menu.csv')



# Check for missing data:)

# In[ ]:


data.isnull().sum()


# Yes! This dataset is perfect. All values are in place. Well let's see what we love to eat :D I see that the dishes are divided into categories. Breakfast, coffee and tea and so on. The first thing is interesting to see how many useful and harmful substances are in dishes from different categories.

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


data.pivot_table('Vitamin A (% Daily Value)', 'Category').plot(kind='bar', stacked=True)


# In[ ]:


data.pivot_table('Vitamin C (% Daily Value)', 'Category').plot(kind='bar', stacked=True, color = 'c')


# In[ ]:


data.pivot_table('Calcium (% Daily Value)', 'Category').plot(kind='bar', stacked=True, color = 'r')


# In[ ]:


data.pivot_table('Iron (% Daily Value)', 'Category').plot(kind='bar', stacked=True, color = 'k')


# In[ ]:


data.pivot_table('Protein', 'Category').plot(kind='bar', stacked=True, color = 'b')


# All this can be displayed on one chart. But please excuse me. I respect every vitamin. Seems pretty logical. We see that salads contain a record amount of vitamin A compared to dishes from other categories. No wonder! After all, vitamin A is rich in many vegetables, as well as fish and eggs. We can also see how similar the distribution graphs of protein and calcium are! Even for me, it's no secret that protein is rich in calcium.
# 
# You can view each substance. Now I am interested in how many harmful substances are in my favorite dishes.

# In[ ]:


data.pivot_table('Trans Fat', 'Category').plot(kind='bar', stacked=True, color = 'c')


# All right. I'll eat desserts :D

# In[ ]:


data.pivot_table('Sugars', 'Category').plot(kind='bar', stacked=True, color = 'g')


# Or not :D

# In[ ]:


data.pivot_table('Cholesterol', 'Category').plot(kind='bar', stacked=True, color = 'c')


# Probably better to have Breakfast at home.
# ![![image.png](attachment:image.png)](https://media.sproutsocial.com/uploads/2010/12/Are-you-making-these-5-social-media-mistakes.jpg)

# In[ ]:


data.pivot_table('Calories', 'Category').plot(kind='bar', stacked=True, color = 'c')


# If you are on a diet but sometimes like to sin with a hamburger then this is for you! You can afford a drink and a salad. But calories are better to look at each product separately. little later:) I'll do something just as interesting.

# In[ ]:


import seaborn as sns
cols = ['Calories','Cholesterol','Trans Fat','Sugars','Dietary Fiber']
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm,cbar = True, annot = True,square = True, fmt = '.2f', annot_kws = {'size':15}, yticklabels = cols, xticklabels = cols)


# Well, no amazing discoveries we have not made. The correlation matrix confirmed the idea that cholesterol and trans fats are best avoided. They are not only harmful to the body, but also increase the caloric content of food hehe
# If you want to know the average calorie content of products, you can see the distribution. Or look at any of the measures of the Central trend.

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution Calories")
ax = sns.distplot(data["Calories"], color = 'r')

print(data.Calories.mean())
print(data.Calories.median())


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution Sugars")
ax = sns.distplot(data["Sugars"], color = 'c')


print(data.Sugars.mean())
print(data.Sugars.median())


# Basically, the products contain relatively little sugar. But remember that most of the sugar contained in shakes, and do not forget about it at the next visit to McDonald's :)
# ![![image.png](attachment:image.png)](http://www.comparechains.com/images/item_images/303_image_01.jpg)

# The enemy must know in person! Look at these brazen sugar delicious foods lol

# In[ ]:


def plot(grouped):
    item = grouped["Item"].sum()
    item_list = item.sort_index()
    item_list = item_list[-20:]
    plt.figure(figsize=(9,10))
    graph = sns.barplot(item_list.index,item_list.values)
    labels = [aj.get_text()[-40:] for aj in graph.get_yticklabels()]
    graph.set_yticklabels(labels)


# In[ ]:


sugar = data.groupby(data["Sugars"])
plot(sugar)


# How many times I bought a shake when I was on a diet. Now everything became clear.....
# ![![image.png](attachment:image.png)](https://avatars.yandex.net/get-music-user-playlist/70586/583840106.1004.99420/m1000x1000?1531833446778&webp=false)

# I think it is important to see what dishes we can meet vitamins

# In[ ]:


vitaminC = data.groupby(data["Vitamin C (% Daily Value)"])
plot(vitaminC)


# Logical! Autumn is coming, don't forget to drink orange juice and be healthy!

# In[ ]:


vitaminA = data.groupby(data["Vitamin A (% Daily Value)"])
plot(vitaminA)


# and about salads, do not forget :3

# In[ ]:


protein = data.groupby(data["Protein"])
plot(protein)


# Protein is a good thing. So say athletes, let us them to believe:) 
# Conscience does not allow to finish without showing you the worst enemies-foods high in cholesterol and TRANS fats!

# In[ ]:


cholesterol = data.groupby(data["Cholesterol"])
plot(cholesterol)


# ![![image.png](attachment:image.png)](https://cosmopolitan.hu/app/uploads/2016/02/mcdonalds-sultkrumpli-cosmopolitan1415x275.png)

# In[ ]:


fats = data.groupby(data["Trans Fat"])
plot(fats)


# In[ ]:


calories = data.groupby(data["Calories"])
plot(calories)


# Oh, chicken mcnuggets! I declare war on you!

# Thanks! 
# ![![image.png](attachment:image.png)](https://vlg.discont-plitki.ru/upload/iblock/861/861d0c45693d75a49f2331d0290f88ab.jpg)

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 
