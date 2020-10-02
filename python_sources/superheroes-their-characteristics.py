#!/usr/bin/env python
# coding: utf-8

# ![](https://i2.wp.com/gentwenty.com/wp-content/uploads/2013/07/Superheroes.jpg?resize=610%2C458)
# **Exploration of Superheroes dataset**
# We will be working with basics of numpy and pandas to explore Super hero dataset. 
# Objective : 
# 1. Understand the depth and characteristics of dataset
# 2. Analyse gender distribution and alignment of super heroes
# 3. Appearance characteristics
# 4. Diving into super powers

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


add = '../input/heroes_information.csv'
hero = pd.read_csv(add)
hero.head()


# In[51]:


import matplotlib.pyplot as plt

hero.nunique()
plt.style.available


# In[72]:


plt.style.use('fivethirtyeight')
hero.Publisher.value_counts().plot(kind='pie', figsize=(6,6),title='Publisherwise SuperHero Distribution')


# ### Lets look into Gender and Alignment 

# In[23]:


gen=hero.Gender.value_counts()
gen


# In[66]:


plt.style.use('seaborn')
gen.plot(kind='pie',figsize=(8,8),legend=True)


# In[8]:


hero.Alignment.value_counts()


# ##### Lets convert them into numerical values for ease of counting
# ##### For Gender
# - Male will be assigned 1 
# - Females will be assinged 0
# ##### For Alignment
# - Good : 1 , Neutral : 0 , Bad : -1
# - This will help in counting and doing operations 

# In[9]:


def gender(s):
    if(s=='Male'):
        return 1
    elif(s=='Female'):
        return 0
    else:
        return -1
    
def align(s):
    if(s=='good'):
        return 1
    elif(s=='neutral'):
        return 0
    else:
        return -1
    


# In[10]:


hero['sex']= hero.apply(lambda hero:gender(hero['Gender']),axis=1)
hero['align'] = hero.apply(lambda hero:align(hero['Alignment']),axis=1)
hero.head(5)


# In[39]:


hero_gender = hero.pivot_table(values='name',index='Publisher',columns=['sex'],aggfunc=np.count_nonzero)
pub = hero.pivot_table(values='name',index='Publisher',aggfunc=np.count_nonzero)
pub.rename(columns={'name':'total'},inplace=True)
pub['male_percent']= hero_gender[1]*100/pub.total
pub_m =pub.sort_values(by=['total'],ascending=False).head(10)
pub_m


# In[46]:


pub_m.sort_values('male_percent', ascending = True,inplace=True )
pub_m['male_percent'].plot(kind='barh')


# #### Insights on gender bias
# 
# 1. Super heroes are dominated by males in every Publisher house (except for ABC studios)
# 2. On the whole 68% of all super heros are Men ! 
# 3. George Lucas and Star Trek has more than 80% of the characters as men

# In[13]:


hero_align = hero.pivot_table(values='name',index='align',columns=['sex'],aggfunc=np.count_nonzero)
h_align=hero_align/pub.total.sum()*100
bad_woman_percent = h_align[0][-1]*100/(h_align[0][1]+h_align[0][-1])
bad_woman_percent


# In[14]:


bad_men_percent = h_align[1][-1]*100/(h_align[1][1]+h_align[1][-1])
bad_men_percent


# In[15]:


bad_men_percent/bad_woman_percent


# In[16]:


h_align


# #### Insight
# - Super hero men are twice as likely to be bad than super hero woman. (You may have thought otherwise)

# #### Are there any correlations ?

# In[17]:


h_corr=hero.corr(method='pearson')
print((h_corr>0.5)|(h_corr<-0.5))


# #### Insight
# - Seems little obvious but there is a + ve correlation for Height and weight

# In[18]:


hero_color =hero[['Eye color','Hair color','Skin color']]
#you can observe some colors have words starting in capital and some in lower case
#lets convert all of them into lower case
hero_color= hero_color.applymap(lambda x : x.lower())
hero_color['combo']=hero_color['Eye color']+"-"+hero_color['Hair color']+"-"+hero_color['Skin color']
hero_color.combo.value_counts().head(10)


# #### Insight
# - Excluding undeclared attributes 15 % of the known super heros have blue eyes and blond hair
# - Example of blue eyes and blond hair - Thor, Captain America, Aquaman and many more
# - Next frequent physical appearance of super heros is brown eyes and black hair. (Note - I too have similar appearance)

# ### Diving into Super powers
# Who has most number of powers ?

# In[20]:


adda="../input/super_hero_powers.csv"
power = pd.read_csv(adda)
power.set_index('hero_names',inplace=True)
power.sum(axis=1).sort_values(ascending=False)[:15]


# ## The answer is Spectre with 49 powers ( but in "reality" One-above-All is the most powerful  even with just 31 powers)
# #### we are considering number of powers and not giving weight to each power for evaluation
