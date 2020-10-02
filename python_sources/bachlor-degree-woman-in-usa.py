#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import some important libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
get_ipython().run_line_magic('matplotlib', 'inline')
data=pd.read_csv("../input/percent-bachelors-degrees-women-usa.csv")   # upload data 


# In[ ]:


data.info()    # General information about data


# In[ ]:


data.head()          # frist 5 columns of data


# In[ ]:


data.shape  # shape of data


# In[ ]:


data.describe()    # digital information about data


# In[ ]:


data.isnull().sum() # lets check null values 


# In[ ]:


# taking out average % of women's b.tech in different field
avg_agr = data['Agriculture'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Agriculture field are of women".format(avg_agr),)

avg_arch = data['Architecture'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Architecture field are of women".format(avg_arch),)

avg_art = data['Art and Performance'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Art and Performance field are of women".format(avg_art),)

avg_bio = data['Biology'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Biology field are of women".format(avg_bio),)

avg_bus = data['Business'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Business field are of women".format(avg_bus),)

avg_com = data['Communications and Journalism'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Communications field are of women".format(avg_com),)

avg_cse = data['Computer Science'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Computer Science are of women".format(avg_cse),)

avg_edu = data['Education'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Education field are of women".format(avg_edu),)

avg_engg = data['Engineering'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Engineering are of women".format(avg_engg),)

avg_eng = data['English'].mean()
print(" On Average {:.2f}% of Bachelor's degree in English are of women".format(avg_eng),)

avg_for = data['Foreign Languages'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Foreign Languages are of women".format(avg_for),)

avg_health = data['Health Professions'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Health Professions are of women".format(avg_health),)

avg_math = data['Math and Statistics'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Math and Statistics are of women".format(avg_agr),)

avg_phy = data['Physical Sciences'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Physical Sciences  are of women".format(avg_phy),)

avg_psy = data['Psychology'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Psychology are of women".format(avg_psy),)

avg_pub = data['Public Administration'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Public Administration are of women".format(avg_pub),)

avg_sch = data['Social Sciences and History'].mean()
print(" On Average {:.2f}% of Bachelor's degree in Social sciences are of women".format(avg_sch),)


# In[ ]:


# visualizing the performance of women in b.tech
size = [33, 33, 61, 49, 40, 56, 25, 76, 72, 66, 71, 82, 44, 31, 68, 76, 45]
labels = "Agriculture","Architecture","Arts","Bio","Business","Communication","CSE","Education","Engg.","English","Foreign Lang.","Health", "Maths","Phy.Science","Psy","Public Admn.","Social sc."
colors = ['red','pink','orange','crimson','purple','violet','lightblue','lightgreen','yellow','cyan','magenta', 'blue', 'green','brown','maroon','grey','darkblue']

my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.rcParams['figure.figsize'] = (15, 15)
plt.pie(size, colors = colors, labels = labels, shadow = True, autopct  = '%.2f%%')
plt.title("A pie chart representing share of women in different Technologies", fontsize = 25)
plt.axis('off')
plt.legend()
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# #Conclussion
# Male Dominating Fields are Technical and, Female Dominating Fields are Non-Technical.

# # let's divide these B.tech sujects into two categories
# Category 1 -> Technical
# 
# Category 2-> Non Technical

# Technical -> Computer Science, Engineering, Biology, Architecture, Agriculture, Physical Sciences, Maths and Statistics and Businness.
# 
# Non Technical-> Arts and Profession, Communication and Journalism, Education, English, Foreign Language, Social Science and History, Health Professions, Psycholofy, Public Administration

# In[ ]:


# checking the time-series growth of women of USA in b.tech for Technical Areas(Male Dominating)

# setting the size of the figure
plt.rcParams['figure.figsize'] = (20, 15)

# for agriculture
plt.subplot(231)
x1 = data['Year']
y1 = data['Agriculture']

plt.plot(x1, y1, '-*')
plt.title("Growth of Women in Agriculture", fontsize = 20)

# for computer science
plt.subplot(232)
x2 = data['Year']
y2 = data['Computer Science']

plt.plot(x2, y2, '-*')
plt.title("Growth of Women in computer science", fontsize = 20)

# for engineering
plt.subplot(233)
x3 = data['Year']
y3 = data['Engineering']

plt.plot(x3, y3, '-*')
plt.title("Growth of Women in engineering", fontsize = 20)

# for Business
plt.subplot(234)
x4 = data['Year']
y4 = data['Business']

plt.plot(x4, y4, '-*')
plt.title("Growth of Women in business", fontsize = 20)

# for physical science
plt.subplot(235)
x5 = data['Year']
y5 = data['Physical Sciences']

plt.plot(x5, y5, '-*')
plt.title("Growth of Women in physical science", fontsize = 20)

# for Maths and Statistics
plt.subplot(236)
x6 = data['Year']
y6 = data['Math and Statistics']

plt.plot(x6, y6, '-*')
plt.title("Growth of Women in maths and statistics", fontsize = 20)

plt.show()


# From above Graph we can see that Agriculture , Business, Engineering, and Physical Sciences Industry is accepting Women as there is an uptrend in the graph but In Field like Computer Science and Maths and Statistics Women are Declining Since 1980-1990

# In[ ]:


# checking the time-series growth of women of USA in b.tech for Technical Areas(Male Dominating)

# setting the size of the figure
plt.rcParams['figure.figsize'] = (20, 15)

# for education
plt.subplot(231)
x1 = data['Year']
y1 = data['Education']

plt.plot(x1, y1, '-*', color = 'black')
plt.title("Growth of Women in Education", fontsize = 20)

# for english
plt.subplot(232)
x2 = data['Year']
y2 = data['English']

plt.plot(x2, y2, '-*', color = 'black')
plt.title("Growth of Women in English", fontsize = 20)

# for foreign languages
plt.subplot(233)
x3 = data['Year']
y3 = data['Foreign Languages']

plt.plot(x3, y3, '-*', color = 'black')
plt.title("Growth of Women in foreign language", fontsize = 20)

# for social science and history
plt.subplot(234)
x4 = data['Year']
y4 = data['Business']

plt.plot(x4, y4, '-*', color = 'black')
plt.title("Growth of Women in Social Sciences and History", fontsize = 20)

# for psychology
plt.subplot(235)
x5 = data['Year']
y5 = data['Psychology']

plt.plot(x5, y5, '-*', color = 'black')
plt.title("Growth of Women in pyschology", fontsize = 20)

# for Health Professions
plt.subplot(236)
x6 = data['Year']
y6 = data['Health Professions']

plt.plot(x6, y6, '-*', color = 'black')
plt.title("Growth of Women in Health Profession", fontsize = 20)




plt.show()


# Except for Foreign Languages, In all the rest of subjects like Education, English, Social Sciences, Psychology, health profession have an increasing graph showing an further increase in women's share in these subjects

# In[ ]:


## time series plot for balanced subbject

# setting the plot size for the graph
plt.rcParams['figure.figsize'] = (20, 10)

# for biology
plt.subplot(131)
x1 = data['Year']
y1 = data['Biology']

plt.plot(x1, y1, '-*', color = 'green')
plt.title("Growth of Women in Biology", fontsize = 20)

# for communication
plt.subplot(132)
x2 = data['Year']
y2 = data['Communications and Journalism']

plt.plot(x2, y2, '-*', color = 'green')
plt.title("Growth of Women in Journalism", fontsize = 20)

# for maths
plt.subplot(133)
x3 = data['Year']
y3 = data['Math and Statistics']

plt.plot(x3, y3, '-*', color = 'green')
plt.title("Growth of Women in maths", fontsize = 20)

plt.show()


# In[ ]:




