#!/usr/bin/env python
# coding: utf-8

# **Learning to code at school correlate with the income at laboral life**

# Every day it's necessary the use of programming tools for work, the computers are substituting people as workforce, so it's almost mandatory to learn (at least) programming for analyzing data.
# 
# The question for Kagglers is: When did you learn Data Science or Machine Learning and how it helps to your laboral life?
# 
# We have a big job: we have to find a correlation (if exist) between when did you learn Data Science or Machine Learning (DS/ML) (before, during o after University) and how it affects in your economical compensation at work. 
# 
# Let's start with question 35 to find out how to Kagglers did learn:
# 
# **What percentage of your current DS/ML training falls under each category?**
# 
# The options are:
# * Self-Taught
# * Online Courses (Coursera, Udemy, edX, etc)
# * Work
# * University
# * Kaggle competitions
# * Others
# 
# Every Kaggler had to sum a total of 100% in their answers.
#  
#  Take a look to the answers of question 35 (I exclude NaN values, because don't work dfor us):
# 

# In[ ]:


#Import the necessay libraries to work
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# In[ ]:


#Import the necessary files from Kaggle's survey
schema = pd.read_csv("../input/SurveySchema.csv")
freeForm = pd.read_csv("../input/freeFormResponses.csv", dtype=np.object)
multiple = pd.read_csv("../input/multipleChoiceResponses.csv", dtype=np.object)


# In[ ]:


#Let's see question 35
dfq35 = multiple.filter(regex="(Q{t}$|Q{t}_)".format(t = 35))[1:]


# In[ ]:


#Eliminate all the NaN values
dfq35 = dfq35.dropna(how='any')
dfq35.head(6)


# Now, we are interested in the mean for each answer, so, we calculate it and present then in a beautiful pie graph.

# In[ ]:


#Get the values for each column
dfq35_1 = dfq35["Q35_Part_1"][1:].values
dfq35_2 = dfq35["Q35_Part_2"][1:].values
dfq35_3 = dfq35["Q35_Part_3"][1:].values
dfq35_4 = dfq35["Q35_Part_4"][1:].values
dfq35_5 = dfq35["Q35_Part_5"][1:].values
dfq35_6 = dfq35["Q35_Part_6"][1:].values

#Convert each value to float and get the mean of the column
dfq35_1 = dfq35_1.astype(np.float).mean()
dfq35_2 = dfq35_2.astype(np.float).mean()
dfq35_3 = dfq35_3.astype(np.float).mean()
dfq35_4 = dfq35_4.astype(np.float).mean()
dfq35_5 = dfq35_5.astype(np.float).mean()
dfq35_6 = dfq35_6.astype(np.float).mean()

#Get the percentage
q35pie = dfq35_1, dfq35_2, dfq35_3, dfq35_4, dfq35_5, dfq35_6
print ("Mean values for all the ages:" + '\n'+ "Self-Taught: {}".format(q35pie[0]) + '\n' + "Online Courses: {}".format(q35pie[1]) + '\n' + "Work: {}".format(q35pie[2]) 
       + '\n' + "University: {}".format(q35pie[3]) + '\n' + "Kaggle Competitions: {}".format(q35pie[4]) + '\n' + "Others: {}".format(q35pie[5]))


# In[ ]:


#Graph 
q35labels = ("Self-taught", "Online courses", "Work", "University", "Kaggle Competitions", "Others")
explode = (0.1, 0, 0, 0, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(q35pie, explode=explode, labels=q35labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# It's easy to see than most people have learned DS/ML by self-taught, and also by Online courses. 
# 
# This is a general view for all the Kagglers, what if we divide them for age range? 
# 
# We can suppose that older people learned DS/ML for self-taught because in the past it not was too common to learn DS/ML at school.

# We define four groups of age by grouping the following age ranges:
# 
# **Youngers**
# -  18 - 21
# -  22 - 24
# -  25 - 29
# 
# **Adults**
# - 30 - 34
# - 35 - 39
# - 40 - 44
# 
# **Medium adults**
# - 45 - 49
# - 50 - 54
# - 55 - 59
# 
# **Older adults**
# - 60 - 69
# - 70 - 79
# - 80+
# 
# Let's calculate the mean percentage of where did each group learned DS/ML:

# In[ ]:


#Make a df for each age group
multjov = multiple[(multiple.Q2 == '18-21')|(multiple.Q2 == '22-24')|(multiple.Q2 == '25-29')]
multadult = multiple[(multiple.Q2 == '30-34')|(multiple.Q2 == '35-39')|(multiple.Q2 == '40-44')]
multmoread = multiple[(multiple.Q2 == '45-49')|(multiple.Q2 == '50-54')|(multiple.Q2 == '55-59')]
multolder = multiple[(multiple.Q2 == '60-69')|(multiple.Q2 == '70-79')|(multiple.Q2 == '80+')]

#Calculate the mean for each group (we use the same method than the previous df )
##Youngers
jovq35 = multjov.filter(regex="(Q{t}$|Q{t}_)".format(t = 35))[1:]
jovq35 = jovq35.dropna(how='any')

jovq35_1 = jovq35["Q35_Part_1"][1:].values
jovq35_2 = jovq35["Q35_Part_2"][1:].values
jovq35_3 = jovq35["Q35_Part_3"][1:].values
jovq35_4 = jovq35["Q35_Part_4"][1:].values
jovq35_5 = jovq35["Q35_Part_5"][1:].values
jovq35_6 = jovq35["Q35_Part_6"][1:].values

jovq35_1 = jovq35_1.astype(np.float).mean()
jovq35_2 = jovq35_2.astype(np.float).mean()
jovq35_3 = jovq35_3.astype(np.float).mean()
jovq35_4 = jovq35_4.astype(np.float).mean()
jovq35_5 = jovq35_5.astype(np.float).mean()
jovq35_6 = jovq35_6.astype(np.float).mean()

jovq35pie = jovq35_1, jovq35_2, jovq35_3, jovq35_4, jovq35_5, jovq35_6

##Adults
adultq35 = multadult.filter(regex="(Q{t}$|Q{t}_)".format(t = 35))[1:]
adultq35 = adultq35.dropna(how='any')

adultq35_1 = adultq35["Q35_Part_1"][1:].values
adultq35_2 = adultq35["Q35_Part_2"][1:].values
adultq35_3 = adultq35["Q35_Part_3"][1:].values
adultq35_4 = adultq35["Q35_Part_4"][1:].values
adultq35_5 = adultq35["Q35_Part_5"][1:].values
adultq35_6 = adultq35["Q35_Part_6"][1:].values

adultq35_1 = adultq35_1.astype(np.float).mean()
adultq35_2 = adultq35_2.astype(np.float).mean()
adultq35_3 = adultq35_3.astype(np.float).mean()
adultq35_4 = adultq35_4.astype(np.float).mean()
adultq35_5 = adultq35_5.astype(np.float).mean()
adultq35_6 = adultq35_6.astype(np.float).mean()

adultq35pie = adultq35_1, adultq35_2, adultq35_3, adultq35_4, adultq35_5, adultq35_6

##Medium Adultos
madultq35 = multmoread.filter(regex="(Q{t}$|Q{t}_)".format(t = 35))[1:]
madultq35 = madultq35.dropna(how='any')

madultq35_1 = madultq35["Q35_Part_1"][1:].values
madultq35_2 = madultq35["Q35_Part_2"][1:].values
madultq35_3 = madultq35["Q35_Part_3"][1:].values
madultq35_4 = madultq35["Q35_Part_4"][1:].values
madultq35_5 = madultq35["Q35_Part_5"][1:].values
madultq35_6 = madultq35["Q35_Part_6"][1:].values

madultq35_1 = madultq35_1.astype(np.float).mean()
madultq35_2 = madultq35_2.astype(np.float).mean()
madultq35_3 = madultq35_3.astype(np.float).mean()
madultq35_4 = madultq35_4.astype(np.float).mean()
madultq35_5 = madultq35_5.astype(np.float).mean()
madultq35_6 = madultq35_6.astype(np.float).mean()

madultq35pie = madultq35_1, madultq35_2, madultq35_3, madultq35_4, madultq35_5, madultq35_6

#Olders
olderq35 = multolder.filter(regex="(Q{t}$|Q{t}_)".format(t = 35))[1:]
olderq35 = olderq35.dropna(how='any')

olderq35_1 = olderq35["Q35_Part_1"][1:].values
olderq35_2 = olderq35["Q35_Part_2"][1:].values
olderq35_3 = olderq35["Q35_Part_3"][1:].values
olderq35_4 = olderq35["Q35_Part_4"][1:].values
olderq35_5 = olderq35["Q35_Part_5"][1:].values
olderq35_6 = olderq35["Q35_Part_6"][1:].values

olderq35_1 = olderq35_1.astype(np.float).mean()
olderq35_2 = olderq35_2.astype(np.float).mean()
olderq35_3 = olderq35_3.astype(np.float).mean()
olderq35_4 = olderq35_4.astype(np.float).mean()
olderq35_5 = olderq35_5.astype(np.float).mean()
olderq35_6 = olderq35_6.astype(np.float).mean()

olderq35pie = olderq35_1, olderq35_2, olderq35_3, olderq35_4, olderq35_5, olderq35_6

jovq35pie, adultq35pie, madultq35pie, olderq35pie

print ("Mean values for age ranges:" + '\n'+ '\n' + "Youngers: " + '\n' + "Self-taught: {}".format(jovq35pie[0]) + '\n' + "Online Courses: {}".format(jovq35pie[1]) + '\n' + "Work: {}".format(jovq35pie[2]) 
       + '\n' + "University: {}".format(jovq35pie[3]) + '\n' + "Kaggle Competitions: {}".format(jovq35pie[4]) + '\n' + "Others: {}".format(jovq35pie[5]) + '\n' + '\n'
       + "Adults: " + '\n' + "Self-taught: {}".format(adultq35pie[0]) + '\n' + "Online Courses: {}".format(adultq35pie[1]) + '\n' + "Work: {}".format(adultq35pie[2]) 
       + '\n' + "University: {}".format(adultq35pie[3]) + '\n' + "Kaggle Competitions: {}".format(adultq35pie[4]) + '\n' + "Others: {}".format(adultq35pie[5]) + '\n' + '\n'
       + "Medium Adults: " + '\n' + "Self-taught: {}".format(madultq35pie[0]) + '\n' + "Online Courses: {}".format(madultq35pie[1]) + '\n' + "Work: {}".format(madultq35pie[2]) 
       + '\n' + "University: {}".format(madultq35pie[3]) + '\n' + "Kaggle Competitions: {}".format(madultq35pie[4]) + '\n' + "Others: {}".format(madultq35pie[5])+ '\n'+ '\n' + "Olders: " 
       + '\n' + "Self-taught: {}".format(olderq35pie[0]) + '\n' + "Online Courses: {}".format(olderq35pie[1]) + '\n' + "Work: {}".format(olderq35pie[2]) 
       + '\n' + "University: {}".format(olderq35pie[3]) + '\n' + "Kaggle Competitions: {}".format(olderq35pie[4]) + '\n' + "Others: {}".format(olderq35pie[5]))


# And now, another beautiful graph:

# In[ ]:


#Graph
##Set width of bar
barWidth = 0.20
 
##Sset height of bar
bars1 = jovq35pie
bars2 = adultq35pie
bars3 = madultq35pie
bars4 = olderq35pie
 
##Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

##Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Youngers')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Adults')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Medium Adults')
plt.bar(r4, bars4, color='#ADFF2F', width=barWidth, edgecolor='white', label='Olders')

#Add xticks on the middle of the group bars
#plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], 
           ['Self-taught', 'Online Courses', 'Work', 'University','Kaggle Competitions', 'Other'],
          rotation=45)
#plt.setp(xtickNames, rotation=45, fontsize=8)
plt.ylabel('Percentage of user than learn by')
plt.xlabel('Platform')
# Create legend & Show graphic
plt.legend()
plt.show()


# The tendence in self-taught increase with the age groups, and the supposition that we make is correct! And, this is related with learning programming at University, because it decrease with the age ranges! :o
# 
# Interesting?
# 
# Now we know,  how did Kagglers learn DS/ML, now let's see when did they start learn programming to analyze data.
# 
# Let's analize question 24 to find out when Kagglers did learn:
# 
# **How long have you been writing code to analyze data?**
# 
# The options are:
# * <1 year
# * 1 - 2 years
# * 3 - 5 years
# * 5 - 10 years
# * 10 - 20 years
# * 20 - 30 years
# * 30 - 40  years
# * 40+ years
# * I have never written code but I want to learn
# * I have never written code and I do not want to learn
# 
# Take a look to the answers for question 24 (again, we exclude NaN values):

# In[ ]:


#Eliminate all the NaN's values
dfq24 = multiple.filter(regex="(Q{t})".format(t = 24))[1:]
dfq24 = dfq24.dropna(how='any')
dfq24.head(6)


# In[ ]:


##We use the same method than the previous df
#Youngers
jovq24 = multjov.filter(regex="(Q{t}$|Q{t}_)".format(t = 24))[1:]
jovq24 = jovq24.dropna(how='any')

jovq24_1 = len(jovq24[jovq24.Q24 == '< 1 year'])-1
jovq24_2 = len(jovq24[jovq24.Q24 == '1-2 years'])-1
jovq24_5 = len(jovq24[jovq24.Q24 == '3-5 years'])-1
jovq24_10 = len(jovq24[jovq24.Q24 == '5-10 years'])-1
jovq24_20 = len(jovq24[jovq24.Q24 == '10-20 years'])-1
jovq24_30 = len(jovq24[jovq24.Q24 == '20-30 years'])
jovq24_40 = len(jovq24[jovq24.Q24 == '30-40 years'])-1
jovq24_50 = len(jovq24[jovq24.Q24 == '40+ years'])-1
jovq24_l = len(jovq24[jovq24.Q24 == 'I have never written code but I want to learn'])-1
jovq24_n = len(jovq24[jovq24.Q24 == 'I have never written code and I do not want to learn'])-1

jovq24_years = (jovq24_1,jovq24_2,jovq24_5,jovq24_10,jovq24_20,jovq24_30,jovq24_40,jovq24_50,jovq24_l,jovq24_n)
jovq24_years = np.asarray(jovq24_years)
jovq24_yearstotal = jovq24_years.sum()
jovq24_porc = (jovq24_years *100 )/jovq24_yearstotal 

##Adults
adultq24 = multadult.filter(regex="(Q{t}$|Q{t}_)".format(t = 24))[1:]
adultq24 = adultq24.dropna(how='any')

adultq24_1  = len(adultq24[adultq24.Q24 == '< 1 year'])-1
adultq24_2  = len(adultq24[adultq24.Q24 == '1-2 years'])-1
adultq24_5  = len(adultq24[adultq24.Q24 == '3-5 years'])-1
adultq24_10 = len(adultq24[adultq24.Q24 == '5-10 years'])-1
adultq24_20 = len(adultq24[adultq24.Q24 == '10-20 years'])-1
adultq24_30 = len(adultq24[adultq24.Q24 == '20-30 years'])
adultq24_40 = len(adultq24[adultq24.Q24 == '30-40 years'])-1
adultq24_50 = len(adultq24[adultq24.Q24 == '40+ years'])-1
adultq24_l  = len(adultq24[adultq24.Q24 == 'I have never written code but I want to learn'])-1
adultq24_n  = len(adultq24[adultq24.Q24 == 'I have never written code and I do not want to learn'])-1

adultq24_years = (adultq24_1,
                  adultq24_2,
                  adultq24_5,
                  adultq24_10,
                  adultq24_20,
                  adultq24_30,
                  adultq24_40,
                  adultq24_50,
                  adultq24_l,
                  adultq24_n)
adultq24_years = np.asarray(adultq24_years)
adultq24_yearstotal = adultq24_years.sum()
adultq24_porc = (adultq24_years *100 )/adultq24_yearstotal 

##Medium Adults
madultq24 = multmoread.filter(regex="(Q{t}$|Q{t}_)".format(t = 24))[1:]
madultq24 = madultq24.dropna(how='any')

madultq24_1  = len(madultq24[madultq24.Q24 == '< 1 year'])-1
madultq24_2  = len(madultq24[madultq24.Q24 == '1-2 years'])-1
madultq24_5  = len(madultq24[madultq24.Q24 == '3-5 years'])-1
madultq24_10 = len(madultq24[madultq24.Q24 == '5-10 years'])-1
madultq24_20 = len(madultq24[madultq24.Q24 == '10-20 years'])-1
madultq24_30 = len(madultq24[madultq24.Q24 == '20-30 years'])
madultq24_40 = len(madultq24[madultq24.Q24 == '30-40 years'])-1
madultq24_50 = len(madultq24[madultq24.Q24 == '40+ years'])-1
madultq24_l  = len(madultq24[madultq24.Q24 == 'I have never written code but I want to learn'])-1
madultq24_n  = len(madultq24[madultq24.Q24 == 'I have never written code and I do not want to learn'])-1

madultq24_years = (madultq24_1,
                   madultq24_2,
                   madultq24_5,
                   madultq24_10,
                   madultq24_20,
                   madultq24_30,
                   madultq24_40,
                   madultq24_50,
                   madultq24_l,
                   madultq24_n)
madultq24_years = np.asarray(madultq24_years)
madultq24_yearstotal = madultq24_years.sum()
madultq24_porc = (madultq24_years *100 )/madultq24_yearstotal 

##Olders
olderq24 = multolder.filter(regex="(Q{t}$|Q{t}_)".format(t = 24))[1:]
olderq24 = olderq24.dropna(how='any')

olderq24_1  = len(olderq24[olderq24.Q24 == '< 1 year'])-1
olderq24_2  = len(olderq24[olderq24.Q24 == '1-2 years'])-1
olderq24_5  = len(olderq24[olderq24.Q24 == '3-5 years'])-1
olderq24_10 = len(olderq24[olderq24.Q24 == '5-10 years'])-1
olderq24_20 = len(olderq24[olderq24.Q24 == '10-20 years'])-1
olderq24_30 = len(olderq24[olderq24.Q24 == '20-30 years'])
olderq24_40 = len(olderq24[olderq24.Q24 == '30-40 years'])-1
olderq24_50 = len(olderq24[olderq24.Q24 == '40+ years'])-1
olderq24_l  = len(olderq24[olderq24.Q24 == 'I have never written code but I want to learn'])-1
olderq24_n  = len(olderq24[olderq24.Q24 == 'I have never written code and I do not want to learn'])-1

olderq24_years = (olderq24_1,
                  olderq24_2,
                  olderq24_5,
                  olderq24_10,
                  olderq24_20,
                  olderq24_30,
                  olderq24_40,
                  olderq24_50,
                  olderq24_l,
                  olderq24_n)

#Get the percentage
olderq24_years = np.asarray(olderq24_years)
olderq24_yearstotal = olderq24_years.sum()
olderq24_porc = (olderq24_years *100 )/olderq24_yearstotal 


# In[ ]:


#Graph
##Set width of bar
barWidth = 0.20
 
##Sset height of bar
bars1 = jovq24_porc
bars2 = adultq24_porc
bars3 = madultq24_porc
bars4 = olderq24_porc
 
##Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

##Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Youngers')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Adults')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Medium Adults')
plt.bar(r4, bars4, color='#ADFF2F', width=barWidth, edgecolor='white', label='Olders')

#Add xticks on the middle of the group bars
#plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], 
           ['<1 year', '1-2 years', '3-5 years', '5-10 years','10-20 years', '20-30 years','30-40 years', '40+ years','I want to learn', 'I don\'t want to learn'],
          rotation=90)
#plt.setp(xtickNames, rotation=45, fontsize=8)

# Create legend & Show graphic
plt.xlabel('Years that started to learn code')
plt.ylabel('Percentage of groups')
plt.legend()
plt.show()


# We can do some conclusions for each age group:
# * For youngers:
#     Most of young people (18-29 years) started to learn programming recently 1-2 years ago, and remember than we already know that they learned at University.
#     
# * For adults:
#     Most of adult people (30-44 years) have been programming since 3-10 years ago, and other big percentage since 1-2 years ago, so, we can say they also started to programming recently, almost a quarter of their life.
# 
# * For medium adults:
#    There is a similiar percentage of medium adults (45-59) who learned 10-20 years ago and 1-5 years ago, so, we can think than maybe they had to learned for requirement at work, remember than most of them learned by self-taught.
#    
# * For olders:
# Most of older people (60+) have been programming since 30 or more years ago, almost a half of their life. It's easy to think why is the only age group that have a percentage in each range of years, this is the group that learned the most by self-taught, so they have been learning but in different stages of ther lifes. It is also the group with the bigger percentage that do not want to learn.
# 
# There is a similar percentage of each group that want to learn programming.

# We know how and when Kaggler's started to programming, it's time for an important question: **Learning to code at school correlates with the income at laboral life?**
# 
# And there is an easy way to analyze that question.
# 
# Question 9: **What is your current yearly compensation(approximate ** \$ USD)?**
# 
# The options are:
# * 0  - 10,000
# * 10 - 20,000
# * 20 - 30,000
# * 30 - 40,000
# * 40 - 50,000
# * 50 - 60,000
# * 60 - 70,000
# * 70  - 80,000
# * 80  - 90,000
# * 90  - 100,000
# * 100 - 125,000
# * 125  - 150,000
# * 150  - 200,000
# * 200 - 250,000
# * 250 - 300,000
# * 300 - 400,000
# * 400 - 500,000
# * 500,000+
# * I do not wish to disclose my approximate yearly compensation
# 
# Let's see the answers for question 9:
# 

# In[ ]:


dfq9 = multiple.filter(regex="(Q{t})".format(t = 9))[1:]
dfq9.head(6)


# In[ ]:


##We use the same method than the previous df

#Youngers
jovq9    = multjov.filter(regex="(Q{t}$|Q{t}_)".format(t = 9))[1:]
jovq9    = jovq9.dropna(how='any')

jovq9_10    = len(jovq9[jovq9.Q9 == '0-10,000'])-1
jovq9_20    = len(jovq9[jovq9.Q9 == '10-20,000'])-1
jovq9_30    = len(jovq9[jovq9.Q9 == '20-30,000'])-1
jovq9_40    = len(jovq9[jovq9.Q9 == '30-40,000'])-1
jovq9_50    = len(jovq9[jovq9.Q9 == '40-50,000'])-1
jovq9_60    = len(jovq9[jovq9.Q9 == '50-60,000'])
jovq9_70    = len(jovq9[jovq9.Q9 == '60-70,000'])-1
jovq9_80    = len(jovq9[jovq9.Q9 == '70-80,000'])-1
jovq9_90    = len(jovq9[jovq9.Q9 == '80-90,000'])-1
jovq9_100   = len(jovq9[jovq9.Q9 == '90-100,000'])-1
jovq9_125   = len(jovq9[jovq9.Q9 == '100-125,000'])-1
jovq9_150   = len(jovq9[jovq9.Q9 == '125-150,000'])-1
jovq9_200   = len(jovq9[jovq9.Q9 == '150-200,000'])-1
jovq9_250   = len(jovq9[jovq9.Q9 == '200-250,000'])-1
jovq9_300   = len(jovq9[jovq9.Q9 == '250-300,000'])-1
jovq9_400   = len(jovq9[jovq9.Q9 == '300-400,000'])
jovq9_500   = len(jovq9[jovq9.Q9 == '400-500,000'])-1
jovq9_600   = len(jovq9[jovq9.Q9 == '500+'])
jovq9_n     = len(jovq9[jovq9.Q9 == 'I do not wish to disclose my approximate yearly compensation'])-1

jovq9_compens = (jovq9_10,jovq9_20,jovq9_30,jovq9_40,jovq9_50,jovq9_60,jovq9_70,jovq9_80,jovq9_90,
              jovq9_100,jovq9_125,jovq9_150,jovq9_200,jovq9_250,jovq9_300,jovq9_400,jovq9_500,
              jovq9_600,jovq9_n)
jovq9_compens = np.asarray(jovq9_compens)
jovq9_compenstotal = jovq9_compens.sum()
jovq9_compensporc = (jovq9_compens*100)/jovq9_compenstotal 

##Adults

adultq9    = multadult.filter(regex="(Q{t}$|Q{t}_)".format(t = 9))[1:]
adultq9    = adultq9.dropna(how='any')

adultq9_10    = len(adultq9[adultq9.Q9 == '0-10,000'])-1
adultq9_20    = len(adultq9[adultq9.Q9 == '10-20,000'])-1
adultq9_30    = len(adultq9[adultq9.Q9 == '20-30,000'])-1
adultq9_40    = len(adultq9[adultq9.Q9 == '30-40,000'])-1
adultq9_50    = len(adultq9[adultq9.Q9 == '40-50,000'])-1
adultq9_60    = len(adultq9[adultq9.Q9 == '50-60,000'])
adultq9_70    = len(adultq9[adultq9.Q9 == '60-70,000'])-1
adultq9_80    = len(adultq9[adultq9.Q9 == '70-80,000'])-1
adultq9_90    = len(adultq9[adultq9.Q9 == '80-90,000'])-1
adultq9_100   = len(adultq9[adultq9.Q9 == '90-100,000'])-1
adultq9_125   = len(adultq9[adultq9.Q9 == '100-125,000'])-1
adultq9_150   = len(adultq9[adultq9.Q9 == '125-150,000'])-1
adultq9_200   = len(adultq9[adultq9.Q9 == '150-200,000'])-1
adultq9_250   = len(adultq9[adultq9.Q9 == '200-250,000'])-1
adultq9_300   = len(adultq9[adultq9.Q9 == '250-300,000'])-1
adultq9_400   = len(adultq9[adultq9.Q9 == '300-400,000'])
adultq9_500   = len(adultq9[adultq9.Q9 == '400-500,000'])-1
adultq9_600   = len(adultq9[adultq9.Q9 == '500+'])
adultq9_n     = len(adultq9[adultq9.Q9 == 'I do not wish to disclose my approximate yearly compensation'])-1

adultq9_compens = (adultq9_10,
                   adultq9_20,
                   adultq9_30,
                   adultq9_40,
                   adultq9_50,
                   adultq9_60,
                   adultq9_70,
                   adultq9_80,
                   adultq9_90,
                   adultq9_100,
                   adultq9_125, 
                   adultq9_150,
                   adultq9_200,
                   adultq9_250,
                   adultq9_300,
                   adultq9_400,
                   adultq9_500,
                   adultq9_600,
                   adultq9_n)
adultq9_compens = np.asarray(adultq9_compens)
adultq9_compenstotal = adultq9_compens.sum()
adultq9_compensporc = (adultq9_compens*100)/adultq9_compenstotal 

##Medium adults
madultq9    = multmoread.filter(regex="(Q{t}$|Q{t}_)".format(t = 9))[1:]
madultq9    = madultq9.dropna(how='any')

madultq9_10    = len(madultq9[madultq9.Q9 == '0-10,000'])-1
madultq9_20    = len(madultq9[madultq9.Q9 == '10-20,000'])-1
madultq9_30    = len(madultq9[madultq9.Q9 == '20-30,000'])-1
madultq9_40    = len(madultq9[madultq9.Q9 == '30-40,000'])-1
madultq9_50    = len(madultq9[madultq9.Q9 == '40-50,000'])-1
madultq9_60    = len(madultq9[madultq9.Q9 == '50-60,000'])
madultq9_70    = len(madultq9[madultq9.Q9 == '60-70,000'])-1
madultq9_80    = len(madultq9[madultq9.Q9 == '70-80,000'])-1
madultq9_90    = len(madultq9[madultq9.Q9 == '80-90,000'])-1
madultq9_100   = len(madultq9[madultq9.Q9 == '90-100,000'])-1
madultq9_125   = len(madultq9[madultq9.Q9 == '100-125,000'])-1
madultq9_150   = len(madultq9[madultq9.Q9 == '125-150,000'])-1
madultq9_200   = len(madultq9[madultq9.Q9 == '150-200,000'])-1
madultq9_250   = len(madultq9[madultq9.Q9 == '200-250,000'])-1
madultq9_300   = len(madultq9[madultq9.Q9 == '250-300,000'])-1
madultq9_400   = len(madultq9[madultq9.Q9 == '300-400,000'])
madultq9_500   = len(madultq9[madultq9.Q9 == '400-500,000'])-1
madultq9_600   = len(madultq9[madultq9.Q9 == '500+'])
madultq9_n     = len(madultq9[madultq9.Q9 == 'I do not wish to disclose my approximate yearly compensation'])-1

madultq9_compens = (madultq9_10,
                   madultq9_20,
                   madultq9_30,
                   madultq9_40,
                   madultq9_50,
                   madultq9_60,
                   madultq9_70,
                   madultq9_80,
                   madultq9_90,
                   madultq9_100,
                   madultq9_125, 
                   madultq9_150,
                   madultq9_200,
                   madultq9_250,
                   madultq9_300,
                   madultq9_400,
                   madultq9_500,
                   madultq9_600,
                   madultq9_n)
madultq9_compens = np.asarray(madultq9_compens)
madultq9_compenstotal = madultq9_compens.sum()
madultq9_compensporc = (madultq9_compens*100)/madultq9_compenstotal

##Olders
olderq9    = multolder.filter(regex="(Q{t}$|Q{t}_)".format(t = 9))[1:]
olderq9    = olderq9.dropna(how='any')

olderq9_10    = len(olderq9[olderq9.Q9 == '0-10,000'])-1
olderq9_20    = len(olderq9[olderq9.Q9 == '10-20,000'])-1
olderq9_30    = len(olderq9[olderq9.Q9 == '20-30,000'])-1
olderq9_40    = len(olderq9[olderq9.Q9 == '30-40,000'])-1
olderq9_50    = len(olderq9[olderq9.Q9 == '40-50,000'])-1
olderq9_60    = len(olderq9[olderq9.Q9 == '50-60,000'])
olderq9_70    = len(olderq9[olderq9.Q9 == '60-70,000'])-1
olderq9_80    = len(olderq9[olderq9.Q9 == '70-80,000'])-1
olderq9_90    = len(olderq9[olderq9.Q9 == '80-90,000'])-1
olderq9_100   = len(olderq9[olderq9.Q9 == '90-100,000'])-1
olderq9_125   = len(olderq9[olderq9.Q9 == '100-125,000'])-1
olderq9_150   = len(olderq9[olderq9.Q9 == '125-150,000'])-1
olderq9_200   = len(olderq9[olderq9.Q9 == '150-200,000'])-1
olderq9_250   = len(olderq9[olderq9.Q9 == '200-250,000'])-1
olderq9_300   = len(olderq9[olderq9.Q9 == '250-300,000'])-1
olderq9_400   = len(olderq9[olderq9.Q9 == '300-400,000'])
olderq9_500   = len(olderq9[olderq9.Q9 == '400-500,000'])-1
olderq9_600   = len(olderq9[olderq9.Q9 == '500+'])
olderq9_n     = len(olderq9[olderq9.Q9 == 'I do not wish to disclose my approximate yearly compensation'])-1

olderq9_compens = (olderq9_10,
                   olderq9_20,
                   olderq9_30,
                   olderq9_40,
                   olderq9_50,
                   olderq9_60,
                   olderq9_70,
                   olderq9_80,
                   olderq9_90,
                   olderq9_100,
                   olderq9_125, 
                   olderq9_150,
                   olderq9_200,
                   olderq9_250,
                   olderq9_300,
                   olderq9_400,
                   olderq9_500,
                   olderq9_600,
                   olderq9_n)
olderq9_compens = np.asarray(olderq9_compens)
olderq9_compenstotal = olderq9_compens.sum()
olderq9_compensporc = (olderq9_compens*100)/olderq9_compenstotal


# In[ ]:


#Graph
##Set width of bar
barWidth = 0.20
 
##Sset height of bar
bars1 = jovq9_compensporc
bars2 = adultq9_compensporc
bars3 = madultq9_compensporc
bars4 = olderq9_compensporc
 
##Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

##Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Youngers')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Adults')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Medium Adults')
plt.bar(r4, bars4, color='#ADFF2F', width=barWidth, edgecolor='white', label='Olders')

#Add xticks on the middle of the group bars
#plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], 
           ['0-10,000','10-20,000','20-30,000','30-40,000','40-50,000','50-60,000','60-70,000',
            '70-80,000','80-90,000','90-100,000','100-125,000','125-150,000','150-200,000',
            '200-250,000','250-300,000','300-400,000','400-500,000','500,000+',
            'I do not wish to disclose'],
          rotation=90)
#plt.setp(xtickNames, rotation=45, fontsize=8)

# Create legend & Show graphic
plt.xlabel('Percentage by group')
plt.ylabel('Compensations')
plt.legend()
plt.show()


# Ok, there are good news, bad news and worst news.
# The good news are that the big percentage of age range with a great compensation (90-100,000)  are the olders, and most of them learn by self-taught, so there is no necessary to learn code at University to get high compensations.
# 
# The bad news are than almost a quarter of the Kaggler's didn't want to give information :(, sorprendently, adults and medium adults are who give more information. The problem with adults and medium adults is when they will start to get a better compensation?
# 
# The worst news are than young people receive the worst compensation (0-10,000)
# 
# So, young people learned to programming recently, are the age range with the worst compensation, and older people are who have the "best" compensation, they started to learning almost 30 years ago.
# 
# What's the perception of each group about traditional education and online school?
# 
# Let's see question 39:
# **How do you perceive the quality of online learning platforms and in-person bootcamps as compares to the quality of the education provided by traditional brick and mortar institutions?**
# 
# The options are:
# * Slighty better
# * Much better
# * Neither better nor worse
# * Slightly worse
# * Much worse
# * No opinion, I do not know
# 
# Take a look of answers for question 39: Take a look of answers for question 39: 

# In[ ]:


dfq39 = multiple.filter(regex="Q{t}_Part_2".format(t = 39))[1:]
dfq39.dropna(how='any').head(6)


# In[ ]:


jovq39 = multjov.filter(regex="(Q39_Part_1)")[1:]
jovq39 = jovq39.dropna(how='any')

jovq39_1    = len(jovq39[jovq39.Q39_Part_1 == 'Slightly better'])-1
jovq39_2    = len(jovq39[jovq39.Q39_Part_1 == 'Much better'])-1
jovq39_3    = len(jovq39[jovq39.Q39_Part_1 == 'Neither better nor worse'])-1
jovq39_4    = len(jovq39[jovq39.Q39_Part_1 == 'Slightly better'])-1
jovq39_5    = len(jovq39[jovq39.Q39_Part_1 == 'Much worse'])-1
jovq39_6    = len(jovq39[jovq39.Q39_Part_1 == 'No opinion; I do not know'])

jovq39_quality = (jovq39_1,
                  jovq39_2,
                  jovq39_3,
                  jovq39_4,
                  jovq39_5,
                  jovq39_6)
jovq39_quality = np.asarray(jovq39_quality)
jovq39_qualitytotal = jovq39_quality.sum()
jovq39_qualityporc = (jovq39_quality*100)/jovq39_qualitytotal 

##

adultq39 = multadult.filter(regex="(Q39_Part_1)")[1:]
adultq39 = adultq39.dropna(how='any')

adultq39_1    = len(adultq39[adultq39.Q39_Part_1 == 'Slightly better'])-1
adultq39_2    = len(adultq39[adultq39.Q39_Part_1 == 'Much better'])-1
adultq39_3    = len(adultq39[adultq39.Q39_Part_1 == 'Neither better nor worse'])-1
adultq39_4    = len(adultq39[adultq39.Q39_Part_1 == 'Slightly better'])-1
adultq39_5    = len(adultq39[adultq39.Q39_Part_1 == 'Much worse'])-1
adultq39_6    = len(adultq39[adultq39.Q39_Part_1 == 'No opinion; I do not know'])

adultq39_quality = (adultq39_1,
                    adultq39_2,
                    adultq39_3,
                    adultq39_4,
                    adultq39_5,
                    adultq39_6)
adultq39_quality = np.asarray(adultq39_quality)
adultq39_qualitytotal = adultq39_quality.sum()
adultq39_qualityporc = (adultq39_quality*100)/adultq39_qualitytotal 

##

madultq39 = multmoread.filter(regex="(Q39_Part_1)")[1:]
madultq39 = madultq39.dropna(how='any')

madultq39_1    = len(madultq39[madultq39.Q39_Part_1 == 'Slightly better'])-1
madultq39_2    = len(madultq39[madultq39.Q39_Part_1 == 'Much better'])-1
madultq39_3    = len(madultq39[madultq39.Q39_Part_1 == 'Neither better nor worse'])-1
madultq39_4    = len(madultq39[madultq39.Q39_Part_1 == 'Slightly better'])-1
madultq39_5    = len(madultq39[madultq39.Q39_Part_1 == 'Much worse'])-1
madultq39_6    = len(madultq39[madultq39.Q39_Part_1 == 'No opinion; I do not know'])

madultq39_quality = (madultq39_1,
                     madultq39_2,
                     madultq39_3,
                     madultq39_4,
                     madultq39_5,
                     madultq39_6)
madultq39_quality = np.asarray(madultq39_quality)
madultq39_qualitytotal = madultq39_quality.sum()
madultq39_qualityporc = (madultq39_quality*100)/madultq39_qualitytotal 

##

olderq39 = multolder.filter(regex="(Q39_Part_1)")[1:]
olderq39 = olderq39.dropna(how='any')

olderq39_1    = len(olderq39[olderq39.Q39_Part_1 == 'Slightly better'])-1
olderq39_2    = len(olderq39[olderq39.Q39_Part_1 == 'Much better'])-1
olderq39_3    = len(olderq39[olderq39.Q39_Part_1 == 'Neither better nor worse'])-1
olderq39_4    = len(olderq39[olderq39.Q39_Part_1 == 'Slightly better'])-1
olderq39_5    = len(olderq39[olderq39.Q39_Part_1 == 'Much worse'])-1
olderq39_6    = len(olderq39[olderq39.Q39_Part_1 == 'No opinion; I do not know'])

olderq39_quality =  (olderq39_1,
                     olderq39_2,
                     olderq39_3,
                     olderq39_4,
                     olderq39_5,
                     olderq39_6)
olderq39_quality = np.asarray(olderq39_quality)
olderq39_qualitytotal = olderq39_quality.sum()
olderq39_qualityporc = (olderq39_quality*100)/olderq39_qualitytotal 


# In[ ]:


#Graph
##Set width of bar
barWidth = 0.20
 
##Set height of bar
bars1 = jovq39_qualityporc
bars2 = adultq39_qualityporc
bars3 = madultq39_qualityporc
bars4 = olderq39_qualityporc
 
##Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

##Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Youngers')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Adults')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Medium Adults')
plt.bar(r4, bars4, color='#ADFF2F', width=barWidth, edgecolor='white', label='Olders')

#Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(bars1))], 
           ['Slightly better','Much better','Neither better nor worse','Slightly worse','Much worse','No opinion; I do not know'],
          rotation=90)
#plt.setp(xtickNames, rotation=45, fontsize=8)

# Create legend & Show graphic
plt.xlabel('Opinions')
plt.ylabel('Percentage by group')
plt.legend()
plt.show()


# There is an tendence to decrease the percentage between age ranges in slightly better opinion, and is the similar tendence in slightly worse opinion, so, there are divided opinions. If we consider the sum of the 'better' opinions (Slightly better and much better), and the sum of the 'worse' opinions (Slightly worse  and much worse), there are better opinions about online platforms than traditional education.
# 
# Most of young people prefer online education, so we must question why?, What are the differences between online education and traditional education?, and why is better one than other?
# 
# There are some question that we must ask to our next Kagglers.  ;)
#  
# 
# 
