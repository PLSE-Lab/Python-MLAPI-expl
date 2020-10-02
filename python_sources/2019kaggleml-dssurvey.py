#!/usr/bin/env python
# coding: utf-8

# 
# # Can a climate modeller (myself) become a Data Scientist in Germany? 
# 
# <img src='https://i.imgur.com/bFrACnF.jpg' width=700 >
# 
# Life was not easy for me but I do not have any concrete problem to complain about. The only challenge for me is that I work at university as a postdoc in climate science and although the salary and the projects are very conviencing, I do not have an unlimited working contract, and I have to search for jobs every 2 to 3 years, which makes my chances of finding new positions less as I age. I will try to find out how the chances are for me to move from academia to industry and find a job as data scientist or machine learning engineer. First I have to list which information do I have and compare my own skills with the survey of  <a href="https://www.kaggle.com/c/kaggle-survey-2019/overview"> Kaggle ML & DS Survey</a> in order to calculate my chances. As a usuall, climate scientist one has a good understanding of numerical model structures, high performance computing, statistical methods and software versioning. The languages used are mostly Fortran and Bash for climate models. For ploting, modellers use python, R, NCL or MatLab. I have also to mention that these models create massive big data problems for the mdoellers. All I heared about the career shift were based on some stories told of some individuals who made this jump. They most tell me, you have to apply, apply and apply. Ok, but wait! There is now this data from Kaggle. It might help me in more details than just apply commands!
# 
# ## What are my preferences?
# First of all I want to stay in Germany (Berlin). Second, it should be at least 50kUSD per year. With this told lets dive into the data.
# 
# 
# 
# 
# 

# In[ ]:


# import the libraries here: 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from collections import Counter
import seaborn as sns


# Well for me it was important to stay in Germany, so I have to make a subset of the data based on germany: 

# In[ ]:


# Load the data files in data-frames: 

files = {}
for filename in os.listdir('../input/kaggle-survey-2019/'):
    if filename.endswith('.csv'):
        files[str(filename[:-4])]    = pd.read_csv('../input/kaggle-survey-2019/'+filename,low_memory=False)
print("all the files have been read: ")        
for keys, value in files.items():
    print(keys)


# In[ ]:


countries = files["multiple_choice_responses"]
germany = countries['Q3'] == "Germany" # you can put other country here!
mcr_germany = files["multiple_choice_responses"][germany]
print('Shape of the subset is:', str(mcr_germany.shape))
print('Shape of the whole data is : ', str(countries.shape))
print('Number of Countries involved in the data is : ', len(set(countries['Q3'])))


# The other important factor for me was the salary, so I have to make a subset of the data based on salary as well: 

# In[ ]:


#set(mcr_germany['Q10'].dropna())
target_salary = ['100,000-124,999','125,000-149,999','150,000-199,999',
                '200,000-249,999','250,000-299,999','300,000-500,000',
                '60,000-69,999','70,000-79,999','80,000-89,999',
                 '90,000-99,999','> $500,000','50,000-59,999']
mcr_final= mcr_germany[mcr_germany['Q10'].isin(target_salary)]
print('shape of the final subset is:', str(mcr_final.shape))


# Now I have 266 observations in Germany with a salary expectation that I have. Lets see how the data looks like for this salary.

# In[ ]:


#Plot the salary category counts:
letter_counts = Counter(mcr_final["Q10"])
df = pd.DataFrame.from_dict(letter_counts, orient='index')
yy = df[0].values[:]
xx = df.index.values[:]
fig=plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='w')
plt.bar(xx,yy, color='grey')
plt.xticks(xx, rotation=80,fontsize=12)
for i, v in enumerate(xx):
    plt.text(i-.25, 
              yy[i]+1, 
              yy[i], 
              fontsize=18, 
              color='k')
plt.ylim([0,65])   
plt.xlabel('Salary',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.title('Salary Category Counts', fontsize=25)
plt.show()


# Well most of data scientist eran something between 50k to 125k USD in year in Germany. That looks great! I will at least try to tell 60k as my salary suggestion by my next job interview. But ofcourse I have to dive into the data to see if I have the same skills as these folk!
# 
# It can be seen from the diagram that there are 21 people in the survey from germany that earn more than 125kUSD in year. That might be my future salary if I was able to get a job as data scientist. For a bit of motivation for myself to continue writing this notebook I will first analyse this very little subset of data! I want curiously know why these guys earn that amount of money?  
# 
# One think that bothers me is the few number of observation. Do I really want to count on the data collected from 266 people? It means roughly, 266/19718 x 100 = 1.35 % of the whole data. And 60 countries were involved in the servey. I think yes, on this subset I will rely, but later I will add some countries similar to Germany from the economical point of view who are within the Europe, like France. 

# In[ ]:


mcr_high_salary = mcr_germany[mcr_germany['Q10'].isin(['125,000-149,999','150,000-199,999',
                                                       '200,000-249,999','250,000-299,999',
                                                       '300,000-500,000','> $500,000'])]
# these rae high salary groups for my decision making


# In[ ]:


#function for plotting bar plots of counts of all available data in a loop :

def plot_bars_for_ds (target_group, lon=15, lat=5, color='red'):
    '''
    lon          = size of x axis in the plot
    lat          = size of y axis in the plot
    target_group = subset of the DataFrame
    color        = color of the bars in barplots
    '''
    
   
    
    
    
    for question in sorted(set(target_group.columns)): # loop over all the columns!
        if question in set(target_group.columns):      # Ceck if I plotted it before, I am droping the colomns which are plotted!

            if target_group[question].isnull().all():  # If all observations in a column are nan, then continue the loop!

                continue

            if len(question) > 5:

                if question[0] == "T":                 # Ignore time from start to finish!
                    continue
                if question[-10:] == "OTHER_TEXT" :    # OTHER_TEXT columns 


                    mcr_hs_part = []
                    if question[2] == "_":
                        mcr_hs_part = target_group[question[0:2]+"_OTHER_TEXT"] #1 to 9



                    else:
                        mcr_hs_part = target_group[question[0:3]+"_OTHER_TEXT"] # 10 to end




                    xx = mcr_hs_part.index.values[:]

                    # now plot it
                    fig=plt.figure(figsize=(lon, lat), dpi= 80, facecolor='w', edgecolor='w')
                    chart = sns.countplot(x=xx, data=mcr_hs_part,color=color )

                    if len(question) > 5:

                        if question[2] == "_":
                            plt.title(files["questions_only"][question[0:2]][0],fontsize=18)
                        else:
                            plt.title(files["questions_only"][question[0:3]][0],fontsize=18)
                    else:
                        plt.title(files["questions_only"][question][0],fontsize=18)
                    plt.ylabel('Number of People', fontsize = 16.0) # Y label
                    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
                    plt.show()

                else:

                    if question[0:3] == "Q14" : #Q14 has 5 parts only




                        mcr_hs_part = target_group["Q14_Part_1_TEXT"]
                        mcr_hs_part.columns = question[0:3]
                        mcr_hs_part.name = "Q14_Part_1_TEXT"
                        for prt in range(2,6):
                            if "Q14_Part_"+str(prt)+"_TEXT" in set(target_group.columns):


                                s = target_group["Q14_Part_"+str(prt)+"_TEXT"]
                                s.name = "Q14_Part_"+str(prt)+"_TEXT"

                                mcr_hs_part = mcr_hs_part.append( s)
                                target_group = target_group.drop(["Q14_Part_"+str(prt)+"_TEXT"], axis=1)
                        target_group = target_group.drop(["Q14_Part_1_TEXT"], axis=1)
                        xx = mcr_hs_part.index.values[:]


                        fig=plt.figure(figsize=(lon, lat), dpi= 80, facecolor='w', edgecolor='w')
                        chart = sns.countplot(x=xx, data=mcr_hs_part, color=color)

                        if len(question) > 5:

                            plt.title(files["questions_only"][question[0:3]][0],fontsize=18)
                        else:
                            plt.title(files["questions_only"][question][0],fontsize=18)
                        plt.ylabel('Number of People', fontsize = 16.0) # Y label

                        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
                        plt.show()
                        continue


                    mcr_hs_part = []
                    if question[2] == "_":
                        mcr_hs_part = target_group[question[0:2]+"_Part_1"]
                        mcr_hs_part.name = question[0:2]+"_Part_1"
                        mcr_hs_part.columns = question[0:2]
                        for prt in range(2,13):
                            if question[0:2]+"_Part_"+str(prt) in set(target_group.columns):


                                s = target_group[question[0:2]+"_Part_"+str(prt)]
                                s.name = question[0:2]+"_Part_"+str(prt)
                                mcr_hs_part = mcr_hs_part.append( s)
                                target_group = target_group.drop([question[0:2]+"_Part_"+str(prt)], axis=1)
                        target_group = target_group.drop([question[0:2]+"_Part_1"], axis=1)


                    else:               
                        mcr_hs_part = target_group[question[0:3]+"_Part_1"]
                        mcr_hs_part.name = question[0:3]+"_Part_1"
                        mcr_hs_part.columns = question[0:3]


                        for prt in range(2,13):
                            if question[0:3]+"_Part_"+str(prt) in set(target_group.columns):


                                s = target_group[question[0:3]+"_Part_"+str(prt)]
                                s.name = question[0:3]+"_Part_"+str(prt)
                                s.columns = question[0:3]
                                mcr_hs_part = mcr_hs_part.append(s)
                                target_group = target_group.drop([question[0:3]+"_Part_"+str(prt)], axis=1)
                        target_group = target_group.drop([question[0:3]+"_Part_1"], axis=1)

                    xx = mcr_hs_part.index.values[:]


                    fig=plt.figure(figsize=(lon, lat), dpi= 80, facecolor='w', edgecolor='w')
                    chart = sns.countplot(x=xx, data=mcr_hs_part , color = color)

                    if len(question) > 5: # there columns which are long including the ones with Part !

                        if question[2] == "_":
                            plt.title(files["questions_only"][question[0:2]][0],fontsize=18)
                        else:
                            plt.title(files["questions_only"][question[0:3]][0],fontsize=18)
                    else:
                        plt.title(files["questions_only"][question][0],fontsize=18)
                    plt.ylabel('Number of People', fontsize = 16.0) # Y label

                    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
                    plt.show()
            else: 

                fig=plt.figure(figsize=(lon, lat), dpi= 80, facecolor='w', edgecolor='w')
                xx = target_group[question].index.values[:]


                chart = sns.countplot(x=xx, data=target_group[question] , color=color)
                if len(question) > 5:

                    plt.title(files["questions_only"][question[0:3]][0],fontsize=18)
                else:
                    plt.title(files["questions_only"][question][0],fontsize=18)
                plt.ylabel('Number of People', fontsize = 16.0) # Y label
                chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
                plt.show()
                
            


# 
# 
# 
# ## First I will plot the <span style="color:red"> high Salary group </span> : 
# 
# 
# 
# 

# In[ ]:


# subset the data for high salaries in Germany: 
mcr_high_salary = mcr_germany[mcr_germany['Q10'].isin(['125,000-149,999','150,000-199,999',
                                                       '200,000-249,999','250,000-299,999',
                                                       '300,000-500,000','> $500,000'])]

plot_bars_for_ds (target_group=mcr_high_salary,lon=15, lat=6, color='salmon')


# 
# 
# ## Now I repeat the plots for the lower salaries with a <span style="color:blue"> lower limit of 50kUSD</span>: 
# 
# I plot the bars in light green to distinguish them to the high salary plots. 
# 
# 

# In[ ]:


# subset the data for normal salaries in Germany: 
mcr_lower_salary = mcr_germany[mcr_germany['Q10'].isin(['60,000-69,999','70,000-79,999','80,000-89,999',
                                                        '90,000-99,999','50,000-59,999'])]

plot_bars_for_ds (target_group=mcr_lower_salary,lon=15, lat=6, color='skyblue')


# # Results
# ## Who are these guys on top of the Data-science-chain in Germany?
# 
# Data scientists with higher salaries (>125,000 USD) are the one with mostly doctoral title and work for big companies with more than 10000 employees. 40% of them are managers. They use the Amazone Web Services, especially EC2 on a regular basis. They keep themself up to date via Blogs and Journal Papers. They are 35 to 44+ years old and are from the beginning of the Data Science application in industry playing a role in the scene. Which means they have more experience than the junior or semi-senior data scientist. They peobably are the team leaders who juniors work for. 
# 
# 
# ## What should I learn or focus on for my next job as data scientist in Germany? 
# 
# Going through all barplots, what I observed is listed in a priority ranking for myself :
# 
# * Start Leaning SQL (PostgresSQL) in more details or put it on my CV: this is one of the skills that is a must to have!
# 
# * Learn anything about Linear regression and decision trees with some knowledge of Neural Nets.
#  
# * I have to master tensorflow and scikit for the next interview. 
# 
# * I have at least to burn some money (100 USD) on AWS EC2 to get used to it.
# 
# *  Use more Seaborn and Matplotlib for my visualization.
# 
# * I can apply for any company small to big ones. People like me can start from everywhere. 
# 
# * At least know word embeding if I apply for jobs in NLP. 
# 
# 
# 
# ## Do I have a chance? 
# 
# Definitely, the data says my chances are still high. 
# 
# ## Aknowledgments
# 
# Special thanks to Kaggle for providing this valuable data set. 

# In[ ]:






# In[ ]:





# In[ ]:




