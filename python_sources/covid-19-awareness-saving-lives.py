#!/usr/bin/env python
# coding: utf-8

# 

# ![](https://media.giphy.com/media/IbmS6XKR5fTVchlxcN/giphy.gif)

# **CORONAVIRUS HAS CLAIMED PRECIOUS LIVES THOUGHOUT THE WORLD. PEOPLE ACROSS THE GLOBE ARE DOING COUNTLESS HOURS OF RESEARCH TO FIND A SOLUTION. MY WORK MEASURES THE AMOUNT OF WORKS DONE BY TITLES PUBLISHED IN JOURNALS ACROSS THE GLOBE AND COMPARES EACH JOURNAL WITH THE AMOUNTS OF TITLES PASSAGES THEY HAVE PRODUCED ON THE CORONA VIRUS WHICH MIGHT BE CLEAR MEASURE OF THERE PARTICIPATION IN CREATING AWARENESS AND DISPLAY OF THE SPIRIT OF JOUNALISM. I AM USING WHO DATA AND MY OWN CREATED DATASET TO DERIVE THE RESULTS **

# <img src="https://www.elsevier.com/__data/assets/image/0012/974739/coronavirus-image-iStock-628925532-full-width-wide.jpg">

# image credits = elsevierdotcom

# ![](https://mail.google.com/mail/u/0?ui=2&ik=17b7366792&attid=0.1&permmsgid=msg-a:r7913645352209974806&th=1725c00afb07d2c0&view=fimg&sz=s0-l75-ft&attbid=ANGjdJ-TKYiM5aCjBsbcntfqIzHDmQd5dgmIF1ysUBmVw71AW20NXX-NvgxvMIfpvn1qj_HpyaTQaGERB6xG75gymW1tMoRTvdZ8OeJWX8E5yWCsKyJe8o-9abrODEo&disp=emb&realattid=ii_kaqynnzc0)

# In[ ]:




import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as aware


# In[ ]:


df = pd.read_csv("../input/who-data-base-for-public-use/2020-Full database.csv")['Journal']


# Lets consider those Journals Who have published atleast 50 or more titles on Covid-19 till now according to WHO data. I am taking the first 37 values as beyond that no Journal has produced atleast 50 or more titles. We may also have to check data for the same jounal with different names repeated more than once.

# In[ ]:


dg = df.value_counts()

journal = dg.head(37)


# In[ ]:


journal


# In[ ]:


word = ','.join(map(str, journal)) #I am joining the number to form a series to put in a dataframe


# In[ ]:


word


# Below is our Dataframe named Awareness_creators and has names of journals who have produced atleast 50 or more titles on covid-19

# In[ ]:


Awareness_creators={'Journals':['BMJ',                                               
'Nature',                                            
'The Lancet',                                        
'Science',                                           
'Journal of medical virology',                       
'Journal of Infection',                              
'The Lancet Infectious Diseases',                    
'JAMA',                                              
'International Journal of Infectious Diseases',      
'New Scientist',                                     
'New England Journal of Medicine',                   
                       
'Clinical Infectious Diseases',                       
'Journal of the American Academy of Dermatology',     
'Infection Control & Hospital Epidemiology',          
'Travel Medicine and Infectious Disease',             
                                
'Revue medicale suisse',                              
'C&EN Global Enterprise',                             
'Science of The Total Environment',                   
'The Lancet Respiratory Medicine',                    
'Psychiatry Research',                                
'Medical Hypotheses',                                
               
'Asian Journal of Psychiatry',                        
'Head & neck',                                        
'Brain, Behavior, and Immunity',                      
'Journal of the American Geriatrics Society',         
'Journal of Clinical Virology',                       
'Gastroenterology',                                   
'European Heart Journal',                             
'Eurosurveillance',                                  
'Radiology',                                          
'Anaesthesia',                                        
'Journal of Hospital Infection',                      
'Critical Care',                                      
'Anesth Analg'],'Number_of_titles_produced':[485,267,226,199,290,146,128,128,104,101,167,97,95,92,83,81,79,72,71,71,69,63,61,61,60,58,58,57,57,57,54,53,53,50]}

Awareness_creators=pd.DataFrame(data=Awareness_creators,index=range(34))
Awareness_creators.style.background_gradient(cmap='plasma')


# lets visualize this data frame with a Bar Chart!!

# In[ ]:


fig = aware.bar(Awareness_creators[['Journals', 'Number_of_titles_produced']].sort_values('Number_of_titles_produced', ascending=False), 
             y="Number_of_titles_produced", x="Journals", color='Journals',  width=1500, height=700,
             log_y=True, template='ggplot2', title='AWARENESS SAVING LIVES')
fig.show()


# In[ ]:


word = ','.join(map(str, df)) # Now we join the words in the Journal column of the who dataset to make a WordCloud visualize 


# In[ ]:


from wordcloud import WordCloud , STOPWORDS #Stopwords to remove unecessary words


# In[ ]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = "White", max_words = 200, stopwords = stopwords).generate(word)


# In[ ]:


plt.figure(1,figsize=(20, 20))
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.show()


# ![](https://mail.google.com/mail/u/0?ui=2&ik=17b7366792&attid=0.1&permmsgid=msg-a:r-7286449435814262293&th=17254dee9cd7195e&view=fimg&sz=s0-l75-ft&attbid=ANGjdJ8AWBE31FEtADkOWpnrZxwIZ6DaNiDj1jMDxNT4zED3n0TBWGk4qzDOPXUGPR5UD_N3VmhBaObSSaf5ZCacbVZPiZAXlriQcK2xWCbzTgqEnszG6oOILU63f0c&disp=emb&realattid=ii_kaozf4dk0)

# ![](https://mail.google.com/mail/u/0?ui=2&ik=17b7366792&attid=0.1&permmsgid=msg-a:r6387276066474129052&th=17254e2351d5a047&view=fimg&sz=s0-l75-ft&attbid=ANGjdJ9C4bkskQAuwn4IaWTybPCzxC03XCsiEBSEinFy1sfruftTAHvolRBeOhdfKnEnK7deJFiKzJlnZlH9wzzeeYYIkvh9oLdSpeCOFRGCVbrNXas0j17jKGGSORU&disp=emb&realattid=ii_kaozjssn0)

# Below is a Pie chart of the title percentage of each selected Journal which shows there contribution to the whole.

# In[ ]:


fig = aware.pie(Awareness_creators,
             values="Number_of_titles_produced",
             names="Journals",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# **BELOW IS A DONUT GRAPH OF THE SAME**

# In[ ]:


from palettable.colorbrewer.qualitative import Pastel1_7
plt.figure(figsize=(25,25))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.pie(Awareness_creators['Number_of_titles_produced'], labels=Awareness_creators.Journals, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Journals by titles produced')
plt.show()


# **Now we read the Awareness Dataset I made which contains the estimated readership of these Jounals. This is the first dataset that I have created, and I have made it public so you can alo use it for understanding how awareness of this apidemic is saving lives**

# In[ ]:


df1 = pd.read_csv("../input/covid19-awareness-dataset/AWARENESS DATASET.csv") 


# In[ ]:


df1=pd.DataFrame(data=df1,index=range(33))
df1


# **Lets make a DataFrame to experiment with our new Dataset **

# In[ ]:


PEOPLE_AWARE={'Journals':['BMJ',                                               
'Nature',                                            
'The Lancet',                                        
'Science',                                           
'Journal of medical virology',                       
'Journal of Infection',                              
'The Lancet Infectious Diseases',                    
'JAMA',                                              
'International Journal of Infectious Diseases',      
'New Scientist',                                     
'New England Journal of Medicine',                   
                       
'Clinical Infectious Diseases',                       
'Journal of the American Academy of Dermatology',     
'Infection Control & Hospital Epidemiology',          
'Travel Medicine and Infectious Disease',             
                                
'Revue medicale suisse',                              
'C&EN Global Enterprise',                             
'Science of The Total Environment',                   
                    
'Psychiatry Research',                                
'Medical Hypotheses',                                
               
'Asian Journal of Psychiatry',                        
'Head & neck',                                        
'Brain, Behavior, and Immunity',                      
'Journal of the American Geriatrics Society',         
'Journal of Clinical Virology',                       
'Gastroenterology',                                   
'European Heart Journal',                             
'Eurosurveillance',                                  
'Radiology',                                          
'Anaesthesia',                                        
'Journal of Hospital Infection',                      
'Critical Care',                                      
'Anesth Analg'],'Number_of_Readers':[50000000,400000,10159512,570400,290796,176796,1019388,20000000,362148,1000000,600000,179928,1942080,496176,3000000,330252,311172,286674,265268,2906226,2000000,500000,150144,320172,11844,30000,359256,2800000,54000,2400000,272268,300068,117120]}

PEOPLE_AWARE=pd.DataFrame(data=PEOPLE_AWARE,index=range(33))
PEOPLE_AWARE.style.background_gradient(cmap='plasma')


# **LETS MAKE A PIE CHART TO SEE THE PERCENTAGE OF READERS TOTAL POPULATION COVERED BY EACH JOURNAL**

# In[ ]:


fig = aware.pie(PEOPLE_AWARE, width=1200, height=1200,
             values="Number_of_Readers",
             names="Journals",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# **LETS MAKE A BAR CHART TO VISUALIZE THE READERS AND SEE WHO COVERED THE MOST AND WHO COVERED THE LEAST NUMBER OF READERS FROM THE READERS POPULATION**

# In[ ]:


fig = aware.bar(PEOPLE_AWARE[['Journals', 'Number_of_Readers']].sort_values('Number_of_Readers', ascending=False), 
             y="Number_of_Readers", x="Journals", color='Journals', width=1500, height=700,
             log_y=True, template='ggplot2', title='READERS WHO READ ABOUT COVID-19 AND MAY TAKE PRECAUTION = SAVING LIVES')
fig.show()


# In[ ]:


fig = aware.bar(PEOPLE_AWARE[['Journals', 'Number_of_Readers']].sort_values('Number_of_Readers', ascending=False), 
             y="Journals", x="Number_of_Readers", color='Journals',orientation = 'h',width=1500, height=1000,
             log_x=True, template='ggplot2', title='READERS WHO READ ABOUT COVID-19 AND MAY TAKE PRECAUTION = SAVING LIVES')
fig.show()


# **Below is a tree representation of the above data**

# In[ ]:


fig = aware.treemap(PEOPLE_AWARE, path=['Journals'], values='Number_of_Readers',title='Tree of READERS WHO READ ABOUT COVID-19 in these Journals',width=1200, height=700)
fig.show()


# **THIS WORK IS IN PROGRESS AND I WILL KEEP ON UPDATING THIS KERNEL**
