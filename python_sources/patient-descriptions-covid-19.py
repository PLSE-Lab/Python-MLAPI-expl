#!/usr/bin/env python
# coding: utf-8

# ### Patient descriptions related to COVID-19 
# By: Myrna M Figueroa Lopez   
# **Task:** Create summary tables that address patient characteristics related to COVID-19.   
# 
# 

# In[ ]:


#libraries
import numpy as np 
import pandas as pd 
import sys
import os


# In[ ]:


# CORD-19-research-challenge METADATA FILE
## a table of 18 columns and 63571 entries
ARTICLES=pd.read_csv('../input/CORD-19-research-challenge/metadata.csv') 
ARTICLES.shape


# In[ ]:


##Cleaned the dataframe (TABLE) above for readability
Articles1= ARTICLES[['title','publish_time','journal','url','abstract','doi','cord_uid']]
Articles1.head(2)


# In[ ]:


##Evaluation of dataframe
##Looking for NaNs
NaNs=Articles1.isnull().sum()
NaNs


# In[ ]:


#I limit to those entries with ABSTRACTS
Abstracts=Articles1.copy()
Abstracts.head(2)


# In[ ]:


#separate each word in the ABSTRACT column
Abstracts['words'] = Abstracts.abstract.str.strip().str.split('[\W_]+')
Abstracts['words']


# In[ ]:


#separate words in the abstract column and create a new column
Abs1 = Abstracts[Abstracts.words.str.len() > 0]
Abs1.head(3)


# In[ ]:


Abs2=Abs1.isnull().sum()
Abs2


# In[ ]:


# saving the dataframe 
Abs1.to_csv('Journal_Abstracts.csv') 


# #### Frequent words in abstracts

# In[ ]:


pip install wordcloud


# In[ ]:


from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 


# In[ ]:


wordcloud = WordCloud(
    width = 2000,
    height = 1000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(Abs1['words']))
fig = plt.figure(
    figsize = (20, 10),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# ### Answers 
# Answers from literature abstracts about:   
# 
# 1. Length of **viral shedding** after illness onset
# 2. **Incubation period** across different age groups
# 3. What is the Incubation Period of the Virus?
# 4. Proportion of patients who were **asymptomatic**
# 5. Pediatric patients who were asymptomatic
# 6. Asymptomatic **transmission** during incubation
# 7. Natural **history of the virus** from an infected person
# 8. What is the median viral shedding duration?
# 9. What is the longest duration of viral shedding?
# 10. **Manifestations**
# 11. How does **viral load relate to disease presentation** which includes likelihood of a positive diagnostic test?
# 

# In[ ]:


#looking through the abstracts with specific terms 
## 
##TABLE OF ABSTRACTS WITH 'sample size' 
Sample_size=Abs1[Abs1['abstract'].str.contains('sample size')]
Sample_size
##111 article ABSTRACTS display the 'sample size' of 51012 entries


# In[ ]:


# saving the dataframe 
Sample_size.to_csv('Sample_size.csv') 


# In[ ]:


##Viral Shedding Table
Viral_shed=Abs1[Abs1['abstract'].str.contains('viral shedding')]
Viral_shed
###157 abstracts include the term 'viral shedding'


# In[ ]:


# saving the dataframe 
Viral_shed.to_csv('Viral_shed.csv') 


# In[ ]:


wordcloud = WordCloud(
    width = 2000,
    height = 1000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(Viral_shed['words']))
fig = plt.figure(
    figsize = (20, 10),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


##1. Length of viral shedding after illness onset
Q1a=Viral_shed[Viral_shed['abstract'].str.contains('viral shedding after')]
Q1a


# In[ ]:


Q1b=Viral_shed[Viral_shed['abstract'].str.contains('viral shedding lasts')]
Q1b


# In[ ]:


Length_viral_shedding_after_onset=pd.concat([Q1a, Q1b])


# In[ ]:


##Table with articles relevant to QUESTION 1
Length_viral_shedding_after_onset.to_csv('Length_viral_shedding_after_onset.csv')


# In[ ]:


#8. What is the median viral shedding duration?
Q8=Viral_shed[Viral_shed['abstract'].str.contains('viral shedding duration')]
Q8


# In[ ]:


Q8.to_csv('viral_shedding_duration.csv')


# In[ ]:


#9. What is the longest duration of viral shedding?
Q9=Viral_shed[Viral_shed['abstract'].str.contains('longest')]
Q9


# In[ ]:


Q9.to_csv('longest_duration_viral_shedding.csv')


# In[ ]:


##Incubation Table
Incubation=Abs1[Abs1['abstract'].str.contains('incubation')]
Incubation
###610 abstracts include the term 'incubation'


# In[ ]:


# saving the dataframe 
Incubation.to_csv('Incubation.csv') 


# In[ ]:


wordcloud = WordCloud(
    width = 2000,
    height = 1000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(Incubation['words']))
fig = plt.figure(
    figsize = (20, 10),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


# Q2 Incubation period across different age groups

Q2a=Incubation[Incubation['abstract'].str.contains('different age')]
Q2a


# In[ ]:


Q2b=Incubation[Incubation['abstract'].str.contains('age group')]
Q2b


# In[ ]:


Q2= pd.concat([Q2a,Q2b])
Q2.to_csv('Incubation_period_across_age_groups.csv')


# In[ ]:


# Q3 What is the Incubation Period of the Virus?
Q3=Incubation[Incubation['abstract'].str.contains('COVID')]
Q3   


# In[ ]:


Q3.to_csv('Incubation_Period_COVID.csv')


# In[ ]:


#Q6 Asymptomatic transmission during incubation
Q6a=Incubation[Incubation['abstract'].str.contains('transmission during')]
Q6a
    


# In[ ]:


Q6b=Incubation[Incubation['abstract'].str.contains('asymptomatic transmission')]
Q6b


# In[ ]:


Q6= pd.concat([Q6a,Q6b])
Q6.to_csv('Asymptomatic_transmission_during_incubation.csv')


# In[ ]:


##Asymptomatic Table

Asymptomatic=Abs1[Abs1['abstract'].str.contains('asymptomatic')]
Asymptomatic
###936 abstracts include the term 'incubation'


# In[ ]:


# saving the dataframe 
Asymptomatic.to_csv('Asymptomatic.csv') 


# In[ ]:


wordcloud = WordCloud(
    width = 2000,
    height = 1000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(Asymptomatic['words']))
fig = plt.figure(
    figsize = (20, 10),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


#Q4. Proportion of patients who were asymptomatic
Q4a=Asymptomatic[Asymptomatic['abstract'].str.contains('number of asymptomatic')]
Q4a.shape


# In[ ]:


Q4a


# In[ ]:


Q4b=Asymptomatic[Asymptomatic['abstract'].str.contains('asymptomatic patients')]
Q4b.shape


# In[ ]:


Q4= pd.concat([Q4a, Q4b])
Q4.to_csv('Proportion_of_patients_were_asymptomatic.csv')


# In[ ]:


#Q5. Pediatric patients who were asymptomatic
Q5a=Asymptomatic[Asymptomatic['abstract'].str.contains('asymptomatic children')]
Q5a.shape


# In[ ]:


Q5a


# In[ ]:


Q5b=Asymptomatic[Asymptomatic['abstract'].str.contains('pediatric')]
Q5b.shape


# In[ ]:


Q5b


# In[ ]:


Q5=pd.concat([Q5a, Q5b])
Q5.to_csv('Pediatric_patients_were_asymptomatic.csv') 


# In[ ]:


##Q7 Natural history of the virus from an infected person
Q7a=Abs1[Abs1['abstract'].str.contains('virus history')]
Q7a


# In[ ]:


Q7b=Abs1[Abs1['abstract'].str.contains('history of the virus')]
Q7b


# In[ ]:


Q7=pd.concat([Q7a, Q7b])
Q7.to_csv('Natural_history_of_virus.csv')


# In[ ]:


##Q10 Manifestations
Q10a=Abs1[Abs1['abstract'].str.contains('manifest')]
Q10a


# In[ ]:


Manifestations=Abs1[Abs1['abstract'].str.contains('manifestation')]
Manifestations


# In[ ]:


Manifestations.to_csv('Manifestations.csv')


# In[ ]:


wordcloud = WordCloud(
    width = 2000,
    height = 1000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(Manifestations['words']))
fig = plt.figure(
    figsize = (20, 10),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


#Viral load table
Viral_LOAD=Abs1[Abs1['abstract'].str.contains('viral load')]
Viral_LOAD
#653 abstracts


# In[ ]:


# saving the dataframe 
Viral_LOAD.to_csv('Viral_LOAD.csv') 


# In[ ]:


wordcloud = WordCloud(
    width = 2000,
    height = 1000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(Viral_LOAD['words']))
fig = plt.figure(
    figsize = (20, 10),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


Q11=Viral_LOAD[Viral_LOAD['abstract'].str.contains('presentation')]
Q11.shape


# In[ ]:


Q11.to_csv('viral_load_relate_to_disease_presentation.csv') 

