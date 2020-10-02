#!/usr/bin/env python
# coding: utf-8

# # COVID-19 EDA
# My exploration so far through the CORD-19 research challenge data set.  
#   
#   
# This is my first kernel on the site and I am relatively new in this world (coming from software engineering). For that reason, any feedback of any kind is welcome.
#   
# If you are curious, the repository for this EDA is on my github at: https://github.com/jbofill10/COVID-19-EDA  
#   
# Thanks for taking the time to check my notebook out!  

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

from tqdm.notebook import tqdm


# In[ ]:


file_paths = list()
for path, dirs, files in os.walk('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv'):
    for file in files:
            file_paths.append(os.path.join(path, file))
for path, dirs, files in os.walk('/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset'):
        for file in files:
            file_paths.append(os.path.join(path, file))
for path, dirs, files in os.walk('/kaggle/input/CORD-19-research-challenge/custom_license/custom_license'):
        for file in files:
            file_paths.append(os.path.join(path, file))
for path, dirs, files in os.walk('/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset'):
        for file in files:
            file_paths.append(os.path.join(path, file))


# In[ ]:


df = pd.DataFrame()
for path in tqdm(file_paths):    
    full_abstract = ''
    full_body = ''
    with open(path.encode('utf-8'), 'r') as file:
        data = json.load(file)
        paper_id=data['paper_id']
        title = data['metadata']['title']
        for x in data['abstract']:
            abstract = x['text']
            full_abstract += abstract

        for x in data['body_text']:
            body = x['text']
            full_body += body

    temp = pd.DataFrame([[paper_id, title, full_abstract, full_body]], columns=['paper_id', 'title', 'abstract', 'body'])
    df = df.append(temp)

df.head()


# # What do we know about Incubation?
# From the data I extrapolated, I arrived at an average of 12 days for how long COVID-19 should take to incubate. In hindsight, that is right around with what is common knowledge now (7-14 days). 
# 
# ### What I did:
# * I searched each sentence of each body text of the articles for the word incubation 
# * Then with regex searched for digits followed by the string "day" or "days". 
# * I stripped all non-numerical characters in order to preserve pharses such as 3-7. There were still some serious outliers in the 600 range, but I ended up trimming the graph before the outliers begun.  

# Import re to find strings that follow this pattern: "(random words) (numbers) day/days". Not the best way, but provides some accuracy at least.

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.style as style
import re


# In[ ]:


style.use('seaborn-poster')
style.use('ggplot')
# I'm thinking of looking for strings with incubation to start
incubation_occurrences = list()
paper_titles = set()

body_text = df['body'].values
index = 0
for text in tqdm(body_text):
    for sent in text.split('. '):
        if 'incubation' in sent:
            temp = re.findall('[1-9]{1,2}\S*\sday|s$', sent)
            if len(temp) > 0:
                for find in temp:
                    find = re.sub('\D', ' ', find)
                    house = find.strip().split(' ')
                    for nums in house:
                        try:
                            incubation_occurrences.append(int(nums))
                            paper_titles.add((df.loc[index]['paper_id'].strip()))
                        except:
                            continue
    index+=1
    
incubation_occurrences
incubation_df = pd.DataFrame({'days': incubation_occurrences}, index=range(0, len(incubation_occurrences)))
incubation_df = incubation_df.sort_index()


# In[ ]:


plt.hist(incubation_df['days'].value_counts(), bins=75, color='purple')
plt.xlabel("Incubation Time in Days", fontsize=20, color='black')
plt.ylabel('Frequency Found in Literature', fontsize=20, color='black')
plt.title('Distribution of Incubation Periods', fontsize=23, color='black')
plt.xlim(1, 62)
plt.rcParams['axes.grid'] = True
plt.show()


# 
# 

# # What do We Know about Transmission?
# 
# ## What I did
# * I used Spacy to search for papers with had words with the lemmatization of "transmit" and "transmission"
#     * I could've improved this search, but it was for the sake of my PC
#     * Any ideas for improvement of the search would be much appreciated here
# * I stored those papers into a dataframe to do some visualization with them along with for retrieving any papers on COVID-19 relating to transmission

# Import spacy related libraries

# In[ ]:


import spacy
from spacy.matcher import Matcher


# In[ ]:


'''
Code for building the dataframe of papers containing words with the lemmatization of transmit and transmission

As much as I'd like it to be live, I also don't want you waiting 45+ minutes for my notebook to load :)

transmit_keywords = ['transmit', 'transmission']
        findings = dict()
        paper_id_tracker = set()
        temp_papers = list()

        transmission_papers = pd.DataFrame(columns=['paper_id', 'title', 'abstract', 'body'])
        body_text = df['body'].values

        nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        matcher = Matcher(nlp.vocab)

        for i in transmit_keywords:
            matcher.add(i, build_transmission_pattern(i))

        index = 0
        for i in tqdm(body_text):
            
            spacy.prefer_gpu()
            
            for docs in nlp.pipe(texts, disable=['parser', 'ner', 'entity_linker']):
                
                for match_id, start, end in matcher(docs):
                    if df.loc[index]['paper_id'] not in paper_id_tracker:
                        
                        paper_id_tracker.add(df.loc[index]['paper_id'])    
                        temp_papers.append([df.loc[index]['paper_id'], df.loc[index]['title'], df.loc[index]['abstract'], df.loc[index]['body']])
                    
                    if str(docs[start:end]) in findings:
                    
                        findings[str(docs[start:end]).lower()] += 1
                    else:
                    
                        findings[str(docs[start:end]).lower()] = 1
            index += 1

        for i in temp_papers:
            transmission_papers = pd.concat([transmission_papers, pd.DataFrame([[i[0], i[1], i[2], i[3]]],
                                                                               columns=['paper_id', 'title', 'abstract',
                                                                                        'body'])])
                                                                                        
def build_transmission_pattern(keyword):
    return [[{'LEMMA': keyword.lower()}]]
'''


# ## How did the Spacy pipeline Perform?
# ![image](https://github.com/jbofill10/COVID-19-EDA/blob/master/Charts/TransmissionFreq.png?raw=true)  
#   
# This graph shows the different variations of the words "transmit" and "transmission" in the papers and the count. I thought this was cool.
# 
# ## Taking it a step further
# I took the dataframe that contained papers relating transmission and then took a subset of those papers that contained common words used to describe the process of transmission.  
# These words were: 'Contact, Aerosol, Surface, and Breathing  
# ![image](https://github.com/jbofill10/COVID-19-EDA/blob/master/Charts/TransmissionMediums.png?raw=true)  
#   
# Even though I know that a lot of the matches found in these papers don't match the context of actual transmission, but I still think it is somewhat accurate.

# # What do we know about COVID-19 itself?
# *This section is still a work in progress*
# 
# I searched for common flu-like symptoms and did some basic string matching  
# I chose not to go with spacy for this because it would just take too long :(  
# 
# I still think I got some pretty neat results in the end.
# 
# 

# In[ ]:


import seaborn as sns


# In[ ]:


common_symptoms = ['fever', 'chills', 'cough', 'sore throat',
                           'runny nose', 'headache', 'fatigue', 'vomiting',
                           'shortness of breath', 'dizziness']
text = df['body'].values

symptom_freq = list()

for x in tqdm(text):
    for sent in x.split('. '):
        for symp in common_symptoms:
            if symp in sent:
                symptom_freq.append(symp)
symptoms_df = pd.DataFrame(symptom_freq, columns=['symptom'])
symp_vals = symptoms_df['symptom'].value_counts().sort_values()


# In[ ]:


plt.figure(figsize=(21, 10))
sns.set(style="darkgrid")
    
sns.barplot(x=symp_vals.values, y=list(range(0, len(symp_vals.index))), orient='h', palette='Spectral')
plt.yticks(list(range(0, len(symp_vals.index))), [i.capitalize() for i in symp_vals.index], fontsize=15)
    
plt.xlabel('Times Mentioned in Literature', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Symptoms', fontsize=20)
    
plt.title('Common Flu Symptoms found in COVID-19 Literature', fontsize=22)
    
plt.show()


# I think most of the data aligns well with what we know about COVID-19... except I was surprised that I found a lot of vomitting related matches in comparison to something like sore throat.  
#   
# I was thinking maybe if I added better searches for cough/throat related symptoms that it would improve a bit.
