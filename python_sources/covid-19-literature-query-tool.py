#!/usr/bin/env python
# coding: utf-8

# # What do we know about COVID-19 risk factors?

# The following notebook is a *work in progress tool* to attempt to complete the tasks defined in the
# 
# **COVID-19 Open Research Dataset Challenge (CORD-19)**
# 
# * The results in the notebook are tailored to the task *"What do we know about COVID-19 risk factors?"* although the same methodology can be used to complete the other 9 tasks
# * At present the implementation is basically a deterministic sentence retrieval tool based on specified keywords (which have to be defined manually by the user, some trial and error is required) with no AI/ML algorithms
# * Some domain knowledge is required to define appropriate key words. For example for "mitigation measures" it is appropriate to specify keywords such as "social distan"," mass gathering"," quarantine"," lockdown"," lock-down"," containment"," shutdown" rather than just look for "mitigation measures"
# * Only the text of the abstract is searched
# * Sentences retrieved are grouped by article with the specified keywords highighted
# * The idea is to just extract key results to be able to answer the different questions by further summarising (manually) the information in the extracted text
# * The final answers to the questions have not yet been drafted
# * Sentence retrieval is very useful for some keywords. For example looking at results <u>for mortality rates or reproductive numbers</u>, it is easy to find a signifcant number of reported rates and it wouldn't be too labour intensive to summarise the results quantitatively (histogram/barplot)
# * In other cases (e.g. keyword pulmonary) the number of sentences retrieved is significant and there is no obvious quantitative measure associated with them
# * Possible improvements are:
#     * Including number of citations of the articles (as a simple gauge of the importance of result)
#     * A further NLP processing of the results to extract numerical quantities (percentages/p-values)
#     * A more refined AI powered query algorithm

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import Markdown, display
import json
from collections import Counter


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#data set credits : https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv
#biorxiv_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv')
#clean_comm_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv')
#clean_noncomm_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv')
#clean_pmc_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv')


# Concatenate all dataframes containing the articles

# In[ ]:


#all_data=pd.concat([biorxiv_data,clean_comm_data,clean_noncomm_data,clean_pmc_data],axis=0).dropna()
#all_data.shape


# In[ ]:


#all_data.shape[0]


# In[ ]:


#all_data.head()


# In[ ]:


filenames_bio = os.listdir('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/')
print("Number of articles retrieved from biorxiv:", len(filenames_bio))
filenames_comm = os.listdir('/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/')
print("Number of articles retrieved from commercial use:", len(filenames_comm))
filenames_custom = os.listdir('/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/')
print("Number of articles retrieved from custom license:", len(filenames_custom))
filenames_noncomm = os.listdir('/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/')
print("Number of articles retrieved from non commercial:", len(filenames_noncomm))


# In[ ]:


all_files = []

for filename in filenames_bio:
    filename = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/' + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
for filename in filenames_comm:
    filename = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/' + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
for filename in filenames_custom:
    filename = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/' + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
for filename in filenames_noncomm:
    filename = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/' + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
    


# In[ ]:


file = all_files[100]
print("Dictionary keys:", file.keys())


# In[ ]:


titles_list=[]
for  file in all_files: 
    for refs in file['bib_entries'].keys(): 
        titles_list.append(file['bib_entries'][refs]["title"]) 


# In[ ]:


freqs = dict(Counter(titles_list))


# In[ ]:


freqs_df=pd.DataFrame.from_dict(freqs,orient='index',columns=['freqs'])


# In[ ]:


freqs_df=freqs_df.sort_values(by='freqs',ascending=False)


# In[ ]:


freqs_df['title']=freqs_df.index


# Read metadata

# In[ ]:


metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')


# Now merge articles citations

# In[ ]:


metadata=metadata.merge(freqs_df,how='left',on='title',suffixes=(False,'_y'))


# In[ ]:


metadata=metadata.sort_values(by='freqs',ascending=False)


# In[ ]:


metadata=metadata.dropna(subset=['abstract']) 


# In[ ]:


metadata.shape


# In[ ]:


metadata.head()


# In[ ]:


list(metadata.columns)


# Define a function for retrieving sentences with provided keywords from the corpus of articles
# Sentences are formatted as bullet points with relevant keywords in bold and ordered by source article (title in italics)
# A list of keywords can be provided for the same query as an input to the function (any match of a single keyword will return a sentence
# Sentences are extracted by splitting text with the "." delimiter
# There is also the option to retrieve the entire abstract (uncomment the line 
# display(Markdown("<sup>"+all_data.iloc[num,4].replace("\n"," ")+"</sup>"))

# In[ ]:


def what_do_we_know(match):
    num=0
    for abstract in metadata.iloc[:,8]:
        matched=False
        for terms in match: 
            if terms in abstract.lower() and "covid" in abstract.lower(): matched=True
        if matched:
            if np.isnan(metadata.iloc[num,17]): citations="NaN" 
            else: citations=str(int(metadata.iloc[num,17]))
            display(Markdown('<i> '+metadata.iloc[num,3]+'</i>'+' - '+citations+' citations'))
            sentence3="<ul>"
            for sentence in abstract.split('. '):
                sentence2=sentence.lower()
                matched2=False
                for terms in match: 
                    if terms in sentence2: 
                        matched2=True
                        sentence2=sentence2.replace(terms, '<b>'+terms+'</b>')
                if matched2: 
                 #   display(Markdown("> "+sentence2))
                    sentence3=sentence3+"<li>"+sentence2.replace("\n","")+"</li>"
            #sentence3=sentence3.replace("****","")
            display(Markdown(sentence3+"</ul>"))       
        #    display(Markdown("<sup>"+abstract.replace("\n"," ")+"</sup>"))
        #print(num)
        num+=1


# # Extract all quantitative results

# In[ ]:


# trial searches
what_do_we_know(["%"," higher than"," lower than"," key result"," equal to"," rate is"," rate was"," p-value"," estimated as"])


# # Smoking, pre-existing pulmonary disease

# In[ ]:


#what_do_we_know([" smok"])
#what_do_we_know([" pre-existing"])


# # Co-infections and other co-morbidities

# In[ ]:


#what_do_we_know([" coinfections"," co-infections"," comorbidities"," co-morbidities"])


# # Neonates and pregnant women

# In[ ]:


#what_do_we_know([" neonat"])
#what_do_we_know([" pregn"])


# # Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.

# In[ ]:


#what_do_we_know([" socioeconomic"," behavioural"])
#what_do_we_know([" economic impact"])


# # Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors

# In[ ]:


#what_do_we_know([" reproductive number"])
#what_do_we_know([" incubation period"])
#what_do_we_know([" serial interval"])
#what_do_we_know([" transmission"])


# In[ ]:


#what_do_we_know([" environmental factor"," environment factor"," environment risk"," food"," climate"," sanitation"])


# # Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups

# In[ ]:


#what_do_we_know([" risk of fatality"," mortality"])
#what_do_we_know([" hospitalized patients"])
#what_do_we_know([" high-risk"," high risk"])


# # Susceptibility of populations

# In[ ]:


#what_do_we_know([" susceptibility"])


# # Public health mitigation measures that could be effective for control

# In[ ]:


#what_do_we_know([" mitigation measures"," social distan"," mass gathering"," quarantine"," lockdown"," lock-down"," containment"," shutdown"])


# In[ ]:




