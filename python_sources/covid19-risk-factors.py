#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
deadline = datetime.strptime('2020-04-16 23:59:00','%Y-%m-%d %H:%M:%S')
print(deadline)
print(datetime.now())
print(deadline - datetime.now(), 'hours')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# imports
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import json
import requests
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import curve_fit
import string
from scipy.integrate import quad

from tqdm import tqdm

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.base import clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder 

from wordcloud import WordCloud

# module settings
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 1000)
plt.rcParams['figure.figsize'] = [15, 8]

# https://images.plot.ly/plotly-documentation/images/python_cheat_sheet.pdf
# https://www.apsnet.org/edcenter/disimpactmngmnt/topc/EpidemiologyTemporal/Pages/ModellingProgress.aspx


# [CORD-19 Challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=558)
# 
# # What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?
# 
# Specifically, we want to know what the literature reports about:
# 
# * Data on potential risks factors
# * Smoking, pre-existing pulmonary disease
# * Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities
# * Neonates and pregnant women
# * Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.
# * Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
# * Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups
# * Susceptibility of populations
# * Public health mitigation measures that could be effective for control
# 
# # Evaluation
# Submissions will be scored using the following grading rubric:
# 
# Accuracy (5 points)
# * Did the participant accomplish the task?
# * Did the participant discuss the pros and cons of their approach?
# 
# Documentation (5 points)
# * Is the methodology well documented?
# * Is the code easy to read and reuse?
# 
# Presentation (5 points)
# * Did the participant communicate their findings in an effective manner?
# * Did the participant make effective use of data visualizations?

# # Who are the most vulnerable?
# 
# * *with rantes -28 cg and gg genotypes had a 3.28-fold (95%ci:2.32-4.64) and 3.06-fold (95%ci:1.47-6.39) increased risk of developing sars respectively (p < 0.0001).*  (source: antivirals for influenza-like illness?)
# * *presence in the room during fiberoptic intubation (or = 2.79, p = .004) or ecg (or = 3.52, p = .002), unprotected eye contact with secretions (or = 7.34, p = .001), patient apache ii score $20 (or = 17.05, p = .009) and patient pa0 2 /fi0 2 ratio #59 (or = 8.65, p = .001) were associated with increased risk of transmission of sars-cov.* (source: systematic review extreme water-related weather events and waterborne disease) 
# * *groups seen as at 'high risk' of infection included the immune compromised (mentioned by 87% respondents), pig farmers (70%), elderly (57%), prostitutes/highly sexually active (53%), and the homeless (53%).* (source: development of a smartphone-based rapid dual fluorescent diagnostic system for the simultaneous detection of influenza a and h5 subtype in avian influenza a-infected patients)
# * [Males (80+ years old) are the most at risk of dying from COVID-19](https://www.kaggle.com/bitsnpieces/covid19-risk-factors/notebook#Males-(80+-years-old)-are-the-most-at-risk-of-dying-from-COVID-19)
# 

# # Literature search
# 
# The literature search is conducted from the CORD-19 Research Challenge dataset. The information is therefore limited to only what is available in the dataset.
# 
# The method that I'll be using includes a simple bag of words overlap to fetch relevant documents. The documents retrieved will be based on which keywords are used. Keywords such as 'risk', 'lung', 'vulnerab', etc. are used to mark the paper as 'is_risk' while keywords such as 'gene', 'molecule' are for 'is_research'. For the preliminary version, only those papers marked 'is_risk' are analyzed. The advantage of this approach is that it's simple and easily understood.
# 
# Future improvements can include using more advanced NLP techniques such as sentence similarity using word and character embeddings which will require more resources. Analyze full text documents using summary tools like [summy](https://pypi.org/project/sumy/). The chosen words were manually selected based on experience however, other synonymous words or words from experts can increase the specificity and accuracy of the results.
# 
# A couple of hand-picked results:
# >==========/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/6c7341f17bfd790cdc05b6d1010c63f4f8eb5890.json==========
# antivirals for influenza-like illness? protocol for a randomized controlled trial of clinical and cost effectiveness in primary care (alic 4 e) antivirals for influenza-like illness? a randomized controlled trial of clinical and cost effectiveness in primary care (alic4e): the alic4e protocol antivirals for influenza-like illness? a randomized controlled trial of clinical and cost effectiveness in primary care (alic 4 e): the alic 4 e protocol antivirals for influenza-like illness? a randomized controlled trial of clinical and cost effectiveness in primary care (alic4e): the alic4e protocol antivirals for influenza-like illness? a randomized controlled trial of clinical and cost effectiveness in primary care (alic 4 e): the alic 4 e protocol
# individuals with rantes -28 cg and gg genotypes had a 3.28-fold (95%ci:2.32-4.64) and 3.06-fold (95%ci:1.47-6.39) increased risk of developing sars respectively (p < 0.0001).
# 
# >==========/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pmc_json/PMC7100800.xml.json==========
# systematic review extreme water-related weather events and waterborne disease
# in multivariate gee logistic regression models, presence in the room during fiberoptic intubation (or = 2.79, p = .004) or ecg (or = 3.52, p = .002), unprotected eye contact with secretions (or = 7.34, p = .001), patient apache ii score $20 (or = 17.05, p = .009) and patient pa0 2 /fi0 2 ratio #59 (or = 8.65, p = .001) were associated with increased risk of transmission of sars-cov.
# 
# >==========/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/6c7341f17bfd790cdc05b6d1010c63f4f8eb5890.json==========
# antivirals for influenza-like illness? protocol for a randomized controlled trial of clinical and cost effectiveness in primary care (alic 4 e) antivirals for influenza-like illness? a randomized controlled trial of clinical and cost effectiveness in primary care (alic4e): the alic4e protocol antivirals for influenza-like illness? a randomized controlled trial of clinical and cost effectiveness in primary care (alic 4 e): the alic 4 e protocol antivirals for influenza-like illness? a randomized controlled trial of clinical and cost effectiveness in primary care (alic4e): the alic4e protocol antivirals for influenza-like illness? a randomized controlled trial of clinical and cost effectiveness in primary care (alic 4 e): the alic 4 e protocol
# coronaviruses (covs) are found in a wide variety of wild and domestic animals, and constitute a risk for zoonotic and emerging infectious disease.
# 
# > ==========/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/f2948dc01fd28774bb06b6c6995bec2a2f466f2a.json==========
# title: household emergency preparedness in china: a cross-sectional survey author names and affiliations
# partially ecologic study based on short-term exposure demonstrated that sars patients from regions with moderate apis had an 84% increased risk of dying from sars compared to those from regions with low apis (rr = 1.84, 95% ci: 1.41-2.40).
# 
# > ==========/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pmc_json/PMC7100800.xml.json==========
# development of a smartphone-based rapid dual fluorescent diagnostic system for the simultaneous detection of influenza a and h5 subtype in avian influenza a-infected patients
# groups seen as at 'high risk' of infection included the immune compromised (mentioned by 87% respondents), pig farmers (70%), elderly (57%), prostitutes/highly sexually active (53%), and the homeless (53%).
# 
# > ==========/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/f2948dc01fd28774bb06b6c6995bec2a2f466f2a.json==========
# title: household emergency preparedness in china: a cross-sectional survey author names and affiliations
# for patients at risk for asthma, or with existing asthma, viral respiratory tract infections can have a profound effect on the expression of disease or loss of control.
# 
# > ==========/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/f2948dc01fd28774bb06b6c6995bec2a2f466f2a.json==========
# title: household emergency preparedness in china: a cross-sectional survey author names and affiliations
# an elevated level of viral diversity was found in some sars-cov-2 infected patients, indicating the risk of rapid evolution of the virus.
# 
# > ==========/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/f2948dc01fd28774bb06b6c6995bec2a2f466f2a.json==========
# title: household emergency preparedness in china: a cross-sectional survey author names and affiliations
# although people at the extremes of age have a greater risk of complications, influenza has been more frequently investigated in the elderly than in children, and inpatients than outpatients.
# 
# > ==========/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/138e18baf12e4e92b67ab7dee321d2b149f236ed.json==========
# title: the changes of prevalence and etiology of pediatric pneumonia from national emergency department information system in korea, between 2007 and 2014
# by multivariate logistic regression, male, older age and comorbidity with diabetes were three important independent risk factors predicting aki among covid-19 patients.
# 
# 

# In[ ]:


meta = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
meta


# In[ ]:


# extracing title from json files
def parse_json(fn):
    with open(fn, 'r') as f:
        data = json.loads(f.read())
#     print(data)
    source = fn.split('/')[4]
#     print('source=',source)
    return([data['paper_id'], data['metadata']['title'], source, fn])

# parse_json('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/2c006d09b6fccc527bf5ee3de0f165b018c39e73.json')

def load_data():
    papers = []
    import os
    import json
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in tqdm(filenames):
            if '.json' not in filename:
                continue
            try:
                fn = os.path.join(dirname, filename)
#                 print(fn)
                papers.append(parse_json(fn))
            except Exception as e:
                print(e)
                continue
    return papers
            
papers = load_data()


# In[ ]:


papers


# In[ ]:



# to data frame
df = pd.DataFrame(papers,columns=['pmcid','title','source', 'file'])
df = df.drop_duplicates()
df = pd.merge(df, meta, on='pmcid', how='left').reset_index().rename(columns={'title_x':'title'})
del df['title_y']

# cleaning titles
titles = df['title']
titles_clean = []
for t in tqdm(titles):
    titles_clean.append(t.replace('"','').lower().strip())
#     for p in parts:

df['title_clean'] = titles_clean


# df = pd.read_csv('/kaggle/input/covid19-uncover-paper-titles/covid19_uncover_paper_titles.csv')
df['title'] = df['title'].astype('str')
df['title_clean'] = df['title_clean'].astype('str')

# text associated that we are interested in
inclusion = ('vulnerab epidem suscept copd male clinical illness female diabetes comorbid mortal prognosis hypertension blood heart liver lung kidney renal brain bladder tuberc drink alcohol mental psychi nerv smoking age stroke cardio cerebr cancer respira chronic factor risk population sex gender hospital health age population ethnic flu death mortality respir disease child pediat adult infant lung resp smok person patient old senior elderly infected individual').split(' ')

# text associated with more research
exclusion = ('pig rat mouse chicken horse mice dog monkey cat cations simulation bird gene cell tissue organism glia phagocy microbiology bacteria eukaryotic assay apoptosis signal protein pathway molecul rna dna monocytes chemokine').split(' ')

relevant_docs = set()
research_docs = set()
df['is_risk'] = 0
df['is_research'] = 0
for i in tqdm(df.index):
    t = df.loc[i,'title_clean']
    if np.nan == t or pd.isna(t):
        continue
    for w in inclusion:
#         print(f't={t}, w={w}')
        if w in t:
            df.loc[i, 'is_risk'] = 1
            relevant_docs.add(t)
            break
    for w in exclusion:
        if w in t:
            df.loc[i, 'is_research'] = 1
            research_docs.add(t)
            break


df.to_csv('paper_titles_is_risk.csv', index=False)
df.shape


# In[ ]:


df


# In[ ]:


df[['source','is_risk','is_research']].groupby('source').sum()


# In[ ]:


meta


# # Title enrichment and filtering

# In[ ]:


# text associated that we are interested in
# inclusion = ('vulnerab suscept factor risk population sex gender hospital health age population ethnic flu death mortality respir disease child pediat adult infant lung resp smok person patient old senior elderly infected individual').split(' ')

# text associated with more research
# exclusion = ('pig', 'rat', 'mouse', 'mice', 'monkey', 'cations', 'gene', 'cell', 'tissue', 'glia', 'phagocy', 'microbiology', 'bacteria', 'eukaryotic', 'apoptosis', 'signal', 'protein', 'pathway', 'molecul', 'rna', 'dna', 'monocytes', 'chemokine')

# relevant_docs = set()
# research_docs = set()
# df['is_risk'] = 0
# df['is_research'] = 0
# for i in tqdm(range(df.shape[0])):
#     t = df.loc[i,'title_clean']
#     for w in inclusion:
#         if w in t:
#             df.loc[i, 'is_risk'] = 1
#             relevant_docs.add(t)
#     for w in exclusion:
#         if w in t:
#             df.loc[i, 'is_research'] = 1
#             research_docs.add(t)

# relevant_docs = list(relevant_docs)
# research_docs = list(research_docs)
# df.to_csv('paper_titles.csv', index=False)


# In[ ]:


df


# # Word cloud of "risk" paper titles

# In[ ]:


df = df.query('is_risk == 1 & is_research == 0')
relevant_docs = df['title'].values

text = ' '.join(relevant_docs)
from wordcloud import WordCloud

# # Build word frequencies on filtered tokens
# freqs = pd.Series(np.concatenate([tokenize(x) for x in articles.Title])).value_counts()
# wordcloud(freqs, "Most frequent words in article titles tagged as COVID-19")

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
# wordcloud = WordCloud(max_font_size=40).generate(text)
# plt.figure()
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()


# In[ ]:


# Fetch full text for 'risk' related papers  
def fetch_abstract(path):
    ret = []
    try:
        
        with open(path, 'r') as f:
            data = json.loads(f.read())
    #     print(data)
    #     for a in data['body_text']:

        for a in data['abstract']:
            ret.append(a['text'])
    except:
        pass
    return (' '.join(ret)).lower().strip()

# fetch_abstract('/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/26aec9a28a4345276498c14e302ead7d96c7feee.json')

# full_text = dict()
df['abstract'] = ''
for i in tqdm(df.index):
    fn = df.loc[i, 'file']
    df.loc[i, 'abstract'] = fetch_abstract(fn)
    
df.to_csv('paper_titles.csv', index=False)
df
    


# # Abstract sentences WordCloud

# In[ ]:


abstract_text = ' '.join(df['abstract'].values)
abstract_text

# remove stop words
words = []
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))
abstract_text = ' '.join([w for w in word_tokenize(abstract_text) if w not in stop_words])

# Generate a word cloud image for the abstract
wordcloud = WordCloud().generate(abstract_text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In[ ]:


pd.set_option('display.max_colwidth', -1)
print(df['title'].head(10))


# # Paper titles that with abstracts that contains *corona, covid, sars, mers, h1n1* and *risk*

# In[ ]:


df


# In[ ]:


from nltk.tokenize import sent_tokenize

results = []
LIMIT = 2000

prev_s = ''
for i in tqdm(df.index):
    a = df.loc[i,'abstract']
    for s in sent_tokenize(a):
#             if 'patient' in s and 'risk' in s:
        prev_s = s.strip()
        if ('covid' is s or 'corona' in s or 'sars' in s or 'mers' in s or 'h1n1' in s) and 'risk' in s:
            f = df.loc[i, 'file']
            title = df.loc[i, 'title']
            pid = df.loc[i, 'pmcid']
            results.append((pid, f,title, prev_s + ' ' + s.strip()))
#                 print(results)
#             print()
#             print('='*10 + fn + '='*10)
#             print(title)
#             print(s.strip())
            if len(results) > LIMIT:
                break
    if len(results) > LIMIT:
        break


with open('results_patient_risk.csv','w') as f:
    for pid,fn,title,s in results:
#         print()
#         print('='*10 + fn + '='*10)
#         print(title)
#         print(s.strip())
        f.write(fn + '\t' + title + '\t' + s.strip() + '\n')


# In[ ]:


# number of abstract sentences that matched grouped by paper titles
df_results = pd.DataFrame(results, columns=['pmcid', 'file','title','sentence'])
df_results.to_csv('results.csv')
print(df_results.shape)
df_results.groupby(['title']).count()[['pmcid']].sort_values(by='pmcid',ascending=False)


# # Abstract sentences matched *corona, covid, sars, mers, h1n1* and *risk*

# In[ ]:


df_results[['pmcid', 'title','sentence']].head(1000)


# In[ ]:


results_text = ' '.join(df_results['sentence'].values)

# Generate a word cloud image for the abstract
wordcloud = WordCloud().generate(results_text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# # Males (80+ years old) are the most at risk of dying from COVID-19
# 
# As shown below, most of the fatalities are Males (80+) with 6,825 fatalities vs Females (80+) 5,322. On the other hand, there were 7,379 Female (80+) survivors vs 4,185 Male (80+) survivors.

# In[ ]:


df_cases = pd.read_csv('/kaggle/input/covid19-european-enhanced-surveillance/covid19_enhanced_surveillance.csv')
df_cases.head(50)


# In[ ]:


pd.set_option('display.max_rows', 1000)
df_cases['cases'] = [ int(str(x).replace('<=5','5')) for x in df_cases['Cases'] ]
df_cases['deaths'] = [ int(str(x).replace('<=5','5')) for x in df_cases['Deaths'] ]
df_cases.groupby(['Outcome', 'Gender','Age group']).sum()

