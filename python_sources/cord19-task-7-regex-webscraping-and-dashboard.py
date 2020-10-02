#!/usr/bin/env python
# coding: utf-8

# Contributions made to this notebook by Colin Lagator (Data Analysis), Emily Morgan (Biomedical/Health SME)

# # Creating Summary Tables Using Regex and Webscraping
# ## CORD-19 Challenge
# #### Contributions made by employees of Booz Allen Hamilton.
# ### Task: Therapeutics, Interventions, and Clinical Studies
# 
# 1. Data Cleaning and Subsetting
#     - Processing Raw Data
#     - Covid-19 Subset
#     - Tasks Subsets
# 2. Finding Summary Table Features
#     - Mentioned Treatments
#     - Study Types
#     - Severity of Disease
#     - Number of Patients
#     - Conclusion Excerpts
# 3. Completing Table with Dashboard
#     - Dashboard made with Plotly's Dash library
# 
# This task focuses on two main questions; What is the best method to combat the hypercoagulable state seen in COVID-19? What is the efficacy of novel therapeutics being tested currently? The task requests that summary tables be provided for each question which contain certain features. This noteboook details a process of finding the summary tables for the two questions above. This process includes keyword searching, regex, webscraping, and an interactive dashboard. Using all these components, the authors feel that useful summary tables can be found. 
# 
# The summary tables are outputted as *results_table_hypercoag.csv* and *results_table_novel_therapy.csv*.

# ----------------

# ### Necessary Imports

# In[ ]:


import numpy as np
import pandas as pd
import re

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.summarization import bm25

import json

import time

from bs4 import BeautifulSoup
import requests as r
import string


# # 1. Data Cleaning and Subsetting

# ## 1.1 Processing Raw Data
# 
# The data was processed into one CSV called full_articles_r2.csv. Only articles that had a PMC or PDF JSON associated with them were kept. The JSON file processing was based on work done in [this notebook](https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool). For consistency, if an article had a PDF JSON file, the PDF was used. Otherwise, the PMC JSON file was used.
# 
# Three features were extracted from the JSON files; the introduction sections, the main body text, and the discussion section. The intro and discussion sections were identified with the section titles, all other text was taken as the body text. If no results section was identified, then the last 6 sentences of the body text were used instead. If no intro section was found, the first 6 sentences of the body text were used.

# *Click CODE to Reveal Code for Data Processing*

# In[ ]:


metadata = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv', dtype={'cord_uid': 'str',
                                                                         'sha': 'str',
                                                                         'source_x': 'str',
                                                                         'title': 'str',
                                                                         'doi': 'str',
                                                                         'pmcid': 'str',
                                                                         'pubmed_id': 'str',
                                                                         'license': 'str',
                                                                         'abstract': 'str',
                                                                         'publish_time': 'str',
                                                                         'authors': 'str',
                                                                         'journal': 'str',
                                                                         'mag_id': 'str',
                                                                         'who_covidence_id': 'str',
                                                                         'arxiv_id': 'str',
                                                                         'pdf_json_files': 'str',
                                                                         'pmc_json_files': 'str',
                                                                         'url': 'str'})

has_pdf_json = metadata[pd.notna(metadata['pdf_json_files'])].reset_index(drop=True)
only_pmc_json = metadata[pd.isna(metadata['pdf_json_files']) & pd.notna(metadata['pmc_json_files'])].reset_index(drop=True)

def semi_colon_split(string):
    return string.split(sep=';')

has_pdf_json['pdf_json_files'] = has_pdf_json['pdf_json_files'].apply(semi_colon_split)
only_pmc_json['pmc_json_files'] = only_pmc_json['pmc_json_files'].apply(semi_colon_split)

class JsonReader:
    def __init__(self,file_path):
        with open(file_path) as file:
            
            article = json.load(file)
            
            intro_text = []
            body_text = []
            results_discussion_text = []
            for section in article['body_text']:
                                
                #identifying introduction sections
                intro_keys = ['introduction']
                intro_keys_filter = []
                for key in intro_keys:
                    intro_keys_filter.append(key in section['section'].lower())
                    
                #identifying disucssion/results sections
                discussion_keys = ['discussion', 'conclusion', 'results']
                discussion_keys_filter = []
                for key in discussion_keys:
                    discussion_keys_filter.append(key in section['section'].lower())
                
                #compiling introduction sections
                if any(intro_keys_filter):
                    intro_text.append(section['text'])
                                
                #compiling discussion sections
                elif (any(discussion_keys_filter) | ('in conclusion' in section['text'].lower())):
                    results_discussion_text.append(section['text'])
                
                #compiling all other sections to body_text
                else:
                    body_text.append(section['text'])
                
            self.intro_text = ' '.join(intro_text)
            self.body_text = ' '.join(body_text)
            self.results_discussion_text = ' '.join(results_discussion_text)
            
#PDF JSON file parsing
            
root_path = '../input/CORD-19-research-challenge/'

pdf_body_text = []
pdf_intro_text = []
pdf_results_discussion_text = []

for paths in has_pdf_json.pdf_json_files:
    file_reader = JsonReader(root_path + paths[0])
    pdf_intro_text.append(file_reader.intro_text)
    pdf_body_text.append(file_reader.body_text)
    pdf_results_discussion_text.append(file_reader.results_discussion_text)
    
has_pdf_json['intro'] = pdf_intro_text
has_pdf_json['body_text'] = pdf_body_text
has_pdf_json['results_discussion'] = pdf_results_discussion_text

#PMC JSON file parsing

pmc_body_text = []
pmc_intro_text = []
pmc_results_discussion_text = []

for paths in only_pmc_json.pmc_json_files:
    file_reader = JsonReader(root_path + paths[0])
    pmc_intro_text.append(file_reader.intro_text)
    pmc_body_text.append(file_reader.body_text)
    pmc_results_discussion_text.append(file_reader.results_discussion_text)
    
only_pmc_json['intro'] = pmc_intro_text
only_pmc_json['body_text'] = pmc_body_text
only_pmc_json['results_discussion'] = pmc_results_discussion_text

#Combining the PMC and PDF features together

full_articles = pd.concat([has_pdf_json, only_pmc_json], axis = 0)
full_articles = full_articles.drop_duplicates(subset='cord_uid', keep='first').reset_index(drop=True)
full_articles.loc[pd.isna(full_articles.title), 'title'] = ''
full_articles.loc[pd.isna(full_articles.abstract), 'abstract'] = ''

def remove_multiple_spaces(string):
    return re.sub(r'\s+', ' ', string)

full_text = full_articles.title + ' ' + full_articles.abstract + ' ' + full_articles.intro +             ' ' +  full_articles.body_text + ' ' + full_articles.results_discussion
full_text = full_text.apply(remove_multiple_spaces)
full_text = full_text.str.strip()

full_articles['full_text'] = full_text

#If no results/discussion section was found, then the last 6 sentences of the paper were used instead.

for i in range(len(full_articles)):
    if full_articles.loc[i, 'results_discussion'] == '':
        full_text = full_articles.loc[i, 'full_text']
        sentences = sent_tokenize(full_text)
        last = sentences[-5:]
        full_articles.at[i, 'results_discussion'] = ' '.join(last)
        
#If no intro section is found, the first 6 sentences of the paper are used.
        
for i in range(len(full_articles)):
    if full_articles.loc[i, 'intro'] == '':
        full_text = full_articles.loc[i, 'full_text']
        sentences = sent_tokenize(full_text)
        first = sentences[:6]
        full_articles.at[i, 'intro'] = ' '.join(first)
        
full_articles.to_csv('full_articles_r2.csv', index=False)


# ## 1.2 Creating a COVID-19 Subset of Articles
# Only articles that were associated with COVID-19 were used in the analysis. These article were identified using a keyword filter. The keyword filter class is based on [this notebook](https://www.kaggle.com/ajrwhite/covid-19-thematic-tagging-with-regular-expressions/notebook).

# *Click CODE to Reveal Keyword Filter Code*

# In[ ]:


class KeywordFilter():
    """
    ***

    Class that returns a df containing the rows in which the specified columns
    (which are assumed to contain strings) contain any one of the keywords.
    Also returns a dataframe containing the amount of articles each keyword is
    contained in.
 
    Param: df - A DataFrame containing columns of text
        keywords - A list of the keywords being looked for.
        columns - A list of column names where the keyword will be 
                  searched for.

    ***
    """
    def __init__(self, df, keywords, columns):
        
        self.df = df.copy(deep=True)
        self.keywords = keywords
        self.columns = columns
        
    # Helper functions that indicates if the indicated columns 
    #    of a row contain the keyword parameter.
    #
    # Param: keyword - The keyword being looked for.
    #
    # Returns: A series of booleans.

    def contains_keyword(self, keyword: str):
        keyword_tags = ([False] * len(self.df))
        for c in self.columns:
            keyword_tags = keyword_tags                 | self.df[c].str.lower().str.contains(keyword, na=False)
        return keyword_tags

    def transform(self):
        """
        Function that filters out rows that do not contain any of the
        keywords in the specified columns.
      
        Returns: The original DataFrame with the added column.
        """
        tag = 'TAG'
        counts = {}
        #set all rows false
        self.df[tag] = False
        for key in self.keywords:
            key_filter = self.contains_keyword(key)
            counts[key] = sum(key_filter)
            #The rows in key filter marked True will indicate that row in df to True
            self.df.loc[key_filter, tag] = True
        self.df = self.df[self.df[tag]]
        self.df = self.df.drop(columns=['TAG'])
        return self.df, pd.Series(counts)


# In[ ]:


df = full_articles[['cord_uid','doi', 'url', 'journal',
                    'publish_time', 'title','abstract',
                    'intro', 'body_text', 'results_discussion',
                    'full_text']]

covid19_keywords =['sars-cov-2', 'covid-19', '2019-ncov', 
                    'novel-coronavirus',
                    'coronavirus 2019','wuhan pneumonia',
                    '2019ncov', 'covid19',
                    'sarscov2', 'coronavirus-2019']

covid19_filter = KeywordFilter(df=df, columns=['abstract'], keywords=covid19_keywords)
covid19_articles, covid19_counts = covid19_filter.transform()


# ## 1.3 Finding Subsets of Articles for Each Task
# A subset of articles was found for each of the questions related to this task. These subsets were found using the same keyword filter as above.

# ### 1.3.1 Keyword Filtering

# In[ ]:


covid19 = covid19_articles.dropna(subset=['title', 'abstract'])
covid19['title_abstract'] = covid19.title + ' ' + covid19.abstract


# #### Articles with Hypercoagulable State Keywords

# In[ ]:


hypercoag_filter = KeywordFilter(df=covid19, columns=['title_abstract'], 
                                 keywords=['anticoagulant therapy', 'treatment',
                                           'hypercoagulable state', 'thrombophilia', 
                                           'anticoagulation'])
hypercoag_articles, _ = hypercoag_filter.transform()


# #### Articles with Novel Treatments and Drug Trial Keywords

# In[ ]:


novel_therapy_filter = KeywordFilter(df=covid19, columns=['title_abstract'], 
                                     keywords=['novel therapeutics', 'clinical testing', 
                                               'treatment', 'clinical trial', 'medicine'])
novel_therapy_articles, _ = novel_therapy_filter.transform()


# ### 1.3.2 BM25 Ranking
# The keywords used above were also used with BM25 ranking. The articles in the subsets were ranked according to their relevance to these keywords. It can be seen in the two histograms below that a small number of articles hold the most relevance. Therefore, the top 125 most relevant articles were taken from each task subset. These top 125 articles approximately represent the 90th percentile of the ranked articles.

# *Click CODE to Reveal BM25 Ranking Code*

# In[ ]:


def ngram(wordlist, n):
    output = []
    for i in range(len(wordlist) - (n-1)):
        output.append(' '.join(wordlist[i:i+n]))
    return output

def clean(string):
    temp = re.sub(r'[^\w\s]', '', string)
    temp = re.sub(r'\b\d+\b', '', temp)
    temp = re.sub(r'\s+', ' ', temp)
    temp = re.sub(r'^\s', '', temp)
    temp = re.sub(r'\s$', '', temp)
    return temp.lower()

def bm25_rank(df, text_column, query):
    data = df.copy(deep=True)
    
    # Clean articles to train the BM25 model
    clean_text = data[text_column].apply(clean)
    tokens = clean_text.apply(word_tokenize)
    two_grams = tokens.apply(ngram, args=(2,))
    tok_articles_list = list(tokens + two_grams)

    # BM25 model
    ranker = bm25.BM25(tok_articles_list)
    ranks = ranker.get_scores(query)
    
    data['bm25_rank'] = ranks
    return data


# #### BM25 Ranking with Hypercoagulable State Keywords

# In[ ]:


hypercoag_articles = bm25_rank(hypercoag_articles, 
                               'title_abstract', 
                               ['anticoagulant therapy', 
                                'treatment', 
                                'hypercoagulable state', 
                                'thrombophilia', 
                                'anticoagulation'])

hypercoag_articles.hist(column='bm25_rank')
hypercoag = hypercoag_articles.sort_values(by='bm25_rank', ascending=False).head(125)


# In[ ]:


hypercoag.head()


# #### BM25 Ranking with Novel Treatments and Drug Trial Keywords

# In[ ]:


novel_therapy_articles = bm25_rank(novel_therapy_articles, 
                                   'title_abstract', 
                                   ['novel therapeutics', 
                                    'clinical testing', 
                                    'treatment', 
                                    'clinical trial', 
                                    'medicine'])

novel_therapy_articles.hist(column='bm25_rank')
novel_therapy = novel_therapy_articles.sort_values(by='bm25_rank', ascending=False).head(125)


# In[ ]:


novel_therapy.head()


# # 2. Finding Summary Table Features Using Regex and Webscraping

# ## 2.1 Finding Mentioned Treatments (Drugs)
# The different drugs mentioned in research papers were found and treated as possible treatment methods.

# ### 2.1.1 Web Scraping Drug Names
# To identify drugs mentioned in articles, a list of known drugs was needed. The [NIH website](https://druginfo.nlm.nih.gov/drugportal/drug/names) was scraped to obtain a list of drugs. The library BeautifulSoup4 was used to do this.

# In[ ]:


alphabet = list(string.ascii_lowercase)

links = []
for letter in alphabet:
    links.append(f'https://druginfo.nlm.nih.gov/drugportal/drug/names/{letter}')
    
pages = []
for link in links:
    page = r.get(link)
    pages.append(page.text)
    time.sleep(3)
    
drug_tables = []
for p in pages:    
    parser = BeautifulSoup(p, 'html.parser')
    tables = parser.find_all('table')
    drug_tables.append(tables[2])
    
drug_names = []
for table in drug_tables:
    a_refs = table.find_all('a')
    for a in a_refs:
        drug_names.append(a.string)
        
drug_names.append('oseltamivir') #adding other relevant drugs by hand
drug_names.append('lopinavir') #adding other relevant drugs by hand

nih_drug_names = pd.DataFrame({'drug_name':drug_names})
nih_drug_names.drug_name = nih_drug_names.drug_name.str.lower()
nih_drug_names.to_csv('nih_drug_names.csv', index=False)


# In[ ]:


nih_drug_names


# ### 2.1.2 Tagging Mentioned Drugs in Articles
# Now that a list of known drugs has been found, it can be used to search articles for mentions of drugs. A new column is created that contains all drugs mentioned by an article separated by semicolons.

# *Click CODE to Reveal Regex Feature Extraction Code*

# In[ ]:


drug_names = nih_drug_names.drug_name

def find_drugs(df):
    mentioned_drugs = []
    for article in df.iterrows():
        article_drugs = ''
        for drug in drug_names:
            if re.search(fr'([\W\b])({drug})([\b\W])', article[1]['title_abstract'].lower()) != None:
                article_drugs = article_drugs + drug + ';'
        if (len(article_drugs) > 0):
            article_drugs = article_drugs[:-1]
        mentioned_drugs.append(article_drugs)
    return mentioned_drugs


# In[ ]:


hypercoag_drugs = find_drugs(hypercoag)
hypercoag['mentioned_drugs'] = hypercoag_drugs
hypercoag.head()


# In[ ]:


novel_drugs = find_drugs(novel_therapy)
novel_therapy['mentioned_drugs'] = novel_drugs
novel_therapy.head()


# ## 2.2 Study Types
# A list of possible study types was found by looking through the titles of papers. The study types were split into adjectives and nouns. For example, the adjective 'observational' and the noun 'study' combine for 'observational study'. Another adjective could be added to make 'retrospective observational sutdy'. Combinations of words like this were searched for in the titles of papers.

# *Click CODE to Reveal Regex Feature Extraction Code*

# In[ ]:


def remove_stopwords(words):
    stop_words = stopwords.words('english')
    clean = [word for word in words if word not in stop_words]
    return clean

study_type_adjs = ['cohort', 'observational', 'clinical', 
              'randomized', 'open-label', 'control',
              'case', 'meta-analysis', 'systematic',
              'narrative', 'literature', 'critical',
              'retrospective', 'controlled', 'non-randomized',
              'secondary', 'rapid', 'double-blinded',
              'open-labelled', 'concurrent', 'pilot', 
              'empirical', 'retrospective', 'single-center',
              'collaborative', 'case-control']
study_type_nouns = ['study', 'trial', 'report', 'review',
                   'analysis', 'studies']

def find_study_types(df):
    study_types = []
    for article in df.iterrows():
        title = article[1].title.lower()
        clean_title = clean(title)
        title_tokens = word_tokenize(clean_title)
        title_tokens = remove_stopwords(title_tokens)
        study_type = ''
        study_type_name = False

        article_study_type = []
        for word in title_tokens:
            if word in study_type_adjs:
                study_type = study_type + word + ' '
                study_type_name = True

            elif word in study_type_nouns:
                study_type = study_type + word + ' '
                study_type_name = True

            if (study_type_name == False):
                study_type = ''

        study_types.append(study_type[:-1])

    return study_types


# In[ ]:


hypercoag_study_types = find_study_types(hypercoag)
hypercoag['study_types'] = hypercoag_study_types
hypercoag.head()


# In[ ]:


novel_study_types = find_study_types(novel_therapy)
novel_therapy['study_types'] = novel_study_types
novel_therapy.head()


# ## 2.3 Severity of Disease
# Four keywords associated with disease severtiy were found in the abstracts of papers. These keywords were; severe, critical, icu, and mild. Papers are tagged by which keywords they contain. Note: The prescence of severe in Severe Acute Respiratory Disease, was not counted in this search.

# *Click CODE to Reveal Regex Feature Extraction Code*

# In[ ]:


severity_types = ['severe', 'critical', 'icu', 'mild']

def find_severities(df):
    severity_of_disease = []
    for abstract in df.abstract:
        text = re.sub('Severe acute respiratory', 'sar', abstract)
        severities = []
        tokens = word_tokenize(text)
        for severity_type in severity_types:
            if severity_type in tokens:
                severities.append(severity_type)
        severity_of_disease.append(';'.join(severities))
    return severity_of_disease


# In[ ]:


hypercoag_severities = find_severities(hypercoag)
hypercoag['severity_of_disease'] = hypercoag_severities
hypercoag.head()


# In[ ]:


novel_severities = find_severities(novel_therapy)
novel_therapy['severity_of_disease'] = novel_severities
novel_therapy.head()


# ## 2.4 Number of Patients
# The abstracts of papers were searched for phrases that resembled '# patients', '#  icu patients', '# hospitalized patients', etc. These types of phrases were extracted and added to a new column.

# *Click CODE to Reveal Regex Feature Extraction Code*

# In[ ]:


page = r.get('https://www.ef.edu/english-resources/english-grammar/numbers-english/')
parser = BeautifulSoup(page.content, 'html.parser')
tables = parser.find_all('table')
cardinal_numbers = pd.read_html(str(tables[0]))[0].loc[0:37, "Cardinal"]

def find_sample_sizes(df):
    patient_num_list = []
    for abstract in df.abstract:
        matches = re.findall(r'(\s)([0-9,]+)(\s|\s[^0-9\s]+\s)(patients)', abstract)
        num_patients = ''
        for match in matches:
            num_patients = num_patients + ''.join(match[1:]) + ';'
        for number in cardinal_numbers:
            cardinal_regex_search = re.search(fr'({number})(\s)(patients)', abstract)
            if cardinal_regex_search != None:
                num_patients = num_patients + cardinal_regex_search[0] + ';'
        num_patients = num_patients[:-1]
        patient_num_list.append(num_patients)
    return patient_num_list


# In[ ]:


hypercoag['sample_size'] = find_sample_sizes(hypercoag)
hypercoag.head()


# In[ ]:


novel_therapy['sample_size'] = find_sample_sizes(novel_therapy)
novel_therapy.head()


# ## 2.5 Conclusion Excerpts
# From the work above, each article is tagged with the drugs that it mentioned along with other details. In this section, excerpts from the conclusion section are found which contain any mentioned drug. For each drug mentioned in a paper, a row is created in the summary table. If no excerpts are found, no summary row is created.

# *Click CODE to Reveal Regex Feature Extraction Code*

# In[ ]:


def discussion_excerpts(df):
    results_table = pd.DataFrame(columns=['date','study', 'study_link', 'journal', 
                        'study_type', 'therapeutic_method', 'sample_size', 
                        'severity_of_disease', 'conclusion_excerpt', 'primary_endpoints', 
                        'clinical_imporovement', 'added_on'])
    for article in df.iterrows():
        
        # Code here looks for the mention of drugs
        drugs_with_abbrv = []
        drugs = article[1].mentioned_drugs.split(sep=',')
        for i in range(len(drugs)):
            abbrv=''
            # This searches for the mentioned drug names that are 
            # followed by abbreviations that a contained in parethesis.
            # Ex. drug (xy-z)
            if (re.search(fr'({drugs[i]})(\s)(\()([a-z-]+)(\))', article[1].full_text.lower())) != None:
                source = re.search(fr'({drugs[i]})(\s)(\()([a-z-]+)(\))', article[1].full_text.lower())
                abbrv = source[4]
            drugs_with_abbrv.append([drugs[i], abbrv])

        sentences = sent_tokenize(article[1].results_discussion)
        for drug in drugs_with_abbrv:
            excerpts = ''
            for sentence in sentences:
                # If the drug name has an abbreviation, sentences 
                # in the results/discussion that contain it are added
                # to a list of excerpts.
                if drug[1] != '':
                    if (re.search(fr'([\W\b])({drug[0]})([\b\W])', sentence.lower()) != None) |                         (re.search(fr'([\W\b])({drug[1]})([\b\W])', sentence.lower()) != None):
                        excerpts += sentence + '\n'
                # If the drug does not have an abbreviation, then
                # only the drug name is searched for.
                if drug[1] == '':
                    if (re.search(fr'([\W\b])({drug[0]})([\b\W])', sentence.lower()) != None):
                        excerpts += sentence + '\n'
            if excerpts != '':
                drug_row = {'date':[article[1].publish_time],'study':[article[1].title], 'study_link':[article[1].url], 
                            'journal':[article[1].journal], 'study_type':[article[1].study_types], 'therapeutic_method':[f'{drug[0]}'], 
                            'sample_size':[''], 'severity_of_disease':[article[1].severity_of_disease], 'conclusion_excerpt':[excerpts], 
                            'primary_endpoints':[''], 'clinical_imporovement':[''], 'added_on':['']}
                results_table = pd.concat([results_table, pd.DataFrame(drug_row)], axis = 0)
                                
    return results_table


# ### What are the best methods to combat the hypercoagulable state associated with COVID-19?

# In[ ]:


hypercoag = hypercoag[hypercoag.mentioned_drugs != '']
results_table_hypercoag = discussion_excerpts(hypercoag)
results_table_hypercoag.reset_index(drop=True).head()


# In[ ]:


results_table_hypercoag.to_csv('results_table_hypercoag.csv', index=False)


# ### What is the efficacy of novel therapeutics being tested currently?

# In[ ]:


novel_therapy = novel_therapy[novel_therapy.mentioned_drugs != '']
results_table_novel = discussion_excerpts(novel_therapy)
results_table_novel.reset_index(drop=True).head()


# In[ ]:


results_table_novel.to_csv('results_table_novel_therapy.csv', index=False)


# # 3 Interactive Dashboard
# 
# Many of the summary table features can be found automatically using regex. However, some of the the features require insight from a researcher or subject matter expert. These being the *Primary Endpoints* of an article and an indication of *Clinical Improvement (Yes or No)*. An interactive dashboard is a potential method to complete the target tables. An example was made using Plotly's Dash library. This example dashboard allows a researcher to fill in the missing features, add to or change existing features, and export their results to a CSV file.
# 
# The code for the dashboard below is included in the input data under **cv19-r2-plotly-dash/dash_app.py.**

# In[ ]:


from IPython.display import Image
Image('../input/cv19-r2-dash/cv19_dash_pt1.png')


# In[ ]:


Image('../input/cv19-r2-dash/cv19_dash_pt2.png')

