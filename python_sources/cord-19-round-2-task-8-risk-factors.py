#!/usr/bin/env python
# coding: utf-8

# # Extracting information on covid-19 risk factors and constructing benchmark data for building machine learning models

# ## Summary Tables of Studies Addressing 24 Different Risk Factors
# 
# We developed a pipeline and a set of approaches to create the summary tables as specified in task 8 using the [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) last updated on June 09. Here we are displaying our manually curated summary tables using the results of our pipeline:

# In[ ]:


# Importing Python Packages and Tools
import pandas as pd
import nltk
from tqdm import tqdm
import os
from IPython.display import display
from ipywidgets import widgets
from ipywidgets import interact, interactive

# Reading summary tables
summarytable_dir = '../input/summarytables/'
sum_table = {}
for file in os.listdir(summarytable_dir):
    fname = file.strip('csv').strip('.')
    full_file = summarytable_dir + file    
    sum_table[fname] = pd.read_csv(full_file)
sum_table.keys()


# In[ ]:


# Displaying summary tables
pd.set_option('display.max_rows', 100)
for fname, df in sum_table.items():
    print(fname,' : ' + str(df.shape[0]) +' rows/row')
    display(df.head())


# # Introduction
# 
# The COVID-19 pandemic has caused nearly 8 million confirmed infected patients and more than 430 thousand deathes worldwide. In response to the pandemic, the [CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) including scholarly articles related to COVID-19 was created for global research community to generate helpful insight for the ongoing combat against this infectious disease using state-of-the-art text mining, NLP and other AI technologies. The [CORD-19 competition](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) was organized by Kaggle as a call for actions to develop tools and information for answering scientific questions with high priority. [Round \#2](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/150921) of the [CORD-19 challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) asked participants to create summary tables with specific structures derived by expert curators.
# 
# To tackle the challenges, we have organized a collaborative team including scientists from [Insilicom Inc.](https://insilicom.com/) and the department of statistics of Florida State University. Insilicom specializes in providing innovative technologies to help scientists effectively use Big Data to accelerate their research and development efforts. It recently developed the [Biomedical Knowledge Discovery Engine (BioKDE)](https://biokde.com/), a deep-learning powered search engine for biomedical literature. 
# 
# Our information extraction pipeline consists of the following components. 
# First, based on the [CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge), we developed a dataset which is clean and annotated with different entities such as genes, disease, chemicals, etc. using [PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/index.html), [BeFree](http://ibi.imim.es/befree/) and [scispacy](https://allenai.github.io/scispacy/) annotated entities; Second, we used extended keywords to query articles relevant to a paticular topic; Third, we used synonyms to further increase the coverage of the retrieved relevant articles; Fourth, we used regular expressions to extract specific information for filling certain columns of the tables; Fifth, we parsed the relevant sentences to obtain typed dependency graphs, which were used to compute the shortest pathes between relevant keywords, such as chemical names and COVID-19 related terms. The shortest pathes are used to further curate the relevant sentences. They will also be used in the future for building predictive models for the corresponding information extraction tasks; Finally, we have manually verified the summary tables before submitting to obtain a manually curated dataset, which can be used in future studies as benchmark data. These manually curated data can also be used to build machine learning models. 
# 
# **Task 8** is to [create summary tables that address risk factor studies related to COVID-19](http://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=888). To be more specific, the goal is to create summary tables report about:
# * Hypertension
# * Diabetes
# * Male gender
# * Heart Disease
# * COPD
# * Smoking Status
# * Age
# * Cerebrovascular disease
# * Cardio- and cerebrovascular disease
# * Cancer
# * Respiratory system diseases
# * Chronic kidney disease
# * Chronic respiratory diseases
# * Drinking
# * Overweight or obese
# * Chronic liver disease
# * Asthma
# * Chronic digestive disorder
# * Dementia
# * Endocrine diseases
# * Heart Failure
# * Immune system disorder
# * Race Black vs White
# * Ethnicity Hispanic vs non-Hispacni
# 

# #### This notebook was organized in the following structure:
# 
# 1. Developing a Cleaned Dataset with Annotated Entity Types
#    - Processing the [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
#    - Entity Annotation
#    - Aggregation and Indexing of CORD-19 Articles
# 2. Creating the Summary Tables
#    - Retrieving COVID-19 Related Articles
#    - Keywords for Retrieving Articles Related to the Summary Table
#    - Extracting relevant infromation
#    - Generating the Summary Tables
# 3. Building machine learning models to answer specific questions
#    - Sentence Parsing
#    - Getting the shortest paths for risk factors

# # Developing a Cleaned Dataset with Annotated Entity Types
# ## Processing the CORD-19 Dataset
# To process the [CORD-19 Dataset](http://https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge), we verified the ids of doi, pmid and pmcid of each article and organized all the articles in a consistent format. For articles with pmids and/or pmcids, the pmids and pmcids will be used for getting entities annotations from PubTator. Each article was stored in a JSON file after the above pre-processing. The codes for extracting ids and article pre-processing can be found as listed:
# 
# code for getting ids
# code for article process.
# - [code for getting ids](https://www.kaggle.com/gdsttian/preprocess-get-ids)
# - [code for article process](https://www.kaggle.com/gdsttian/preprocess-cord-data).

# In[ ]:


# !python preprocess_get_ids.py
# !python preprocess_cord_data.py


# ## Entity Annotation
# 
# Entity annotation is very helpful for extracting relevant information. [PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/) provides annotations of biomedical concepts in PubMed abstracts and PMC full-text articles. Using the [PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/) API, we acquired annotations for the articles in the [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) when they are available. For those without pre-calculated annotations, we used the [PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/) web interface to retrieve the annotations. All [PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/) annotations were then parsed and organized in a consistent format for each article. The entities annotated by [PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/) include:
# 
# - Genes
# - Diseases
# - Chemicals
# - Species
# - Mutation
# - Cellline
# 
# Beside [PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/) annotations, we also used [BeFree](http://ibi.imim.es/befree/) and [scispacy](https://allenai.github.io/scispacy/) to annotate additional entities. [BeFree](http://ibi.imim.es/befree/) annotates entities of **genes** and **diseases** (a python package needs to be installed from the [BeFree repo](https://bitbucket.org/nmonath/befree/src/master/)). In order to use [BeFree](http://ibi.imim.es/befree/) in our pipeline, we modified a function of the package, which can be found [here](https://www.kaggle.com/gdsttian/befree-ner-covid19).
# 
# [scispacy](https://allenai.github.io/scispacy/) includes different models for biomedical concept annotation, among which two were used in our pipeline. The two models and the entities annotated by each model are listed as follows:
# 
# - en_ner_craft_md: genes, taxonomies, sequence ontologies, chemicals, gene ontologies and cellline
# - en_ner_jnlpba_md: DNA, cell type, cellline, RNA and protein
# 
# All annotations were combined into a final set of annotations. When annotations by different tools overlap with each other, we selected the annotations with the largest span. When different tools annotate entities at the same span, we gave priority to [PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/).
# 
# All codes for annotations are available through the following links:
# 
# - [code for acquiring existing PubTator annotations](https://www.kaggle.com/gdsttian/entities-get-pubtator-annotation)
# - [code for posting titles abd abstracts to PubTator for annotations](https://www.kaggle.com/gdsttian/entities-post-tiabs-to-pubtator)
# - [code for retrieving completed title and abstract annotations from PubTator](https://www.kaggle.com/gdsttian/entities-retrieve-tiabs-from-pubtator)
# - [code for parsing PubTator annotations](https://www.kaggle.com/gdsttian/entities-process-pubtator-annotation)
# - [code for adding BeFree and scispacy annotations](https://www.kaggle.com/gdsttian/entities-additional-annotation)esearch/pubtator/) include:
# 
# 

# In[ ]:


# !pip install git+https://bitbucket.org/nmonath/befree.git
# !python entities_get_pubtator_annotation.py
# !python entities_post_tiabs_to_pubtator.py
# !python entities_retrieve_tiabs_from_pubtator.py
# !python entities_process_pubtator_annotation.py
# !python entities_additional_annotation.py


# ## Aggregation and Indexing of CORD-19 Articles
# 
# With all the CORD-19 articles processed and the annotations combined, they were aggregated into a single JSON file. For each article, the entities identified were summarized and relations between entities were extracted if they co-occur in the same sentence. 
# 
# Query of relevant articles plays an important role in creating the summary tables. In order to retrieve target articles, we created indices of articles by publication time and keywords in titles and abstracts. We used [spaCy](https://spacy.io/) for tokenization of titles and abstracts. Articles returned from the query were ranked by the counts of the keywords occuring in the articles.
# 
# All codes for data aggregation and indexing were given as follows:
# 
# - [code for data aggregation](https://www.kaggle.com/gdsttian/data-aggregation)
# - [code for entities summary and relation building](https://www.kaggle.com/gdsttian/data-nodes-relations)
# - [code for index by time](https://www.kaggle.com/gdsttian/data-indexing-time)
# - [code for index by words](https://www.kaggle.com/gdsttian/data-indexing-word)

# In[ ]:


# !python data_aggregation.py
# !python data_nodes_relations.py
# !python data_indexing_time.py
# !python data_indexing_word.py


# # Creating the Summary Tables
# 
# After the clean annotated dataset was created, we can start the query and information extraction process.

# In[ ]:


# data pathes
data_path = '/kaggle/input/cord-19-data-with-tagged-named-entities/data' # folder for system data
json_path = '/kaggle/input/cord-19-data-with-tagged-named-entities/data/json_files/json_files' # path of final json files
mapping_pnid = 'mapping_corduid2nid.json' # dictionary mapping cord_uid to numeric id for each paper

index_year = 'index_time_year.json' # dictionary of list of papers for each publish year
index_title = 'index_word_title.json' # dictionary of list of papers for each word in title
index_abstract = 'index_word_abstract.json' # dictionary of list of papers for each word in abstract
word_counts = 'paper_word_counts.json' # word counts by paper
index_table = 'index_word_table.json'
paper_tables = 'paper_tables.json'

entity_lists = 'entity_lists.json' # entity checking lists including disease list, blacklist etc.
entity_nodes = 'entity_nodes.json' # entities dictionary
entity_relations = 'entity_relations.json' # entity relation dictionary

mapping_sents = 'mapping_sents2nid.json' # mapping sent id to numeric id
index_sents = 'index_word_sents.json' # mapping word to a list of numeric sent id
sentences = 'sentences.json' # dictionary of all sentences with unique id


# We import the python packages and the tools we developed for the process. These tools can be used for data loading, article query and display. Codes of the tools can be accessed by the following links:
# 
# - [code for utility tools](https://www.kaggle.com/gdsttian/utils)
# - [code for search tools](https://www.kaggle.com/gdsttian/mining-search-tool)

# In[ ]:


# packages
from utils import *
from mining_search_tool import *
csv_path = 'csv'
if not os.path.exists(csv_path): os.makedirs(csv_path)


# ## Loading the Data

# In[ ]:


# load dataset for search and display
papers = SearchPapers(data_path, json_path, mapping_pnid, index_year,
                      index_title, index_abstract, word_counts, index_table, paper_tables,
                      entity_lists, entity_nodes, entity_relations, index_sents, mapping_sents, sentences)


# ## Retrieving COVID-19 Related Articles
# 
# Information for the summary tables need to be extracted from the articles relevant to COVID-19. These articles were queried using a list of keywords. We defined the list of extended keywords based on those used by the [PMC COVID-19 Initiative](https://www.ncbi.nlm.nih.gov/pmc/about/covid-19/) and manual reading of some relevant articles.
# 
# As most of the articles associated with COVID-19 were published in 2020, we limited the publication time to year 2020.
# 
# There are more than 35 thousand articles identified as relevant to COVID-19 in the [CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) updated on June 9.
# 
# One of the articles was displayed as follows.

# In[ ]:


covid19_names = """covid-19, covid19, covid, sars-cov-2, sars-cov2, sarscov2,
                   novel coronavirus, 2019-ncov, 2019ncov, wuhan coronavirus
                """
papers_covid19 = papers.search_papers(covid19_names, section = None, publish_year = '2020')
print(f"{'Total papers related to COVID-19:':20}{len(papers_covid19):6}")


# In[ ]:


papers.display_papers(papers_covid19[:1])


# ## Keywords for Retrieving Articles Related to the Summary Table
# 
# ### Defining extended keyword lists for risk factors, severity terms and mortality terms
# 

# In[ ]:


# Defining extended keyword quries for risk factors
Age_query = """ age, ages, infant, infancy, child, children, adolescent, adolescents, young, youth, old, olds, elderly, senior, pediatric, middle-age, aging, senescence """
Asthma_query = """ asthma, asthma attack, bronchial asthma, allergy, allergic asthma """
CKD_query = """ ckd, chronic kidney, chronic kidney disease, chronic renal disease """
COPD_query =  """ copd, chronic obstructive pulmonary disease """
Cancer_query = """ malignant neoplastic disease, cancer, tumor, carcinogenesis, melanoma, leukemia, benign, terminal """
CardioCerebrovascular_query = """  cardio cerebrovascular disease, vascular disease, cerebrovascular disease, cardiovascular disease, hypercholesterolemia, CVD """
Cerebrovascular_query = """ cerebrovascular, stroke, ischemic, hemorragic """
ChronicDigestiveDisorder_query = """ chronic digestive,  digestion, absorption, celiac disease, ibs, irretable bowel syndrome """
ChronicLiverDisease_query = """ chronic liver,  chronic liver disease, cirrhosis """
ChronicRespiratoryDisease_query = """ chronic respiratory, chronic respiratory disease """
Dementia_query = """ dementia, alzheimer's disease """
Diabete_query = """ diabetes, diebete, insulin resistance, prediabetes, diabetic, diabetic complications, blood glucose, fasting blood glucose, insulin sensitivity, hyperglycemia """
Drinking_query = """ alcohol, alcohol intake, alcoholic, alcoholic drinks, alcoholic beverage, alcoholic consumption, intoxicant, inebriant, binge drinking"""
Endocrine_query = """ hormone, endocrine, endocrine gland, endocrine, endocrinal, endocrinal disorder """
HeartDisease_query = """ heart, heart disease, chd,  coronary heart disease, arrhythmia, atherosclerosis, ischemia, angina """
HeartFailure_query = """ heart failure"""
Hispani_query = """ spanish american, hispanic american,hispanic, latino """
Hypertension_query = """high blood pressure, hypertension, metabolic syndrome, blood pressure """
Immune_query = """ immune system, immune system disorder, autoimmune disease, autoimmne thyroiditis, inflammation, inflammatory, gout, arthritis """
Male_query = """ male, man, gender, female, sex """
Obese_query =  """ bmi, heavy, obese, obesity, body mass index, fat, overweight, abdominal obesity, pear shape, apple shape """
Race_black_query = """ black, white, african american, afro-american, race, caucasian, caucasoid race, negroid race """
RespiratorySystemDisease_query = """ respiratory system, respiratory system disease, pneumonia """
Smoking_query = """ smoke,smoking, tobacco """


# In[ ]:


# Collecting the query name as a list for future usage.
query_list =  ['Age_query' ,'Asthma_query', 'CKD_query','COPD_query','Cancer_query' ,'CardioCerebrovascular_query' ,'Cerebrovascular_query',
               'ChronicDigestiveDisorder_query','ChronicLiverDisease_query','ChronicRespiratoryDisease_query','Dementia_query','Diabete_query' ,
               'Drinking_query','Endocrine_query','HeartDisease_query','HeartFailure_query','Hispani_query','Hypertension_query',
               'Immune_query' ,'Male_query','Obese_query' ,'Race_black_query','RespiratorySystemDisease_query','Smoking_query']


# In[ ]:


# Defining extended keyword list for risk factors
syn_list = []
for query_name in query_list:
    syn_name = query_name.strip('query')+'syn'
    syn_list.append(syn_name)    
    globals()[syn_name] = [sent.strip() for sent in globals()[query_name].split(',')]


# In[ ]:


# Defining extended keyword lists for severity terms and mortality terms
severe_syn = ['mild', 'moderate', 'severe', 'varied', 'critical', 'icu', 'non-icu','positive','positive testing','hospitalization','hospitalized']
fatality_syn = ['fatality','mortality','mortalities','death','deaths','dead','casualty']
combined_syn = severe_syn + fatality_syn


# ## Extracting relevant information
# We will first take 'age' as an example, and later generalize to other risk factors

# In[ ]:


Age_query


# In[ ]:


# Getting papers related with 'age', searching over titles and abstracts, published in 2020
riskfactor_papers = papers.search_papers(Age_query, section = 'tiabs', publish_year = '2020')
print('There are ' + str(len(riskfactor_papers)) + ' papers in the dataset that are related to age.')
# Get the subset of Covid-19 related papers
riskfactor_papers = list(set(riskfactor_papers) & set(papers_covid19))
print('Among them, ' + str(len(riskfactor_papers)) + ' papers are related to COVID-19.')


# With 'age' related papers, we select those papers with patterns indicating numerical outcome of the studies, for example, containing 'OR', 'AOR', 'HR', 'AHR', 'RR', and 'RH'. We call this as ratio pattern.

# In[ ]:


ratios = ['OR', 'AOR', 'HR', 'AHR', 'RR', 'RH', 'odds ratio', 'hazard ratio', 'relative ratio','odds']
# The following pattern is looking for pattern: ( ratio keywords + numbers )
extract_pattern = '|'.join([f'\([^()]*\\b{ratio}\\b\s?[=:]?\s?\d+\.\d+.*?\)' for ratio in ratios])


# Defining a function called get_odds to select papers with ratio pattern. Click code to see the code.

# In[ ]:


# Search over full text to get papers with pattern ( ratios such as 'OR', 'AOR', 'HR', 'AHR', 'RR', 'RH' )
# and save these to a dictionary
def get_odds(riskfactor_papers):
    or_riskf = {}
    for paper_id in riskfactor_papers:
        paper = papers.get_paper(str(paper_id))
        date = paper['publish_time']
        study = paper['title']['text']
        study_link = paper['url']
        journal = paper['journal']
        doi = paper['doi']
        cord_uid = paper['cord_uid']
        pmc_link = paper['pmcid']
        abstract = paper['abstract']['text']

        if pmc_link != '':
            pmc_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_link}/"

        if abstract != '':
            rate = re.findall(extract_pattern, abstract)
            if rate != []:
                # if any word in combined synonyms list is in abstract, then we store the paper
                if (any(word in nltk.word_tokenize(abstract.lower()) for word in combined_syn)):
                    or_riskf[f'{str(paper_id)}|a|0|0'] = {'Date': date,
                                                          'Study': study,
                                                          'Study Link': study_link,
                                                          'Journal': journal,                                                        
                                                          'DOI': doi,
                                                          'CORD_UID': cord_uid,
                                                          'PMC_Link': pmc_link,
                                                          'Abstract': abstract,                                                         
                                                          'Text':abstract,
                                                          'Ratio':rate}                
#                 print(rate)
        bodytext = paper['body_text']
        if bodytext != []:
            for sec_id, section in enumerate(bodytext):
                for txt_id, text in enumerate(section['text']):
                    rate = re.findall(extract_pattern, text)
                    if rate != []:
                        # if any word in combined synonyms list is in body text, then we store the paper
                         if (any(word in nltk.word_tokenize(abstract.lower()) for word in combined_syn)):
                            or_riskf[f'{str(paper_id)}|b|{sec_id}|{txt_id}'] = {'Date': date,
                                                                                'Study': study,
                                                                                'Study Link': study_link,
                                                                                'Journal': journal,                                                        
                                                                                'DOI': doi,
                                                                                'CORD_UID': cord_uid,
                                                                                'PMC_Link': pmc_link,
                                                                                'Abstract': abstract,                                                         
                                                                                'Text':text,
                                                                                'Ratio':rate}
#                         print(rate)
    return or_riskf


# In[ ]:


# Transfering dictionary to a dataframe
riskfactor_dic = get_odds(riskfactor_papers)
for k,v in riskfactor_dic.items():
    df = pd.DataFrame.from_dict(riskfactor_dic,orient='index')
df = df.reset_index().rename(columns={'index':'Ratio_ID'})
df.head(2)


# In the above dataframe, the column 'Text' stores the paragraph with a matching pattern; The column 'Ratio' stores all the numerical results of the corresponding ratio pattern.

# ### Extracting sentences with ratio pattern from 'Text' column and add a new column called Sentence
# 
# Following regex will find pattern like   ( ratio keywords  (any word) (any word) ) or ( ratio keywords ) 

# In[ ]:


keep_pattern = '|'.join([f'(\([^\(\)]*\\b{ratio}\\b([^\(\)]*\(\w+\)[^\(\)]*)*\))|(\([^\(\)]*\\b{ratio}\\b[^\(\)]*\))' for ratio in ratios])


# In[ ]:


# If there are more than one sentences matching the pattern, then store it in another column (Sentence2)
df['Sentence'] = ''; df['Sentence2'] = ''; 
for idx,text in df.Text.items():
    sents = nltk.sent_tokenize(text)  
    for sent in sents:
        if (re.findall(keep_pattern,sent,re.I)):
            if any(word in nltk.word_tokenize(sent.lower()) for word in Age_syn) and any(word in sent.lower() for word in combined_syn):
                if df['Sentence'][idx] == '':
                    df['Sentence'][idx] += sent + ' '   
                else:
                    df['Sentence2'][idx] += sent + ' '
                    
# Combining column 'Setence' and 'Sentence2'
df1 = df.loc[:,'Ratio_ID':'Sentence']
df2 = df.loc[:,'Ratio_ID':'Sentence2'].drop(columns='Sentence')
df2 = df2.rename(columns = {'Sentence2':'Sentence'})
dfn = pd.concat([df1,df2])
dfn = dfn[dfn['Sentence'] != '']
dfn = dfn.sort_values('Study')
dfn = dfn.drop_duplicates(subset = 'Sentence',ignore_index =True)
dfn.shape


# In[ ]:


dfn.head(2)


# In the dataframe defined above, we store the sentences with the ratio pattern in the column 'Sentence'. Note that these setences are all from the column 'Text'. Addtional work can be done to decide if the paper is truly relevant to the target table based on the column 'Sentence'.

# ### Running the pipeline for all the other risk factors

# In[ ]:


# Here we use df_dic to store the dataframe.
df_dic = {}
for riskfactor_query in query_list:
    riskfactor_name = riskfactor_query.strip('query').strip('_')
    riskfactor_query = globals()[riskfactor_query]
    riskfactor_papers = papers.search_papers(riskfactor_query, section = 'tiabs', publish_year = '2020')
    riskfactor_papers = list(set(riskfactor_papers) & set(papers_covid19))
    print('-------------------------------------------------------------------------------------------')
    print('There are '+str(len(riskfactor_papers)) + ' Covid-19 papers in the dataset related to ' + riskfactor_name )      
    
    # Transfer dictionary to a dataframe
    riskfactor_dic = get_odds(riskfactor_papers)
    for k,v in riskfactor_dic.items():
        df = pd.DataFrame.from_dict(riskfactor_dic,orient='index')
    df = df.reset_index().rename(columns={'index':'Ratio_ID'})
    
    # Adding column 'Sentence' to store the sentences with pattern ( ratio keywords )
    df['Sentence'] = ''; df['Sentence2'] = ''; 
    for idx,text in df.Text.items():
        sents = nltk.sent_tokenize(text)  
        for sent in sents:
            if (re.findall(keep_pattern,sent,re.I)):
                if any(word in nltk.word_tokenize(sent.lower()) for word in Age_syn) and any(word in sent.lower() for word in combined_syn):
                    if df['Sentence'][idx] == '':
                        df['Sentence'][idx] += sent + ' '   
                    else:
                        df['Sentence2'][idx] += sent + ' '
                        
    # Modifying the dataframe                    
    df1 = df.loc[:,'Ratio_ID':'Sentence']
    df2 = df.loc[:,'Ratio_ID':'Sentence2'].drop(columns='Sentence')
    df2 = df2.rename(columns = {'Sentence2':'Sentence'})
    dfn = pd.concat([df1,df2])
    dfn = dfn[dfn['Sentence'] != '']
    dfn = dfn.sort_values('Study')
    dfn = dfn.drop_duplicates(subset = 'Sentence',ignore_index =True)
    
    print('The shape of the dataframe of ' + riskfactor_name + ' is :')
    print(dfn.shape)
    
    # Save dataframes as csv files
#     dfn.to_csv('csv/' + riskfactor_name + '.csv')
    
    # Save dataframes in a dictionary
    df_dic[riskfactor_name] = dfn


# To this point, we have dataframes with information needed to fill the summary table for different risk factors.

# In[ ]:


# Get dataframe by risk factor name
df_dic['Dementia']


# ## Generating Summary Tables

# In[ ]:


# Study type extend keywords
sys_review = ['systematic review', 'meta-analysis',
              'search: pubmed, pmc, medline, embase, google scholar, pptodate, web of science',
              'searched: pubmed, pmc, medline, embase, google scholar, uptodate, web of science',
              'in: pubmed, pmc, medline, embase, google scholar, uptodate, web of science']
retro_study = ['record review','retrospective', 'observational cohort', 'scoping review']
simulation = ['modelling','model','molecular docking','modeling','immunoinformatics', 'simulation', 'in silico', 'in vitro']


# In[ ]:


# Regex for extracting sample size
ss_patient = re.compile(r'(\s)([0-9,]+)(\s|\s[^0-9,\.\s]+\s|\s[^0-9,\.\s]+\s[^0-9,\.\s]+\s)(patients|persons|cases|subjects|records)')
ss_review = re.compile(r'(\s)([0-9,]+)(\s|\s[^0-9,\.\s]+\s|\s[^0-9,\.\s]+\s[^0-9,\.\s]+\s)(studies|papers|articles|publications|reports|records)')


# In[ ]:


df_dic.keys()


# In[ ]:


s_table = {}
for name,dfs in df_dic.items():
    dfs['Severity of Disease'] = ''
    dfs['Fatality'] = ''
    dfs['Study Type'] = ''
    dfs['Sample Size'] = ''      
    
    for idx,row in dfs.iterrows():  
    
        abstract = row['Abstract']
        sentence = row['Sentence']
        ratio = row['Ratio']
        
        # Filling Study Type
        for pharase in sys_review:
            if(pharase in abstract):
                dfs.loc[idx,'Study Type'] = 'Systematic Review'
        for pharase in retro_study:
            if(pharase in abstract):
                dfs.loc[idx,'Study Type'] = 'Retrospective Study'
        for pharase in simulation:
            if(pharase in abstract):
                dfs.loc[idx,'Study Type'] = 'Simulation'

        #Filling Sample Size
        study_type = dfs.loc[idx,'Study Type']
        sample_size = ''
        if study_type == 'Systematic Review':
            matches = re.findall(ss_review, abstract)
            for match in matches:
                if match[1].isdigit() and int(match[1]) != 2019:
                    dfs.loc[idx, 'Sample Size'] = sample_size + ''.join(match[1:]) + '; '
        elif study_type == 'Retrospective Study' or study_type == 'Other' :
            matches = re.findall(ss_patient, abstract)
            for match in matches:
                if match[1].isdigit() and int(match[1]) != 2019:
                    dfs.loc[idx, 'Sample Size'] = sample_size + ''.join(match[1:]) + '; '

        # Filling Ratios
        if (any(word in nltk.word_tokenize(sentence.lower()) for word in severe_syn )):
            dfs.loc[idx,'Severity of Disease'] = ratio
        elif (any(word in nltk.word_tokenize(sentence.lower()) for word in fatality_syn )):
            dfs.loc[idx,'Fatality'] = ratio

    cols = ['Date','Study','Study Link','Journal','Study Type','Severity of Disease','Fatality','Sample Size','DOI','CORD_UID']
    dfw = dfs[cols]
    s_table[name] = dfw
    display(name + ' : ' + str(dfw.shape[0]) + ' rows/row' )
    display(dfw.head())    


# In[ ]:


# Saving risk factor dataframes into csv files
for fname, df in s_table.items():
    df.to_csv(f'{fname}.csv')


# # Building machine learning models to answer specific questions

# Filling some columns of the summary table can be considered as answering certain questions. For example, whether a sentence should be an entry in the table can be determined by answering the following questions: 1. Does the sentence describe the risk factor relationship beween the risk factor term and the disease term (or severity/fatality)? 2. Does the sentence contain numerical result for such relationship? 3. Is a tagged number in the sentence the numerical result for such relationship? If the answers to the above three questions are true, then we can fill some important columns of the summary table. To answer these questions, the current question-answering systems do not work well because the answers are highly specific to the questions.
# 
# To tackle such problem, we formulated it as building machine learning models to perform classification tasks. Our current pipeline can extract sentences with both risk factor keywords (i.e. age, hypertension, etc.) and disease keywords (COVID-19, severity, mortality, etc.), which also contain numerical results in the ratio pattern. Such sentences are stored in the Sentence column in the generated dataframes. If we manually annotate the sentences based on the questions we have proposed above, then we can build machine learning models to extract the specific information required by the summary table.
# 
# We manually verified the summary tables generated by our pipeline for the following purposes:
#     1. The manually verified information can be more valuable for scientists studying COVID-19;
#     2. The manually verified data can be used as benchmark data for evaluating future studies on this type of problems;
#     3. The manually verified data can be used to build machine learning models to develop better information extraction systems.

# At the moment, we are still working on generating high quality labelled data. Once the labelled training data are available, we will build the model and update our notebook.

# ## Sentence parsing
# 
# To build the machine learning models, since the training sample size will be relatively small, we will perform sentence parsing to generate more informative input for the machine learning model. One type of information we extract from sentence result is the shortest pathes between risk factor terms and disease terms. Other information can also be extracted and used in the model building process. We will first try some pre-trained deep learning models. Sentence parsing was performed using Stanford Parser. After parsing the sentence on local machine, we stored the files as json files.

# #### Loading the json files

# In[ ]:


import json
import csv
risk_factor_json_dir = '../input/riskfactorjson/'


# In[ ]:


# Storing the files in a list and a dictionary
file_dic = {}
for fname in os.listdir(risk_factor_json_dir):
    filename = fname.strip('json').strip('.')
    full_fname = risk_factor_json_dir + fname    
    files = json.load(open(full_fname, 'rb'))    
    file_dic[filename] = files


# In[ ]:


# Get the file from the dictionary
file_dic['Age']['0']


# ## Getting the Shortest Paths for Risk Factors
# 
# ### Finding the shortest pathes in the typed dependency graph between risk factor terms and disease/severity/fatality terms

# In[ ]:


# Defining Graph data structure
from collections import defaultdict
class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        self.dep sotres the dependency between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 'nsubj'}
        """
        self.edges = defaultdict(list)
        self.weights = {}
        self.dep = {}
    
    def add_edge(self, from_node, to_node, weight, dep):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight
        self.dep[(from_node, to_node)] = dep
        self.dep[(to_node, from_node)] = dep

# Defining dijsktra to get the shortest path from initial node to target node        
def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

# Switching the order of a tuple
def switch(x):
    return(x[1],x[0])

# Getting the intersection of two lists
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

# Getting the shortest path from initial node to target node, with the dependecies
def shortest_path(file,from_node,to_node):
    
    print(file['text'])
    edges = file['edges']
    edge_list = []
    for idx,edge in edges.items():
        edge_list.append((edge['target'],edge['source'],1,edge['dep']))
        
    g = Graph()
    for edge in edge_list:
        g.add_edge(*edge)
    
    path = dijsktra(g, from_node, to_node)
    
    tpl = []
    for i in range(len(path) - 1):
        value = tuple(path[i:i+2])
        tpl.append(value)

    new_path = []
    for chunk in tpl:
        if chunk in g.dep.keys():
            if chunk[0] not in new_path:
                new_path.append(chunk[0])
            new_path.append(g.dep[chunk])
            new_path.append(chunk[1])
        elif switch(chunk) in g.dep.keys():
            if switch(chunk)[1] not in new_path:
                new_path.append(switch(chunk)[1])
            new_path.append(g.dep[switch(chunk)])
            new_path.append(switch(chunk)[0])
    
    return new_path


# ### Let's take risk factor 'age' as an example: finding the shortest path for 'age' to severity/fatality terms.

# In[ ]:


npaths = []
for i in range(len(files)):
    file = files[str(i)]
    sent_list = nltk.word_tokenize(file['text'].lower())
    if intersection(sent_list, Age_syn):
        age_related =intersection(sent_list, Age_syn)[0]
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')  
#     print(intersection(sent_list, combined_syn))
   
    for severe_related in intersection(sent_list, combined_syn):
        print(age_related,severe_related)
        path = shortest_path(file,age_related,severe_related)        
        print(path)
        print('----------------------')
        npaths.append(path) 


# In[ ]:


npaths


# ### Getting the shortest paths for all the risk factors

# In[ ]:


file_dic.keys()


# In[ ]:


# short_path dictionary: take file name as a key and list of path as values
# gen_dic dictionary: take touple (from_node, to_node, file index) as a key and path as a value; 
#                     file index is to distinguish those same (from_node, to_node) 
short_path = {} ; gen_dic ={}
for riskfactor_query in query_list:
    
    fname = riskfactor_query.strip('query').strip('_')
    files = file_dic[fname]    
    fname_syn = globals()[fname + '_syn']
    
    npaths = []
    for i in range(len(files)):
        file = files[str(i)]
        sent_list = nltk.word_tokenize(file['text'].lower())
        
        if intersection(sent_list, fname_syn):
            fname_related = intersection(sent_list, fname_syn)[0] 
            
            if (intersection(sent_list, combined_syn)):
                print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*') 
#                 print(intersection(sent_list, combined_syn))
                
                for severe_related in intersection(sent_list, combined_syn):
                    tpl = (fname_related,severe_related, str(i))
                    path = shortest_path(file,fname_related,severe_related) 
                    if path:
                        print(path)
                        npaths.append(path) 
                        gen_dic[tpl] = [path, file['text']]

                            
    short_path[fname] = npaths


# short_path.keys()

# In[ ]:


short_path['Asthma']


# In[ ]:


# Saving gen_dic into a csv file
import csv
with open('shortest_path.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Sentence','Nodes','Shortest Path'])
    for key,value in gen_dic.items():
        writer.writerow([value[1], key, value[0]])

