#!/usr/bin/env python
# coding: utf-8

# # Risk Factors Analysis

# # Introduction
# 
# A part of **CoronaWhy Team**, this notebook is used to produce the most relevant research papers for the various risk factors which are associated with Coronavirus. 
# 
# - It is a part of main [CoronaWhy Task Risk Notebook](https://www.kaggle.com/arturkiulian/coronawhy-org-task-risk-factors)
# - Visit our [website](https://www.coronawhy.org) to learn more.
# - Read our [story](https://medium.com/@arturkiulian/im-an-ai-researcher-and-here-s-how-i-fight-corona-1e0aa8f3e714).
# - Visit our [main notebook](https://www.kaggle.com/arturkiulian/coronawhy-org-global-collaboration-join-slack) for historical context on how this community started.
# 

# # Inputs:
# 
# *All of the data used is public to the world and present in Kaggle.*
# 
# - metadata : [allen-institute-for-ai/CORD-19-research-challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
# - dataset v6 : [skylord/coronawhy](https://www.kaggle.com/skylord/coronawhy)
# - dataset v7 : [skylord/coronawhy](https://www.kaggle.com/skylord/coronawhy)

# # Setup

# ## Download packages

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install pandas==1.0.3\n!pip install spacy_langdetect\n!pip install nltk\n!pip install scispacy\n!pip install kaggle\n!python -m spacy download en_core_web_lg')


# ## Download data

# Due to storage limitations and accessing only the required data from various datasets available in Kaggle, we used the Kaggle tokens to download the specific datatet files, i.e. for accessing latest metadata (v7).

# In[ ]:


# Downloading all datasets

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_key = user_secrets.get_secret("key")

f = open("kaggle.json", "w")
f.write('{"username":"pranjalya","key":"'+secret_key+'"}')
f.close()

get_ipython().system('mkdir ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')
get_ipython().system('kaggle datasets download allen-institute-for-ai/CORD-19-research-challenge -f metadata.csv')
get_ipython().system('unzip metadata.csv.zip')
get_ipython().system('rm -rf metadata.csv.zip')

PATH = '../input/coronawhy/v6_text/v6_text/'


# ## Import packages

# In[ ]:


# Importing standard libraries
import os
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
import warnings
warnings.filterwarnings('ignore')


# # Helper functions
# - `find_ngrams`: Given a dataframe, a column to search and some keywords, extract the parts of the dataframe that contain **all keywords**
# - `keywordcounter`: Given a list of sentences and keywords, return the count of keywords contained in each sentence.
# - `aggregation`: Given the sentences of a paper, compose a dataframe following the standardized output for results (ToDo := Add this format)

# In[ ]:


def find_ngrams(dataframe, columnToSearch, keywords):
    '''
    Input : Complete Dataframe, Column to search keywords in, Keywords to search for
    Returns : Reduced dataframe which contains those keywords in given column
    '''
    df_w_ngrams = dataframe[dataframe[columnToSearch].str.contains('|'.join(keywords), case=False) == True]
    return df_w_ngrams

def keywordcounter(sentences, keywords_list):
    '''
    Input : List of sentences, List of keywords
    Returns : Keywords present in sentences, Total count of all keywords present in Input
    '''
    keyword = {}
    sent = " ".join(sentences)
    for pol in keywords_list:
        counter = sent.lower().count(pol)
        if (counter > 0):
            keyword[pol] = counter
    return list(keyword.keys()), sum(keyword.values())

def aggregation(item, keyWordList, RiskFactor):
    '''
    Input : Dataframe of sentences of a paper
    Return : Datframe in Standard Output format
    '''
    dfo = {}
    
    dfo['Risk Factor'] = RiskFactor
    dfo['Title'] = item['title'].iloc[0]
    dfo['Keyword/Ngram'], dfo['No of keyword occurence in Paper'] = keywordcounter(item['sentence'].tolist() + [item['abstract'].iloc[0]], keyWordList)
    dfo['paper_id'] = item['paper_id'].iloc[0]
    
    if (pd.isnull(item['url'].iloc[0])==False):
        dfo['URL'] = item['url'].iloc[0]
    else:
        dfo['URL'] = ''

    dfo['Sentences'] = item[item['section']=='results']['sentence'].tolist()
    
    if (item['authors'].iloc[0].isnull().any()==False):
        dfo['Authors'] = item['authors'].iloc[0].tolist()
    else:
         dfo['Authors'] = ''
        
    dfo['Correlation'] = item['causality_type'].iloc[0]
    dfo['Design Methodology'] = item['methodology'].iloc[0]
    dfo['Sample Size'] = item['sample_size'].iloc[0]
    dfo['Coronavirus'] = item['coronavirus'].iloc[0]
    dfo['Fatality'] = item['fatality'].iloc[0]
    dfo['TAXON'] =item['TAXON'].iloc[0]
    
    return dfo


# # NLP Time
# 

# ## Feature Engineering: Methodology, Sample Size, Correlation, Outcome
# We manually engineered some features, they can be extracted from a dataframe using `extract_features` function

# In[ ]:


def extract_features(ngramDf, allSentdataFrame):
    # extracting methodology
    methods_list = ['regression','OLS','logistic','time series','model','modelling','simulation','forecast','forecasting']
    methodology = find_ngrams(allSentdataFrame, 'sentence', methods_list)

    # extracting sample size
    sample_size_list = ['population size','sample size','number of samples','number of observations', 'number of subjects']
    sample_size = find_ngrams(allSentdataFrame, 'sentence', sample_size_list)

    # extracting nature of correlation
    causal_list =['statistically significant','statistical significance','correlation','positively correlated','negatively correlated','correlated','p value','p-value','chi square','chi-square','confidence interval','CI','odds ratio','OR','coefficient']

    causality_type = find_ngrams(allSentdataFrame, 'sentence', causal_list)

    # extracting coronavirus related sentence
    coronavirus_list = ['severe acute respiratory syndrome','sars-cov','sars-like','middle east respiratory syndrome','mers-cov','mers-like','covid-19','sars-cov-2','2019-ncov','sars-2','sarscov-2','novel coronavirus','corona virus','coronaviruses','sars','mers','covid19','covid 19']

    coronavirus = find_ngrams(allSentdataFrame, 'sentence', coronavirus_list)

    # extracting outcome
    disease_stage_list = ['lethal', 'morbid',"death", "fatality", "mortality","lethal", "lethality", "morbidity"]

    fatality = find_ngrams(allSentdataFrame, 'sentence', disease_stage_list)

    df_list = [methodology,sample_size,causality_type,coronavirus,fatality]
    df_list_name = ['methodology','sample_size','causality_type','coronavirus','fatality']
    i=0
    for one_df in df_list:
        one_df.rename(columns={'sentence':df_list_name[i]},inplace=True)
        grouped_one_df = one_df.groupby('paper_id')[df_list_name[i]].sum()
        ngramDf = pd.merge(ngramDf,grouped_one_df,on='paper_id',how='left')
        i=i+1
    return ngramDf


# ## Pipelining function

# We made a pipeline so we can plug in the risk factor and the dataframe, and it returns a dataframe with the relevant papers for that specific risk. This function `get_relevant_papers` acts as a pipelining function.

# In[ ]:


def get_relevant_papers(df, ngrams, risk):
    '''
    Input : 
        df -> Dataframe containing sentences from papers with metadata
        ngrams -> Dictionary with keys as Risk Factors
        risk -> The risk factor to be searched
    Returns : Dataframe containing papers in Output format
    '''
    
    # Extracting relevant papers in seperate dataframes from Result section and Abstract
    result_df = find_ngrams(df, 'sentence', ngrams[risk])
    abstract_df = find_ngrams(df, 'abstract', ngrams[risk])

    print("There are {} sentences containing keywords/ngrams in Result section for {}".format(len(result_df), risk))
    print("There are {} sentences containing keywords/ngrams in Abstract section for {}.".format(len(abstract_df), risk))

    
    # Merging the result section and abstract sentences into single dataframe
    df_r = pd.concat([result_df, abstract_df])
    df_r = df_r.loc[df_r.astype(str).drop_duplicates().index]

    print("Total unique papers in Result section : {}".format(result_df['paper_id'].nunique()))
    print("Total unique papers in Abstract section : {}".format(abstract_df['paper_id'].nunique()))
    print("Total unique papers in total : {}".format(df_r['paper_id'].nunique()))

    
    # Getting all sentences from papers containing the keywords in result section for feature extraction
    df_body_all_sentence = pd.merge(df[['paper_id','sentence']], result_df['paper_id'], on='paper_id', how='right')
    df_body_all_sentence.rename(columns={'sentence_x':'all_sentences','sentence_y':'ngram_sentence'}, inplace=True)

    df_abstract_all_sentence = pd.merge(df[['paper_id','abstract']], abstract_df['paper_id'], on='paper_id', how='right')
    df_abstract_all_sentence.rename(columns={'abstract_x':'all_sentences','abstract_y':'ngram_sentence'}, inplace=True)

    
    # Merging these sentences in single dataframe
    df_all_sentences = pd.concat([df_body_all_sentence, df_abstract_all_sentence])
    df_all_sentences = df_all_sentences.loc[df_all_sentences.astype(str).drop_duplicates().index]

    print("Total unique papers in combined section : {}".format(df_all_sentences['paper_id'].nunique()))

    
    # Extracting features from these sentences
    df_real = extract_features(df_r, df_all_sentences)
    df_real = df_real[['paper_id','language', 'section', 'sentence', 'lemma', 'UMLS', 'sentence_id', 'publish_time', 'authors', 'methodology','sample_size', 'causality_type','coronavirus','fatality','title','abstract','publish_time','authors', 'url', 'TAXON']]


    # Preparing the output in format
    grouped = df_real.groupby('paper_id')
    df_output = pd.DataFrame(columns=['Risk Factor', 'Title','Keyword/Ngram', 'No of keyword occurence in Paper', 'paper_id', 'URL', 'Sentences', 'Authors', 'Correlation', 'Design Methodology', 'Sample Size','Coronavirus','Fatality','TAXON'])

    for key, item in grouped:
        df_output = pd.concat([df_output, pd.DataFrame([aggregation(item, ngrams[risk], risk)])])

    df_output = df_output.reset_index()

    print("There are {} papers for Risk Factor : {}\n\n".format(len(df_output), risk))

    # Cleaning some memory
    del df_output['index']
    del df_r
    del df_real
    del df_all_sentences

    return df_output


# ## Feature Engineering: Risk factor n-grams list

# Also, we identified various risk factors and crafted a list of most relevant n-grams for each one:
# - `pollution` = ["air pollution and", "indoor air pollutants", "indoor air pollution", "household air pollution", "air pollution is", "between air pollution", "of air pollution", "particulate air pollution", "pollution and the", "air pollutant data"]
# - `population density`: ["population density","number of people in","number of people per",
#   "highly populated areas","highly populated countries",
#   "densely populated countries","densely populated areas",
#   "high density areas","high density countries"
#   ,"population densities", "density of population","sparsely populated",
#   "densely populated","density of the population","dense population",
#   "populated areas", "densely inhabited","housing density",
#   "densely-populated","concentration of people", "population pressure",
#   "population studies","populated regions","populous",
#   "high population densities","residential densities","overpopulated"]
# - `humidity` = ["humidity","monsoon","rainy","vapour","rainfall"]
# - `air temperature` = ['heated climate','cold temperatures','hot weather','cold weather', 'tropical climate', 'tropical weather', 'temperate', 'tropic', 'sunlight', 'summer', 'winter', 'spring', 'autumn', 'weather', 'in the season of', 'climate', 'local temperature']
# - `heart/lungs diseases`: ['congestive heart failure', 'complete atrioventricular block', 'myocardial ischemia', 'rheumatic heart disease', 'junctional premature complex', 'pulmonary heart disease', 'myocardial disease', 'sick sinus syndrome', 'hypertensive disorder', 'cardiac arrhythmia', 'supraventricular tachycardia','heart disease', 'cardiac arrest', 'supraventricular premature beats', 'ventricular premature complex', 'endocardial fibroelastosis', 'primary pulmonary hypertension', 'mitral valve regurgitation', 'heart failure', 'hypertensive renal disease', 'pulmonic valve stenosis', 'left heart failure', 'primary dilated cardiomyopathy', 'ischemic chest pain']
# - `age`: ['persons older than', 'patients older than', 'patients not younger', 'patients not younger', 'above 65 years',
# 'over 65 years', '65 years old', 'over 65 years','above 60 years', 'over 60 years', '60 years old', 
# 'over 60 years','above 70 years', 'over 70 years', '70 years old', 'over 70 years',
# 'among the elderly', 'among the aged','60 years and over', '65 years and over',
# 'aging population', 'older age group','circa 65 years']
# 

# In[ ]:


# Risk Factors list

riskfactors = ['pollution', 'population density', 'humidity', 'age', 'temperature', 'heart risks']


# In[ ]:


# N-grams list for each risk factor as dictionary

ngrams = {}

ngrams['population density'] = ['high density areas','high density countries', 'population densities', 'density of population', 'sparsely populated','densely populated', 'density of the population','dense population', 'populated areas','densely inhabited','housing density', 'densely-populated','concentration of people','population pressure','population studies','populated regions', 'populous','high population densities','residential densities']

ngrams['pollution'] = ['air pollution and', 'indoor air pollutants', 'indoor air pollution', 'household air pollution', 'air pollution is', 'between air pollution', 'of air pollution', 'particulate air pollution', 'pollution and the', 'air pollutant data', 'water pollution']

ngrams['humidity'] = ['humidity','monsoon','rainy','vapour','rainfall']

ngrams['heart risks'] = ['congestive heart failure', 'complete atrioventricular block','myocardial ischemia', 'rheumatic heart disease', 'junctional premature complex','pulmonary heart disease', 'myocardial disease', 'sick sinus syndrome', 'hypertensive disorder', 'cardiac arrhythmia', 'supraventricular tachycardia','heart disease', 'cardiac arrest', 'supraventricular premature beats','ventricular premature complex', 'endocardial fibroelastosis','primary pulmonary hypertension', 'mitral valve regurgitation', 'heart failure', 'hypertensive renal disease', 'pulmonic valve stenosis', 'left heart failure', 'primary dilated cardiomyopathy', 'ischemic chest pain']
                      
ngrams['temperature'] = ['heated climate', 'cold temperatures', 'hot weather', 'cold weather', 'tropical climate', 'tropical weather', 'temperate','tropic','sunlight', 'summer','winter','spring','autumn','weather','in the season of','climate', 'local temperature']

ngrams['age'] =  ['persons older than', 'patients older than', 'patients not younger', 'patients not younger', 'above 65 years', 'over 65 years', '65 years old', 'over 65 years','above 60 years', 'over 60 years', '60 years old', 'over 60 years','above 70 years', 'over 70 years', '70 years old', 'over 70 years', 'among the elderly', 'among the aged','60 years and over', '65 years and over', 'aging population', 'older age group','circa 65 years']


# ## Load data from articles
# 
# To gather the relevant papers we will search in: abstracts, and result sections, as we've found these contain the most relevant keywords. Abstracts are retrieved from the metadata file. The rest come from a mix of dataset v6 and v7.

# In[ ]:


# Load metadata and keep only the required columns

metadata = pd.read_csv('metadata.csv')
metadata.rename(columns={'sha':'paper_id'}, inplace = True)
metadata = metadata[['paper_id', 'title', 'abstract', 'publish_time', 'authors', 'url']]
metadata['paper_id'] = metadata['paper_id'].astype("str")
metadata['title'] = metadata['title'].fillna('')
metadata['abstract'] = metadata['abstract'].fillna('')


# In[ ]:


# Then : Loading result section

firstpass = True

for pkl in os.listdir(PATH):
    df = pd.read_pickle(PATH+pkl, compression='gzip')
    if(firstpass):
        v7_data = pd.read_json('../input/coronawhy/v7_text.json')
        df_result = pd.concat([v7_data[v7_data['section']=='results'], df[df['section']=='results']])
        firstpass = False
    else:
        df_result = pd.concat([df_result, df[df['section']=='results']])

df_result['paper_id'] = df_result['paper_id'].astype("str")

print("No of unique papers in result section : ", df_result['paper_id'].nunique(), " out of ", len(df_result), " rows in dataframe")
print("There are metadata present for ", metadata['paper_id'].nunique(), " papers.")


# ### Enrich with metadata

# We merge the metadata and the sentences from result section, to enrich the details. In this process, we also filter out all the papers for which no metadata is available.

# In[ ]:


df = df_result.merge(metadata, how='inner', on='paper_id')

df['paper_id'] = df['paper_id'].astype("str")
df['title'] = df['title'].fillna('')
df['abstract'] = df['abstract'].fillna('')

print("There are ", df['paper_id'].nunique(), " papers available in both metadata and papers extracted.")


# In[ ]:


# Helper code to search for keywords in a better manner

rx = r"\.(?=\D)"
df['sentence'] = df['sentence'].str.replace(rx,' . ')
df['sentence'] = df['sentence'].str.replace(',',' , ')
df['abstract'] = df['abstract'].str.replace(rx,' . ')
df['abstract'] = df['abstract'].str.replace(',',' , ')


# ## Prepare output
# 

# Now we extract the papers for each risk and store it in *relevant_papers* dictionary, after that, we sort the papers list for each factor based on whether any Coronavirus terms are mentioned, and after that, how many occurences of keywords took place in result section and abstract combined.

# In[ ]:


relevant_papers = {}

for risk in riskfactors:
    relevant_papers[risk] = get_relevant_papers(df, ngrams, risk)
    relevant_papers[risk] = relevant_papers[risk].sort_values(['Coronavirus', 'No of keyword occurence in Paper'], ascending=[False, False]).reset_index()
    del relevant_papers[risk]['index']
    relevant_papers[risk].to_csv('{}.csv'.format(risk))


# ## Printing the top 10 papers (before annotation) for each risk factor

# In[ ]:


relevant_papers['age'].head(10)


# In[ ]:


relevant_papers['pollution'].head(10)


# In[ ]:


relevant_papers['population density'].head(10)


# In[ ]:


relevant_papers['humidity'].head(10)


# In[ ]:


relevant_papers['heart risks'].head(10)


# In[ ]:


relevant_papers['temperature'].head(10)


# # Acknowledgements:
# 
# We gently acknowledge the work of these persons, it would have never been possible without their hard work and effort:
# 
# [Artur Kiulian](https://www.linkedin.com/in/artur-kiulian/)
# - Helped with initial task exploration, risk factor related knowledge base and notebook structure for submission.
# 
# [Mayya Lihovodov](https://www.linkedin.com/in/mayya-lihovodov-6b47b279?originalSubdomain=il)
# - Risk Task Lead. Helped to organize a general flow of work process and find an approach.  
# 
# [Iason Konstantinidis](https://www.linkedin.com/in/konstantinidis/)
# - Helped with project management and setup of development process
# 
# [Kriti Mahajan](https://www.linkedin.com/in/kriti-mahajan-174101b9/)
# - Helped with extracting output for 3 risk factors: population density,temperature and humidity
# - Exploratory data analysis
# - Topic modelling
# - Extraction methodology, correlation, statistical significance , sample size , fatal outcomes and strains of coronaviruses from papers
# 
# [Robbie Edwards](https://www.linkedin.com/in/robbie-edwards/)
# - Created LDA notebook and Bigram/Trigram notebook
# - Doing whatever Queen Mayya needs
# 
# Vijay Daita
# - Helped with checking data twitter.com/voojbot
# 
# [Michael Wang](https://www.linkedin.com/in/michaelhechengwang)
# - Created processing notebook that finds papers matching risk factor ngrams, combining metadata for Kaggle submission
# 
# [Ansun Sujoe](https://www.linkedin.com/in/ansun-sujoe-9b752b157/)
# - Helped come up with custom keywords to input to N-grams models
# - Created a flowchart to help organize aspects of risk factors and people responsible
# 
# [Pranjalya Tiwari](https://www.linkedin.com/in/pranjalya-tiwari-456a7a179)
# - Paper extraction by sections and sentences, and Abstracts and Titles
# - Generalised code for feeding extracted N-grams keywords and getting output in format
# - GitHub code maintenance
# 
# [Lukasz Gagala](https://www.linkedin.com/in/%C5%82ukasz-g%C4%85ga%C5%82a-8847b88a/)
# - General purpose search engine, semantic search
# - Text summerisation tool
# - LDA for UMSL terms
# - Paragraph extraction and encoding
# 
# [Kevin Li](https://www.linkedin.com/in/kevin-li-70843b86/)
# - Created notebook for bigram/trigram extraction for heart disease and cross validation to ICD/CUI codes
# 
# [Samtha Reddy](https://www.linkedin.com/in/samthareddy83/)
# - Extracted sentences for papers claiming that age above 60 (65, 70?)  is a risk factor with high amount of complications and deaths" using NLP
# 
# [Guillermo Blanco](https://www.linkedin.com/in/guillermo-blanco-354a6a51/)
# - Help with organization in github
# - Help formalizing i/o for each subtask in risk-factors
# - Add bi/trigrams code to github
# - Preparation of submission notebook
# 
# [Mohammad Tanweer](https://www.linkedin.com/in/mohammadtanweer/)
# - Helped with filtering documents for risk factor smoking
# - Generated bigrams and trigrams for the filtered documents
# - Generated topics using LDA for the filtered documents
# 
# [Andrew Wood](https://www.linkedin.com/in/andrew-wood-87910921)
# - Heart n-gram creation
# - Age n-gram updatation
# 
# [Brandon Eychaner](https://www.linkedin.com/in/haroldeychaner/)
# - Text Preprocessing, cleaning, lemmatization
# - Named Entity Recognition, Biomedical terminology normalization
# 
# [Mike Honey](https://www.linkedin.com/in/mikehoney/)
# - Data Visualization
# 
# [Mandlenkosi Ngwenya](https://www.linkedin.com/in/mandla-ngwenya-47b60027)
# - Helped create consolidated dataset
# - Created a Doc2Vec model for finding related comorbidities
# 

# In[ ]:




