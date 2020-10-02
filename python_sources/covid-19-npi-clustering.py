#!/usr/bin/env python
# coding: utf-8

# # **COVID-19 NPI Clustering**
# You can find the complete version of our visualization tool here https://kaggle-covid-npi.herokuapp.com/

# # **Table of Contents**
# 
# 1. [Introduction](#introduction)
# 1. [Goals](#goals)
# 1. [Approach](#approach)
# 1. [Load the Data](#load-data)
#     * [Load Metadata](#load-metadata)
#     * [Load JSON Papers](#load-jsondata)
#     * [Slice by Keywords](#first-slice)
# 1. [Text Filtering](#text-filtering)
#     * [Remove Stop Words](#remove-stopwords)
#     * [TF-IDF Vectorization](#tfidf)
#     * [Tokenize Title](#tokens)
# 1. [NPI Prediction](#npi-predict)
#     * [Label NPI](#label-npi)
#     * [PCA](#pca)
#     * [XG Boost Model](#xgboost)
#     * [Slice NPI](#slice-npi)
# 1. [Clustering](#clustering)
# 1. [Geolocation](#geolocation)
# 1. [Visualization](#visualization)
# 1. [Conclusion](#conclusion)
# 1. [Citation/Sources](#citation)

# <a id="introduction"></a>
# # **Introduction**
# **Coronavirus Disease 2019 (COVID-19)** was first identified in December 2019 in Wuhan, the capital of China's Hubei province and has since then spread to be identified as a Global Pandemic.
# 
# ### *What is COVID-19*:
# COVID-19 is a novel coronavirus which expresses itself in humans as a respiratory illness.
# 
# ### *How does it spread:*
# According to the World Health Organization as of 15/04/2020 "This disease can spread from person to person through small droplets from the nose or mouth which are spread when a person with COVID-19 coughs or exhales. These droplets land on objects and surfaces around the person. Other people then catch COVID-19 by touching these objects or surfaces, then touching their eyes, nose or mouth. People can also catch COVID-19 if they breathe in droplets from a person with COVID-19 who coughs out or exhales droplets."
# 
# The question that is on everyones mind is **"How could we have prevented the spread?"**
# We hope to allow the research to answer this.
# 
# ### *What can we do to stop it*:
# *For the purpose of this challenge we will be focusing on Non-pharmaceutical Interventions in preventing the spread of COVID-19.*
# 
# As per the CDC: **Non-pharmaceutical Interventions(NPIs)** are actions, apart from getting vaccinated and taking medicine, that people and communities can take to help slow the spread of illnesses like pandemic influenza (flu).
# 
# NPI's are typically broken up into 3 categories listed with examples:
# 
#     1. Personal: 
#         -Staying home when you are sick
#         -Covering coughs and sneezes with a tissue
#         -Washing hands with soap and water or using hand sanitizer when soap and water is not available
#     2. Community:
#         -Social Distancing
#         -Closures (child care centers, schools, places of worship, sporting events, etc)
#     3. Environmental:
#         -Routine surface cleaning 
# 
# The goal of our submission is to analyze the research done on the implementation, efficacy and scaling of NPI's, isolate these papers from other COVID-19 research articles and derive meaningful insights to help us mitigate the existing situaiton and prepare our systems for any future pandemic which may come our way.
# 
# <a id="goals"></a>
# # Goals:
# The goal of our submission is to analyze the research done on the implementation, efficacy and scaling of NPI's, isolate these papers from other COVID-19 research articles and derive meaningful insights to help us mitigate the existing situation and prepare our systems for any future pandemic which may come our way.
# 
# We aimed to create a tool to allow interested parties to visualize and investigate patterns in published data, in this case focusing on non-pharmaceutical interventions to address pandemics. This tool can be adapted to other subject areas by adapting the training of the classifier and keywords used. Through a series of clustering methods, we hope to provide users a more focused means of asking questions of large datasets, and understanding relationships between publications. Ultimately, researchers should be able to learn from the vast amounts of papers being published, not get lost in the flood, or have their work go unused.
# 
# <a id="approach"></a>
# # Approach:
# To tackle this challenge, we created a classifier to front-load the work of sorting NPI vs. non-NPI articles. This aided us in refining our dataset in preparation for cleaning and standardization, as well as improved the outcomes of our clustering model. As a tool which needs to be trusted in crisis situations, we embedded a human-in-the-loop mechanism, in the form of label definition for the classifier. By using a supervised and human-driven method up front, we can train the model to ensure that only relevant articles are being aggregated. Clustering can then begin, with the goal of placing cluster centers as far apart as possible. We then allow a second step of human guidance by enabling users to investigate the components and metadata of clusters, so they understand what they are looking at on both micro and macro levels. Users can further refine model output by setting cluster parameters, as described below.
# 

# <a id="load-data"></a>
# # **Load the Data** 

# <a id="load-metadata"></a>
# ### *Load Metadata*
# The code to load the metadata is based on the notebook by Ivan Ega Pratama, from Kaggle.
# #### Source: [COVID EDA: Initial Exploration Tool](https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
import pickle


# In[ ]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', -1)


# In[ ]:


# Load metadata from Kaggle
root_path = '/kaggle/input/CORD-19-research-challenge'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df = meta_df.drop_duplicates().reset_index(drop=True)


# The title, abstract and author columns have characters that are not needs for this analysis and the remove of such characters (e.g. :, &, -) is imperative to run the models in this notebook

# In[ ]:


# Clean numbers and special characters
regex_arr = ['\d','\t','\r','\)','\(','\/', ':', ';', '&', '#', '-', '\.']
def clean_numbers_and_special_characters (df, col):
    meta_df[col] = meta_df[col].replace(regex=regex_arr, value='')
    meta_df[col] = meta_df[col].str.strip()
    meta_df[col] = meta_df[col].str.lower()
    return meta_df    

meta_df = clean_numbers_and_special_characters(meta_df, 'title')
meta_df = clean_numbers_and_special_characters(meta_df, 'abstract')
meta_df = clean_numbers_and_special_characters(meta_df, 'authors')


# <a id="load-jsondata"></a>
# ### *Load JSON Papers*
# The code to fetch all the json data and load into a dataframe is loosely based on the same notebook by Ivan Ega Pratama. JSON files were not only parsed but were merged with the metadata to create a dataframe to be used for the rest of this notebook.

# In[ ]:


# Obtain file paths for all .json files
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)


# In[ ]:


# Remove .xml.json files because most of the these files have duplicate information compared to the .json files
all_json = [x for x in all_json if 'xml.json' not in x ]
len(all_json)


# In[ ]:


# Method is needed to group doi into a list to avoid duplicate rows for title, abstract, and body_text
def agg_doi(df):
    no_doi = list(df.copy())
    no_doi.remove('doi')
    df_doi = df.groupby(no_doi)['doi'].apply(list).reset_index(name = "doi")
    return df_doi


# In[ ]:


class FileReader:
    def __init__(self, all_json):
        self.all_json = all_json
    
    def extract_all_json(self, meta_df):
        meta_df = meta_df.fillna('')
        sha_meta = agg_doi(meta_df[['sha', 'doi']])
        pmc_meta = agg_doi(meta_df[['pmcid', 'doi']])
        json_dict = []
        all_json_div = len(all_json) // 10
        for file_path in all_json:
            with open(file_path) as file:
                if len(json_dict) % all_json_div == 0:
                    print(f'{len(json_dict)} of {len(all_json)} json files processed')
                data = json.load(file)
                paper_id = data['paper_id']
                title = data['metadata']['title'].lower().strip()
                author_list = self._extract_name_list(data)
                body_text = self._extract_json_text(data, 'body_text')
                abstract= self._extract_json_text(data, 'abstract')
                doi = []
                if len(meta_df[meta_df.sha == paper_id]):
                    doi = sha_meta[sha_meta.sha == paper_id]['doi'].tolist()[0]
                    if not title:
                        if len(meta_df[(meta_df.sha == paper_id) & (~meta_df.abstract.isna())]):
                            title = meta_df[(meta_df.sha == paper_id) & (~meta_df.title.isna())]['title'].tolist()[0]
                    if not abstract:
                        if len(meta_df[(meta_df.sha == paper_id) & (~meta_df.abstract.isna())]):
                            abstract = meta_df[(meta_df.sha == paper_id) & (~meta_df.abstract.isna())]['abstract'].tolist()[0]
                elif len(meta_df[meta_df.pmcid == paper_id]):
                    doi = pmc_meta[pmc_meta.pmcid == paper_id]['doi'].tolist()[0]
                    if not title:
                        if len(meta_df[(meta_df.pmcid == paper_id) & (~meta_df.title.isna())]):
                            title = meta_df[(meta_df.pmcid == paper_id) & (~meta_df.title.isna())]['title'].tolist()[0]
                    if not abstract:
                        if len(meta_df[(meta_df.pmcid == paper_id) & (~meta_df.title.isna())]):
                            abstract = meta_df[(meta_df.pmcid == paper_id) & (~meta_df.title.isna())]['abstract'].tolist()[0]
                json_dict.append({'paper_id': paper_id,
                                  'title': title,
                                  'author_list': author_list,
                                  'abstract': abstract,
                                  'body_text': body_text,
                                  'doi': doi})
        return pd.DataFrame(json_dict)

    def _extract_name_list(self, data):
        name_list = []
        for a in data['metadata']['authors']:
            first = a['first']
            middle = ''
            if a['middle']:
                middle = a['middle'][0] + ' '
            last = a['last']
            name_list.append(f'{first} {middle}{last}')
        return name_list
    
    def _extract_json_text(self, data, key):
        return ' '.join(str(item['text']).lower().strip() for item in data[key])

files = FileReader(all_json)
df_covid = files.extract_all_json(meta_df)


# <a id="first-slice"></a>
# ### *Slice by Keywords*
# Through online research and interviewing healthcare providers, we generated a list of keywords to train a classifier to identify NPI-related papers. This first pass of data slicing was performed on paper abstracts, as titles were not consistently populated, and body text contained a lower signal to noise ratio.

# In[ ]:


keywords = ['incident command system',
            'emergency operations',
            'joint information center',
            'social distancing',
            'childcare closers',
            'travel advisory',
            'travel warning',
            'isolation',
            'quarantine',
            'mass gathering cancellations',
            'school closures',
            'facility closures'
            'evacuation',
            'relocation',
            'restricting travel',
            'travel ban',
            'patient cohort',
            'npi']


# In[ ]:


all_json_df = df_covid[df_covid['abstract'].str.contains('|'.join(keywords), na=False, regex=True)].reset_index(drop=True)


# In[ ]:


len(all_json_df)


# In[ ]:


all_json_df.to_pickle('data.pkl')


# The code above is to pickle the data for use in an external notebook. As json parsing may take a long time, this notebook can be started from this point on with old data.

# In[ ]:


# Run this cell to read a previously created pickle file instead of re-running the script
# all_json_df = pd.read_pickle('/kaggle/working/data.pkl')


# <a id="text-filtering"></a>
# # **Text Filtering**

# In[ ]:


import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import re


# <a id="remove-stopwords"></a>
# ### *Remove Stop Words*
# Finding and removing stopwords is vital to removing noise as these words most likely will not contribute to the summary, meaning, scope of the text.

# In[ ]:


def remove_punc(df, columns):
    for col in columns:
        df[col] = df[col].str.replace('[^a-zA-Z\s]+','')
    return df


# In[ ]:


def remove_stopwords(df, columns):
    stop = stopwords.words('english')
    for col in columns:
        df[col] = df[col].astype(str).apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return df


# In[ ]:


all_json_df = remove_punc(all_json_df, ['body_text','abstract'])
all_json_df = remove_stopwords(all_json_df, ['body_text','abstract'])


# <a id="tfidf"></a>
# ### *TF-IDF Vectorization*
# Text data cannot be used in a model as is, so this data needs to be converted. We will use TF-IDF to convert the text into a measure of the relative importance of the words in the text.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


def to_tfidf(df, columns):
    for col in columns:
        tfidfv = TfidfVectorizer()
        df[col + '_tfidf'] = list(tfidfv.fit_transform(df[col]).toarray())
    return df


# In[ ]:


all_json_df = to_tfidf(all_json_df, ['body_text','abstract'])


# <a id="tokens"></a>
# ### *Tokenize Title*
# Let's create a list of the top keywords from each paper. The tokenize method below will do just that.

# In[ ]:


def tokenize(row):
    title_tokens = []
    title = row['title']
    if title == title:
        title = re.sub('(/|\|:|&|#|-|\.)', '', title)
        tokens = word_tokenize(title)
        remove_sw = [word for word in tokens if word not in stopwords.words('english')]
        remove_numbers = [word for word in remove_sw if not word.isnumeric()]
        remove_comas = [word for word in remove_numbers if not word in [',', '(', ')', '"', ':', '``', '.', '?']]
        title_tokens.extend(remove_comas)
    return [value[0] for value in Counter(title_tokens).most_common()[0:30]]


# In[ ]:


all_json_df['tokens'] = all_json_df.apply(tokenize, axis=1)


# <a id="npi-predict"></a>
# # **NPI Prediction**
# In this section, a copy of `all_json_df` is created to be used to predict NPI. With the help of several people, 1300+ papers were identified as NPI, non-NPI or NULL. We will merge this dataset with the copied dataframe to model NPI. Thereafter, the data from all_json_df will be used to predict NPI based on that model.

# In[ ]:


model_df = all_json_df.copy()


# <a id="label-npi"></a>
# ### *Label NPI*

# In[ ]:


LABELED_FILE = '/kaggle/input/labeled-data/labeled_npi.csv'
df_labels = pd.read_csv(LABELED_FILE)


# In[ ]:


model_df = model_df.merge(df_labels, on="title", how="inner")
model_df = model_df.loc[model_df['isNPI'].notna()]


# <a id="pca"></a>
# ### *PCA*
# For the vectorized data, we will use Principle Component Analysis (PCA) to reduce the dimensions of the data.

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[ ]:


def pca_apply(df, columns, n_comp):
    new_df = df.copy()
    for col in columns:
        pca = PCA(n_components=n_comp, random_state=1)
        new_df[col+'_pca'] = list(pca.fit_transform(np.stack(df[col].to_numpy())))
    return new_df.reset_index(drop=True)


# In[ ]:


def apply_scaler(df, columns):
    new_df = df.copy()
    for col in columns:
        scaler = StandardScaler()
        new_df[col + '_scaled'] = list(scaler.fit_transform(np.stack(df[col].to_numpy())))
    return new_df.reset_index(drop=True)


# In[ ]:


model_df = pca_apply(model_df, ['abstract_tfidf','body_text_tfidf'], 10)
model_df = apply_scaler(model_df,['abstract_tfidf_pca','body_text_tfidf_pca'])


# <a id="xgboost"></a>
# ### *XGBoost Model*

# In[ ]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


# In[ ]:


# Set X and y for model development
X = np.stack(model_df['body_text_tfidf_pca_scaled'].to_numpy())
y = model_df["isNPI"]

# Set the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)


# In[ ]:


# Set params and XGBoost classifier
clf_xgb = xgb.XGBClassifier(max_depth=6, learning_rate=0.1,silent=False, objective='binary:logistic',                   booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,                   subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1, reg_alpha=0, reg_lambda=1)

# Fit training sets into the model
clf_xgb.fit(X_train, y_train)


# In[ ]:


y_pred = clf_xgb.predict(X_test)
precision_recall_fscore_support(y_test, y_pred, average='macro')


# In[ ]:


score = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {score}')


# In[ ]:


core_booster = clf_xgb.get_booster()


# In[ ]:


xgb.plot_importance(core_booster)
plt.show()


# In[ ]:


from sklearn.metrics import plot_roc_curve


# In[ ]:


plot_roc_curve(clf_xgb, X_test, y_test)
plt.show()


# <a id="slice-npi"></a>
# ### *Slice NPI*
# The slicing of `all_json_df` will be done in two steps.
# 1. Remove all rows that do not have more than 1 occurance of a keyword in the abstract
# 2. Remove all rows that are predicted to not be an NPI by the XGBoost model

# In[ ]:


final_df = all_json_df.copy()


# In[ ]:


def npi_slice(df):
    def get_count(row):
        return sum([row['abstract'].count(keyword) for keyword in keywords])
    df = df[df.apply(get_count, axis=1) > 1]
    return df
final_df = npi_slice(final_df)
len(final_df)


# In[ ]:


# Apply PCE without merging the pre-labeled dataframe. Slicing is done on the full dataset
final_df = pca_apply(final_df, ['abstract_tfidf','body_text_tfidf'], 10)
final_df = apply_scaler(final_df,['abstract_tfidf_pca','body_text_tfidf_pca'])


# In[ ]:


def npi_col(row):
    x = np.array([row['body_text_tfidf_pca_scaled']])
    y_pred = clf_xgb.predict(x)[0]
    if y_pred > 0:
        return True
    return False


# In[ ]:


final_df['npi_pred'] = final_df.apply(npi_col, axis=1)
final_df = final_df[final_df['npi_pred']].reset_index(drop=True)
final_df = final_df.drop(columns=['npi_pred'])


# In[ ]:


len(final_df)


# <a id="clustering"></a>
# # **Clustering**

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


def cluster(df, columns, clust_nums):
    new_df = df.copy()
    for col in columns:
        kmeans = KMeans(n_clusters = clust_nums)
        new_df[col + "_clusterID"] = list(kmeans.fit_predict(np.stack(df[col].to_numpy())))
    return new_df


# In[ ]:


# Let's try to create 10 clusters
clustered_df = cluster(final_df, ['abstract_tfidf_pca_scaled', 'body_text_tfidf_pca_scaled'], 10)


# In[ ]:


clustered_df.body_text_tfidf_pca_scaled_clusterID.value_counts()


# In[ ]:


clustered_df[clustered_df.body_text_tfidf_pca_scaled_clusterID == 3][['title']].head()


# In[ ]:


# Reduce dimensions for plotting later
def reduce_dimension(row):
    return row[:3]
clustered_df['abstract_tfidf_pca_scaled'] = clustered_df['abstract_tfidf_pca_scaled'].apply(reduce_dimension)
clustered_df['body_text_tfidf_pca_scaled'] = clustered_df['body_text_tfidf_pca_scaled'].apply(reduce_dimension)


# <a id="geolocation"></a>
# # **Geolocation**

# In[ ]:


get_ipython().system('pip install country_list')
from country_list import countries_for_language


# In[ ]:


countries = set([k.lower() for k in dict(countries_for_language('en')).values()])


# In[ ]:


def set_country_columns(df):
    def get_country(row):
        text_set = set(row['body_text'].split(' '))
        return list(countries.intersection(text_set))
    df['countries'] = df.apply(get_country, axis=1)
    return df
clustered_df = set_country_columns(clustered_df)


# In[ ]:


def get_country_df(df):
    country = []
    count = []
    for k in dict(countries_for_language('en')).values():
        len_country = len(df[df['countries'].map(set([k.lower()]).issubset)])
        country.append(k.lower())
        count.append(len_country)
    return pd.DataFrame({'country': country, 'count': count})
country_frequency = get_country_df(clustered_df)


# In[ ]:


country_frequency.sort_values(by='count', ascending=False)


# In[ ]:


import plotly.express as px


# In[ ]:


fig = px.scatter_geo(country_frequency,
                     locationmode='country names',
                     locations='country',
                     hover_name='country',
                     size='count',
                     projection='natural earth')
fig.update_layout(title='Research Papers Mentioned by Country',
                  autosize=False,
                  width=500,
                  height=250,
                  paper_bgcolor='rgba(0,0,0,0)'
                 )
fig.show()


# <a id="visualization"></a>
# # **Visualization**

# In[ ]:


clustered_df[['x', 'y', 'z']] = pd.DataFrame(clustered_df['body_text_tfidf_pca_scaled'].values.tolist(),
                                             index = clustered_df.index)


# In[ ]:


fig = px.scatter(clustered_df, x='x', y='y',
                 color='body_text_tfidf_pca_scaled_clusterID',
                 hover_name='title',
                 hover_data=['paper_id', 'doi'])
fig.update_layout(title = '2D cluster of research papers',
                  xaxis = dict(dtick=1, range=[-5,5], scaleratio = 1),
                  yaxis = dict(dtick=1, range=[-5,5], scaleratio = 1),
                  hoverlabel=dict(
                    bgcolor='white', 
                    font_size=8, 
                    font_family='Rockwell'
                  ),
                  coloraxis=dict(
                    colorbar=dict(title='Cluster ID')
                  ))
fig.show()


# In[ ]:


fig = px.scatter_3d(clustered_df, x='x', y='y', z='z',
                    color='body_text_tfidf_pca_scaled_clusterID',
                    hover_name='title',
                    hover_data=['paper_id', 'doi'])
fig.update_layout(title = '3D cluster of research papers by body_text',
                  paper_bgcolor='rgba(0,0,0,0)',
                  scene = dict(
                    xaxis = dict(dtick=1, range=[-5,5],),
                    yaxis = dict(dtick=1, range=[-5,5],),
                    zaxis = dict(dtick=1, range=[-5,5],),),
                  hoverlabel=dict(
                    bgcolor='white', 
                    font_size=8, 
                    font_family='Rockwell'
                  ),
                  coloraxis=dict(
                    colorbar=dict(title='Cluster ID')
                  ))
fig.show()


# The visualizations above can be located on the following website. https://kaggle-covid-npi.herokuapp.com/
# 
# This site can do the following:
# * Select a 2d or 3d visualize
# * Select Number of clusters to display
# * Select the cluster by cluster id
# * Search by keyword
# * Display text data after click
# * Display geographical map and query 2d/3d visualization by geo-graph click

# <a id="conclusion"></a>
# # **Conclusion**
# Machine learning methods have enabled us to structure and make explorable vast amounts of data. As detailed above, we have produced a classifier-fed clustering alogrithm which enables users to specify the category across which to find clusters (in this case, the body text or abstract of papers), specify number of clusters, and visually represent each cluster spatially amongst the others. By enabling researchers to search for specific terms and see matching and adjacent publications to provide context to their understandings, we produce natural and interpretable groupings of knowledge.
# 
# Progress in the fight against COVID-19 will be fought on many fronts. First responders, manufacturers, essential workers, researchers, governmental bodies, and more all each have their part to play. Thankfully, we are seeing global collaboration at all time highs, but alongside this, we are seeing masses of data being produced. Unfortunately, the sheer volume of this research means that it cannot possible be digested and synthesized by any individual. Researchers, government officials, and medical professionals might benefit from having certain information, but the forest is too dense. By allowing these parties to aggregate, visualize, and parse this mass of data, we hope to enable new insights and initiatives to be taken.

# <a id="citation"></a>
# # **Citation/Sources**

# https://www.cdc.gov/nonpharmaceutical-interventions/index.html
# 
# https://www.who.int/news-room/q-a-detail/q-a-coronaviruses
# 
# https://www.cdc.gov/nonpharmaceutical-interventions/tools-resources/published-research.html
# 
# https://www.azdhs.gov/documents/preparedness/emergency-preparedness/response-plans/adhs-npi-playbook.pdf
# 
# https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf
# 
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3291414/
# 
# https://www.medrxiv.org/content/10.1101/2020.03.03.20029843v3
# 
# https://bmcpublichealth.biomedcentral.com/articles/10.1186/1471-2458-14-1328
# 
# https://www.who.int/influenza/resources/documents/RapidContProtOct15.pdf
# 
# https://stacks.cdc.gov/view/cdc/11425
