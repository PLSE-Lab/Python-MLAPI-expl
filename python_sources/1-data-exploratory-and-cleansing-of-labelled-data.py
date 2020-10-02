#!/usr/bin/env python
# coding: utf-8

# # 1. Data Exploratory and Cleansing of Labelled Data
# **Prerequisite**: Please Download the CORD-19 Dataset at https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=882
# <br> **File name**: 551982_1230614_bundle_archive
# <br>**Key files / folders used for the Project:**
# 1.     **Meta data**                : metadata.csv
# 2.     **Priority Questions files** : Kaggle -> target_tables -> 2_relevant_factors
# 3.     **Research Paper Locations**    : document_parses -> pdf_json & pmc_json

# In[ ]:


# Import and read Metadata
import pandas as pd
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from os import listdir
from tqdm import tqdm
import numpy as np

# Key Data Paths
data_path = '../input/CORD-19-research-challenge'
# data_path = './551982_1230614_bundle_archive/'
priority_question_path = data_path + '/Kaggle/target_tables/2_relevant_factors/'


# ## 1.1. Metadata Analysis 
# <br> Key Observations: there are lots of missing values even from the source json files. This means that the database is not up to date and some literatures might require exteral resources i.e. web scrapping to obtain the information.
# <br> This could Impact on whether we will be able to collect enough information from this dataset

# In[ ]:


# Loading the meta data and relative questions path

meta_data=pd.read_csv(data_path + '/metadata.csv')
print("Column names: {}".format(meta_data.columns))
print("number of rows: ", len(meta_data))
meta_data.head(5)


# In[ ]:


plt.figure(figsize=(20,10))
na_analysis = meta_data.isna().sum()
na_analysis.sort_values().plot(kind='bar', stacked=True, x = 'columns', y = 'count')


# ## 1.2. Priority Question (Target_table) Analysis
# Key Observations: This is a One to many relationship between a priority question to the literatures.
# A literature can be referenced multiple times within a question.
# This observation provides us with the information that we will need to break down each document into sections

# In[ ]:


question_1 = 'Effectiveness of a multifactorial strategy to prevent secondary transmission.csv'
priority_question_1 =pd.read_csv(priority_question_path + question_1,index_col=0)
print("Column names: {}".format(priority_question_1.columns))
print("number of rows: ", len(priority_question_1))
priority_question_1.head(10)
priority_question_1.shape


# ### 1.2.1 Matching between Metadata and Priority data
# key observation: Not all literatures referenced in Priority data is in the Metadata. We might need to acquire the literature from external sources

# In[ ]:


# Testing match between metadata and question 1
match_list = []
for index, row in priority_question_1.iterrows():
    url_row = meta_data.loc[meta_data['url'] == row['Study Link']]['url'].any()
    title_row = meta_data.loc[meta_data['title'] == row['Study']]['title'].any()
    if url_row != False or title_row != False:
        match_list.append(True)
    else:
        match_list.append(False)
print("percentage of matched list is", sum(match_list), "/", len(match_list))


# #### 1.2.1.1. Observation from above few observations can be made
# 1. A literature can be repeated multiple times within each priority question - therefore the articles are segmented into sections for analysis
# 2. matching meta data with question using both title and url as criteria -> if found then we will be able to identify the JSON within our database
# 3. there is only one mismatch in priority_question1 that does not have the article within the database
# 4. Number of rows for each topics are different - data imbalancing.
# 

# In[ ]:


# Testing the duplication relationship between priority question 1 and literatures
duplicateRowsDF = priority_question_1[priority_question_1.duplicated(['Study'])]
print("Duplicate Rows based on a single column are:", duplicateRowsDF, sep='\n')


# In[ ]:


# Listing all the priority questions 
priority_question_list = [f for f in listdir(priority_question_path) if isfile(join(priority_question_path, f))]
for question in priority_question_list:
    match_list = []
    priority_question_df = pd.read_csv(priority_question_path + question,index_col=0)
    for index, row in priority_question_df.iterrows():
        url_row = meta_data.loc[meta_data['url'] == row['Study Link']]['url'].any()
        title_row = meta_data.loc[meta_data['title'] == row['Study']]['title'].any()
        if url_row != False or title_row != False:
            match_list.append(True)
        else:
            match_list.append(False)
    print(question)
    print("percentage of matched list is", sum(match_list), "/", len(match_list), " = ", sum(match_list)/len(match_list))


# ### 1.2.2. Checking any N/As in the priority questions list
# From the observations below, there are some missing values for certain questions in Measure of Evidence

# In[ ]:


# Checking the quality of the csv file for each questions
for question in priority_question_list:
    print("file name is: ",question)
    priority_question_df = pd.read_csv(priority_question_path + question,index_col=0)
    priority_question_df.info()


# ## 1.3. Matching metadata and Priority question data by URL or the Title of the article
# Trying to find a match between the question file and the metadata to find whether the downloaded database contains the file, if not then web scraping might be required to obtain the information from the URL provided

# In[ ]:


# Obtaining all files names from all priority_question list into one dataframe
# Metadata -> We want pdf_json_files, pmc_json_files

pdf_json_files_list = [];
pmc_json_files_list = [];
research_topic_list = [];
topic_id_list = [];
topic_id = 1
combined_ques_literature = [];
for question in tqdm(priority_question_list):
    print("file name is: ",question)
    priority_question_df = pd.read_csv(priority_question_path + question,index_col=0)
    combined_ques_literature.append(priority_question_df)
    for index, row in priority_question_df.iterrows():
        # Match using URL or title
        url_match_row = meta_data.loc[meta_data['url'] == row['Study Link']]
        title_match_row = meta_data.loc[meta_data['title'] == row['Study']]
        if not url_match_row.empty or not title_match_row.empty:
            if not url_match_row.empty:
                pdf_json_files_list.append(url_match_row['pdf_json_files'].values[0])
                pmc_json_files_list.append(url_match_row['pmc_json_files'].values[0])
            else:
                pdf_json_files_list.append(title_match_row['pdf_json_files'].values[0])
                pmc_json_files_list.append(title_match_row['pmc_json_files'].values[0])
        else:
            pdf_json_files_list.append(" ")
            pmc_json_files_list.append(" ")
        topic_id_list.append(topic_id)
        research_topic_list.append(question.split('.')[0])
    topic_id = topic_id + 1

# combining all datarfame into one big dataframe
combined_ques_literature = pd.concat(combined_ques_literature);


# In[ ]:


# Checking if previous result provides the correct dimension for the output
print(len(topic_id_list), len(pmc_json_files_list), len(pdf_json_files_list), len(research_topic_list))
print("before adding columns: ", combined_ques_literature.shape)
print(combined_ques_literature.columns)
combined_ques_literature.insert(1, "topic_id", topic_id_list, True)
combined_ques_literature.insert(2, "research_topic", research_topic_list, True)
combined_ques_literature.insert(3, "pdf_json_files", pdf_json_files_list, True)
combined_ques_literature.insert(4, "pmc_json_files", pmc_json_files_list, True)

print("after adding columns: ", combined_ques_literature.shape)


# ### 1.3.1. Observation on merging metadata and priority questions
# Based on observations:
# 1. the tables within priority questions are inconsistent with each other -> i.e. the columns are referring to the same thing
# Factors, Factors Described
# Influential,Infuential,Influential (Y/N)
# 
# <br> Hence, next step would be to perform some cleaning to merge the duplicated rows 

# In[ ]:


combined_ques_literature[['Influential','Infuential','Influential (Y/N)']].info()
combined_ques_literature[['Factors', 'Factors Described']].info()
combined_ques_literature[['Date', 'Date Published']].info()


# ### 1.3.2. Observation on Duplicated Rows
# Merging the Influential columns together will yield a complete 547 rows of non- null values. which is good. Same Approach will be peformed for observations

# In[ ]:


combined_ques_literature['Influential'] = combined_ques_literature['Influential'].fillna(combined_ques_literature['Infuential'])
combined_ques_literature['Influential'] = combined_ques_literature['Influential'].fillna(combined_ques_literature['Influential (Y/N)'])            
combined_ques_literature['Factors'] = combined_ques_literature['Factors'].fillna(combined_ques_literature['Factors Described'])
combined_ques_literature['Date'] = combined_ques_literature['Date'].fillna(combined_ques_literature['Date Published'])
combined_ques_literature = combined_ques_literature.drop(['Infuential', 'Influential (Y/N)', 'Factors Described', 'Date Published'], axis=1)


# In[ ]:


print(combined_ques_literature.info())


# ### 1.4. Scoping and finalising the Categorisation the data
# Since Not all rows contains the literature within the database, we will remove the labelled data for ones that does not have data in pdf_json_files. This is to simplify this solution for the purpose of machine learning
# This can potentially be future work to develop a methodology for web scrapping techniques to obtain the literature from external sources outside CORD-19

# In[ ]:


# Drop all the na and spaces on the pdf_json_file to make sure all labelled data is linked to a literature stored in database

scoped_categorised_literature = combined_ques_literature.dropna(subset=['pdf_json_files'])
scoped_categorised_literature = scoped_categorised_literature[~scoped_categorised_literature['pdf_json_files'].str.isspace()] 
scoped_categorised_literature.info()


# ### 1.5. Exporting the cleansed and mapped data into a pickle file

# In[ ]:


scoped_categorised_literature.to_pickle("./1_scoped_cat_lit.pkl")


# In[ ]:




