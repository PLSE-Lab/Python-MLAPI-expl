#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Task1 Let's make Study Summary Tutorial
# 
# 1st public version: 16/06/2020
# 
# This notebook is for Task 1 of COVID-19 Challenge.  
# In this task, we are supposed to make literature summary.  
# I use the techniques of treating json file and web scraping. 
# 
# The overview of my process in this notebook is like below:
# 1. Get articles related to certain topic by filtering title (In this notebook, I chose 'school closure')
# 2. Get general information from metadata.csv
# 3. Get more specific information by using web scraping
# 4. Fill out some other topics by rule-based algorithm
# 
# Each step contains incompleteness, however, this approach works to some extent, I guess.
# 
# Here is table of contents:
# - [Data Check and Treating Json Example](#Data-Check-and-Treating-Json-Example)
# - [Data Visualization](# Data-Visualization)
# - [Task 1 Challenge](#Task-1-Challenge)
#     -  [General Information](#General-Information)
#     -  [Web Scraping](#Web-Scraping)
#     -  [Study Type, Factors](#Study-Type,-Factors)
#     -  [Influential](#Influential)
#     -  [Excerpt](#Excerpt)
# - [Acknowledgement](#Acknowledgement)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import time
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import requests    # For web scraping
from bs4 import BeautifulSoup

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[ ]:


INPUT_TARGET_DIR = "/kaggle/input/CORD-19-research-challenge/Kaggle/target_tables/2_relevant_factors/"

for file in os.listdir(INPUT_TARGET_DIR):
    print(file)


# ## Data Check and Treating Json Example

# In[ ]:


# Let's take a look at target csv example.
target_df = pd.read_csv(INPUT_TARGET_DIR + "Effectiveness of school distancing.csv", index_col=0)
target_df.head()


# In[ ]:


INPUT_JSON_DIR =  "/kaggle/input/CORD-19-research-challenge/document_parses/pdf_json/"
json_file = open(INPUT_JSON_DIR + "566b5c62fc77292ebe09295d59e7fbf6fc914260.json", "r")
json_data = json.load(json_file)
json_data


# As we can see the result above, JSON file is comprised of some dictionaries.  
# You can find json format in json_schema.txt.  
# If you want to know the format in the code, you can run the cell below.

# In[ ]:


# !cat "/kaggle/input/CORD-19-research-challenge/json_schema.txt"


# We can access each object by key like below.

# In[ ]:


json_data["metadata"]


# In[ ]:


json_data["abstract"]


# In[ ]:


json_data["body_text"]


# In[ ]:


json_data["bib_entries"]


# # Data Visualization

# In[ ]:


metadata_df = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
metadata_df.head()


# In[ ]:


# General Info of metadata
metadata_df.info()


# 17.url has more non-null objects than 15.pdf_json_files and 16.pmc_json_files.  
# So in order to collect data, it is better to use scraping techniques than just searching pdf file.

# In[ ]:


metadata_df.nunique()


# In these columns, countplots of source_x and license are seemed to be visualized.  
# Other columns have too many non-unique values to use countplot.

# In[ ]:


# Source Count Plot
plt.figure(figsize=(8, 6))
sns.countplot(metadata_df["source_x"])
plt.xticks(rotation=90)


# From the above graph, we can easily find that most of sources are combination of some basic sources.  
# We should decompose these combined sources to plot count more clearly.

# In[ ]:


# We can get decomposed sources by using split method.
temp_df = metadata_df["source_x"].str.split(";", expand=True)
temp_df.tail()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ncount_dict = {}\n\nfor row in range(len(temp_df)):\n    for col in range(len(temp_df.columns)):\n        key = temp_df.iloc[row, col]\n        if key != None:\n            key = key.lstrip()\n        count_dict.setdefault(key, 0)\n        count_dict[key] += 1\n\ndel count_dict[None]    # We delete key:None \ncount_dict')


# In[ ]:


count_dict_sorted = dict(sorted(count_dict.items(), key=lambda x:x[1], reverse=True))

plt.bar(count_dict_sorted.keys(), count_dict_sorted.values())
plt.xticks(rotation=90)
plt.ylabel("Count")
plt.xlabel("Source")


# In[ ]:


# License Count Plot
sns.countplot(metadata_df["license"])
plt.xticks(rotation=90)


# In[ ]:


metadata_df["year"] = metadata_df["publish_time"].str[:4].astype("float")
metadata_df["year"].unique()


# In[ ]:


temp_series = metadata_df.groupby("year")["cord_uid"].count()


# In[ ]:


plt.figure(figsize=(12, 6))
sns.lineplot(x=temp_series.index, y=temp_series.values)
plt.xticks(rotation=90)
plt.ylabel("Count")
plt.title("Total Record Count Transition")


# We can easily find that record hits increase rapidly in 2020.

# # Task 1 Challenge
# From now on, let's move on to task challenge itself!
# 
# ## School Distancing Literature Summary
# First, I extract the articles related to school distancing.

# In[ ]:


# We change title to lowercase.
metadata_df["title"] = metadata_df["title"].str.lower()
metadata_df["title"]


# In[ ]:


# First, we extract the records that contains 'school' and 'distancing' keywords in paper title. 
# Machine Learning, especially, Topic Finding from text dataset can be useful in this task.
# However, let's make it as simple as possible for my first step! 
# Actually, this rule base algorithm can be strong enough to extract the articles related to 'school distancing' from all datasets (about length 140k).

school_row = []

for row in range(len(metadata_df)):
    try:
        if ("school" in metadata_df.loc[row, "title"]) & ("clos" in metadata_df.loc[row, "title"]):
            school_row.append(row)
    except: 
        continue

print("We hit {} records when searching 'school' and 'closure'".format(len(school_row)))
print("This is {:.2f} % of this dataset".format(len(school_row) / len(metadata_df) * 100, 2))


# In[ ]:


# Let's check the metadata in school_rows
metadata_df.loc[school_row, :].head()


# We can find that articles regarding school closure can be extracted from its title.

# In[ ]:


# Let's check what we have to fill out for submission
target_df.head(1)


# ### General Information

# In[ ]:


# Submission DataaFrame, first we make its format and fill out the content later
summary_df = pd.DataFrame(columns = target_df.columns)

# General Info
summary_df["Date"] = metadata_df.loc[school_row, "publish_time"]
summary_df["Study"] = metadata_df.loc[school_row, "title"]
summary_df["Study Link"] = metadata_df.loc[school_row, "url"]
summary_df["Journal"] = metadata_df.loc[school_row, "journal"]
summary_df["pdf_json_file"] = metadata_df.loc[school_row, "pdf_json_files"]
summary_df["pmc_json_file"] = metadata_df.loc[school_row, "pmc_json_files"]
summary_df["abstract"] = metadata_df.loc[school_row, "abstract"]
summary_df.head()


# We have to fill out Study Type, Factors, Influential Excerpt, Measure of Evidence and Added on columns.  
# But for added_on column (i.e. date when the article is added on this Kaggle dataset), I couldn't find where I can get source.

# In[ ]:


list(summary_df["Study Link"])


# In[ ]:


concated_url = 'https://doi.org/10.1186/s12879-017-2934-3; https://www.ncbi.nlm.nih.gov/pubmed/29321005/'

url_list = concated_url.split('; ')
for url in url_list:
    if 'ncbi' in url:
        print(url)


# In[ ]:


# We have to deal with errors 
# concated_url = np.nan

# url_list = concated_url.split('; ')
# for url in url_list:
#     if 'ncbi' in url:
#         print(url)


# In[ ]:


# Extract only NCBI data because I couldn't check other sites' web scraping policies
ncbi_url = []

for concated_url in list(summary_df["Study Link"]):
    isncbi = 0
    try:
        url_list = concated_url.split('; ')
        for url in url_list:
            if 'ncbi' in url:
                ncbi_url.append(url)
                isncbi = 1
            
        if isncbi == 0:   
            ncbi_url.append(np.nan)
    except AttributeError:
        ncbi_url.append(np.nan)


# In[ ]:


ncbi_url


# In[ ]:


# this number does match to the number of records
len(ncbi_url)


# ### Web Scraping
# 
# #### Web Scraping Tutorial
# In the following cells, the tutorial of web scraping is executed.  
# 
# **Please Remind the Followings**
#  - In some website, web scraping is prohibited, so we have to take the websites' policies. 
#  - In this tutorial, I access NCBI website by using scraping.
#    The website declares about web scraping in the policies and disclaimers tab. (https://www.ncbi.nlm.nih.gov/home/about/policies/)
#  - It says too many web scraping queries is not allowed, but in this tutorial, we only access the website for a few times. So it seems OK.
#    
# > Guidelines for Scripting Calls to NCBI Servers
# Do not overload NCBI's systems. Users intending to send numerous queries and/or retrieve large numbers of records should comply with the following:
# > - Run retrieval scripts on weekends or between 9 pm and 5 am Eastern Time weekdays for any series of more than 100 requests.  
# > - Send E-utilities requests to https://eutils.ncbi.nlm.nih.gov, not the standard NCBI Web address.  
# > - Make no more than 3 requests every 1 second.  
# > - Use the URL parameter email, and tool for distributed software, so that we can track your project and contact you if there is a problem. For more information, please see the Usage Guidelines and Requirements section in the Entrez Programming Utilities Help Manual.  
# > - NCBI's Disclaimer and Copyright notice must be evident to users of your service. NLM does not claim the copyright on the abstracts in PubMed; however, journal publishers or authors may. NLM provides no legal advice concerning distribution of copyrighted materials, consult your legal counsel.   

# In[ ]:


url = list(summary_df["Study Link"])[0] 
r = requests.get(url)


# In[ ]:


print(url)    # The first article URL


# In[ ]:


# print(r.headers)      # Header Information
print(r.content)


# This result is difficult to read, so we use BeautifulSoup Library 

# In[ ]:


soup = BeautifulSoup(r.content, "html.parser")
print(soup.prettify())


# The result seems more beautiful with BeautifulSoup

# In[ ]:


# You can access each object like below using BeautifulSoup instance
print(soup.title)
print(soup.title.name)
print(soup.title.string)

# You can access 'a' tag  in html like below
print(soup.a)    # It just returns the first link.
print(soup.find_all('a'))    # Returns all the links


# ### Study Type, Factors

# In[ ]:


soup.get_text()


# I'd like to get some important keyword from this whole text.

# In[ ]:


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
print("Every 10th stopword:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))


# In[ ]:


from wordcloud import WordCloud

wordcloud = WordCloud(background_color="white", stopwords=ENGLISH_STOP_WORDS)
wordcloud.generate(soup.body.get_text().replace('\n','').replace('\t',''))


# In[ ]:


plt.figure(figsize=(12, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Show Description WordCloud in the first article")


# We extracted the article related to 'school closure' (i.e. the article whose title includes 'school' and 'closur').  
# Thus, the thema of this article is 'school closure', and we can see this phrase in the center.  
# Of course, the factor can be concluded as school closure.   
# But in order to differentiate other articles in this dataframe, we try to extract the 2nd or 3rd keywords.

# In[ ]:


wordcount_dict = wordcloud.process_text(soup.body.get_text().replace('\n','').replace('\t',''))
# wordcount_dict


# In[ ]:


# For tutorial, I picked up ten keywords.
wordcount_dict_sorted = dict(sorted(wordcount_dict.items(), key=lambda x:x[1], reverse=True))
result = {k:wordcount_dict_sorted [k] for k in list(wordcount_dict_sorted )[:10]}
result


# In[ ]:


if "school" in result.keys():
    del result["school"]
if "closure" in result.keys():
    del result["closure"]


# In[ ]:


# For submission, I picked up five keywords.
result = {k:result[k] for k in list(result)[:5]}
result


# In[ ]:


summary_df.loc[1459, "Factors"] = ', '.join(list(result.keys()))


# In[ ]:


summary_df.head(1)


# In[ ]:


summary_df.loc[1459, "abstract"].lower()


# In[ ]:


# Study Type
# I defined that study type could be classified into three types. (Modelling, Review or Summary, Others)
# I created some basic rule regarding classification of these three types.
# First, if 'modeling' in abstract,we think this article is about modeling.
# Second, if 'review' or 'summary' in abstract, we think this article is about review or summary.
# Third, we dive into the article body and search 'model' or 'review, summary' keywords.
# If we can't find any words, we define it as Others category.

study_type = 'Others'

if 'model' in summary_df.loc[1459, "abstract"].lower():
    print("This article abstract contains 'model' keywords")
    study_type = 'Modeling'
elif 'review' in summary_df.loc[1459, "abstract"].lower() or 'summary' in summary_df.loc[1459, "abstract"].lower():
    print("This article abstract contains 'review' or 'summary' keywords")
    study_type = 'Review'

elif 'model' in wordcount_dict:
    print("This article body contains 'model' keywords")
    study_type = 'Modeling'

elif 'review' in wordcount_dict:
    print("This article body contains 'review' or 'summary' keywords")
    study_type = 'Review'

    
summary_df.loc[1459, "Study Type"] = study_type


# In[ ]:


summary_df.head(1)


# ### Influential	
# 
# When you access the first article webpage of NCBI with your browser, you can find a block like below:
# > This article has been cited by other articles in PMC.
# 
# Example in the first article case:
# 
# >![](http://)<div class="fm-panel"><div>This article has been <a href="/pmc/articles/PMC4021091/citedby/">cited by</a> other articles in PMC.</div></div></div></div>
# 
# Actually, this is a link to the page which shows other articles that cited this original article.  
# We can use this information to decide whether the article is influential or not.  
# Specifically, the number of articles that cited the document can be one effective signal for estimating the study's importance.   
# So from mow on, I'd like to get the number of times cited of this first document.

# In[ ]:


cited_url = 'https://www.ncbi.nlm.nih.gov' + soup.find(id="pmclinksbox").find("a").get("href")
cited_url


# Then we access this link.

# In[ ]:


r_cited = requests.get(cited_url)
soup_cited = BeautifulSoup(r_cited.content, "html.parser")


# In[ ]:


print(soup_cited.prettify())


# In[ ]:


# We can use .find method to get the target sentence.
text = soup_cited.find("h2").get_text()
print(text)


# In[ ]:


# We extract the number of cited using regex
regex = re.compile('\d+')

number_cited = regex.findall(text)
number_cited = int(number_cited[0])
print(number_cited)


# If this number is larger than five, I regard this article as influential

# In[ ]:


is_influential = 'N'
if number_cited >= 5:
    is_influential = 'Y'

summary_df.loc[1459, "Influential"] = is_influential


# In[ ]:


summary_df.head(1)


# ### Excerpt
# This excerpt is the last two lines of Abstract.  
# This rule is because conclusions are often written in the latter half of the abstract.  

# In[ ]:


abstract_text = summary_df.loc[1459, "abstract"] 
abstract_text


# In[ ]:


'. '.join(abstract_text.split('. ')[-2:])


# This is a good excerpt of this article.

# In[ ]:


summary_df.loc[1459, "Excerpt"] = '. '.join(abstract_text.split('. ')[-2:])
summary_df.head(1)


# Sorry, I couldn't come up with any idea to fill out Measure of Evidence and Added on column.  
# Then I do this process so far to all records.  
# And I only tried this method to pmc articles.  
# Sorry for incomplete result. (I don't have time to examine how I should get important features from other websites.)

# In[ ]:


record_count = 0

for target_row in summary_df.index[:5]:
    print(target_row)
    print("Start Web Scraping")
    article_url = ncbi_url[record_count]
    r = requests.get(article_url)
    soup = BeautifulSoup(r.content, "html.parser")
    
    # Factors
    # We extracted the article related to 'school closure' (i.e. the article whose title includes 'school' and 'closur'). 
    # Thus, the thema of this article is 'school closure', and we can see this phrase in the center.  
    # Of course, the factor can be concluded as school closure.   
    # But in order to differentiate other articles in this dataframe, we try to extract the 2nd or 3rd keywords.
    
    wordcloud = WordCloud(background_color="white", stopwords=ENGLISH_STOP_WORDS)
    wordcount_dict = wordcloud.process_text(soup.body.get_text().replace('\n','').replace('\t',''))
    
    wordcount_dict_sorted = dict(sorted(wordcount_dict.items(), key=lambda x:x[1], reverse=True))
    result = {k:wordcount_dict_sorted [k] for k in list(wordcount_dict_sorted )[:10]}

    if "school" in result.keys():
        del result["school"]
    if "closure" in result.keys():
        del result["closure"]
    
    result = {k:result[k] for k in list(result)[:5]}
    summary_df.loc[target_row, "Factors"] = ', '.join(list(result.keys()))
    
    
    # Study Type
    # I defined that study type could be classified into three types. (Modelling, Review or Summary, Others)
    # I created some basic rule regarding classification of these three types.
    # First, if 'modeling' in abstract,we think this article is about modeling.
    # Second, if 'review' or 'summary' in abstract, we think this article is about review or summary.
    # Third, we dive into the article body and search 'model' or 'review, summary' keywords.
    # If we can't find any words, we define it as Others category.

    study_type = 'Others'

    if 'model' in wordcount_dict:
        print("This article body contains 'model' keywords")
        study_type = 'Modeling'

    elif 'review' in wordcount_dict:
        print("This article body contains 'review' or 'summary' keywords")
        study_type = 'Review'
    
    try:
        if 'model' in summary_df.loc[target_row, "abstract"].lower():
            print("This article abstract contains 'model' keywords")
            study_type = 'Modeling'
        elif 'review' in summary_df.loc[target_row, "abstract"].lower() or 'summary' in summary_df.loc[target_row, "abstract"].lower():
            print("This article abstract contains 'review' or 'summary' keywords")
            study_type = 'Review'
    except AttributeError:
        pass



    summary_df.loc[target_row, "Study Type"] = study_type
    
    # Influential
    # Actually, this is a link to the page which shows other articles that cited this original article.  
    # We can use this information to decide whether the article is influential or not.  
    # Specifically, the number of articles that cited the document can be one effective signal for estimating the study's importance.   
    # So from mow on, I'd like to get the number of times cited of this first document.
    
    try:
        cited_url = 'https://www.ncbi.nlm.nih.gov' + soup.find(id="pmclinksbox").find("a").get("href")
        r_cited = requests.get(cited_url)
        soup_cited = BeautifulSoup(r_cited.content, "html.parser")

        text = soup_cited.find("h2").get_text()
        regex = re.compile('\d+')

        number_cited = regex.findall(text)
        number_cited = int(number_cited[0])
        print("number_cited: " + str(number_cited))
        
    except:
        number_cited = 0
        pass
    
    is_influential = 'N'
    if number_cited >= 5:
        is_influential = 'Y'

    summary_df.loc[target_row, "Influential"] = is_influential
    
    # Excerpt
    # This excerpt is the last two lines of Abstract.  
    # This rule is because conclusions are often written in the latter half of the abstract.  
    
    try:
        abstract_text = summary_df.loc[target_row, "abstract"] 
        summary_df.loc[target_row, "Excerpt"] = '. '.join(abstract_text.split('. ')[-2:])
        
    except AttributeError:
        pass
    
    # For not accessing website so many times in a short time.
    time.sleep(60)
    
    record_count += 1


# In[ ]:


summary_df.head(5)


# In[ ]:


# Delete unnecessary cols
summary_df = summary_df.drop(columns=["pdf_json_file", "pmc_json_file", "abstract"])
summary_df.head(1)


# In[ ]:


summary_df.to_csv("Effectiveness of school distancing.csv", index=False)


# # Acknowledgement
# 
# I think this notebook is incomplete one, but I think this works for your tutorial of this task 1.  
# I hope this tutorial is somewhat useful for beginners who have difficulty getting started.  
# However, I need more time to complete this task...
# 
# If you like it, I appreciate all of your comments and upvotes, thank you for reading my notebook!

# In[ ]:




