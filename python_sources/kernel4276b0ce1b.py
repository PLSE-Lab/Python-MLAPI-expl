#!/usr/bin/env python
# coding: utf-8

# # Task "What do we know about COVID-19 risk factors?"

# **Subtask "Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors"**
# 
# *A model of generic document crawler.*

# *NOTE: This approach may be employed for any other subtask (i.e. bullet point) in the task. This notebook showcases the issue of transmission dynamics. Moreover, this approach is fully customizable meaning that the user of this notebook may fine-tune parameters to achieve better results.*

# Special credits to:
# - https://www.kaggle.com/acmiyaguchi for providing an excellent notebook with pyspark data import 
# - https://www.merriam-webster.com for providing a free API and outstanding word base I used to seek for synonyms
# - https://wordassociations.net/ for providing a free API and outstanding word base I used to seek for associations

# ## Workflow

# The diagram below presents the workflow of browsing the documents.

# ![](https://www.lucidchart.com/publicSegments/view/f95a3e22-8511-49e6-b412-f5a8c1ef1f39)

# *NOTE: To print out intermediate results and control values, uncomment print() commands.*

# ## Notebook

# In[ ]:


get_ipython().system('pip install pyspark')
get_ipython().system('pip install pyarrow')


# In[ ]:


import pandas as pd
import os
import pyspark
from pyspark.sql.functions import lit
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from nltk.corpus import stopwords
import requests
import json
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pandas.core.common import flatten


# ### Import Data

# Credits: https://www.kaggle.com/acmiyaguchi

# In[ ]:


from pyspark.sql.functions import lit
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)


def generate_cord19_schema():
    """Generate a Spark schema based on the semi-textual description of CORD-19 Dataset.

    This captures most of the structure from the crawled documents, and has been
    tested with the 2020-03-13 dump provided by the CORD-19 Kaggle competition.
    The schema is available at [1], and is also provided in a copy of the
    challenge dataset.

    One improvement that could be made to the original schema is to write it as
    JSON schema, which could be used to validate the structure of the dumps. I
    also noticed that the schema incorrectly nests fields that appear after the
    `metadata` section e.g. `abstract`.
    
    [1] https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-13/json_schema.txt
    """

    # shared by `metadata.authors` and `bib_entries.[].authors`
    author_fields = [
        StructField("first", StringType()),
        StructField("middle", ArrayType(StringType())),
        StructField("last", StringType()),
        StructField("suffix", StringType()),
    ]

    authors_schema = ArrayType(
        StructType(
            author_fields
            + [
                # Uncomment to cast field into a JSON string. This field is not
                # well-specified in the source.
                StructField(
                    "affiliation",
                    StructType(
                        [
                            StructField("laboratory", StringType()),
                            StructField("institution", StringType()),
                            StructField(
                                "location",
                                StructType(
                                    [
                                        StructField("settlement", StringType()),
                                        StructField("country", StringType()),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                StructField("email", StringType()),
            ]
        )
    )

    # used in `section_schema` for citations, references, and equations
    spans_schema = ArrayType(
        StructType(
            [
                # character indices of inline citations
                StructField("start", IntegerType()),
                StructField("end", IntegerType()),
                StructField("text", StringType()),
                StructField("ref_id", StringType()),
            ]
        )
    )

    # A section of the paper, which includes the abstract, body, and back matter.
    section_schema = ArrayType(
        StructType(
            [
                StructField("text", StringType()),
                StructField("cite_spans", spans_schema),
                StructField("ref_spans", spans_schema),
                # While equations don't appear in the abstract, but appear here
                # for consistency
                StructField("eq_spans", spans_schema),
                StructField("section", StringType()),
            ]
        )
    )

    bib_schema = MapType(
        StringType(),
        StructType(
            [
                StructField("ref_id", StringType()),
                StructField("title", StringType()),
                StructField("authors", ArrayType(StructType(author_fields))),
                StructField("year", IntegerType()),
                StructField("venue", StringType()),
                StructField("volume", StringType()),
                StructField("issn", StringType()),
                StructField("pages", StringType()),
                StructField(
                    "other_ids",
                    StructType([StructField("DOI", ArrayType(StringType()))]),
                ),
            ]
        ),
        True,
    )

    # Can be one of table or figure captions
    ref_schema = MapType(
        StringType(),
        StructType(
            [
                StructField("text", StringType()),
                # Likely equation spans, not included in source schema, but
                # appears in JSON
                StructField("latex", StringType()),
                StructField("type", StringType()),
            ]
        ),
    )

    return StructType(
        [
            StructField("paper_id", StringType()),
            StructField(
                "metadata",
                StructType(
                    [
                        StructField("title", StringType()),
                        StructField("authors", authors_schema),
                    ]
                ),
                True,
            ),
            StructField("abstract", section_schema),
            StructField("body_text", section_schema),
            StructField("bib_entries", bib_schema),
            StructField("ref_entries", ref_schema),
            StructField("back_matter", section_schema),
        ]
    )


def extract_dataframe_kaggle(spark):
    """Extract a structured DataFrame from the semi-structured document dump.

    It should be fairly straightforward to modify this once there are new
    documents available. The date of availability (`crawl_date`) and `source`
    are available as metadata.
    """
    base = "/kaggle/input/CORD-19-research-challenge"
    crawled_date = "2020-03-13"
    sources = [
        "noncomm_use_subset",
        "comm_use_subset",
        "biorxiv_medrxiv",
        "custom_license",
    ]

    dataframe = None
    for source in sources:
        #path = f"{base}/{crawled_date}/{source}/{source}"
        path = f"{base}/{source}/{source}"
        df = (
            spark.read.json(path, schema=generate_cord19_schema(), multiLine=True)
            .withColumn("crawled_date", lit(crawled_date))
            .withColumn("source", lit(source))
        )
        if not dataframe:
            dataframe = df
        else:
            dataframe = dataframe.union(df)
    return dataframe


# In[ ]:


spark = SparkSession.builder.getOrCreate()
df = extract_dataframe_kaggle(spark)


# In[ ]:


# df.printSchema()


# In[ ]:


df.createOrReplaceTempView("cord19")


# Extracting authors (in case they were needed later to human-assess the result), abstracts (so that human can faster reject inrellevant ones in case they were reported as relevant) and texts.

# In[ ]:


text = (
    df.select("paper_id", F.posexplode("body_text").alias("pos", "value"))
    .select("paper_id", "pos", "value.text")
    .withColumn("ordered_text", F.collect_list("text").over(Window.partitionBy("paper_id").orderBy("pos")))
    .groupBy("paper_id")
    .agg(F.max("ordered_text").alias("sentences"))
    .select("paper_id", F.array_join("sentences", " ").alias("text"))
    .withColumn("words", F.size(F.split("text", "\s+")))
)


# In[ ]:


# text.show(n=5)


# In[ ]:


# if you want to limit the number of rows sent for processing, uncomment this line
 source_texts = text.limit(20)


# ### Retrieving Subtask Words

# On the basis of the bullet point in the task (starting with "Specifically, we want to know what the literature reports about: ...") I build a list of words I am going to look for in the text. I clean the list from stopwords, duplicates and get it lowercased.
# As mentioned before I focus on transmission dynamics, but this workflow may be used for any question in the task and other tasks as well.

# In[ ]:


task_specs = ('Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors'.replace(',', '')).lower()

transmission_dynamics_words = task_specs.split(' ')


# In[ ]:


#print(transmission_dynamics_words)


# Remove stopwords.

# In[ ]:


for stopword in stopwords.words('english'):
    if transmission_dynamics_words.__contains__(stopword):
        transmission_dynamics_words.remove(stopword)


# In[ ]:


# print(transmission_dynamics_words)


# Discarding duplicate words.

# In[ ]:


transmission_dynamics_words = list(set(transmission_dynamics_words)) 


# In[ ]:


# print(transmission_dynamics_words)


# Lowercase words.

# In[ ]:


for word in transmission_dynamics_words: 
    word.lower()


# ### Synonyms

# As presented on the diagram this step covers looking for synonyms to words in transmission_dynamics_words. It queries Merriam Webster via API to give me a JSON with the words with free Merriam Webster subscription and then performs initial data pre-processing.

# In[ ]:


synonyms_ragged = [] #temporary structure

for word in transmission_dynamics_words:
    url = 'https://www.dictionaryapi.com/api/v3/references/thesaurus/json/'+ word +'?key=26e5f7f5-056c-42e3-85ba-956a86beef64'
    synonyms_work = requests.get(url)
    synonyms_dictionary = synonyms_work.json()
    for i in (0, len(synonyms_dictionary)-1):
        try:
            dic = synonyms_dictionary[i]
            meta = dic['meta']
            syns = meta['syns']
            synonyms_ragged.append(syns)
        except:
            continue


synonyms = list(flatten(synonyms_ragged))


# In[ ]:


# print(synonyms)


# In[ ]:


for word in synonyms: 
    word.lower()


# ### Associations

# Calling Word Associations to seek for accociations for each word in transmission_dynamics_words.

# In[ ]:


associations = pd.DataFrame(['item', 'weight', 'pos'])
df_associations = pd.DataFrame()

for word in transmission_dynamics_words:
    url = 'https://api.wordassociations.net/associations/v1.0/json/search?apikey=786154c9-f2dd-483c-b5f1-6eb934c67275&text=' + word + '&lang=en'
    associations_work = requests.get(url)
    associations_as_dictionary = associations_work.json()
    response_body = associations_as_dictionary['response']
    # get the content of the response that is saved as list with one element 
    # clean response until it can be a dataframe
    response_content = response_body[0]
    response_content_str = response_content['items']
    response_content_str = response_content_str[1 : len(response_content_str)]
    for association in response_content_str:
        list_to_merge = [association['item'], association['weight'], association['pos']]
        df_to_merge = pd.DataFrame(list_to_merge)
        df_associations = pd.concat([df_associations, df_to_merge], axis=1)
        associations_chunk = df_associations.transpose()
        pd.concat([associations, associations_chunk], axis=0)


associations = df_associations.transpose()
    
associations.columns = ['item', 'weight', 'pos']


# In[ ]:


# print(associations.info())


# In[ ]:


associations.reset_index(inplace=True)


# In[ ]:


def lower_case_item(item):
    return item.lower()


associations['item'] = associations['item'].apply(lambda item : lower_case_item(item))


# In[ ]:


# print(associations.head())


# In[ ]:


# remove duplicates that may occur within associations
associations.drop_duplicates(subset='item', keep='first', inplace=True)


# ### Tokenizing, Removing Stopwords, Stemming

# Preparing texts and lists for analysis. 

# First full texts - will be tokenized, cleaned from stopwords and stemmed.

# In[ ]:


spark.conf.set("spark.sql.execution.arrow.enabled", "true")


# In[ ]:


# convert Spark format to Pandas
source_texts_df = source_texts.limit(10).toPandas()


# In[ ]:


source_texts_df['text'] = source_texts_df['text'].apply(lambda item : lower_case_item(item))


# In[ ]:


def tokenize_remove_stopwords(text):
    tokenized_text = word_tokenize(text)
    for stopword in stopwords.words('english'):
        if tokenized_text.__contains__(stopword):
            tokenized_text.remove(stopword)
    tokenized_text=[word.lower() for word in tokenized_text if word.isalpha()]
    return tokenized_text


source_texts_df['text_tokens'] = source_texts_df['text'].apply(lambda text: tokenize_remove_stopwords(text))


# In[ ]:


# print(source_texts_df['text_tokens'])


# In[ ]:


ps = PorterStemmer()


# In[ ]:


def stem_words(text):
    stemmed_text = []
    for word in text:
        stemmed_text.append(ps.stem(word))

    stemmed_text_string = ' '.join([str(elem) for elem in stemmed_text])

    return stemmed_text_string


source_texts_df['text_stemmed'] = source_texts_df['text_tokens'].apply(lambda text: stem_words(text))


# In[ ]:


# print(source_texts_df['text_stemmed'])


# Stem transmission_dynamics_words, synonyms and associations.

# In[ ]:


transmission_dynamics_words_stemmed = []

for word in transmission_dynamics_words:
    transmission_dynamics_words_stemmed.append(ps.stem(word))


synonyms_stemmed = []

for word in synonyms:
    synonyms_stemmed.append(ps.stem(word))
    

associations['item_stemmed'] = associations['item'].apply(lambda item: ps.stem(item))


# # Browsing texts for words

# Now that we've got:
# - texts 
# - list of words from the task - transmission_dynamics_words_stemmed
# - list of associations with words on transmission_dynamics_words - associations_stemmed
# - list of synonyms - synonyms_stemmed
# 
# we can proceed to sweeping texts for these words (let's call them 'keywords' from now on).
# 
# We do it in 3 steps:
# 1. Browse for words from task. Score texts: for every word found 1 point (no matter how many occurences). Pick 1000 best matches.
# 2. Browse for synonyms. Pick top scoring 700 texts from step 1 (or any other number you like, this is a parameter). Score texts: for every close synonym give 0.75 point, for every far synonym give 0.25 point. Add points to points from step 1.
# 3. Browse for associations. Pick top scoring texts from step 2. Score texts: every association has weight assigned up-front by Word Associations. Use this exact value for scoring. Total points. Pick 100 winners.

# In[ ]:


# define cutoff - number of texts to keep after browsing texts for key words
cutoff = 1000
per_word = 2
per_synonym = 1
# per association are defined in their respective df


# In[ ]:


# add columns to keep text scores to full_table with all the texts
source_texts_df['score_task_words'] = 0.00
source_texts_df['score_synonyms'] = 0.00
source_texts_df['score_associations'] = 0.00
source_texts_df['score_total'] = 0.00


# In[ ]:


# source_texts_df.info()


# In[ ]:


# browse by words from task
for index, row in source_texts_df.iterrows():
    row_score = 0
    for word in transmission_dynamics_words_stemmed:
        if str(row['text_stemmed']).find(word) != -1:
            row_score += per_word
            source_texts_df.at[index, 'score_task_words'] = row_score
            source_texts_df['score_total'] = source_texts_df['score_task_words'] + source_texts_df['score_synonyms'] + source_texts_df['score_associations']


# In[ ]:


# print(source_texts_df.head())


# In[ ]:


# browse by words from synonyms
for index, row in source_texts_df.iterrows():
    row_score = 0
    for word in synonyms_stemmed:
        if str(row['text_stemmed']).find(word) != -1:
            row_score += per_word
            source_texts_df.at[index, 'score_synonyms'] = row_score
            source_texts_df['score_total'] = source_texts_df['score_task_words'] + source_texts_df['score_synonyms'] + source_texts_df['score_associations']


# In[ ]:


# print(source_texts_df.head())


# In[ ]:


# browse by words from associations
for index, row in source_texts_df.iterrows():
    row_score = 0.00
    for word in associations['item_stemmed']:
        if str(row['text_stemmed']).find(word) != -1:
            row_score += float((associations.at[index, 'weight'])/100)
            source_texts_df.at[index, 'score_associations'] = row_score
            source_texts_df['score_total'] = source_texts_df['score_task_words'] + source_texts_df['score_synonyms'] + source_texts_df['score_associations']


# In[ ]:


# print(source_texts_df.head())


# In[ ]:


# get top texts (by cutoff parameter)
top_texts = source_texts_df.nlargest(cutoff, 'score_total', keep="all")


# In[ ]:


# print(top_texts['score_total'].head(20))


# In[ ]:


# export to .csv file
top_texts.to_csv('top_texts.csv')

