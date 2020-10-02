#!/usr/bin/env python
# coding: utf-8

# # Using Novel Language Models and Web scraping to Effectively Identify Articles related to Therapeutics and Vaccines
#  * Team: MD-Lab, ASU
#  * Author: Jitesh Pabla, Email: jpabla1@asu.edu, Kaggle ID: jiteshpabla
#  * Team Members: Rishab Banerjee, Hong Guan, Ashwin Karthik Ambalavanan, Mihir Parmar, Murthy Devarakonda
#  * Email ID: loccapollo@gmail.com, hguan6@asu.edu, aambalav@asu.edu, mparmar3@asu.edu, Murthy.Devarakonda@asu.edu
#  * Kaggle ID: loccapollo, hongguan, ashwinambal96, mihir3031, murthydevarakonda
#  * This is a Team Submission
#  * Here are the links to our teams Kernels:
#      - https://www.kaggle.com/jiteshpabla/classifying-cord-19-articles-using-elasticbert/edit
#      - https://www.kaggle.com/ashwinambal96/scibert-based-article-identification
#      - https://www.kaggle.com/hongguan/micro-scorers-for-covid-19-open-challenge/
#      - https://www.kaggle.com/loccapollo/lexicon-based-similarity-scoring-with-bert-biobert
#      - https://www.kaggle.com/mihir3031/bert-sts-for-searching-relevant-research-papers
#      - The final ensembling that combines everything together: http://https://www.kaggle.com/hongguan/ensemble-model-for-covid-19-open-challenge/

# # Introduction
# This repository deals with the "cord19-vaccines-and-therapeutics" dataset which is based on the ["What do we know about vaccines and therapeutics?"](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=561) task of the COVID-19 Open Research Dataset Challenge (CORD-19).

# # part1: Extracting Articles from Google

# The "cord19-vaccines-and-therapeutics" dataset is extracted from Google scholar using the [Publish or Perish tool](https://harzing.com/resources/publish-or-perish) by using the queries like:
#  - "vaccine"
#  - "therapeutics"
#  - "Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers"
#  - etc.
#  along with the keyword "coronavirus"

# ## Code

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob

# Input data files are available in the "../input/" directory.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# load the metadata file and remove duplicates (if any)

# In[ ]:


metadf = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
metadf = metadf.drop_duplicates()
metadf


# load all the files from the google search results dataset into a dictionary of dataframes (key being the query; value being the dataframe)

# In[ ]:


DATA_DIR = "/kaggle/input/cord19-vaccines-and-therapeutics-google-results/"
google_files = glob.glob(DATA_DIR + "4*.csv")
google_files


# In[ ]:


google_files_df = {}
for filename in google_files:
    print(filename)
    df = pd.read_csv(filename)
    #remove unnecessary or empty columns
    df = df.drop(columns=['QueryDate', "Type", "DOI", "ISSN", "CitationURL", "Volume", "Issue", "StartPage", "EndPage", "ECC", "CitesPerAuthor", "AuthorCount"])
    # remove all empty rows
    df = df.dropna(how='all')
    # rename title coumns
    df = df.rename(columns={"Title": "title"}, errors="raise")
    # only take first 400
    df = df[:400]
    # string matching for keys
    result_index = filename.find('results/')
    google_files_df[filename[result_index+8:-4]] = df
google_files_df


# Generate a normalized score from 0-1 based on the ranks using the formula:
# ```
# 1 - (rank/total)
# ```
# where
# - rank = rank of article
# - total = total number of articles

# In[ ]:


def get_score(rank_col):
    score = 1 - (rank_col/len(rank_col))
    return score


# In[ ]:


for current_item in google_files_df.items():
    current_query, current_df = current_item
    current_df['score from rank'] = get_score(current_df['GSRank'])
google_files_df


# Perform a merge on the "title" column the metadata and the Google search results data to find the common articles

# In[ ]:


matched_df = {}
for current_item in google_files_df.items():
    current_query, current_df = current_item
    matched_df[current_query] = pd.merge(current_df, metadf, on="title")
matched_df


# View a single row

# In[ ]:


x = matched_df['400Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers'].loc[1]
x


# ## Analysis

# Find the number of common articles in each dataframe (per query dataframe) to guage how much overlap of information exists between each query

# In[ ]:


for i, current_item in enumerate(matched_df.items()):
    current_query, current_df = current_item
    print("#######Common articles for ", current_query)
    #matched_df[current_query] = pd.merge(current_df, metadf, on="title")
    for j, compare_item in enumerate(matched_df.items()):
        #if not i==j:
        compare_query, compare_df = compare_item
        merged_df = current_df.merge(compare_df, on=['title'], how='inner', indicator=True)
        print(compare_query[2:40], "--", len(merged_df))


# Using word cloud to visualize the common wordsin the articles of various queries

# In[ ]:


from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt


# In[ ]:


for i, current_item in enumerate(matched_df.items()):
    current_query, tempdf = current_item
    tit_text = " ".join(str(tit) for tit in tempdf.title)
    abs_text = " ".join(str(abs) for abs in tempdf.Abstract)
    text = tit_text + " " + abs_text 
    print(current_query)
    # Create and generate a word cloud image:=
    # lower max_font_size, change the maximum number of word and lighten the background:
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    fig = plt.figure(figsize=(10,5))
    #fig.title(current_query)
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# # Part2: creating training and testing dataset

# Give class=1 to vaccine; class=2 for therpeutics; and class=0 for other

# In[ ]:


vacc_df = matched_df["420vaccine"]
vacc_df["class"] = 1
ther_df = matched_df["410therapeutics"]
ther_df["class"] = 2


# In[ ]:


#positive samples
concat_google_df = pd.concat([vacc_df, ther_df], axis=0, ignore_index=True)
concat_google_df = concat_google_df.drop_duplicates(subset='title', keep="first")
concat_google_df


# The nagative samples are extracted by taking a completely seperate query from the 9 given challenges of the CORD-19 challenge which are semantically very different from "vaccines and therapeutics" to make sure that the negative samples do not contain any information that should be a part of the positive samples.
# 
# They are extracted and matched in the same manner as the postive samples.

# In[ ]:


#negative samples
google_files_neg = glob.glob(DATA_DIR + "NEGATIVE*.csv")

google_df_neg = []
for filename in google_files_neg:
    df = pd.read_csv(filename)
    google_df_neg.append(df)

concat_google_df_neg = pd.concat(google_df_neg, axis=0, ignore_index=True)
concat_google_df_neg = concat_google_df_neg.drop_duplicates(subset="title")
concat_google_df_neg["class"] = 0
concat_google_df_neg


# In[ ]:


# find comman samples between postive and negative samples
merged_df = concat_google_df.merge(concat_google_df_neg, on=['title'], 
                   how='inner', indicator=True, suffixes=('', '_y'))
merged_df.drop(list(merged_df.filter(regex='_y$')), axis=1, inplace=True)
merged_df


# In[ ]:


#removing the common articles from the nagative samples
without_commom_df = pd.concat([merged_df, concat_google_df_neg])
without_commom_df = without_commom_df.drop_duplicates(keep=False, subset="title")
without_commom_df =  without_commom_df.drop(columns=["_merge"])
without_commom_df


# splitting the samples equally into train (80%) and test data (20%)

# In[ ]:


#split into train test
def split_traintest(df):
  #shuffle
  df = df.sample(frac=1)
  #first 20%
  index = int(len(df)*0.2)
  print(len(df))
  print(index)
  test_df = df.iloc[:index]
  train_df = df.iloc[index:]
  return test_df, train_df

ther_test, ther_train = split_traintest(ther_df)
vacc_test, vacc_train = split_traintest(vacc_df)
neg_test, neg_train = split_traintest(without_commom_df)


# In[ ]:


def save_csv(df_list, name):
  final_df = pd.concat([df_list[0],df_list[1], df_list[2]])
  final_df.to_csv("GOOGLE_CLASSIFIED_samples_"+name+'.csv')
  print(len(final_df))

save_csv([ther_test, vacc_test, neg_test], "test")
save_csv([ther_train, vacc_train, neg_train], "train")


# # Part 3: Training SciBERT on the Google extracted labels to get labels for entire dataset

# The analysis shows that there is not many common articles between "vaccine" and other long queries like "Approaches to evaluate risk for enhanced disease after vaccination" etc. The same goes for "therapeutics" query.
# Since the long queries are very complex and specific we decide to classify the articles into the "vaccines" and "therapeutics" categories first.

# ### part 3-a: load/install

# In[ ]:


get_ipython().system('pip install -q tensorflow_gpu>=2.0')


# In[ ]:


import tensorflow as tf
print(tf.__version__)


# In[ ]:


get_ipython().system('pip install -q ktrain')


# In[ ]:


import pandas as pd
import numpy as np
import glob


# In[ ]:


test = pd.read_csv("GOOGLE_CLASSIFIED_samples_test.csv")
train = pd.read_csv("GOOGLE_CLASSIFIED_samples_train.csv")


# In[ ]:


test = test.fillna("")
train = train.fillna("")

test["text"] = test["title"] +";"+ test["abstract"] +";"+ test["journal"]
train["text"] = train["title"] +";"+ train["abstract"] +";"+ train["journal"]


print('size of training set: %s' % (len(train)))
print('size of validation set: %s' % (len(test)))

x_train = train.text.to_list()
y_train = train["class"].to_list()
x_test = test.text.to_list()
y_test = test["class"].to_list()

for x in x_train[:10]:
  print(x)

print(y_train[:10])


# In[ ]:


import ktrain
from ktrain import text


# ### part 3-b: train/load model
# ### (Option 1) finetuning the SciBERT model on the dataset

# In[ ]:


MODEL_NAME = 'allenai/scibert_scivocab_uncased' #'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=500, classes=['0', '1', '2'])
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)


# In[ ]:


learner.fit_onecycle(5e-5, 4)


# In[ ]:


learner.validate(class_names=t.get_classes())


# ### (Option 2) load the pretrained model from the dataset files

# In[ ]:


predictor = ktrain.load_predictor(DATA_DIR+'GOOGLE_scibert_predictor/GOOGLE_scibert_predictor')


# ### part 3-c: do the final predictions

# In[ ]:


metadf = metadf.fillna("")

metadf["text"] = metadf["title"] +";"+ metadf["abstract"] +";"+ metadf["journal"]

metadf["class"] = ""
metadf["0"] = 0
metadf["1"] = 0
metadf["2"] = 0

metadf


# In[ ]:


# test on 1 sample
predictor.predict_proba(metadf.text.iloc[0])


# In[ ]:


i=0
for index, row in metadf.iterrows():
    probs = predictor.predict_proba(row["text"])
    classif = predictor.predict(row["text"])
    metadf.loc[index, "class"] = classif
    metadf.loc[index, "0"] = probs[0]
    metadf.loc[index, "1"] = probs[1]
    metadf.loc[index, "2"] = probs[2]
    if i%200 == 0:
        print(index)
    #if i%500 == 0:
        # uncomment to save after every 500 samples
        #metadf_diff.to_csv("GOOGLE_CLASSIFIED_metadata_diff_"+str(i)+'.csv')
    i = i+1

metadf_diff
#metadf_diff.to_csv("GOOGLE_CLASSIFIED_metadata_"+"final"+'.csv')

