#!/usr/bin/env python
# coding: utf-8

# # SBERT+BERT for Cord-19 Data
# ## Introduction
# 
# The development of **Question Answering** (QA) system is necessary for rapidly emerging domains, such as the ongoing **coronavirus disease of 2019** (COVID-19) pandemic. In particular, when no suitable domain-specific resources are likely available at the starting. 
# To respond to the needs of medical experts to quickly and accurately receive answers to their scientific questions related to coronaviruses, we can develop QA systems based on articles related to COVID-19. Thus, the Kaggle opened this competition named the **COVID-19 Open Research Dataset (CORD-19)** and proposed the **CORD-19** dataset that encompasses 120K articles about coronaviruses and other diseases. The competition offered more than ten tasks to cover some fundamental questions related to COVID 19 and provided the chance for the ML community to develop QA systems and employ them on the CORD-19 dataset.  In this notebook, we have been focused on finding answers for **What is the efficacy of novel therapeutics being tested currently?**. 
# " 
# 
# ## Approaches
# 
# To do so, we implemented two frameworks (SBERT+BERT and LDA+ALBERT) that are based on advanced NLP and ML tools. This notebook is focusing on SBERT+BERT. In another notebook, we are using [LDA+ALBERT](https://www.kaggle.com/parkyoona/lda-albert-for-cord-19-data). 
# 
# ### SBERT+BERT
# The first framework based on [BERT](https://arxiv.org/abs/1810.04805) model. BERT (Bidirectional Encoder Representations from Transformers) is a well-known language representation model that is designed to pre-train deep bidirectional representations from the unlabeled text by jointly conditioning on both left and right context in all layers. The pre-trained BERT model can be fine-tuned with just adding one additional output layer and can be used for a wide range of tasks, such as question answering and language inference. 
# In this notebook, we are using Sentence-BERT ([SBERT](https://arxiv.org/abs/1908.10084)) as well. SBER is an enhancement of the BERT and is useful in developing a semantic-based sentence embedding.
# 
# ### LDA+ALBERT
# 
# The second framework is based on [ALBERT](https://arxiv.org/abs/1909.11942). 
# ALBERT has similar architecture as other BERT models, but it is based on a transformer encoder with Gaussian Error Linear Units (GELU) nonlinearities. ALBERT uses a different embedding method than BERT. In more detail, ALBERT uses two-step word embedding that first projects a word into a lower-dimensional embedding space and then extends it to the hidden space. Furthermore, ALBERT uses a cross-layer parameter sharing to improve parameter efficiency; it only uses feed-forward network (FFN) parameters across layers to share attention parameters. Another difference between ALBER and BERT is that ALBERT uses a sentence-order prediction (SOP) loss to avoid topic prediction and focus on modeling inter-sentence coherence. 

# In[ ]:


from IPython.display import Image
Image('../input/system-diagram/BERT-Based QA System Diagram.jpg')


# The QA system diagram is shown in Figure 1. The articles were first filtered using a keyword search which filters out articles that do not contain COVID-19 keywords in the titles. This reduced the number of scholarly articles from 120,464 to 31,450. Once filtered, the top n articles were extracted by embedding the query and the article titles using [Sentence-BERT](https://arxiv.org/abs/1908.10084), measuring the cosine similarity between the query embedding and each article title embedding, and sorting the articles by the cosine similarity score. Only the top n articles with the highest cosine similarity scores were kept. In this experiment, we set n as 100.  We used a pre-trained Sentence-BERT model, ['bert-base-nli-stsb-mean-tokens'](https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/sts-models.md). This model is a BERT-base model with mean-tokens trained on AllNLI and then on STS benchmark training set.
# 
# Once the top articles were extracted, the primary endpoint of each article was extracted using QA BERT. To do this, we set the question as the article title, the context as the article conclusion, and the answer as the primary endpoint of the article. We used a pre-trained QA BERT-large-uncased model with whole word masking fine-tuned on SQuAD, ['bert-large-uncased-whole-word-masking-finetuned-squad'](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad#). This whole process from input to output took 532.13 seconds.

# Figure 1: BERT-based QA System Diagram

# ## Step 0. Install and Import All Required Libraries 

# In[ ]:


# turn the internet on for this to install properly
get_ipython().system(' pip install -U sentence-transformers')
import scipy.spatial
import numpy as np
import os, json
import glob
import re
import torch
import pandas as pd


# ### Step 0-1. Read Input Data

# In[ ]:


import scipy.spatial
import numpy as np
import os, json
import glob
import re
import torch
import pandas as pd

data_dirs = ['../input/CORD-19-research-challenge/document_parses/pmc_json', 
             '../input/CORD-19-research-challenge/document_parses/pdf_json'
            ]
json_article_paths = []
for data_dir in data_dirs:
    json_article_paths = json_article_paths + glob.glob(os.path.join(data_dir, "*.json"))


# ## Step 1. The First Filtering Method
# 
# The Cord-19 dataset encompasses around 120K articles. We use keyword such as keywords **RNA virus, SARS, coronavirus, COVID, SARS-Cov-2, -Co, 2019-nCoV, vaccine, Antibody-Dependent Enhancement, therapeutic, prophylaxis clinical, naproxen, clarithromycin, minocyclinethat** to filter articles that are not related to the the following question:  
# 1. What is the efficacy of novel therapeutics being tested currently?
# 
# As the results of our filtering method we find the titles and paper ids of paper that are relevent to our questions. 

# In[ ]:


print(len(json_article_paths))


# There are initially 120,464 scholarly articles in the CORD-19 dataset.

# In[ ]:


import os, json

if not os.path.exists('../input/filtered-data/filtered_df.csv'):
    # synonyms to COVID-19 according to wikipedia
    keywords = ['persistence','decontamination','RNA virus',' SARS','coronavirus', 'COVID', 'SARS-Cov-2', '-CoV', '2019-nCoV','coronavirus vaccine','Antibody-Dependent Enhancement','therapeutic','prophylaxis clinical','naproxen','clarithromycin','minocyclinethat']

    #keywords
    titles = []
    paper_ids = []
    #json_article_paths[0:10000]: #TODO Change to all articles when done
    for json_file in json_article_paths: 
        # read json file into doc
        doc = json.load(open(json_file))

        # clean title
        title = doc['metadata']['title']  
        title = re.sub(r'[^\x00-\x7F]',' ', title)

        # append article only if it contains any of the keywords in its title
        if title != '' and any(keyword.lower() in title.lower() for keyword in keywords):
            titles.append(title)
            paper_ids.append(doc['paper_id'])

    print(len(paper_ids))


# After filtering out articles that do not contain COVID-19 keywords in the titles, the number of scholarly articles was reduced from 120,464 to 31.450.

# In[ ]:


if os.path.exists('../input/filtered-data/filtered_df.csv'):
    filtered_df = pd.read_csv('../input/filtered-data/filtered_df.csv')

else:
    keyword_articles_df = pd.DataFrame({
        'title': titles, 
        'paper_id': paper_ids
    })

    meta_df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')

    filtered_df = pd.merge(meta_df, keyword_articles_df)
    filtered_df = filtered_df.drop_duplicates(subset='title')
    filtered_df = filtered_df.dropna(subset=['abstract'])
    
filtered_df.head()


# Out of the filtered articles, only 11,468 articles contain metadata, have unique titles, and contain abstracts.

# In[ ]:


titles = list(filtered_df.title)
paper_ids = list(filtered_df.paper_id)


# ## Step 2. The First Filtering Method Extract Answers by Finding Semantically Similar PubMed Articles Using SBERT
# 
# We are using [SBERT](http://https://pypi.org/project/sentence-transformers/) to derive sentence embeddings from our query set and the titles that have been collected in the pervious step. We use computes the cosine similarity between the sentence embeddings. 

# In[ ]:


from sentence_transformers import SentenceTransformer
from transformers import BertForQuestionAnswering


# In[ ]:


def get_top_n_similar_articles_df(question, titles):
    embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    query_embeddings = embedder.encode([question])

    # list of article titles
    title_embeddings = embedder.encode(titles)

    # get top 50 article titles based on cosine similarity
    closest_n = 50
    distances = scipy.spatial.distance.cdist(query_embeddings, title_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    # save similar articles info
    top_paper_ids = []
    top_titles = []
    top_similarity_scores = []
    top_abstracts = []
    abstracts = list(filtered_df.abstract)

    print('Query: ' + question + '\n')

    # Find the closest 50 article titles for each query sentence based on cosine similarity
    for idx, distance in results[0:closest_n]:
        top_paper_ids.append(paper_ids[idx])
        top_titles.append(titles[idx])
        top_similarity_scores.append(round((1-distance), 4))
        top_abstracts.append(abstracts[idx])
        print('Paper ID: ' + paper_ids[idx])
        print('PubMed Article Title: ' + titles[idx])
        print('Similarity Score: ' + "%.4f" % (1-distance))
        print('\n')
        
    top_50_similar_articles_df = pd.DataFrame({
        'paper_id': top_paper_ids,
        'cosine_similarity': top_similarity_scores,
        'title': top_titles,
        'abstract': top_abstracts
    })
    
    return top_50_similar_articles_df


# ## Step 2-1. Get Top Articles for Question 1

# In[ ]:


# research question
queries = ['What is the efficacy of novel therapeutics being tested currently?', 
           'What is the best method to combat the hypercoagulable state seen in COVID-19?']

if os.path.exists('../input/top-50-similar-articles/top_50_similar_articles_df.csv'):
    q1_top_50_similar_articles_df = pd.read_csv('../input/top-50-similar-articles/top_50_similar_articles_df.csv')

else:
    q1_top_50_similar_articles_df = get_top_n_similar_articles_df(queries[0], titles)


# In[ ]:


q1_top_50_similar_articles_df[['cosine_similarity', 'title']].head()


# ## Step 2-2. Get Top Articles for Question 2

# In[ ]:


# research question
queries = ['What is the efficacy of novel therapeutics being tested currently?', 
           'What is the best method to combat the hypercoagulable state?']

if os.path.exists('../input/q2-top-50-similar-articles-df/q2_top_50_similar_articles_df.csv'):
    q2_top_50_similar_articles_df = pd.read_csv('../input/q2-top-50-similar-articles-df/q2_top_50_similar_articles_df.csv')

else:
    q2_top_50_similar_articles_df = get_top_n_similar_articles_df(queries[1], titles)


# In[ ]:


q2_top_50_similar_articles_df.head()


# ## Step 3. Extract Excerpt From Abstracts of Relevant Articles Using QA BERT
# The article title is the question and the excerpt is the answer to the question. The answer to the question is extracted from the abstract of the article. We used the pre-trained QA BERT model "bert-large-uncased-whole-word-masking-finetuned-squad". 

# In[ ]:


model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


# In[ ]:


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


# In[ ]:


def extract_answer_from_text(question, text):
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, text)
    input_ids = input_ids[0:512]
    
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)
    
    # Run our embeddings through the model
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from texts
    
    # BERT only needs the token IDs, but for the purpose of inspecting the 
    # tokenizer's behavior, let's also get the token strings and display them.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    
    if answer_start <= 0 or answer_end <= 0 or answer_end <= answer_start:
        answer = "Not Relevant"
        score = float('-inf')
    
    else:
        # Start with the first token.
        answer = tokens[answer_start]

        # Select the remaining answer tokens and join them with whitespace.
        for i in range(answer_start + 1, answer_end + 1):

            # If it's a subword token, then recombine it with the previous token.
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]

            # Otherwise, add a space then the token.
            else:
                answer += ' ' + tokens[i]

        # extract answer
        answer = answer.replace('[CLS]', '')
        answer = answer.replace('[SEP]', '').strip()

        # extract score
        score = (start_scores.max() + end_scores.max()) / 2
        score = score.item()

    return answer, score


# In[ ]:


def get_excerpts_and_scores(question, abstracts):
    excerpts = []
    scores = []

    for index, abstract in enumerate(abstracts):
        # extract excerpt from abstract
        excerpt, score = extract_answer_from_text(question, abstract)
        excerpts.append(excerpt)
        scores.append(score)

    return excerpts, scores


# In[ ]:


import colorama
import re 

def print_top_n_articles(question, top_n_articles, top_n_similar_articles_df, scores, excerpts, top_indices):
    print("Prediction highlighted in red....")
    print("========  " + question + "  ======== \n")

    for i, top_idx in enumerate(top_indices):
        print("Rank: " + str(i+1))
        
        # get top 50 articles
        data = top_n_similar_articles_df.iloc[top_idx]
        
        print("Title : " + data['title'])
        print("Confidence: " + str(scores[top_idx]))
        
        abstract = data['abstract']
        
        # clearn excerpt
        excerpt = excerpts[top_idx]
        excerpt = re.sub(' -', '-', excerpt)
        excerpt = re.sub('- ', '-', excerpt)
        excerpt = re.sub(' ,', ',', excerpt)
        excerpt = re.sub(r'\s([?.!"](?:\s|$))', r'\1', excerpt)
        excerpt = re.sub('\( ', '(', excerpt)
        excerpt = re.sub(' \)', ')', excerpt)
        
        # put excerpt in red font
        insensitive_excerpt = re.compile(re.escape(excerpt), re.IGNORECASE)
        highlighted_txt = insensitive_excerpt.sub('\033[31m' + excerpt + '\033[39m', abstract)
        print("Abstract: " + highlighted_txt)
        print('\n')


# ## Step 3-1. Print Top 10 Answers to Question 1
# The answers are the excerpts of the abstracts highlighted in red.

# In[ ]:


# get excerpts and scores for question 1
q1_excerpts, q1_scores = get_excerpts_and_scores(queries[0], q1_top_50_similar_articles_df.abstract[0:40])
n = 10
q1_top_indices = [i[0] for i in sorted(enumerate(q1_scores), key=lambda x:-x[1])][0:n]


# In[ ]:


print_top_n_articles(queries[0], n, q1_top_50_similar_articles_df, q1_scores, q1_excerpts, q1_top_indices)


# ## Step 3-1. Print Top 10 Answers to Question 2
# The answers are the excerpts of the abstracts highlighted in red.

# In[ ]:


# get excerpts and scores for question 1
q2_excerpts, q2_scores = get_excerpts_and_scores(queries[1], q2_top_50_similar_articles_df.abstract[0:40])
n = 10
q2_top_indices = [i[0] for i in sorted(enumerate(q2_scores), key=lambda x:-x[1])][0:n]


# In[ ]:


print_top_n_articles(queries[1], n, q2_top_50_similar_articles_df, q2_scores, q2_excerpts, q2_top_indices)

