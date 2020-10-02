#!/usr/bin/env python
# coding: utf-8

# # CORD-19 Challenge
# In this notebook we attempt to answer key scientific questions for COVID-19 using the COVID-19 Open Research Dataset (CORD-19) by using a combination of **Text-mining** and **Natural Language Processin**g (NLP) tools such as** TF-IDF** and **BERT (finetuned on Question Answering Tasks)**. The key steps in our approach are illustrated in the following pipeline:
# ![cordpipeline](https://user-images.githubusercontent.com/55004415/79482903-7f654800-7fdf-11ea-95c9-a022a9ed2e3b.png)
# 

# # Results:

# In[ ]:


from IPython.core.display import display, HTML
import glob
for filename in glob.glob('/kaggle/input/cord-answers1/*.txt'):
    f = open(filename,"r")
    ans = f.read()
    display(HTML(ans))


# # Detailed Approach:

# ## 1. Find articles relevant to Coronavirus and Covid-19 by matching abstracts using **TF-IDF**
# 
# ### Loading Data into DF
# We load data from both summary csv file and all availiable text / abstracts from additional .json files. A quick scan through all the artcles in the dataset reveals that only a fraction in the dataset are pertaining to Coronaviruses. We thus propose the following:
# * Perform a query on all Abstracts to find matches of articles discussing Coronavirus and dump the rest. We make the assumption that if Coronavirus/Covid is not mentioned in the abstract, the article is likely not about Coronavirus/Covid.
# * After finding the relevant abstracts, we extract the body/full-text of the article to perform Q&A subsequently. Since not all abstracts have a full-text, we will only look for abstracts with full-text via their SHA. 

# In[ ]:


import numpy as np
import pandas as pd 
import glob
import json
import math

root_dir = '/kaggle/input/CORD-19-research-challenge' 
df = pd.read_csv(f'{root_dir}/metadata.csv') # Reading the metadata of the data set
sha_abstract = df[['sha', 'abstract']] # Filtering out rows with both `SHA` and `Abstract`
sha_abstract = sha_abstract.dropna().reset_index()[['sha', 'abstract']] # Separating `SHA` and `Abstract` columns into separate lists

corpus = list(sha_abstract.values[:,1])
sha = list(sha_abstract.values[:,0])


# ### Use TF-IDF to perform a Query search on all the abstracts to find relevant articles for Coronavirus
# TF-IDF (Term Frequency - Inverse Document Frequency) is a statistical measure that is often used to evaluate how important a word is to a document (here, the word is the query and the documents are the abstracts). The formula for computing the TF-IDF score of word $i$ in document $j$ is:
# 
# $$ \text{TF-IDF score} = TF(i,j)*IDF(i)\text{, where:}$$
# 
# $$TF(i,j) = \frac{\text{word i frequency in document j}}{\text{total # words in document j}}$$
# 
# $$IDF(i) = \log_2\left( \frac{\text{Total # of documents}}{\text{# documents with word i}}\right)$$
# 
# In addtition to accounting for the word importance to the document, we also make use of vectorization to convert the query and the individual abstracts into feature vectors which can be matched via their cosine similarity.
# 
# * We will perform TF-IDF + vectorization on the query and all the abstract paragraphs individually
#     * Remove common 'stop words' since we only care about the topic. We use the list from: https://gist.github.com/sebleier/554280
#     * Select a suitable feature size (need to be large enough to capture enough vocab)    
# * Then compute cosine-similarity to compare the similarity between the query vector and all the abstract vectors
# * Record SHAs of all the **non-zero** cosine-similarity

# In[ ]:


fs = open('/kaggle/input/stopwords/stopwordlist.txt','r') # Source of stopwordlist: https://gist.github.com/sebleier/554280
stopWordsList = fs.read().split(" ")


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

query = ['coronavirus coronaviruses cov covid']
qC = query+corpus
featuresize = 512 # It was found that increasing the featuresize did not change the number of abstracts found
vectorizer = TfidfVectorizer(stopWordsList,max_features=featuresize)
X = vectorizer.fit_transform(qC).toarray()
#print(vectorizer.get_feature_names())

covidCS = []
covid_SHA = []
ID = []
for i in range(len(corpus)):
    cossim = cosine_similarity(X[0].reshape(1,-1),X[i+1].reshape(1,-1))
    if cossim > 0.: 
        covidCS.append(cossim)
        covid_SHA.append(sha[i])
        ID.append(i)
print("{} out of {} abstracts contains your query for featuresize = {}".format(len(covidCS),len(corpus),featuresize))


# ### Extracting all paragraphs of all the relevant full-text to form our Q&A Database

# In[ ]:


# Each covid_SHA may have multiple SHAs. 
# Create a list of SHAs, with each element containing a list of SHAs pointing to the same article
full_sha = []
for sha in covid_SHA:
    newsha = sha.split('; ')
    full_sha.append(newsha)
    
doc_paths = glob.glob(f'{root_dir}/*/*/*/*.json')
root_dir = '/kaggle/input/CORD-19-research-challenge'
df = pd.read_csv(f'{root_dir}/metadata.csv')
sub_df = df[['sha', 'title', 'authors', 'url']]
def get_text(full_sha):
    # Initialise the full_text array of articles
    full_text=[]
    for full_sha_num in full_sha: # for each list of SHA pointing to the same article
        sha = full_sha_num[0]
        document_path = [path for path in doc_paths if sha in path] # we find the document path for the first SHA in the list as all SHA points to the same article
        with open(document_path[0]) as f:
            file = json.load(f)
            article_series = sub_df[sub_df['sha'].str.contains(sha, na=False)]
            sha = article_series.values[0][0]
            title = article_series.values[0][1]
            authors = article_series.values[0][2]
            url = article_series.values[0][3]
            for text_part in file['body_text']:
                text = text_part['text']
                # remove citations from each paragraph
                for citation in text_part['cite_spans']:
                    text = text.replace(citation['text'], "")
                full_text.append([sha, title, authors, url, text])
    return full_text
full_text = get_text(full_sha)
full_text_df = pd.DataFrame(full_text, columns=['sha','title','authors', 'url', 'text'])
pd.set_option('display.max_colwidth', 50)
full_text_df.head()
print("No. of paragraphs: ",len(full_text_df))


# ## 2. Find the top k paragraphs for each task topic and feed paragraphs into pre-trained BERT model to perform Q&A

# ### Import BERT from Huggingface Transformers
# BERT (Bidirectional Encoder Representations from Transformers), is a language model that has acheived state-of-the-art results in many natural language processing (NLP) tasks. It makes use of the attention mechanism to learn contextual relations between words in a text. We will not explain the details of the workings of the transformer, but refer the reader to good resources such as: http://jalammar.github.io/illustrated-transformer/ 
# 
# For the purpose of our tasks, we will use the BERT large-uncased pre-trained model config 'bert-large-cased' that has been fine-tuned on the Stanford Question Answering Dataset (SQuaD). This model is available from Huggingface transformers (see https://huggingface.co/transformers/) using the PyTorch framework.

# In[ ]:


import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')
model.eval(); # set model in eval mode


# In[ ]:


# Formatting functions to be used later

def highlight_paragraph(paragraph, token_start, token_end,num_seg_a):
    para =""
    para = para + " ".join(paragraph[num_seg_a:token_start])
    para = para + "[start]"  + " ".join(paragraph[token_start:token_end+1])+ "[end]"
    para = para + " ".join(paragraph[token_end+1:])
    return para

def split_paragraph(paragraph, qns):
    input_ids = tokenizer.encode(paragraph)
    #print(len(input_ids))
    qns_ids = tokenizer.encode(qns)
    #print(len(qns_ids))
    total_ids = qns_ids + input_ids
    tokens = tokenizer.convert_ids_to_tokens(total_ids)
    assert len(tokens) == len(qns_ids) + len(input_ids)
    num_sections = math.ceil(len(tokens)/512) 
    sections = []
    while len(tokens) > 512: 
        one_section = tokens[:512]
        one_sect_id = total_ids[:512]
        found_end = False
        i = 0
        last_index = len(one_section)
        while not found_end:
            i += 1
            if (one_section[last_index-i-1][-1] == "." and one_section[last_index-i][0].isupper()):
                found_end = True               
                sent_end = i
                sections.append((one_section[:last_index-sent_end],one_sect_id[:last_index-sent_end]))
                tokens = tokens[:len(qns_ids)+1]+tokens[last_index-sent_end:]
                total_ids = total_ids[:len(qns_ids)+1]+tokens[last_index-sent_end:]
    sections.append((tokens,total_ids))
    return sections


# ### Matching Top k = 20 paragraphs
# To avoid searching through all 190k paragraphs in our Q&A database (it is likely that most will not contain the answer), we first use TF-IDF and match the paragraphs to the Task topic and sort them in decreasing cosine similarity in order to find the top k relevant paragraphs to a single task topic.
# 
# ### Using BERT+Q&A:
# For each of the top k=20 paragraphs we will attempt to use BERT Q&A to find answers to the specific task question. Note that while we have used the task topics to shortlist the paragraphs based on relevancy, there is no guarantee that the shortlisted paragraphs will contain answers to our specific questions and thus answers would need to be manually inspected and assessed. The following was done to help filter 'good' answers:
# * We require answers to be between 1-10 words long
# * We modify our stop-word list to include to the stop list words that frequently occur but do not answer task questions and confuses the model

# In[ ]:


def askQuestion(taskno, paragraphs, titles, authors, urls):
    task = taskno[0] # element 0 is the task topic where tf-idf is performed as first filter
    question = taskno[1] # element 1 is the specific question
    tP = [task]+paragraphs
    vectorizer = TfidfVectorizer(stopWordsList,max_features=featuresize) 
    Y = vectorizer.fit_transform(tP).toarray() # vectorize task topic and every paragraph
    taskCS = []
    
    counter_threshold = 0
    # use cosine similarity to find top k paragraphs relevant to task topic
    for i in range(len(paragraphs)):
        cossim = cosine_similarity(Y[0].reshape(1,-1),Y[i+1].reshape(1,-1))
        taskCS.append(cossim[0][0])
        
    ## Find top k similar paragraphs
    k = 20

    ranked = np.argsort(taskCS)[::-1][:k]
    ranked_CS = [taskCS[i] for i in ranked]
    ranked_para = [paragraphs[i] for i in ranked]
    ranked_url = [urls[i] for i in ranked]
    ranked_title = [titles[i] for i in ranked]
    ranked_author = [authors[i] for i in ranked]
    
    sectionlist = []
    answerlist=[]
    urllist=[]
    authorlist=[]
    titlelist=[]
    
    for i in range(k):
        answer_text = ranked_para[i]
        sections = split_paragraph(answer_text,question)
        
        for s in range(len(sections)):
            urllist.append(ranked_url[i])
            authorlist.append(ranked_author[i])
            titlelist.append(ranked_title[i])
            
            tokens, input_ids = sections[s]

            sep_index = input_ids.index(tokenizer.sep_token_id) # Search the input_ids for the first instance of the `[SEP]` token.
            num_seg_a = sep_index + 1 # The number of segment A tokens includes the [SEP] token istelf
            num_seg_b = len(input_ids) - num_seg_a # The remainder are segment B.   
            segment_ids = [0]*num_seg_a + [1]*num_seg_b # Construct the list of 0s and 1s.
            assert len(segment_ids) == len(input_ids) # There should be a segment_id for every input token.
    
            start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                     token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

            answer_start = torch.argmax(start_scores) # Find the tokens with the highest `start` and `end` scores.
            answer_end = torch.argmax(end_scores)   
            answer = ' '.join(tokens[answer_start:answer_end+1]) # Combine the tokens in the answer and print it out.

            para = highlight_paragraph(tokens,answer_start,answer_end,num_seg_a)
            para = para.replace(' ##', '')
            para = para.replace('[CLS]', '')
            para = para.replace('[SEP]', '')
            max_length = 10 # we limit answer lengths to < 10 words
            if (answer_start != answer_end & answer_end - answer_start < max_length):  
                answerlist.append(answer)
                sectionlist.append(para)
                
            else:
                sectionlist.append("Answer not found")
                answerlist.append("Answer not found")
    
    new_df = pd.DataFrame(list(zip(answerlist,sectionlist,titlelist,authorlist,urllist)),columns=['Answers','Evidence','Title','Authors','URL'])
    
    new_df = new_df[~new_df['Answers'].str.match("Answer not found")]
    new_df = new_df.iloc[:,1:]
    return new_df


# We list the Task topic and Task questions we hope to address and perform them one by one sequentially

# In[ ]:


paragraphs = list(full_text_df.values[:,4])
SHAp = list(full_text_df.values[:,0])
titles = list(full_text_df.values[:,1])
authors = list(full_text_df.values[:,2])
urls = list(full_text_df.values[:,3])

featuresize = 1024

task1 = ['incubation', "How long is the incubation period?"]
task2 = ['movement strategies',"Effectiveness of movement control strategies"]
task3 = ['transmission asymptomatic',"Prevalence of asymptomatic shedding and transmission"]
task4 = ['transmission', 'mode of transmission']


# In[ ]:


#for taskno in [task1,task2,task3]:
taskno = task1
answers_df = askQuestion(taskno, paragraphs, titles, authors, urls)

html_df = answers_df.to_html()
html_df = html_df.replace('[start]', '<span style="color: blue;"> ')
html_df = html_df.replace('[end]', ' </span>')
html_df = "<h2>Question: "+taskno[1]+"</h2>"+html_df
html_df = "<h2>Topics: "+taskno[0]+"</h2>"+html_df
display(HTML(html_df))


# We manually filter the 3 best answers for each question and present it below

# In[ ]:


filtered = answers_df.loc[[7,9,11],['Evidence','Title','Authors', 'URL']]
html_df = filtered.to_html()
html_df = html_df.replace('[start]', '<span style="color: blue;"> ')
html_df = html_df.replace('[end]', ' </span>')
html_df = "<h2>Question: "+taskno[1]+"</h2>"+html_df
html_df = "<h2>Topics: "+taskno[0]+"</h2>"+html_df
display(HTML(html_df))

# Save answers as txt file
# with open("incubation_answers.txt", "w+") as f:
#     f.write(html_df)


# The same proceedure was performed for the rest of the tasks and results were saved to file. We present the rest of the results below: 

# In[ ]:


for filename in glob.glob('/kaggle/input/cord-answers1/*.txt'):
    f = open(filename,"r")
    ans = f.read()
    display(HTML(ans))


# ## Concluding Thoughts
# * Simple approach of using TF-IDF to sequentially filter the corpus of documents + using a pre-trained language model finetuned on Q&A tasks to highlight answers from the shortlisted paragraphs allowed us to answer some of the key questions with reasonable accuracy.
# * However, this approach was effective only to answer very specific questions (such as identifying the length of incubation periods, etc) and would not be very effective in answering broader questions such as "What do we know about the relationship between the environment and the virus?" (unless an article specifically reports that word-for-word of course). 
# * In this approach, most of the Information Retrieval was performed in the TF-IDF step, which converts queries and documents into a simple Bag-of-Words representation. As opposed to current state-of-the-art language models that makes use of attention mechanisms to learn contextual representations, a simple Bag-of-Words representation has poor contextual representation.
# * For convenience, we used the BERT-finetuned-on-SQuAD model available from Huggingface transformers. This model was pretrained on corpora of texts from Wikipedia and fine-tuned on questions posed on Wikipedia articles (SQuAD). A possible improvement would be to use models that are pre-trained on more relevant domain-specific corpus such as BioBERT or SciBERT (which have been pre-trained on biomedical and scientific publications respectively), and/or further fine-tuning on domain-specific Q&A datasets. One such example could be the PubMedQA Dataset (https://arxiv.org/abs/1909.06146). 

# ---

# ## Credits:
# 
# This work is part of a collaborative effort with @jingjielim, @strifonov, @atanasova, @darumen, @preslav.
# 
# It covers the first branch in our diagram of contribution shown below:
# 
# ![cord-logo-final](https://user-images.githubusercontent.com/55004415/79496823-b42fca00-7ff4-11ea-993f-cd5792955641.png)
