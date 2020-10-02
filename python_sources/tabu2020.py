#!/usr/bin/env python
# coding: utf-8

# # COVID-19 - Information Extraction Combining Unsupervised and Supervised Learning
# 
# ## What is known about transmission, incubation, and environmental stability?

# # Content
# 
# ## A. Methodology
# 
# 1. Keyword selection from Task's questions
# 2. Extraction of contexts containing keywords 
# 3. Representation of contexts in a vector space
# 4. Clustering of contexts using LDA (Latent Dirichlet Allocation)
# 5. Building a database for supervised learning
#    - Input: contexts (see 2 above)
#    - Output: groups of contexts (see 4 above)
# 6. Training classifiers using Bag of Words
# 7. Query-based retrieval of relevant contexts using logistic regression
# 8. Ranking and returning contexts using cosine distance between query and retrieved contexts
#    - Wordcloud visualization of results
# 
# ## B. Findings
# 
# ## C. Pros and Cons

# ## 1. Keyword selection from Task's questions
# 
# Transmission, transmitted, transmitting, incubate, incubation, incubated, environmental, stability, asymptomatic, hysical science,	protection, health status, distribution, hydrophilic, phobic, environment, decontamination, nasal, fecal, model, phenotypic, phenotype, immunity, protective.

# ## 2. Extraction of contexts containing keywords from dataset

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# In[ ]:


#List to save contexts
contexts = []


# The code chunk below takes time to process due to its computational cost. However, it is not neccessary to run it as we have created and uploaded the two files that it is intended to create. That is one containing all the paths to relevant articles, and another one cotaining all the contexts relevant to the defined keywords. <b>Therefore this step may be skipped and proceed to 2.1</b>. If you want to run it, please uncomment the code below.

# In[ ]:


import json
import spacy

print('It\'s running!\n')

nlp = spacy.load('en')
vocab = ['transmission', 'transmitted', 'transmitting', 'incubation', 'incubate', 'incubated', 'environmental', 'stability', 'health status', 'asymptomatic', 'physical science', 'distribution', 'hydrophilic', 'phobic', 'environment', 'decontamination', 'nasal', 'fecal', 'model', 'phenotypic', 'phenotype', 'immunity', 'protective', 'protection']

paths = open('/kaggle/input/file-paths/file_paths.txt', 'r')
count = 0
for p in paths:
    
    if count >=-1:
        
        text = ""
        try:
            data = open(p[:-1]).read()
        except:
            continue
        
        data_dic = json.loads(data)
        try:
            title = data_dic['metadata']['title']
        except:
            title = ""
        try:
            abstract = data_dic['abstract'][0]['text']
        except:
            abstract = ""
        
        body_structure = data_dic['body_text']
        body = ""
        for i in range(0,len(body_structure)):
            body+= body_structure[i]['text'] + " "
        
        text += title + ". " + abstract + " " + body + "\n"

        try:
            doc = nlp(text)
        except:
            print('Article skipped due to the number of tokens, more than 1 million!')
            continue

        for token in doc:
            if str(token) in vocab:
                contexts.append(str(token.sent)) #Contexts, that is full sentences containing keywords,
                                                 #extracted here.
    count+=1
    if count == 29335:
        break
paths.close()
print('Finished!')


# ## 2.1 Filtering out irrelevant contexts

# In[ ]:


import re

if len(contexts)==0:
    ctx = open('/kaggle/input/all-contexts/Contexts_Kaggle_Covid-19_2.txt', 'r')
else:
    ctx = contexts

contx = []
for c in ctx:
    c = re.sub('Context: ','',c)
    c = c[:-1]
    #Do not take into account contexs which contains the pronoun "we"...
    pron = c.lower().split()
    if "we" in pron:
        continue
    elif "COVID-19".lower() in c.lower() or "SARS-COV-2".lower() in c.lower(): # make sure the contexts talk about COVID-19 and not about other viruses. 
        contx.append(c.lower())

print("Number of contexts: ", len(contx))


# ## 3. Representation of contexts in a vector space. Bag-of-Words representation.

# In[ ]:


import sklearn
from sklearn.feature_extraction.text import CountVectorizer as BoW
from sklearn.feature_extraction.text import TfidfVectorizer as TF_IDF

bow = BoW(ngram_range=(1,1)).fit(contx)
bw = bow.transform(contx)
X = bw.toarray()
print("Dimension of the matrix of Data: ", X.shape)


# ## 4. Clustering of contexts using LDA (Latent Dirichlet Allocation)

# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=5) #Number of clusters (5)
lda.fit(bw)

#get clusters for some given samples:
clusters_ids = []
for i in range(bw.shape[0]):
    prob = lda.transform(bw[i])
    clusters_ids.append((i,np.argmax(prob)))


# ## 5. Building a database for supervised learning
#     Database format:
#    - Input: contexts (see 2 above)
#    - Output: groups of contexts (see 4 above)
# 
# ### Building the output variables for a supervised dataset from the clustering process

# In[ ]:


samplesXcluster = [0,0,0,0,0]    # 5 clusters
y = []
for c in clusters_ids:
    y.append(c[1])
    samplesXcluster[c[1]]+=1
y = np.asarray(y)


# ## Exploring the clusters

# In[ ]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
clt = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
samples = [samplesXcluster[0],samplesXcluster[1],samplesXcluster[2],samplesXcluster[3],samplesXcluster[4]]
ax.bar(clt,samples)
plt.show()


# ## 6. Training classifiers using Bag of Words

# In[ ]:


from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import KFold
from sklearn.svm import SVC

kf = KFold(n_splits=50, shuffle=True)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #Logistic regression classifier
    clf = LR(C=1, max_iter=200, solver='liblinear').fit(X_train, y_train)
    
    scores.append(clf.score(X_test, y_test))
print('Score = ', np.round(np.mean(np.asarray(scores)),2))


# ## 7. Query-based retrieval of relevant contexts using logistic regression

# In[ ]:


## Test classification using some of the Task's keywords and phrases.

## Comment or uncomment according to the query to be used.

#query = ["incubation period in days"]
#query = ["how long individuals are contagious."]
#query = ["how long individuals are contagious after recovery."]
query = ["Prevalence of asymptomatic shedding"]
#query = ["Prevalence of asymptomatic transmission in children"]
#query = ["transmission in children"]
#query = ["seasonality of transmission"]
#query = ["physical science of the coronavirus"]
#query = ["Smoking, pre-existing lung disease"]
#query = ["Role of the environment in transmission"]
#quetion = ["Effectiveness of personal protective equipment"]
#query = ["control strategies"]
#query = ["how long individuals are contagious, even after recovery"]
#query = ["Physical science"]
#query = ["Persistence and stability"]
#query = ["Persistence of virus on surfaces"]
#query = ["history of the virus"]
#query = ["diagnostic"]
#query = ["phenotypic change"]


qf = bow.transform(query)
ind = clf.predict(qf)
out = []
for c in clusters_ids:
    if c[1] == ind:
        out.append(contx[c[0]][1:])


# ## 8. Ranking and returning contexts using cosine distance between query and retrieved contexts
# 

# In[ ]:


from scipy.spatial.distance import cosine as dist
import operator

answer = bow.transform(out).toarray()
qq = qf.toarray()
distances = np.zeros(answer.shape[0])
ans_dist = dict()
i = 0
for a in answer:
    d = dist(a,qq)
    distances[i] = d 
    ans_dist.update({i:d})
    i+=1
    
dist_sort = sorted(ans_dist.items(), key=operator.itemgetter(1), reverse=False)

answer_words = ""
results2show = 10
for k in range(results2show):
    try:
        ans = out[dist_sort[k][0]]
        print("- ", ans + "\n")
        answer_words += ans + ' ' 
    except:
        continue


#    - Wordcloud visualization of results

# In[ ]:


from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

wordcloud = WordCloud(width = 600, height = 600,
                background_color ='black',
                stopwords = stopwords,
                min_font_size = 10).generate(answer_words)
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()


# ### Findings
# 
# 1. Efficiency of context classification with $5$ clusters is $0.78$.
# 2. A lower number of clusters seems to improve efficiency, but it may affect relevance of returned results. Relevance has been evaluated qualitatively only.
# 3. We have observed that the answer to a specific query using the Task's keywords is often found explicitely (or implied) in the results. See below a couple of examples:
# 
#    - Query: Range of incubation periods for the disease in humans
#      - Answer: the symptoms of covid-19 infection appear after an incubation period of approximately 5.2 days.
#    - Query: how long individuals are contagious?
#      - Answer: sars-cov-2 is contagious even in the incubation (sars was not).

# ## Pros and Cons
# 
# ### Pros
# 1. Simple and completly automatic approach combining unsupervised and supervised techniques.
# 2. The system returns detailed information (not complete articles) about a topic or keyword. 
# 3. The system allows queries using natural language.
# 4. Low computation cost
# 5. No feature engineering
# 6. Highly interpretable features
# 7. Can be easily adapted to the other tasks in this challenge with different questions.
# 8. The system does not require external resources besides the dataset.
# 9. The number of results to show can be define as desired.
# 
# ### Cons
# 1. No query reformulation implemented.
# 2. No metadata included in the results (i.e., source of the information, year, etc.).
# 3. The number of clusters has to be defined manually.
# 4. Relevant information may have been left out during filtering of contexts.
# 5. No keyword stemming or lemmatization
# 6. No NER or relation extraction
