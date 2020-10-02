#!/usr/bin/env python
# coding: utf-8

# In this notebook we will,
# 1. Try to find relationships between relevant entities (animals and coronavirus, in this case) using co-occurance of keywords. 
#     Note: Keywords can co-occur at different levels viz. document, paragraph and sentence. 
# 2. We will further dive down into the results using setence level __Bert Encodings__ and __Plotly Vizualisations__.
# 

# Let us first install all the required libraries 

# In[ ]:


get_ipython().system(' wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip')
get_ipython().system(' unzip stanford-corenlp-full-2018-10-05.zip')
get_ipython().system(' pip install stanfordcorenlp')
get_ipython().system(' pip install stanford-openie')
get_ipython().system(' pip install sentence-transformers')
get_ipython().system(' pip install circlify')


# I start with a dataframe that contains text from the documents made available as part of the Kaggle COVID-19 challange (I have made this dataframe available here : https://www.kaggle.com/asitang/data-covid). I create an index/lookup where that can give me all the documents and their paragraphs in which that keyword is present.

# Below code reads the dataframe, where each row is information about a document.   
# 1. I split the text from each row/document into paragraphs.  
# 2. Each paragraph is then cleaned (which involves only keeping Nouns).  
# 3. The clean paragraph text is then added to another dataframe called "df_para", where each row will contain information about a paragraph (will also contain the document id).  
# 
# No need to run the below code, as it takes quite a bit of time :)

# In[ ]:


# import pandas as pd
# from stanfordnlp.server import CoreNLPClient
# os.environ["CORENLP_HOME"]='../working/stanford-corenlp-full-2018-10-05'

# df=pd.read_parquet('data/doc_processed.parquet', engine='pyarrow')
# df['doc_id']=df.index
# df['text']=df['body_text']+'\n'+df['abstract']
# print(len(df))
# df_para=pd.DataFrame()

# def clean_text(text, client):
#     # remove anything that is not noun or a special entity
#     ann = client.annotate(text)
#     clean_tokens = set()
#     for sentence in ann.sentence:
#         length = len(sentence.token)
#         clean_tokens.update([sentence.token[i].word for i in range(length) if ('NN' in sentence.token[i].pos) and not has_numbers(sentence.token[i].word) and not is_pun(sentence.token[i].word) ])
#     return list(clean_tokens)

# with CoreNLPClient(annotators=['tokenize','ssplit', 'pos'], timeout=30000, memory='16G') as client:
#     for index, row in tqdm(df.iterrows(), total=len(df)):
#         text=row['text']
#         paragraphs=text.split('\n')
#         for paragraph in paragraphs:
#             df_para=df_para.append({'doc_id':row['doc_id'], 'text':paragraph, 'clean_text':' '.join(clean_text(paragraph, client))}, ignore_index=True)

# df_para['para_id']=df_para.index
# print(len(df_para))
# df_para.to_parquet('data/para_processed.parquet', engine='pyarrow')
# display(df_para.head(5)['clean_text'])


# In[ ]:


# load datasets
import pandas as pd
df_doc=pd.read_parquet('../input/data-covid/doc_processed.parquet', engine='pyarrow')
print('Number of documents', len(df_doc))
df_para=pd.read_parquet('../input/data-covid/para_processed.parquet', engine='pyarrow')
print('Number of paragraphs', len(df_para))


# We create an index/mapping, which will give us the paragraph id for any given keyword 

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import numpy as np

# create indexes for search
para_id_lookup={}
vectorizer={}

vectorizer_keyword = CountVectorizer(tokenizer=lambda x: list(set(x.split(' '))))
keyword_to_para=sparse.csr_matrix(np.transpose(vectorizer_keyword.fit_transform(df_para['clean_text'].values)))
print('Index created! Shape:', keyword_to_para.shape)
para_id_lookup['keyword']=keyword_to_para
vectorizer['keyword']=vectorizer_keyword


# Some utility functions to help me process data (ignore please :) )

# In[ ]:


# some utility functions
def find_indices(entity_aliases,type, para_id_lookup, vectorizer):
    ent_ids = set()
    ent_doc_ids = set()
    for entity_alias in entity_aliases:
        if entity_alias not in vectorizer[type].vocabulary_.keys():
            continue
        ent_ids_tmp = para_id_lookup[type][vectorizer[type].vocabulary_[entity_alias]].indices
        ent_doc_ids_tmp = set(df_para[df_para['para_id'].isin(ent_ids_tmp)]['doc_id'].values)
        ent_ids.update(ent_ids_tmp)
        ent_doc_ids.update(ent_doc_ids_tmp)

    return ent_ids, ent_doc_ids

def find_pairwise_strengths(entities_l, entities_r, para_id_lookup, vectorizer):
    results = pd.DataFrame()
    for entity_l, type_l in entities_l:
        entity_l_title = entity_l[0]
        entity_l_aliases = entity_l[1]
        ent_l_ids, ent_l_doc_ids = find_indices(entity_l_aliases, type_l, para_id_lookup, vectorizer)
        for entity_r, type_r in entities_r:
            entity_r_title = entity_r[0]
            entity_r_aliases = entity_r[1]
            ent_r_ids, ent_r_doc_ids = find_indices(entity_r_aliases, type_r, para_id_lookup, vectorizer)

            matched_para_ids = set(ent_l_ids).intersection(ent_r_ids)
            matched_doc_ids = set(ent_l_doc_ids).intersection(ent_r_doc_ids)

            score_para = len(matched_para_ids)
            score_doc = len(matched_doc_ids)
            
            if score_doc > 0 or score_para > 0:
                results = results.append(
                    {'entity_l': entity_l_title, 'entity_r': entity_r_title, 'score_doc': score_doc, 'score_para':score_para, 'matched_para_ids': matched_para_ids, 'matched_doc_ids':matched_doc_ids},
                    ignore_index=True)
            
        return results


# let us find all the coronavirus references in the text.
# Todo this, we will find all keywords that have 'covid' or 'coronavirus' in them. 
# We will store the entities in a dictionary.

# In[ ]:


"""For this analysis, we will create a list for each entity group (CoronaVirus and Animals) with the following format:
[
('<entity_name>', {<aliases>}, '<entity_type viz. keyword location etc.>'),
('<entity_name>', {<aliases>}, '<entity_type>')
]

# Do not worry to much about this format, it will be usuful in future if data is pre tagged with location, disease etc. 
"""


# In[ ]:


expanded_keywords=[]

keyword='covid'
# find all tokens from the text that have the above keyword in them 
expanded_keywords.extend([item for item in vectorizer_keyword.get_feature_names() if keyword.lower() in item.lower()])

keyword='coronavirus'
# find all tokens from the text that have the above keyword in them 
expanded_keywords.extend([item for item in vectorizer_keyword.get_feature_names() if keyword.lower() in item.lower()])

# put them in a particular format that is accepted by the 'find_pairwise_strengths' funtion
# all these keywords will be used to find the entity 'CoronaVirus'
entity_l_='CoronaVirus' # give a name to the this entity
entities_l=[((entity_l_, set(expanded_keywords)), 'keyword')]

print(entities_l)


# Similar to above, we create another dictionary that contains various animals and their aliases. I use a list of animal names gotten from here: https://gist.github.com/atduskgreg/3cf8ef48cb0d29cf151bedad81553a54 

# In[ ]:


# read a list of animal names and create tokens from it (1-gram)
animal_words=[line.strip() for line in open('../input/nlp-lists/animals.txt','r').readlines()]
animal_words=set(animal_words)
entities_r_dict={}
for animal_word in animal_words:
    for token in animal_word.split(' '):
        token=token.lower()
        if token not in entities_r_dict.keys():
            entities_r_dict[token]=set()
        entities_r_dict[token].add(token)

# put them in a particular format that is accepted by the 'find_pairwise_strengths' funtion
# all these keywords will be used to find the entity type 'Animal' 
#        consisting of several entites (bats, cats, rats and so on)
entities_r=[((k,v), 'keyword') for k, v in entities_r_dict.items()]
print(entities_r)


# Let us find relationships between CoronaVirus and some common animals:  
# To do this we **find the paragraphs and documents, where these entities co-occur** and **rank** the entities based on their co-occurance.

# In[ ]:


# find pairwise strengths between coronavirus and several animals
results=find_pairwise_strengths(entities_l, entities_r, para_id_lookup, vectorizer)

# rank animal entities (animals) based on co-occurance at different levels 
# show results

print("Ranking based on co-occurance at document level")
results_=results.sort_values(by=['score_doc'], ascending=False).head(10)
display(results_[['entity_r','score_doc']])
top_contenders=set(results_['entity_r'].values)

print("Ranking based on co-occurance at paragraph level")
results_=results.sort_values(by=['score_para'], ascending=False).head(10)
display(results_[['entity_r','score_para']])
top_contenders.update(results_['entity_r'].values)

print('List of top Contenders (sum of above list):',', '.join(top_contenders))


# Further, to better understand the relationship of CoronaVirus with the top contenders, we will go at the sentence level. We will find all the sentences that contains both a mention of CoronaVirus and a mention of an Animal. Now, we will:
# 1. Use our index to quickly **find all those paragraphs** first.
# 2. Use Stanford's **Open IE to create simpler sentences** (from the compound and complex sentences) of that paragraph.
# 3. **Filter out all sentences that do not have both CoronaVirus and an Animal mention** in them.
# 4. **Convert the mentions of aliases of CoronaVirus with 'CoronaVirus'** and **an animal name (say 'bat') with the 'Animal'** keywords. This will be useful for clustering without too much bias on the extact name of the entity.
# 5. **Create Bert embeddings** of each of the sentences and save them.
# 
# ***(Note: things will definitely be faster if you run Open IE beforehand on all paragraphs)***

# In[ ]:


from openie import StanfordOpenIE
import json
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer

# set some properties for the extractors
bertify=SentenceTransformer('bert-base-nli-mean-tokens')
os.environ["CORENLP_HOME"]='../input/stanford-resources/stanford-corenlp-full-2018-10-05/'
ie_properties={"openie.triple.strict":"true", "openie.max_entailments_per_clause":"1", "splitter.disable":"true"}
    

with StanfordOpenIE() as client: # start the Stanford IE engine for Subject, relation, Object extraction
    triple_results = pd.DataFrame()
    for _, row in results[(results['entity_l']==entity_l_) & (results['entity_r'].isin(top_contenders))].iterrows():
        
        # for each animal find all the paragraphs where it occurs
        entity_r_title=row['entity_r']
        print('\n',entity_r_title)
        for index in tqdm(row['matched_para_ids'], total=len(row['matched_para_ids'])):
            text=df_para.iloc[index]['text']
            entity_aliases=entities_r_dict[entity_r_title]
            
            
            def check_alias(triple, expanded_keywords, aliases):
                # chech within a triple, if both types of enitties (animal and coronavirus) are mentioned
                to_serach=triple['subject']+' '+triple['relation']+' '+triple['object']
                for alias_a in aliases:
                    if alias_a in to_serach:
                        for alias_b in expanded_keywords:
                            if alias_b in to_serach:
                                return True

                return False
            
            # mask the occurance of the entity by its type (camel->animal, 'coronavirus' et al.-> CoronaV)
            def mask_instance(sentence, expanded_keywords, aliases):
                for alias in aliases:
                    if alias in sentence:
                        sentence=sentence.replace(alias, 'Animal')
                        break
                for alias in expanded_keywords:
                    if alias in sentence:
                        sentence=sentence.replace(alias, entity_l_)
                        break
                return sentence
            
            # collect all the sentenses (clauses) where both the entity types are present (corona and animal)
            for triple in [triple for triple in client.annotate(text , properties=ie_properties)
                            if check_alias(triple, expanded_keywords, entity_aliases)]:
                triple_results= triple_results.append({'sentence':mask_instance(triple['subject']+' '+triple['relation']+' '+triple['object'], expanded_keywords, entity_aliases), 'relation':triple['relation'],'subject':triple['subject'],'object':triple['object'], 'triple':json.dumps(triple) , 'entity':entity_r_title}, ignore_index=True)
    
    print(len(triple_results), 'sentences found!')
    print('calculating bert embeddings..')
    triple_results['embedding_bert']=bertify.encode(triple_results['sentence'].values)
    triple_results.to_parquet('../working/sentences_bert.parquet', engine='pyarrow')


# * We will now cluster (**t-SNE** custering algorithm) the sentences (using their **Bert embeddings**). So, **sentences** that are trying to convey **similar ideas should be clustered together**. 
# * We will also color code each sentence based on the animal it is talking about.
# * Now, if we see a **distinct cluster that has the same color**, this means that cluster represent a **uniqe relation** of that animal with CoronaVirus!

# In[ ]:


import plotly.express as px
from sklearn.manifold import TSNE
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)         # initiate notebook for offline plot

triple_results=pd.read_parquet('../working/sentences_bert.parquet', engine='pyarrow')
X = np.array(list(triple_results['embedding_bert'].values))
triple_results_sampled=triple_results
# print('data shape', X.shape)

# cluster sentences: run t-SNE on bert vectors of sentences
X_embedded = TSNE(n_components=2).fit_transform(X)
# print('TSNE embedding shape', X_embedded.shape)

triple_results_sampled['x_axis'] = X_embedded[:, 0]
triple_results_sampled['y_axis'] = X_embedded[:, 1]
triple_results.to_parquet('../working/sentences_bert_tsne.parquet', engine='pyarrow')

fig = px.scatter(triple_results_sampled, x='x_axis', y='y_axis', color='entity', hover_data=['entity','sentence'])
entity_color_lookup={item.legendgroup: item.marker.color for item in fig.data}
iplot(fig)


# ![Screen%20Shot%202020-04-08%20at%204.03.44%20PM.png](attachment:Screen%20Shot%202020-04-08%20at%204.03.44%20PM.png)There is a **cluster in the far left**, where it talks about coronavirus and **bats**, and hovering over them and looking at the sentences, it seems like they are clustered together because they all talk about **CoronaVirus**, that was found in bats In **China**.
# 
# (Note: the plot may change every run, so I have a screenshot here)

# The above cluster reresents all kinds of relations between CoronaVirus and Animals. What if we are only interested in certain relations!
# * Now, we will create a **Heat Map of the sentences**, based on their closeness to a **'posed relation'** which can be asumed **like a question**.
# * We will use a **Bubble Packing vizualisation** to show a heat map of sentences for each Animal.
# 
# (Note: the circle packing algorithm takes some time :P)

# In[ ]:


posed_relation="CoronaVirus in Animals caused outbreak"


# In[ ]:


import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import math
from random import seed, random
import circlify
from sentence_transformers import SentenceTransformer


# initialize/load stuff
seed(1)
bertify=SentenceTransformer('bert-base-nli-mean-tokens')
triple_results=pd.read_parquet('../working/sentences_bert_tsne.parquet', engine='pyarrow')
fig = go.Figure()

# create a bert vector for the posed-relation
relation_embedding=bertify.encode([posed_relation])[0]

# score each sentence in the results based on the question/posed-relation
scores=[cosine_similarity([relation_embedding, embedding])[0][1] for embedding in triple_results['embedding_bert'].values]
triple_results['score']=scores

# circle pack all the sentences belonging to a particular animal
data=[]
for entity in set(triple_results['entity'].values):
    entity_sentences=triple_results[triple_results['entity']==entity]
    temp_data={}
    temp_data['id']=entity
    temp_data['datum']=1
    temp_data['children']=[]
    for indx, row in entity_sentences.iterrows():
        temp_data['children'].append({'id':indx, 'datum':1})
    data.append(temp_data)

circles=circlify.circlify(data)
for circle in circles:
    if circle.level==2:
        triple_results.loc[circle.ex.get('id',None), 'x_temp']=circle.x
        triple_results.loc[circle.ex.get('id',None), 'y_temp']=circle.y

# create a bounding circle with the color of the animal from previous graph to encapsulate all the sentences for that animal
shapes=[]
buffer=.02
for entity in set(triple_results['entity'].values):
    entity_sentences=triple_results[triple_results['entity']==entity]
    shapes.append(dict(
            type="circle",
            xref="x",
            yref="y",
            x0=min(entity_sentences['x_temp'].values)-buffer,
            y0=min(entity_sentences['y_temp'].values)-buffer,
            x1=max(entity_sentences['x_temp'].values)+buffer,
            y1=max(entity_sentences['y_temp'].values)+buffer,
            line_color=entity_color_lookup[entity],
        ))

# add all to plotly scatter plot and plot away
fig.add_trace(go.Scatter(x=triple_results['x_temp'].values, y=triple_results['y_temp'].values,
                             mode='markers',
                             marker_color=triple_results['score'].values,
                             hoverinfo='text',
                            hovertext=['entity='+entity+'\n'+'sentence='+sentence for entity,sentence in  zip(triple_results['entity'].values, triple_results['sentence'].values)],
                    marker=dict(
                        colorscale="Gray",
                        reversescale=True,
                        showscale=True
                        )))
fig.update_layout(
    shapes=shapes
)

    

iplot(fig)


# Non visually speaking, we can rank the animal now based on the relation/quiestion we posed:

# In[ ]:


triple_results_=triple_results.loc[:, ['score', 'entity']]
print('\n\nRanking by averaging:')
display(triple_results_.groupby('entity').mean().sort_values(by=['score'], ascending=False))
print('\n\nRanking by taking max:')
display(triple_results_.groupby('entity').max().sort_values(by=['score'], ascending=False))


# Let us see what are the best answers/sentences to the posed relation:

# In[ ]:


print('\n\nClosest results for relation:', posed_relation)
triple_results_answers=triple_results.sort_values(by=['score'], ascending=False).head(5)[['entity','sentence']]

print('\n\nIn general:\n')
display(triple_results_answers)

print('\n\nPer entity basis:\n')
triple_results.sort_values(by=['entity','score'], ascending=False, inplace=True)
for entity in set(triple_results['entity'].values):
    triple_results_answers=triple_results[triple_results['entity']==entity].head(5)[['sentence']]
    print('\n\nentity:',entity)                                                                                
    display(triple_results_answers)
    


# In[ ]:




