#!/usr/bin/env python
# coding: utf-8

# **<center style="font-size: 16pt;"><a href="https://www.kaggle.com/atmarouane/covid-19-search-engine-indexing-by-lda-enm">Ensemble Model (EnM) for document retrieval results</a></center>**

# <h1><span class="tocSkip"></span>Table of Contents</h1>
# <div id="toc-wrapper"></div>
# <div id="toc"></div>

# # Technical

# ## Configuration class
# 
# We set variables like from where we load, where to store and some parameters (Explained later).

# In[ ]:


class config():
    CORPUS_FN = '/kaggle/input/cord-19-step2-corpus/corpus.pkl'
    ENM_FN = '/kaggle/input/cord-19-step3-enm/ranker_enm.pickle'
    TOC2_FN='/kaggle/input/toc2js/toc2.js'
    
    n_relevant = 150
    rm1_lambda = 0.6
    rm3_lambda = 0.7
    
    n_display = 15
    
    query_txt = 'Tools and studies to monitor phenotypic change and potential adaptation of the virus' 


# ## Libraries

# ### Our libraries
# 
# All our libraries are made public under open source.

# In[ ]:


import cord_19_container as container
import cord_19_rankers as rankers
import cord_19_lm as lm
import cord_19_vis as vis

from cord_19_container import Sentence, Document, Paper, Corpus

from cord_19_metrics import compute_queries_perf

from cord_19_helpers import load, save
from cord_19_text_cleaner import Cleaner
from cord_19_wn_phrases import wn_phrases


# ### Commun libraries

# In[ ]:


from gensim import matutils
from sklearn.metrics.pairwise import cosine_similarity

import copy
from collections import defaultdict
import re

import numpy as np
import pandas as pd


# ### Visualization libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import display, HTML, Markdown, Latex

import wordcloud

import matplotlib.pyplot as plt
import bokeh
import holoviews as hv

hv.extension('bokeh', logo=False)
hv.output(size=260)

HTML("""
<style>
.output_png {
    text-align: center;
    vertical-align: middle;
}

.rendered_html table{
    display: table;
}
</style>
""")


# ## Load data

# Load the corpus, papers talking about COVID-19/SARS-CoV-2, done in our previous kernel.

# In[ ]:


corpus = load(config.CORPUS_FN)
dictionary = corpus.dictionary

# Rebuild id2token from token2id, only token2id is saved
for k,v in dictionary.token2id.items():
    dictionary.id2token[v]=k

# Set the dictionary as global, we have to find better way
container.dictionary = dictionary
rankers.dictionary = dictionary
vis.dictionary = dictionary

print(f'#Papers {len(corpus)}, #Tokens {len(dictionary)}')


# ## Load model

# Loading our model.

# In[ ]:


ranker_enm = load(config.ENM_FN)
ranker_nmf = ranker_enm.models['NMF']
ranker_ldi = ranker_enm.models['LDI']

ranker_ql = rankers.ranker_QL(corpus, config.rm1_lambda)


# # Original query

# In[ ]:


query = container.Document([Cleaner(True).clean(config.query_txt)])
query.tokenize()
wn_phrases(query)

display(HTML(f'We are searching for:<br><br>'))
q_original_text = '<br>'.join([s.original_text for s in query.sentences])
display(HTML(f'<p style="font-size: 18pt;">{q_original_text}</p>'))

#When debuging print query.text


# ## Topics importance

# In[ ]:


def plot_topics_dist(q):
    score = ranker_nmf[q]
    _, R = lm.get_relevant(corpus, score, config.n_relevant)
    
    fig_1, (ax1_nmf, ax1_ldi) = plt.subplots(1, 2, figsize=(14,6), sharey=True)
    fig_2, (ax2_nmf, ax2_ldi) = plt.subplots(1, 2, figsize=(14,6), sharey=True)

    vis.plot_topics_dist(ranker_nmf, R, ax1_nmf, ax2_nmf, "NMF", set_y_label=True)
    fig_1.suptitle(f'Number of Documents by Dominant Topic.')

    vis.plot_topics_dist(ranker_ldi, R, ax1_ldi, ax2_ldi, "LDI", set_y_label=False)
    fig_2.suptitle(f'Mean topic probability over corpus.')

    plt.show()
    
plot_topics_dist(query)


# ## Cooccurrences importance

# In[ ]:


# https://notes.mikejarrett.ca/connecting-neighbourhoods/
def rotate_label(plot, element):
    text_cds = plot.handles['text_1_source']
    length = len(text_cds.data['angle'])
    text_cds.data['angle'] = [0]*length
    xs = text_cds.data['x']
    text = np.array(text_cds.data['text'])
    xs[xs<0] -= np.array([len(t)*0.03 for t in text[xs<0]])

def display_coi(ranker, q, n_words=100):
    """Cooccurrences importance
    """
    
    score = ranker[q]
    query_likelihood = ranker_ql[q]
    
    I, R = lm.get_relevant(corpus, score, config.n_relevant)
    query_likelihood = query_likelihood[I]
    query_likelihood = query_likelihood / sum(query_likelihood)
    
    rm1 = lm.compute_rm1(R.TRF, corpus.p_coll, query_likelihood, lambda_=config.rm1_lambda)
    
    # NOTE: Here we are using RM1 and not RM3, we haven't to emphasize query terms
    lda_tm_rm = lm.compute_tm_rm(ranker_ldi, R, q, query_likelihood, rm1, lambda_=0.7)
    nmf_tm_rm = lm.compute_tm_rm(ranker_nmf, R, q, query_likelihood, rm1, lambda_=0.7)
    tm_rm = 0.7*nmf_tm_rm + 0.3*lda_tm_rm
    
    top_tokens = np.argsort(tm_rm)[::-1][:n_words]
    top_tokens_set = set(top_tokens)
    tokens_subset_id = list(range(len(top_tokens_set)))
    
    map_to_new = dict(zip(top_tokens_set, tokens_subset_id))
    map_to_old = dict(zip(tokens_subset_id, top_tokens_set))

    def to_new_id(X):
        return [map_to_new[x] for x in X]
    
    def to_old_id(X):
        return [map_to_old[x] for x in X]
    
    def to_word(X):
        return [dictionary.id2token[x] for x in X]
    
    # Cooccurrences matrix
    CoM = np.zeros([n_words, n_words])
    
    for paper in R:
        for sent in paper:
            sent_words = list(sent.tokensid_set.intersection(top_tokens_set))
            words_p = [tm_rm[w] for w in sent_words]
            sent_words = to_new_id(sent_words)
            
            pseudo_corpus = list(zip(sent_words, words_p))
            vec = matutils.corpus2dense([pseudo_corpus], n_words)
            
            # [n_words,1]*[1,n_words]=[n_words,n_words]
            CoM += vec@vec.T
    
    I,J = np.triu_indices(n_words, 1)
    co_score = CoM[I,J]
    
    df = pd.DataFrame({'source':to_word(to_old_id(I)),
                       'target':to_word(to_old_id(J)),
                       'value':co_score})
    df.sort_values(by=['value'], inplace=True, ascending=False)
    
    links = 0
    words_set = set()
    max_words = 20 # It can be 21 or 22
    for row_i, row in df.iterrows():
        words_set.update([row.source, row.target])
        if len(words_set) < max_words:
            links += 1
        else:
            break
    
    # print(f'#links {links}')
    df = df.head(links)
    
    #display(df)
    
    chord = hv.Chord(df)
    chord.opts(
        hv.opts.Chord(cmap='Category20', edge_cmap='Category20', 
                      node_color='index', labels='index', edge_color='source',
                      edge_line_width=3,
                      hooks=[rotate_label])
    )
    
    display(chord)


# In[ ]:


display_coi(ranker_enm, query)


# ## Words importance

# In[ ]:


def display_wi(ranker, q):
    """
    Word importance
    """

    score = ranker[q]
    query_likelihood = ranker_ql[q]
    
    I, R = lm.get_relevant(corpus, score, config.n_relevant)
    query_likelihood = query_likelihood[I]
    query_likelihood = query_likelihood / sum(query_likelihood)
    
    rm1 = lm.compute_rm1(R.TRF, corpus.p_coll, query_likelihood, lambda_=config.rm1_lambda)
    
    # NOTE: Here we are using RM1 and not RM3, we haven't to emphasize query terms
    lda_tm_rm = lm.compute_tm_rm(ranker_ldi, R, q, query_likelihood, rm1, lambda_=0.7)
    nmf_tm_rm = lm.compute_tm_rm(ranker_nmf, R, q, query_likelihood, rm1, lambda_=0.7)
    tm_rm = 0.7*nmf_tm_rm + 0.3*lda_tm_rm
    
    freq = {}
    for i,p in enumerate(tm_rm):
        token = dictionary.id2token[i]
        freq[token] = p
    
    tm_rm = np.sort(tm_rm)
    
    wc = wordcloud.WordCloud(width=800, height=400, max_words=100).generate_from_frequencies(freq)
    plt.figure(figsize=[7,3], dpi=120)
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'Top100 Word/Relevants probality [{tm_rm[-100]:.3f}, {tm_rm[-1]:.3f}]')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    


# In[ ]:


display_wi(ranker_enm, query)


# ## Results

# In[ ]:


def display_results(ranker, q):
    
    scores = ranker[q]
    
    I, R = lm.get_relevant(corpus, scores, config.n_display)
    
    for i, paper in enumerate(R):
        paper_id = I[i]
        enm = scores[paper_id]
        
        table_html = "<table>"
        
        # Please add the link (Kaggle kernel) from where you got the 'doi link'
        link='<a href="https://doi.org/'+paper.doi+'" target=blank>'+paper.title+'</a>'
        table_html += f"<tr><th style='text-align:left; width: 5%;'>Title:</th><td style='text-align:left;'><b>{link}</b></td></tr>"
        table_html += f"<tr><th style='text-align:left;'>Score:</th><td style='text-align:left;'>{enm:.3f}</td></tr>"
        
        sentences = [sent for sent in paper if len(sent.bow)>1]
        
        a = ranker_nmf.project(sentences)
        b = ranker_nmf.project(q)
        
        sim = cosine_similarity(a, b)
        sim = sim[:,0]
        
        for j,sent in enumerate(sentences):
            jaccard_sim = 1-matutils.jaccard(q.bow, sent.bow)
            sim[j] = (sim[j] + jaccard_sim)*0.5
        
        sI = np.argsort(sim)[::-1]
        top5_tbl = "<table>"
        top5_tbl += "<tr><th>Score</th><th style='text-align:left;'>Sentence</th></tr>"
        for j in sI[:5]:
            top5_tbl += f"<tr><td>{sim[j]:.3f}</td><td style='text-align:left;'>{sentences[j].original_text}</td></tr>"
        top5_tbl += "</table>"
        
        table_html += "<tr><td colspan='2' style='text-align:left;'><b>Top5 sentences:</b></td></tr>"
        table_html += "<tr><td colspan='2'>"+top5_tbl+"</td></tr>"
        
        table_html += "</table>"
        
        display(HTML(table_html))
        #display(HTML('<hr>'))


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndisplay_results(ranker_enm, query)')


# # Expanded query

# In[ ]:


def get_top_terms(q, score, query_likelihood, lambda_ = 0.6, topk=20):
    I, R = lm.get_relevant(corpus, score, config.n_relevant)
    
    query_likelihood = query_likelihood[I]
    query_likelihood = query_likelihood / sum(query_likelihood)
    
    rm1 = lm.compute_rm1(R.TRF, corpus.p_coll,
                         query_likelihood,
                         lambda_=config.rm1_lambda)
    
    TF = R.TF.copy()
    # Set to 0 words not in query
    mask = np.isin(np.arange(len(corpus.dictionary)), list(q.tokensid_set), invert=True)
    TF[:,mask] = 0
    p_w_q = lm.mle(TF) # P_mle(w|Q)
    rm3 = lm.compute_rm3(rm1, p_w_q, lambda_=config.rm3_lambda)

    # Combine topics models with RM
    lda_tm_rm = lm.compute_tm_rm(ranker_ldi, R, q, query_likelihood, rm3, lambda_=lambda_)
    nmf_tm_rm = lm.compute_tm_rm(ranker_nmf, R, q, query_likelihood, rm3, lambda_=lambda_)
    tm_rm = 0.7*nmf_tm_rm + 0.3*lda_tm_rm
    
    return np.argsort(tm_rm)[::-1][:topk]

def expand_query(q):
    oq = copy.deepcopy(q)
    best_q = oq
    
    n_expanded = 0

    best_clarity_score = 0
    max_iter = 3

    while max_iter:
        max_iter -= 1
        expanded = False

        q_score = ranker_enm[q]
        query_likelihood = ranker_ql[q]
        
        prev_clarity_score = compute_queries_perf(corpus, q, q_score, query_likelihood,
                                                  kind='uef', n_relevant=config.n_relevant,
                                                  rm1_lambda=config.rm1_lambda, rm3_lambda=config.rm3_lambda).item()

        best_clarity_score = prev_clarity_score
        best_lambda = 0
        best_topk = 0
        min_k = len(q.text)+1
        for topk in np.arange(min_k, 20):
            for lambda_ in np.linspace(0.2,0.8,7):
                top_terms = get_top_terms(q, q_score, query_likelihood, lambda_=lambda_, topk=topk)
                new_q = defaultdict(int, q.bow)
                for tokenid in top_terms:
                    new_q[tokenid] += 1
                new_q_bow = list(new_q.items())
                new_q = copy.deepcopy(q)
                new_q._bow = new_q_bow

                new_q_score = ranker_enm[new_q]
                new_query_likelihood = ranker_ql[new_q]

                clarity_score = compute_queries_perf(corpus, new_q, new_q_score, query_likelihood,
                                                     kind='uef', n_relevant=config.n_relevant,
                                                     rm1_lambda=config.rm1_lambda,
                                                     rm3_lambda=config.rm3_lambda).item()
                if clarity_score>best_clarity_score:
                    best_clarity_score = clarity_score
                    best_q = new_q
                    best_lambda = lambda_
                    best_topk = topk
                    break

        if best_topk != 0:
            q = best_q
            delta = best_clarity_score - prev_clarity_score
        else:
            delta = 0
            best_lambda = 0
        print(f'CS: {best_clarity_score:.6f} \u0394: +{delta:.6f} Lambda: {best_lambda:.2f}')

        if best_topk == 0:
            break
                
    return best_q


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nexpanded_query = expand_query(query)')


# In[ ]:


df_o = pd.DataFrame([(dictionary.id2token[k], v) for k,v in query.bow], columns=['word', 'original'])
df_e = pd.DataFrame([(dictionary.id2token[k], v) for k,v in expanded_query.bow], columns=['word', 'expanded'])

df = pd.merge(df_e, df_o, how='left', on='word')
df.fillna(0, inplace=True)
df.sort_values(by=['expanded', 'word'], ascending=[True, False], inplace=True)
df['expanded'] -= df['original']

df.plot.barh(x='word', y=['original', 'expanded'], colormap='bwr', stacked=True,
             figsize=[12,8], rot=0, title="Expanded" )
plt.tight_layout(pad=0)
plt.show()


# ## Topics importance

# In[ ]:


def plot_topics_dist(q):
    score = ranker_nmf[q]
    _, R = lm.get_relevant(corpus, score, config.n_relevant)
    
    fig_1, (ax1_nmf, ax1_ldi) = plt.subplots(1, 2, figsize=(14,6), sharey=True)
    fig_2, (ax2_nmf, ax2_ldi) = plt.subplots(1, 2, figsize=(14,6), sharey=True)

    vis.plot_topics_dist(ranker_nmf, R, ax1_nmf, ax2_nmf, "NMF", set_y_label=True)
    fig_1.suptitle(f'Number of Documents by Dominant Topic.')

    vis.plot_topics_dist(ranker_ldi, R, ax1_ldi, ax2_ldi, "LDI", set_y_label=False)
    fig_2.suptitle(f'Mean topic probability over corpus.')

    plt.show()
    
plot_topics_dist(query)


# ## Cooccurrences importance

# In[ ]:


display_coi(ranker_enm, expanded_query)


# ## Words importance

# In[ ]:


display_wi(ranker_enm, expanded_query)


# ## Results

# In[ ]:


display_results(ranker_enm, expanded_query)


# In[ ]:


from IPython.display import HTML

with open(config.TOC2_FN, 'r') as file:
    js = file.read()

    display(HTML('<script type="text/Javascript">'+js+'</script>'))
    
    del js


# In[ ]:


get_ipython().run_cell_magic('javascript', '', '\n// Autonumbering & Table of Contents\n// Using: https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tree/master/src/jupyter_contrib_nbextensions/nbextensions/toc2\ntable_of_contents(default_cfg);')


# In[ ]:




