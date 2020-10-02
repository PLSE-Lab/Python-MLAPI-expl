#!/usr/bin/env python
# coding: utf-8

# # Interactive Search using BioBERT and CorEx Topic Modeling
# 
# The end result of this notebook is an interactive search engine for article abstracts. As a user types their question, 22,000+ abstracts are ranked in realtime, and the top result is displayed. Additionally, users may filter by automatically determined topics. To run, log in to Kaggle, click the "Copy and Edit", and run all cells (takes about 3 minutes). Scroll to the bottom to see the following search widget:
# 
# ![](https://i.imgur.com/ptrEnu0.png)
# 
# This notebook strives to demonstrate the simplest way to use BioBERT for vectorizing academic medical texts, and how to use CorEx for topic modeling. I encourage you to fork this notebook, and let me know in the comments how you're using it!

# # Methodology
# 
# The starting point is the "all_sources_metadata_2020-03-13.csv" file. Title and Abstract are combined into the "document", which is vectorized using a pretrained BERT model called BioBERT. This model was finetuned on PubMed text. While you may use CorEx topic modeling on the raw text of the "document", I saw better results after preprocessing the tokens, which are cached in the file "*cached-data-interactive-abstract-and-expert-finder/df_final_covid_clean_topics.pkl*". My [other notebook](https://www.kaggle.com/jdparsons/interactive-abstract-and-expert-finder) contains the methodology I used for preprocessing.
# 
# ## Resources
# 
#  * Preprocessing text: https://www.kaggle.com/jdparsons/interactive-abstract-and-expert-finder
#  * Pre-trained BioBERT weights: https://github.com/naver/biobert-pretrained
#  * Simplified usage of BERT using BERT-as-a-service: https://github.com/hanxiao/bert-as-service
#  * Example of using BERT-as-a-service in Kaggle notebooks: https://www.kaggle.com/brendanhasz/bert-in-kernels
#  * CorEx library: https://github.com/gregversteeg/corex_topic
#  * Interactive search widget inspired by: https://www.kaggle.com/dgunning/browsing-research-papers-with-a-bm25-search-engine

# In[ ]:


load_cached_file = True # if True, vectorize_subset is ignored
vectorize_subset = 100 # for quicker testing, only vectorize a subset of documents. Otherwise, set to -1 to process all 22,000 (takes about 6 hours)


# # Imports and Setup
# 
# A lot of experimentation was required to find the right combination of Tensorflow version and filenames to get BioBERT to work with BERT-as-a-Service.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# takes 2 min to install everything\n\nfrom scipy.spatial.distance import cdist\nimport subprocess\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport pickle as pkl\nfrom ipywidgets import interact, widgets # this is what makes the dataframe interactive\n\n# BioBERT dependencies\n# Tensorflow 2.0 didn't work with the pretrained BioBERT weights\n!pip install tensorflow==1.15\n# Install bert-as-service\n!pip install bert-serving-server==1.10.0\n!pip install bert-serving-client==1.10.0\n\n# We need to rename some files to get them to work with the naming conventions expected by bert-serving-start\n!cp /kaggle/input/biobert-pretrained /kaggle/working -r\n%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.index /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.index\n%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.data-00000-of-00001 /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.data-00000-of-00001\n%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.meta /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.meta\n\n# CorEx topic modeling dependencies\n# https://github.com/gregversteeg/corex_topic\n!pip install 'corextopic'\nfrom sklearn.feature_extraction.text import CountVectorizer\nimport scipy.sparse as ss\nfrom corextopic import corextopic as ct")


# The cached file is from my [other notebook](https://www.kaggle.com/jdparsons/interactive-abstract-and-expert-finder). That notebook was focused on TF-IDF, and as such, involved lots of preprocessing of text tokens. These clean tokens are beneficial for CorEx topic modeling. BERT doesn't need this type of preprocessing because the neural networks learn to suppress stopwords and other noise. Therefore, vectorization is applied to the raw "document" text and not the preprocessed text.

# In[ ]:


if load_cached_file is True:
    df = pkl.load(open('/kaggle/input/biobert-vectors-cache/df_biobert_vectors.pkl', "rb"))
    unvectorized_df = pkl.load(open('/kaggle/input/cached-data-interactive-abstract-and-expert-finder/df_final_covid_clean_topics.pkl', "rb"))
    df['clean_tfidf'] = unvectorized_df['clean_tfidf'] # copy over the heavily preprocessed and already tokenized words for CorEx
else:
    df = pkl.load(open('/kaggle/input/cached-data-interactive-abstract-and-expert-finder/df_final_covid_clean_topics.pkl', "rb"))
    df = df.reset_index() # just in case the indices aren't sequential


# # CoreEx Topic Modeling
# 
# I found the optimal number of topics was between 16-24, but you can experiment with different values of target_num_topics. Because I'm using pre-tokenized data from my cached dataframe, I use a dummy function in CountVectorizer.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# takes 20-60 seconds depending on how many CorEx topics you select\n\ntarget_num_topics = 20 # num topics CorEx will identify\n\ndef dummy(doc):\n    return doc\n\nvectorizer = CountVectorizer(\n    tokenizer=dummy,\n    preprocessor=dummy,\n)\n\n# to process without my cached file, you can comment out the dummy params in CountVectorizer and change this line to use df['document'].\n# NOTE: this will greatly increase the run time of this cell.\ncorex_docs = df['clean_tfidf'].tolist()\ndoc_word = vectorizer.fit_transform(corex_docs)\n\ndoc_word = ss.csr_matrix(doc_word)\n\n# Get words that label the columns (needed to extract readable topics and make anchoring easier)\nwords = list(np.asarray(vectorizer.get_feature_names()))\n\n# https://github.com/gregversteeg/corex_topic\n# Train the CorEx topic model with x topics (n_hidden)\ntopic_model = ct.Corex(n_hidden=target_num_topics, words=words, max_iter=1000, verbose=False, seed=2020)\n\n# You can seed CorEx with anchor words for some topics if you find it is struggling in certain areas\n# domain expertise is important to choose the right anchor words\n#topic_model.fit(doc_word, words=words, anchors=[\n#    ['transmission', 'incubation'],\n#    ['bat', 'pig', 'porcine'],\n#    ['national','international','policy', 'public_health', 'public']\n#], anchor_strength=2)\n\n# or use the default where it is unguided\ntopic_model.fit(doc_word, words=words)\n\n# plot overall topic scores\nplt.figure(figsize=(10,5))\nplt.bar(range(topic_model.tcs.shape[0]), topic_model.tcs, color='#4e79a7', width=0.5)\nplt.xlabel('Topic', fontsize=16)\nplt.ylabel('Total Correlation (nats)', fontsize=16);\n# no single topic should contribute too much. If one does, that indicates more investigation for boilerplate text, more preprocessing required\n# To find optimal num of topics, we should keep adding topics until additional topics do not significantly contribute to the overall TC\n\ntopics = topic_model.get_topics()\ntopic_list = []\n\nfor n,topic in enumerate(topics):\n    topic_words,_ = zip(*topic)\n    print('{}: '.format(n) + ','.join(topic_words))\n    topic_list.append('topic_' + str(n) + ': ' + ', '.join(topic_words))")


# While I ended up not using the hierarchical feature of CorEx, it's neat to see how it would group each of the topics into a parent topic.

# In[ ]:


# Train successive hierarchical layers
tm_layer2 = ct.Corex(n_hidden=4)
tm_layer2.fit(topic_model.labels)

layer2_topics = tm_layer2.get_topics()
parents = []

for parent_topic in layer2_topics:
    layer_obj = {
        'keys': [],
        'words': []
    }
    
    for ind, _ in parent_topic:
        layer_obj['keys'] += [ind]
        layer_obj['words'] += [w[0] for w in topics[ind]][0:3]
    
    parents.append(layer_obj)

print('\n')

for p_topic in parents:
    key_str = [str(k) for k in p_topic['keys']]
    keys = ','.join(key_str)
    top_words = ','.join(p_topic['words'])
    
    print('PARENT GROUP: ' + keys)
    print(top_words + '\n')


# With these topics, tag each document with the CorEx topic likelihood scores. Each document gets a score for each topic, so you can have a single document that scores highly across multiple topics.

# In[ ]:


# remove any existing topic columns from previous runs
for c in [col for col in df.columns if col.startswith('topic_')]:
    del df[c]

for topic_num in range(0, len(topics)):
    # CorEx stores the likelihood scores in the same order as the source document, so the index will match
    df['topic_' + str(topic_num)] = topic_model.log_p_y_given_x[:,topic_num]

# For display purposes, create a final "best_topic" column which is the highest scoring topic for a row.
# The search UI will allow you to optionally view all topic scores
corex_cols = [col for col in df if col.startswith('topic_')]
df['best_topic'] = df[corex_cols].idxmax(axis=1)


# # BioBERT Vectorization
# 
# Start BERT-as-a-service. Even if you're using the cached file where all rows have been vectorized, you'll still need this to vectorize new search queries.
# 
# I was unable to get the GPU version working. If you know how to do this, please let me know in the comments!

# In[ ]:


get_ipython().run_cell_magic('time', '', "# takes 20 sec to start the BERT server\n\n# the documentation recommends batch size of 16 for CPU, 256 for GPU\n# Kaggle notebooks have 2 cpus, which is the num_worker param\n\nbert_command = 'bert-serving-start -model_dir /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed -max_seq_len=None -max_batch_size=32 -num_worker=2'\nprocess = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)\n\n# Start the BERT client. It takes about 10 seconds for the bert server to start, which delays the client\nfrom bert_serving.client import BertClient\n\nbc = BertClient()")


# If **not** using the cached file, vectorize each document using BioBERT so we can do cosine/Euclidean similarity on user queries. On the full 22k documents, this cell takes about 6 hours to run.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif load_cached_file is False:\n    # FOR TESTING - shuffle rows and select n\n    if (vectorize_subset > -1):\n        df = df.sample(frac=1).head(vectorize_subset).reset_index(drop=True)\n\n    abstracts = df[\'document\'].tolist()\n\n    # add the BioBERT vector to each row\n    embeddings = bc.encode(abstracts)\n\n    select_cols = [\'title\', \'abstract\', \'authors\', \'document\', \'clean_tfidf\']\n\n    # slim down filesize of dataframe by only selecting the cols we need\n    df = df[select_cols]\n\n    df[\'biobert_vector\'] = embeddings.tolist()\n    pkl.dump(df, open(\'df_biobert_vectors.pkl\', "wb"))')


# # Interactive Search
# 
# Try copy/pasting some of the prompts from https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks
# 
# As soon as you change one of the parameters, notice the CPU meter will spike until processing is complete.

# In[ ]:


print('To focus on a CorEx topic area, select it from the dropdown and drag the threshold slider to the right.')
print('A higher threshold value will filter out results that are less likely to be belong to the topic.')

default_question = 'Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.'

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('max_colwidth', 120)
# anybody know how to left-align text cols? The options I found on stack overflow didn't work in these Kaggle notebooks

biobert_vectors = np.array(df.biobert_vector.tolist())
total_docs = len(df.index)

@interact
def search_articles(
    query=default_question,
    topic=topic_list,
    #topic_threshold=(-20, 0.00, 0.5),
    topic_threshold=widgets.IntSlider(min=-15,max=0,step=0.5,value=-15),
    num_results=[10, 25, 100],
    show_topic_scores=[False, True],
    score_type=['cosine', 'Euclidean'],
    
):

    query_embedding = bc.encode([query])[0]

    sort_ascending = False
    score = []
    
    if score_type is 'Euclidean':
        score = np.sum(query_embedding * biobert_vectors, axis=1) / np.linalg.norm(biobert_vectors, axis=1)
        sort_ascending = False
    else:
        score = cdist(np.stack(df.biobert_vector), [query_embedding], metric="cosine")
        sort_ascending = True
    
    df["score"] = score
    
    # smaller corex_topic scores means more likely to be of that topic
    corex_cols = []
    if show_topic_scores is True:
        corex_cols = [col for col in df if col.startswith('topic_')]
        
    select_cols = ['title', 'abstract', 'authors', 'score', 'best_topic'] + corex_cols
    
    results = df[select_cols].loc[df[topic.split(':')[0]] > topic_threshold].sort_values(by=['score'], ascending=sort_ascending).head(num_results)
    
    if (len(results.index) == 0):
        print('NO RESULTS')
        
        return None
    else:

        top_row = results.iloc[0]

        print('TOP RESULT OUT OF ' + str(total_docs) + ' DOCS FOR QUESTION:\n' + query + '\n')
        print('TITLE: ' + top_row['title'] + '\n')
        print('ABSTRACT: ' + top_row['abstract'] + '\n')
        print('PREDICTED TOPIC: ' + topic_list[int(top_row['best_topic'].replace('topic_', ''))])

        print('\nAUTHORS: ' + str(top_row['authors']))

        select_cols.remove('authors')
        
        return results[select_cols]


# # Questions
# 
# Thank you for checking out my notebook! I have a few lingering questions I'd like to get your feedback on in the comments:
# 
# * Is there any automated way to determine the optimal number of CorEx topics?
# * I couldn't get BERT-as-a-Service to run using the GPU - any ideas what I did wrong? In theory, running with a GPU should dramatically reduce the 6 hour vectorization time of the full 22k documents.
# * Is there an Authors dataset I could join to the results? I'd like to provide full names and contact info for the authors. This strikes me as useful for journalists who search for a topic and wish to contact an expert.
