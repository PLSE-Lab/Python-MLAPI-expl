#!/usr/bin/env python
# coding: utf-8

# # Notebook purpose:
# 
# Encoding Covid-19 data set using 2 different methedolgies. Namely, [BERT](http://https://github.com/google-research/bert) and [BioBERT](http://https://github.com/dmis-lab/biobert) pretrained models. 
# 
# The output will be a dataframe with two added columns of BERT and BioBERT vectors of each row of COVID_19 dataframe after removing title, abstract, full-body Nans. 
# 
# ## General Notes:
# 
# 1.The BERT vector based on the paper text, and BioBERT vector based on the paper title & abstract.
# 
# 2.This kernel runtime exceeds allowable kaggle run time. Probably, you will need to download it and run on your pc or divide the process into multiple sessions.
# 
# 3.We show sample of running in this notebook. However, We created public dataset of whole generated results [BioBERT + BERT Encoding](https://www.kaggle.com/fatma98/datasets)

# # Methodology
# ## 1.Data preparation:
# In this kernel we use the output dataframe from [CORD-19: Create Dataframe notebook](https://www.kaggle.com/danielwolffram/cord-19-create-dataframe). 
# 
# First, We clean the important columns from nans. 
# 
# Then, combined Title and Abstract is vectorized using a pretrained BERT model called BioBERT, A fine-tuned model on PubMed text. 
# 
# Finally, full-body text is vectorized based on BERT Model.
# 
# ## 2.Modeling:
# BioBert & BERT vectorizaion based  

# In[ ]:


get_ipython().run_cell_magic('time', '', "# takes 44.7 s to install everything \n\n# BioBERT dependencies\n# Tensorflow 2.0 didn't work with the pretrained BioBERT weights\n!pip install tensorflow==1.15\n# Install bert-as-service\n!pip install bert-serving-server==1.10.0\n!pip install bert-serving-client==1.10.0\n\n# We need to rename some files to get them to work with the naming conventions expected by bert-serving-start\n!cp /kaggle/input/biobert-pretrained /kaggle/working -r\n%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.index /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.index\n%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.data-00000-of-00001 /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.data-00000-of-00001\n%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.meta /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.meta\n\n!pip install transformers\n!pip install sentence-transformers\n\nfrom IPython.utils import io\nwith io.capture_output() as captured:\n    !pip install scispacy\n    !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz\n        \nprint('installation done')")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# takes 5.14 s to import everything\n\nimport subprocess\nimport pickle as pkl\nimport pandas as pd\nimport numpy as np \nfrom sentence_transformers import SentenceTransformer\nfrom transformers import BertTokenizer, BertModel\nimport pandas as pd\nfrom scipy.spatial.distance import jensenshannon\nfrom IPython.display import HTML, display\nfrom tqdm import tqdm\nimport en_core_sci_lg\nimport pickle as pkl\nfrom sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\nfrom sklearn.decomposition import LatentDirichletAllocation')


# In[ ]:


df = pd.read_csv('../input/cord19createdataframe/cord19_df.csv')
#comment the following line if you want to run the all data on your pc
df=df[:1000]
df.columns


# In[ ]:


#drop na and concatinate title and abstract to encode both of them at same time
df = df.dropna(subset=['abstract','body_text','url'])
df['document'] = df['title'] + '. ' + df['abstract']
df['document'] =  df['document'].astype(str)


# # BioBERT Model

# In[ ]:


bert_command = 'bert-serving-start -model_dir /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed -max_seq_len=None -max_batch_size=32 -num_worker=4'
process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)

# Start the BERT client. It takes about 10 seconds for the bert server to start, which delays the client
from bert_serving.client import BertClient

bc = BertClient(ignore_all_checks = True)


# # BERT Model

# In[ ]:


model_bert = SentenceTransformer('bert-base-nli-max-tokens')


# In[ ]:


abstracts = df[0:]['document'].tolist()
full_text = df['body_text'].tolist()

j = 0
bert_text_vec = np.zeros((df.shape[0],768))
biobert_doc_vec = np.zeros((df.shape[0],768))
n = 100

while(j < len(abstracts) - n + 1):
    print('Encoding abstracts & full_text number {} to {}'.format(j, j + n))
    abstracts_temp = abstracts[j:j+n]
    full_text_temp = abstracts[j:j+n]

    encoded_abstract = bc.encode(abstracts_temp)
    encoded_txt =model_bert.encode(full_text_temp)

    bert_text_vec[j:j+n,:] =  encoded_txt
    biobert_doc_vec[j:j+n,:] = encoded_abstract
    j += n
    # save after each 1000 
    if j % 1000 == 0:
        print('Updating output pickle file at j = {}...'.format(j))
        with open('vector_df_j_{}.pkl'.format(j), "wb") as fp:
            pkl.dump(bert_text_vec, fp, protocol=pkl.HIGHEST_PROTOCOL)
            pkl.dump(biobert_doc_vec, fp, protocol=pkl.HIGHEST_PROTOCOL)
        print('Updating done')

if j < df.shape[0]:
    print('Encoding abstracts & Full_text number {} to {}'.format(j, df.shape[0]))
    abstracts_temp = abstracts[j:df.shape[0]]
    full_text_temp = full_text[j:df.shape[0]]

    abstracts_temp = bc.encode(abstracts_temp)
    full_text_temp =model_bert.encode(full_text_temp)

    biobert_doc_vec[j:df.shape[0],:] = abstracts_temp
    bert_text_vec[j:df.shape[0],:] = full_text_temp

print('Encoding df done')


# In[ ]:


bio_vec = biobert_doc_vec
bio_vec


# In[ ]:


bert_vec = bert_text_vec
bert_vec


# In[ ]:


bert_vec.shape


# In[ ]:


df['bert_vector']=bert_vec.tolist()
df['biobert_vector']=bio_vec.tolist()


# In[ ]:


df.head()


# ## save your new dataframe

# In[ ]:


pkl.dump(df, open('BERT-BioBERT-Daraframe'.format(j), "wb"))


# ***

# # LDA

# ### 1.Data -we test with a subset to test it-

# In[ ]:


data = pd.read_csv('../input/cord19createdataframe/cord19_df.csv')
df_lda=data[:5] 
df_lda = df_lda.dropna(subset=['abstract'])
df_lda = df_lda.dropna(subset=['body_text'])
df_lda = df_lda.dropna(subset=['url'])
all_texts_lda = df_lda.body_text
df_lda.shape


# 
# ### 2.Nlp & Stop Words

# In[ ]:


# medium model
nlp = en_core_sci_lg.load(disable=["tagger", "parser", "ner"])
nlp.max_length = 3000000
# New stop words list 
customize_stop_words = [
    'doi', 'preprint', 'copyright', 'org', 'https', 'et', 'al', 'author', 'figure', 'table',
    'rights', 'reserved', 'permission', 'use', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI',
    '-PRON-', 'usually',
    r'\usepackage{amsbsy', r'\usepackage{amsfonts', r'\usepackage{mathrsfs', r'\usepackage{amssymb', r'\usepackage{wasysym',
    r'\setlength{\oddsidemargin}{-69pt',  r'\usepackage{upgreek', r'\documentclass[12pt]{minimal'
]

# Mark them as stop words
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True


# ### 3.Tokenizer 

# In[ ]:


def spacy_tokenizer(sentence):
    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]


# ### 4.Vectorizer

# In[ ]:


vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, min_df=2)
data_vectorized_lda = vectorizer.fit_transform(tqdm(all_texts_lda))


# ### 5.LDA

# In[ ]:


lda = LatentDirichletAllocation(n_components=50, random_state=0,verbose=1)
lda.fit(data_vectorized_lda)
lda_vec=lda.transform(data_vectorized_lda)
doc_topic_dist_lda = pd.DataFrame(lda_vec)
pkl.dump(doc_topic_dist_lda, open('lda_output_final.pkl', "wb"))
lda_pkl_read = pkl.load(open('lda_output_final.pkl', "rb"))


# ### 6.Get Nearest Papers (in Topic Space)

# In[ ]:


def get_k_nearest_docs_lda(doc_dist, k=5):
    
    temp = lda_pkl_read
        
    distances = temp.apply(lambda x: jensenshannon(x, doc_dist), axis=1)
    k_nearest = distances[distances != 0].nsmallest(n=k).index
    k_distances = distances[distances != 0].nsmallest(n=k)
    
    return k_nearest, k_distances
    


# ### 7.Search

# ### Search for risk factor and save them as pkl to be used in [mixer kernel](https://www.kaggle.com/marinamaher/team-final/edit)
# 
# ### LDA must be always fitted so we did this method to run risk factors fast.

# In[ ]:


def relevant_articles_lda(tasks, k=10):
    
    tasks2 = [tasks] if type(tasks) is str else tasks 
     
    tasks_vectorized = vectorizer.transform(tasks2)
    tasks_topic_dist = pd.DataFrame(lda.transform(tasks_vectorized))
    
    pkl.dump(tasks_topic_dist, open('put your query here', "wb"))
    #tasks_topic_dist = pkl.load(open(''+tasks+'.pkl'+'', "rb"))


    for index, bullet in enumerate(tasks2):
        
        recommended_index,distance = get_k_nearest_docs_lda(tasks_topic_dist.iloc[index], k)
        recommended = df_lda.iloc[recommended_index]
        recommended["index"]=recommended_index
        
        h = '<br/>'.join([str(i)+ '<a href="' + str(l) + '" target="_blank">'+ str(n) +'</a>'  for l, n ,i in recommended[['url','title','index']].values])

        display(HTML(h))


# In[ ]:


query='Liverrr'
relevant_articles_lda(query)


# # finally...
# open [the following notebook](https://www.kaggle.com/marinamaher/bert-models-mixer-lda)to know the output of mixing this two vectors to get the relevent papers of your questions about COVID_19

# In[ ]:





# In[ ]:




