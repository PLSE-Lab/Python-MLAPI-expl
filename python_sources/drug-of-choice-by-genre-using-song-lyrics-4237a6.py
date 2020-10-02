#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import and preview the data
import pandas as pd
df = pd.read_csv('../input/lyrics.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


#replace carriage returns
df = df.replace({'\n': ' '}, regex=True)
df.head()


# In[ ]:


#count the words in each song
df['word_count'] = df['lyrics'].str.split().str.len()
df.head()


# In[ ]:


#check the word counts by genre
df['word_count'].groupby(df['genre']).describe()
genres = df['word_count'].groupby(df['genre'])


# In[ ]:


#let's see what the songs with 1 word look like
df1 = df.loc[df['word_count'] == 1]
df1


# In[ ]:


#elimintate the 1-word songs and review the data again
df = df[df['word_count'] != 1]
df['word_count'].groupby(df['genre']).describe()


# In[ ]:


#There are still some outliers on the low end. Reviewing songs with less than 100 words.
df100 = df.loc[df['word_count'] <= 100]
df100


# In[ ]:


#let's check on the high end
df1000 = df.loc[df['word_count'] >= 1000]
df1000


# In[ ]:


#let's get rid of the outliers on the low and high end using somewhat randomly selected points
del df1, df100, df1000 
df_clean = df[df['word_count'] >= 100]
df_clean = df[df['word_count'] <= 1000]
df_clean['word_count'].groupby(df_clean['genre']).describe()


# In[ ]:


#let's see how much smaller the data set is now
df.info()


# In[ ]:


#check the overall distribution of the cleaned dataset
import seaborn as sns
sns.violinplot(x=df_clean["word_count"])


# In[ ]:


#compare wordcounts by genre
import matplotlib as mpl
mpl.rc("figure", figsize=(12, 6))
sns.boxplot(x="genre", y="word_count", data=df_clean)


# In[ ]:


genre = df_clean.groupby(['genre'],as_index=False).count()
genre2 = genre[['genre','song']]
genre2


# In[ ]:


liquor = pd.DataFrame(df_clean.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('liquor')].count()))
liquor.reset_index(inplace=True)
liquor.columns = ['genre', 'liquor_lyrics']
liquor


# In[ ]:


beer = pd.DataFrame(df_clean.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('beer')].count()))
beer.reset_index(inplace=True)
beer.columns = ['genre', 'beer_lyrics']
beer


# In[ ]:


wine = pd.DataFrame(df_clean.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('wine')].count()))
wine.reset_index(inplace=True)
wine.columns = ['genre', 'wine_lyrics']
wine


# In[ ]:


pills = pd.DataFrame(df_clean.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('pills')].count()))
pills.reset_index(inplace=True)
pills.columns = ['genre', 'pills_lyrics']
pills


# In[ ]:


weed = pd.DataFrame(df_clean.groupby(['genre'])['lyrics'].apply(lambda x: x[x.str.contains('weed')].count()))
weed.reset_index(inplace=True)
weed.columns = ['genre', 'weed_lyrics']
weed


# In[ ]:


import functools
dfs = [genre2,beer,wine,liquor,pills,weed]
genre3 = functools.reduce(lambda left,right: pd.merge(left,right,on='genre', how='outer'), dfs)
genre3


# In[ ]:


genre3['beer_ratio'] = genre3['beer_lyrics'] / genre3['song']
genre3['wine_ratio'] = genre3['wine_lyrics'] / genre3['song']
genre3['liquor_ratio'] = genre3['liquor_lyrics'] / genre3['song']<
genre3['pills_ratio'] = genre3['pills_lyrics'] / genre3['song']
genre3['weed_ratio'] = genre3['weed_lyrics'] / genre3['song']
genre3


# In[ ]:


from textgenrnn import textgenrnn


# In[ ]:


model_cfg = {
    'rnn_size': 500,
    'rnn_layers': 12,
    'rnn_bidirectional': True,
    'max_length': 15,
    'max_words': 10000,
    'dim_embeddings': 100,
    'word_level': False,
}
train_cfg = {
    'line_delimited': True,
    'num_epochs': 100,
    'gen_epochs': 25,
    'batch_size': 750,
    'train_size': 0.8,
    'dropout': 0.0,
    'max_gen_length': 300,
    'validation': True,
    'is_csv': False
}


# In[ ]:


uploaded = files.upload()
all_files = [(name, os.path.getmtime(name)) for name in os.listdir()]
latest_file = sorted(all_files, key=lambda x: -x[1])[0][0]


# In[ ]:


model_name = '500nds_12Lrs_100epchs_Model'
textgen = textgenrnn(name=model_name)
train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file
train_function(
    file_path=latest_file,
    new_model=True,
    num_epochs=train_cfg['num_epochs'],
    gen_epochs=train_cfg['gen_epochs'],
    batch_size=train_cfg['batch_size'],
    train_size=train_cfg['train_size'],
    dropout=train_cfg['dropout'],
    max_gen_length=train_cfg['max_gen_length'],
    validation=train_cfg['validation'],
    is_csv=train_cfg['is_csv'],
    rnn_layers=model_cfg['rnn_layers'],
    rnn_size=model_cfg['rnn_size'],
    rnn_bidirectional=model_cfg['rnn_bidirectional'],
    max_length=model_cfg['max_length'],
    dim_embeddings=model_cfg['dim_embeddings'],
    word_level=model_cfg['word_level'])


# In[ ]:


print(textgen.model.summary())


# In[ ]:


files.download('{}_weights.hdf5'.format(model_name))
files.download('{}_vocab.json'.format(model_name))
files.download('{}_config.json'.format(model_name))


# In[ ]:


textgen = textgenrnn(weights_path='6layers30EpochsModel_weights.hdf5',
                       vocab_path='6layers30EpochsModel_vocab.json',
                       config_path='6layers30EpochsModel_config.json')
generated_characters = 300
textgen.generate_samples(300)
textgen.generate_to_file('lyrics.txt', 300)


# In[ ]:




