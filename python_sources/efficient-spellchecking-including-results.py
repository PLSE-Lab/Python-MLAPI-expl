#!/usr/bin/env python
# coding: utf-8

# To Spellcheck or Not?
# ===
# 
# **Pros: **
# * Fewer unknown tokens
# * Reduced noise
# * More uniform train and test sets (maybe)
# 
# **Cons: **
# * Misspellings are a potential feature, by correcting them we could be *losing* information!
# * Many bad spelling corrections could add noise
# 
# How much spelling correction helps (or hurts) probably depends somewhat on your model and what other pre-processing you're doing. For me, training on spell checked inputs didn't make any noticable difference with a single model. But, I saw a decent improvement when ensembling the results of spellchecked models with non-spellchecked models.
# 
# The Spellchecker...
# ===================
# I found a nice python package that does a solid job of spellchecking individual words: [Autocorrect](https://github.com/phatpiglet/autocorrect/). Huge thanks to [phatpiglet](http://phatpiglet.com/) and the rest of the contributors. Given a word, it returns its best guess at a spelling correction. If the word isn't misspelled or it's so badly misspelled that the module can't come up with any suggestions, it just returns the input word unmodified.
# 
# Unfortunately, it's not installed on Kaggle and I can't seem to install it. So I added the stub below to get it working here. If you want to run it yourself locally just install autocorrect and change `RUNNING_ON_KAGGLE` to `False`

# In[ ]:


RUNNING_ON_KAGGLE = True
if RUNNING_ON_KAGGLE:
    def spell(word):
        return word
else: 
    from autocorrect import spell
    
spell('horse')


# ... is slow
# ===================
# Running on my machine it takes around 0.001 second to correct one, 5-letter word, 0.17 seconds to correct an 8-letter word and 0.26 seconds for a 10-letter word. If we were to try correcting every word in every comment in both the training and test sets it would take *roughly*: 
# 
# ~0.01 sec/word \* ~100 words/comment \* 320,000 comments = 320,000 seconds or **88 hours!**
# 
# Brute force is pretty obviously a bad idea. Fortunatly with a few tweaks we can get it running much faster:
# * Only try to spellcheck words that are missing form the word embedding
# * Skip words that are >24 characters long, they take *forever* to correct and are usually beyond repair anyway
# * Multiple processes
# 
# The result:

# In[ ]:


from multiprocessing import Pool, cpu_count
import pandas as pd
import re


def get_known_words(word_embeddings_file):
    words = set()
    with open(word_embeddings_file,encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            words.add(values[0].lower())
    return words


EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
if RUNNING_ON_KAGGLE:
    words = set()
else:
    words = get_known_words(EMBEDDING_FILE)


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def spell_check(chunk):
    fixed_rows = []
    for i,row in chunk.iterrows():
        fxd_words = []
        comment = row['comment_text'].lower()
        comment = re.sub('[^a-zA-Z ]+', '', comment)
        for w in comment.split():
            if w is None:
                continue
            if w in words or len(w) > 24:
                fxd_words.append(w)
            else:
                fxd_words.append(spell(w).lower())
        sp_comment = ' '.join(fxd_words)
        fixed_rows.append((row[0],sp_comment))
    return fixed_rows


PROC_COUNT = cpu_count()
CHUNK_SIZE = 1024
pool = Pool(PROC_COUNT)

# Uncomment line below to run
for set_name in [] #['train', 'test']:
    source = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/'+set_name+'.csv') # remove [:100] to proces all examples
    source['comment_text'] = source['comment_text'].astype(str)
    result = source.copy()

    fixed_rows = pool.map(spell_check,chunker(source,CHUNK_SIZE))
    for fxd_row in fixed_rows:
        for index,fixed_comment in fxd_row:
            result.set_value(index,'comment_text',fixed_comment)

    if RUNNING_ON_KAGGLE:
        print(result)
    else:
        result.to_csv('sp_check_'+set_name+'.csv')


# On my machine it took just **a little over 2 hours **to run, much better than 88. 
# 
# I included my outputs in the data section of this kernal feel free to use them with your own models. Note: I also used 's [Prashant Kikani](https://www.kaggle.com/prashantkikani)'s [super-awesome preprocessing code](https://www.kaggle.com/prashantkikani/pooled-gru-glove-with-preprocessing) on my inputs first. Thanks Prashant!
# 
# Let me know if you have any questions, comments, suggestions or improvements! I'd love to get your feedback. Thanks for reading.

# In[ ]:




