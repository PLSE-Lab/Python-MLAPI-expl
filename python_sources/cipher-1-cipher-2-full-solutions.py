#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import os
import re


# In[ ]:


from sklearn.datasets import fetch_20newsgroups
from collections import Counter


# This kernel focuses on the cryptanalysis of the ciphers.
# * It contains a **full manual cryptanalysis of cipher #1** before we move on to an automated one (cipher1_map.csv).
# * It contains also the results of a **full cryptanalysis of cipher #2** (cipher2_map.csv)
# 
# The sole purpose of the manual cryptanalysis illustrated on cipher #1 are:
# * To give an idea of what's doable manually on cipher #1 (spoiler alert: a lot)
# * To demonstrate how tedious and time-consuming this manual analysis can be
# * To illustrate the needs and requirements for automated cryptanalysis tools  (spoiler alert: fuzzywuzzy)
# 
# There is already a very nice automated kernel that has been published: https://www.kaggle.com/rturley/a-first-crack-tools-and-first-cipher-solution
# 
# Here is the cryptanalysis plan we will follow:
# * Take advantage of the existing knowledge on the issue at hand, here we know that:
#   * The plaintexts corresponding to our ciphertexts are part of the 20 newsgroups dataset
#   * The cipher #1 is a substitution cipher: it is a simple table wich maps a plaintext character to a ciphertext character  
# * Analyze the most frequent characters in the ciphertexts and compare them to the plaintexts ones. 
#   * Our hope is to be able to match a few usual suspects that pop out statistically in English like e or t. 
#   * And more importantly to identify words separators (the space or newline characters) which would allow us to move on to the next step
# * Analyze the most frequent words in the ciphertexts and compare them to the plaintexts ones.
#   * If we have partially decoded words that we can recognize then we can increase of knowledge of the cipher map
# * Complete the cryptanalysis by specifically looking for less frequent characters and matching plaintexts and ciphertexts pairs
# 
# In short: Characters > Words > Messages
#     

# # Loading the Datasets

# In[ ]:


print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/20-newsgroups-ciphertext-challenge/train.csv')
test = pd.read_csv('../input/20-newsgroups-ciphertext-challenge/test.csv')


# In[ ]:


train_chars = train['ciphertext'].map(len).sum()
test_chars = test['ciphertext'].map(len).sum()


# In[ ]:


train_plain = fetch_20newsgroups(subset='train')
test_plain = fetch_20newsgroups(subset='test')


# In[ ]:


train_plain = pd.DataFrame(data = np.c_[train_plain['data'], train_plain['target']],
                    columns= ['plaintext','target'])

test_plain = pd.DataFrame(data = np.c_[test_plain['data'], test_plain['target']],
                    columns= ['plaintext','target'])


# In[ ]:


train_plain_chars = train_plain['plaintext'].map(len).sum()
test_plain_chars = test_plain['plaintext'].map(len).sum()


# In[ ]:


print('# of characters in train ciphertexts: {:,}'.format(train_chars))
print('# of characters in test ciphertexts: {:,}'.format(test_chars))
print('# of characters in train&test ciphertexts: {:,}'.format(train_chars + test_chars))


# In[ ]:


print('# of characters in train plaintexts: {:,}'.format(train_plain_chars))
print('# of characters in test plaintexts: {:,}'.format(test_plain_chars))
print('# of characters in train&test plaintexts: {:,}'.format(train_plain_chars + test_plain_chars))


# So as discussed https://www.kaggle.com/c/20-newsgroups-ciphertext-challenge/discussion/74701 , the mapping between plain and cipher texts will probably not be exact as their number of characters are not equal.

# We focus on cipher #1

# In[ ]:


difficulty_level = 1

df_c = pd.concat([train[train['difficulty']==difficulty_level][['ciphertext']],
                  test[test['difficulty']==difficulty_level][['ciphertext']]], axis = 0).rename(columns={'ciphertext':'text'})

df_p = pd.concat([train_plain[['plaintext']],
                  test_plain[['plaintext']]], axis = 0).rename(columns={'plaintext':'text'})


# # Character Analysis
# 
# As discussed in other kernels, like https://www.kaggle.com/mithrillion/enigma-was-gimped-by-weather-reports, cipher #1 is very probably a substitution cipher, so we start our cryptanalysis by attempting to match character frequencies.

# In[ ]:


def char_freqs(df):
    text = ''.join(df['text'])
    freqs = 100 * pd.Series(Counter(text)) / len(text)
    freqs.sort_values(ascending= False, inplace=True)
    return(freqs)


# In[ ]:


c_freqs = char_freqs(df_c)
p_freqs = char_freqs(df_p)


# In[ ]:


c_freqs.head()


# In[ ]:


p_freqs.head()


# In[ ]:


c_freqs.describe()


# In[ ]:


p_freqs.describe()


# In[ ]:


def filter_freqs(freqs, freqs_thresh, freqs_sep_thresh):
    freqs = freqs[freqs > freqs_thresh]
    freqs_sep = pd.concat([freqs.diff(-1),-freqs.diff(1)],axis=1).apply(np.nanmin, axis=1)
    freqs_sep = freqs_sep[freqs_sep > freqs_sep_thresh]
    return(freqs[freqs_sep.index])


# In[ ]:


c_freqs = filter_freqs(c_freqs, 0.25, 0.10)
p_freqs = filter_freqs(p_freqs, 0.25, 0.10)


# In[ ]:


char_dic = pd.concat([pd.DataFrame(c_freqs).reset_index().rename(columns={'index' : 'c',0:'c_freq'}),
                      pd.DataFrame(p_freqs).reset_index().rename(columns={'index' : 'p',0:'p_freq'})],
                      axis=1).dropna()


# In[ ]:


char_dic


# # Word Analysis
# 
# Thanks to the previous step, we have been able to identify some important characters, namely some word separators, blank space and newline. We can now move on to words frequency analysis.
# 
# We start by looking at the most frequent words overall, then we look at the most frequent words by word length and last we look at words that contain specific characters.

# In[ ]:


ciphertext = '1'.join(df_c['text'])
plaintext = ' '.join(df_p['text'])


# In[ ]:


def word_freqs(s, seps):
    words = list(filter(None, re.split('[' + ''.join(seps) + ']+',s)))
    freqs = pd.Series(words).value_counts()
    freqs = freqs.reset_index().rename(columns={'index' : 'word', 0:'count'})
    freqs['word_len'] = freqs['word'].map(len)
    freqs['abs_freq'] = 100 * freqs['count'] / len(words)
    freqs = pd.merge(freqs,
                     freqs.groupby('word_len')[['count']].sum().reset_index().rename(columns={'count' : 'word_len_count'}),
                     on='word_len')
    freqs['rel_freq'] = 100 * freqs['count'] / freqs['word_len_count']
    freqs.sort_values(by='abs_freq',ascending=False,inplace=True)
    return(freqs)


# In[ ]:


c_words = word_freqs(ciphertext,['1','s'])


# In[ ]:


p_words = word_freqs(plaintext,[' ','\n'])


# In[ ]:


c_words.head(15)


# In[ ]:


p_words.head(15)


# In[ ]:


c_words.describe()


# In[ ]:


p_words.describe()


# In[ ]:


c_freqs = filter_freqs(c_words.set_index('word')['abs_freq'], 0.25, 0.05)
p_freqs = filter_freqs(p_words.set_index('word')['abs_freq'], 0.25, 0.05)


# In[ ]:


words_dic = pd.concat([pd.DataFrame(c_freqs).reset_index().rename(columns={'index' : 'c',0:'c_freq'}),
                      pd.DataFrame(p_freqs).reset_index().rename(columns={'index' : 'p',0:'p_freq'})],
                      axis=1).dropna()


# In[ ]:


words_dic


# In[ ]:


words_dic = words_dic.loc[words_dic.iloc[:,0].map(len) == words_dic.iloc[:,2].map(len)]


# In[ ]:


def dico_update(c_list, p_list, dico):
    for i, w in enumerate(c_list):
        for j, c in enumerate(w):
            if c in dico:
                assert(dico[c] == p_list[i][j])
            elif p_list[i][j] in dico.values():
                assert(c == p_list[i][j])
            else:
                dico[c] =  p_list[i][j]
    return(dico)


# In[ ]:


cp_dico = dico_update(words_dic.iloc[:,0].values,words_dic.iloc[:,2].values,dict(zip(''.join(char_dic['c']), ''.join(char_dic['p']))))


# In[ ]:


cp_dico


# In[ ]:


translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


c_words.head()


# In[ ]:


word_len = 3


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(10)


# In[ ]:


p_words[p_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(10)


# In[ ]:


cp_dico['o'] = 'y'
cp_dico['c'] = 'u'
cp_dico['_'] = 'c'
cp_dico['{'] = 'T'
cp_dico['\x03'] = 'b' 


# In[ ]:


ciphertext = '1'.join(df_c['text'])
c_words = word_freqs(ciphertext,['1','s'])
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


word_len = 4


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(10)


# In[ ]:


p_words[p_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(10)


# In[ ]:


cp_dico['z'] = 'v'
cp_dico['a'] = 'i'
cp_dico['W'] = 'w'
cp_dico['-'] = 'm'


# In[ ]:


ciphertext = '1'.join(df_c['text'])
c_words = word_freqs(ciphertext,['1','s'])
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


word_len = 5


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(10)


# In[ ]:


p_words[p_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(10)


# In[ ]:


cp_dico['*'] = 'F'
cp_dico['G'] = ':'
cp_dico[';'] = '\''
cp_dico['f'] = 'k'


# In[ ]:


ciphertext = '1'.join(df_c['text'])
c_words = word_freqs(ciphertext,['1','s'])
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


word_len = 6


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(10)


# In[ ]:


p_words[p_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(10)


# In[ ]:


cp_dico['F'] = 'L'
cp_dico['\''] = 'p'
cp_dico['d'] = 'g'


# In[ ]:


ciphertext = '1'.join(df_c['text'])
c_words = word_freqs(ciphertext,['1','s'])
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


word_len = 8


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(10)


# In[ ]:


p_words[p_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(10)


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(10).iloc[1,0][0]


# In[ ]:


cp_dico['>'] = 'S'
cp_dico['X'] = 'j'
cp_dico['d'] = 'g'
cp_dico['\x1a'] = 'q' 


# In[ ]:


ciphertext = '1'.join(df_c['text'])
c_words = word_freqs(ciphertext,['1','s'])
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


word_len = 9


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


p_words[p_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


cp_dico['w'] = 'K'
cp_dico['2'] = 'C'
cp_dico[':'] = 'I'
cp_dico['9'] = 'P'


# In[ ]:


ciphertext = '1'.join(df_c['text'])
c_words = word_freqs(ciphertext,['1','s'])
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


word_len = 10


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


p_words[p_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


cp_dico['@'] = 'U'
cp_dico['x'] = 'D'
cp_dico['+'] = 'x'


# In[ ]:


ciphertext = '1'.join(df_c['text'])
c_words = word_freqs(ciphertext,['1','s'])
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


word_len = 11


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


p_words[p_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


cp_dico['T'] = ','
cp_dico['%'] = 'O'


# In[ ]:


ciphertext = '1'.join(df_c['text'])
c_words = word_freqs(ciphertext,['1','s'])
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


word_len = 12


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


p_words[p_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15).iloc[1,0][-1]


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15).iloc[3,0][3]


# In[ ]:


cp_dico['\x08'] = '.'
cp_dico['q'] = '-'
cp_dico['\x1e'] = 'R' 
cp_dico['h'] = 'z'


# In[ ]:


ciphertext = '1'.join(df_c['text'])
c_words = word_freqs(ciphertext,['1','s'])
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


word_len = 13


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


p_words[p_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15).iloc[2,0][2]


# In[ ]:


cp_dico['!'] = 'X'
cp_dico['\x7f'] = 'N'
cp_dico['/'] = 'A'


# In[ ]:


ciphertext = '1'.join(df_c['text'])
c_words = word_freqs(ciphertext,['1','s'])
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


word_len = 14


# In[ ]:


c_words[c_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


p_words[p_words['word_len'] == word_len].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


cp_dico['b'] = '@'


# Now that our mapping dictionary has reached a decent size, we have to refine our search to avoid stumbling only upon fully decoded characters, since we started by the most frequent ones.

# In[ ]:


ciphertext = '1'.join(df_c['text'])

undecoded = ''.join(set(ciphertext).difference(set(cp_dico.keys())))
set_undecoded = set(undecoded)

c_words = word_freqs(ciphertext,['1','s'])
c_words['to_decode'] = c_words['word'].map(lambda x: any((c in set(undecoded)) for c in x))
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
c_words['word'] = c_words['word'].map(lambda x: x.translate(translation))


# In[ ]:


word_len = 6


# In[ ]:


c_words[(c_words['word_len'] == word_len) & (c_words['to_decode'] == True)].sort_values(by='rel_freq',ascending=False).head(15)


# In[ ]:


cp_dico['}'] = 'J'
cp_dico['J'] = 'B'
cp_dico['e'] = 'M'
cp_dico['"'] = 'G'


# # Message Analysis
# 
# After the previous step, we could move one to fully pairing messages together

# In[ ]:


target = 1


# In[ ]:


df_c = train[(train['difficulty'] == difficulty_level) & (train['target'] == target)].copy()
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c['ciphertext'] = df_c['ciphertext'].map(lambda x: x.translate(translation))


# In[ ]:


df_c.head()


# In[ ]:


df_p = pd.concat([train_plain[train_plain['target'] == str(target)],
                  test_plain[test_plain['target'] == str(target)]], axis = 0)


# In[ ]:


def complete_dico(c,p):
    cp_dico_temp = {}
    for i,cc in enumerate(c.replace('\n ','\n')):
        if cc != p[i]:
            if cc not in cp_dico_temp:
                cp_dico_temp[cc] = p[i]
                print('cp_dico[\'{}\'] = \'{}\''.format(cc,p[i]))


# In[ ]:


c_index = 398
c = df_c.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'dfr@usna.navy.mil'


# In[ ]:


df_p.head()


# In[ ]:


df_p[df_p['plaintext'].str.contains(c_sample)]


# In[ ]:


p_index = 5620
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


complete_dico(c,p)


# In[ ]:


c


# In[ ]:


p


# In[ ]:


cp_dico['|'] = '('
cp_dico['y'] = ')'
cp_dico['g'] = 'H'
cp_dico['u'] = '3'
cp_dico['\x06'] = '7'
cp_dico['\t'] = '5'
cp_dico[','] = '4'
cp_dico['L'] = '1'
cp_dico['\\'] = '0'
cp_dico['n'] = '8'
cp_dico['['] = '>'
cp_dico[r' '] = '<'


# In[ ]:


df_c = train[(train['difficulty'] == difficulty_level) & (train['target'] == target)].copy()
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c['ciphertext'] = df_c['ciphertext'].map(lambda x: x.translate(translation))


# In[ ]:


df_c.head()


# In[ ]:


c_index = 146
c = df_c.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'now back to lurking'


# In[ ]:


df_p[df_p['plaintext'].str.contains(c_sample)]


# In[ ]:


p_index = 1156
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


p = p[p.find(c_sample):]


# In[ ]:


complete_dico(c,p)


# In[ ]:


cp_dico['r'] = '&'
cp_dico['l'] = 'W'
cp_dico['\x18'] = 'V'


# In[ ]:


df_c = train[(train['difficulty'] == difficulty_level) & (train['target'] == target)].copy()
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c['ciphertext'] = df_c['ciphertext'].map(lambda x: x.translate(translation))


# In[ ]:


df_c.head(15)


# In[ ]:


c_index = 807
c = df_c.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'bpirenne@eso.org'


# In[ ]:


df_p[df_p['plaintext'].str.contains(c_sample)]


# In[ ]:


p_index = 3864
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


c_sample = 'iga computer.'
p = p[p.find(c_sample):]
print(p)


# In[ ]:


complete_dico(c,p)


# In[ ]:


cp_dico['U'] = '['
cp_dico['i'] = ']'
cp_dico['3'] = ';'
cp_dico[' '] = '<'
cp_dico['~'] = '+'
cp_dico['<'] = '9'
cp_dico['H'] = '2'
cp_dico['5'] = '6'


# In[ ]:


df_c = train[(train['difficulty'] == difficulty_level) & (train['target'] == target)].copy()
translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c['ciphertext'] = df_c['ciphertext'].map(lambda x: x.translate(translation))


# In[ ]:


df_c.tail(15)


# In[ ]:


c_index = 38295
c = df_c.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'ones I know about are from Maximum Strategy'


# In[ ]:


df_p[df_p['plaintext'].str.contains(c_sample)]


# In[ ]:


p_index = 3474
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


c_sample = 'y that can do 180MB'
p = p[p.find(c_sample):]
print(p)


# In[ ]:


complete_dico(c,p)


# In[ ]:


cp_dico['?'] = '/'


# Where do we stand now?

# In[ ]:


df_c = train[(train['difficulty'] == difficulty_level)].copy()
c_alphabet = pd.Series(Counter(''.join(df_c['ciphertext'])))
c_alphabet.shape


# In[ ]:


len(cp_dico)


# In[ ]:


print('We have manually decoded {:.0%} of the cipher #1'.format(len(cp_dico)/c_alphabet.shape[0]))


# To finish the cryptanalysis we have to look specifically for the undecoded characters in plaintext/ciphertext message pairs

# In[ ]:


df_p = pd.concat([train_plain,test_plain], axis = 0)


# In[ ]:


undecoded = set(c_alphabet.index).difference(cp_dico.keys())
undecoded = list(undecoded)
print(undecoded)


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('4')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_sample = 'Imagination is more important than knowledge'
df_c_focus[df_c_focus['ciphertext'].str.contains(c_sample)]


# In[ ]:


c_index = 34448
c = df_c_focus.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'Imagination is more important than knowledge'


# In[ ]:


df_p[df_p['plaintext'].str.contains(c_sample)]


# In[ ]:


p_index = 7391
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


p = df_p.loc[p_index,'plaintext'].iloc[1]


# In[ ]:


print(p)


# In[ ]:


complete_dico(c,p[-350:]) 


# In[ ]:


cp_dico['4'] = '|'
cp_dico['Z'] = '='
cp_dico['.'] = '~'
cp_dico['m'] = '\\'
cp_dico['\x10'] = 'Y'
cp_dico[')'] = '_'


# In[ ]:


undecoded = set(c_alphabet.index).difference(cp_dico.keys())
undecoded = list(undecoded)
print(undecoded)


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('S')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 15690
c = df_c_focus.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'I know there are lots of graphics-board companies out'
df_p[df_p['plaintext'].str.contains(c_sample)]


# In[ ]:


p_index = 9998
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


complete_dico(c,p) 


# In[ ]:


p


# In[ ]:


c


# In[ ]:


cp_dico['S'] = '\x08'
cp_dico['6'] = '"'
cp_dico['&'] = '?'


# In[ ]:


undecoded = set(c_alphabet.index).difference(cp_dico.keys())
undecoded = list(undecoded)
print(undecoded)


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('I')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 1501
c = df_c_focus.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'got the same error when I tried to build'
df_p[df_p['plaintext'].str.contains(c_sample)]


# In[ ]:


p_index = 9368
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


complete_dico(c,p[862:])  


# In[ ]:


cp_dico['\x1c'] = '*'
cp_dico['Q'] = '\t'
cp_dico['I'] = '}'


# In[ ]:


undecoded = set(c_alphabet.index).difference(cp_dico.keys())
undecoded = list(undecoded)
print(undecoded)


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('B')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 37028
c = df_c_focus.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'to think about for the remote machine'
df_p[df_p['plaintext'].str.contains(c_sample)]


# In[ ]:


p_index = 2593
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


p = df_p.loc[p_index,'plaintext'].iloc[1]
print(p)


# In[ ]:


complete_dico(c,p[23543:])   


# In[ ]:


cp_dico['`'] = '#'
cp_dico['B'] = '$'
cp_dico['P'] = '!'
cp_dico['k'] = '{'
cp_dico['Y'] = '`'


# In[ ]:


undecoded = set(c_alphabet.index).difference(cp_dico.keys())
undecoded = list(undecoded)
print(undecoded)


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('V')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 3
c = df_c_focus.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'also hearty proponents of'
df_p[df_p['plaintext'].str.contains(c_sample)] 


# In[ ]:


p_index = 928
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


p = df_p.loc[p_index,'plaintext'].iloc[0]
print(p)


# In[ ]:


complete_dico(c,p[2617:])   


# In[ ]:


cp_dico['V'] = 'Q'


# In[ ]:


undecoded = set(c_alphabet.index).difference(cp_dico.keys())
undecoded = list(undecoded)
print(undecoded)


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('D')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 4840
c = df_c_focus.loc[c_index,'ciphertext']
c


# In[ ]:


c_sample = 'out the environment and their future'
df_p[df_p['plaintext'].str.contains(c_sample)] 


# In[ ]:


p_index = 11142
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


p[2642:]


# In[ ]:


cp_dico['D'] = '\x0c'   


# In[ ]:


undecoded = set(c_alphabet.index).difference(cp_dico.keys())
undecoded = list(undecoded)
print(undecoded)


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('=')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 31228
c = df_c_focus.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'Kyle P Hunter'
df_p[df_p['plaintext'].str.contains(c_sample)] 


# In[ ]:


p_index = 5507
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


p = df_p.loc[p_index,'plaintext'].iloc[1]


# In[ ]:


print(p)


# In[ ]:


complete_dico(c,p)   


# In[ ]:


cp_dico['='] = '\x10'


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('\$')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 11727
c = df_c_focus.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'georgel@NeoSoft.com'
df_p[df_p['plaintext'].str.contains(c_sample)] 


# In[ ]:


p_index = 5150
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


p = df_p.loc[p_index,'plaintext'].iloc[1]


# In[ ]:


complete_dico(c,p[292:])   


# In[ ]:


cp_dico['$'] = '\x02'


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('K')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 232
c = df_c_focus.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'From: szh@zcon.com'
df_p[df_p['plaintext'].str.contains(c_sample)] 


# In[ ]:


p_index = 7361
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


p = df_p.loc[p_index,'plaintext'].iloc[1]


# In[ ]:


complete_dico(c,p)   


# In[ ]:


cp_dico['K'] = 'Z'


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('p')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 9981
c = df_c_focus.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'Since it is a Life Time membership, you won'
df_p[df_p['plaintext'].str.contains(c_sample)] 


# In[ ]:


p_index = 4938
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


p = df_p.loc[p_index,'plaintext'].iloc[0]


# In[ ]:


print(p)


# In[ ]:


cp_dico['p'] = '\x1e'  


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('\(')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 38567
c = df_c_focus.loc[c_index,'ciphertext']
print(c)


# In[ ]:


c_sample = 'After all the space walking,  they are going to  re-boost the HST'
df_p[df_p['plaintext'].str.contains(c_sample)] 


# In[ ]:


p_index = 810
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


p = df_p.loc[p_index,'plaintext'].iloc[1]


# In[ ]:


print(p)


# In[ ]:


cp_dico[r'('] = '^'


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('\x0c')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 38788
c = df_c_focus.loc[c_index,'ciphertext']


# In[ ]:


c


# In[ ]:


c_sample = 'From: dbl@visual.com'
df_p[df_p['plaintext'].str.contains(c_sample)] 


# In[ ]:


p_index = 1459
p = df_p.loc[p_index,'plaintext']
print(p)


# In[ ]:


p = df_p.loc[p_index,'plaintext'].iloc[0]


# In[ ]:


p


# In[ ]:


cp_dico['\x0c'] = r'%'


# In[ ]:


df_c_focus = df_c[df_c['ciphertext'].str.contains('E')].copy()

translation = str.maketrans(''.join(cp_dico.keys()), ''.join(cp_dico.values()))
df_c_focus['ciphertext'] = df_c_focus['ciphertext'].map(lambda x: x.translate(translation))

df_c_focus


# In[ ]:


c_index = 392
c = df_c_focus.loc[c_index,'ciphertext']
c


# In[ ]:


cp_dico['E'] = 'E'


# # And we are done!

# In[ ]:


print('We have manually decoded {:.2%} of the cipher #1'.format(len(cp_dico)/c_alphabet.shape[0]))


# In[ ]:


cipher1_df = pd.DataFrame(list(cp_dico.values()),index=list(cp_dico.keys()),columns=['plain']).reset_index().rename(columns={'index' : 'cipher'})


# In[ ]:


cipher1_df.head()


# In[ ]:


cipher1_df.to_csv('cipher1_map.csv', index=False)


# # And using either manual or automated cryptanalysis...
# It is as easy to decrypt cipher #2

# In[ ]:


cipher2_df = pd.read_csv('../input/cipher-2-full-solution/cipher2_map.csv')


# In[ ]:


cipher2_df.to_csv('cipher2_map.csv', index=False)


# Stay tuned for the next kernel on automated cryptanalysis in which we will sum up the lessons learnt during the above manual work... and for the cryptanalysis of cipher #3 if you request it enough ;-)

# In[ ]:




