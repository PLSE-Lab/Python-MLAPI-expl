#!/usr/bin/env python
# coding: utf-8

# This is an attempt to solve difficulty 3. It builds on the tools provided by this awesome kernel to solve difficulty 1. 
# https://www.kaggle.com/rturley/a-first-crack-tools-and-first-cipher-solution

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

import collections, itertools
import sklearn.feature_extraction.text as sktext
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
train_data.head()


# In[ ]:


test_data = pd.read_csv("../input/test.csv")
test_data.head()


# In[ ]:


from sklearn.datasets import fetch_20newsgroups
plaintext_data = fetch_20newsgroups(subset='all', download_if_missing=True)
category_names = plaintext_data.target_names


# Some count stastics for each difficulty level and plaintext. As has been solved before, level 1 and 2 are substitution ciphers. Level 4 clearly looks like a transposition/shuffling cipher. For the level 3, it definitely is not a transposition cipher, and it does not look like simple substitution either. Some letters do have exact same counts as level 2 like 8, $, e, }, 4, * etc.  
# 
# Also https://www.kaggle.com/lbronchal/without-breaking-ciphers-0-48-lb kernel explores simple tf-idf based features for classification. For the level 3, the notebook achieves considerable better performance then level 4 and is less than simple substitution ciphers of level 2 and level 1. 
# 
# One could make a guess that level 3 is some complex substitution cipher (for the lack of proper word), where some chars are left unchanged.

# In[ ]:


plain_counts = pd.Series(collections.Counter(itertools.chain.from_iterable(plaintext_data.data)))     .rename("counts").to_frame()     .sort_values("counts", ascending = False)
plain_counts = 1000000 * plain_counts / plain_counts.sum()
plain_counts = plain_counts.reset_index().rename(columns = {"index":"plain_char"})

diff_counts = []
for i in range(1,5):
    counts = pd.Series(
        collections.Counter(itertools.chain.from_iterable(train_data.query("difficulty == @i")["ciphertext"].values)) + \
        collections.Counter(itertools.chain.from_iterable(test_data.query("difficulty == @i")["ciphertext"].values))
        ).rename("counts").to_frame() \
        .sort_values("counts", ascending = False)
    counts = 1000000 * counts / counts.sum()
    counts = counts.reset_index().rename(columns = {"index":"diff_{}".format(i)})
    diff_counts.append(counts)

pd.concat([plain_counts] + diff_counts, axis = 1).head(20)


# Lets start doing some exploration.
# 
# I am beginning with the decryption map, which deciphers level 2 -> plaintext. Apply the same map and see if we can find something.  

# In[ ]:


##diff2 -> plain, subs
key = [['8', ' '],['$', 'e'],['{', 't'],['V', 'a'],['e', 'o'],['h', 'i'],['\x10', 'n'],['}', 's'],['*', 'r'],['7', 'h'],['?', 'l'],['z', '\n'],['H', 'd'],['f', 'c'],['j', 'u'],['4', 'm'],['\x1a', '.'],['x', '-'],['.', 'p'],['k', 'g'],['v', 'y'],['d', 'f'],['^', 'w'],['\x18', 'b'],['b', '>'],['[', ','],['\x03', 'v'],['6', 'A'],['A', 'I'],['m', 'k'],['B', "'"],['N', ':'],['E', 'S'],['S', '1'],['\x06', 'T'],['(', 'X'],['c', '0'],['l', 'M'],['9', 'C'],['%', '*'],['\x02', ')'],['\x08', '('],['O', '2'],['a', '='],['\x0c', 'N'],['&', 'R'],['@', 'P'],['|', '3'],['\x7f', 'D'],[',', 'O'],['i', '@'],['L', 'E'],['M', 'L'],['=', '"'],['C', '9'],['X', '\t'],['\x1b', '5'],['1', 'F'],['n', 'H'],['Q', 'B'],['3', '4'],['0', '_'],['2', 'x'],['s', 'W'],['<', '6'],[')', 'G'],['_', 'j'],['G', 'U'],['u', '8'],['\x19', '?'],['-', '?'],['o', 'z'],['F', '/'],[';', '|'],['\t', 'J'],['~', 'K'],['W', '!'],['!', 'V'],["'", '<'],[' ', 'Y'],['\n', '+'],['#', 'q'],['I', '$'],[':', '#'],[']', 'Q'],['/', '^'],['g', '#'],['\x1e', '%'],['p', ']'],['5', ']'],['\\', '['],['`', 'Z'],['t', '&'],['y', '&'],['R', 'Z'],['P', '}'],['r', '{'],['"', '\r'],['T', 'u'],['Z', '\x02']]
decrypt_map_2 = {i:j for i,j in key}


# In[ ]:


diff_3_data = train_data.query("difficulty == 3").copy()
diff_3_data["trans"] = diff_3_data["ciphertext"].apply(
    lambda x:''.join([decrypt_map_2.get(k,'?') for k in x])
)

##top starting letters 
diff_3_data.trans.str[:5].value_counts().head()


# In[ ]:


collections.Counter([i[:5] for i in plaintext_data.data]).most_common(5)


# In[ ]:


diff_3_data.loc[diff_3_data["trans"].apply(lambda x:x[:5] == 'FrMmZ')].head(3)


# A sample translated text, some words do make sense. One can find matching plaintext by some trial and error. This particular cipher belongs to target 11, is part of first 300 charecters of a plaintext document (starting from "From:"), Surname could be Russell, second last word could be 'whatever'. If you search for this, you will find exact match in plaintext data. 
# I think I got lucky with this, but then this gave confidence that you can try and figure out more pairs this way.
# 
# I went ahead and found 5 pairs, it hardly took 10 minutes. 

# In[ ]:


diff_3_data.query("Id == 'ID_fb163c212'").trans.iloc[0]


# In[ ]:


target_11_data = [i[:300] for i,j in zip(plaintext_data.data, plaintext_data.target) if j == 11] 
[i for i in target_11_data if i.find('Russell') > 0 and i.find("whatever") > 0]


# In[ ]:


pd.options.display.max_columns = 300

pd.options.display.max_rows = 300

plain_1, cipher_1 = [
    '''From: trussell@cwis.unomaha.edu (Tim Russell)\nSubject: Re: Once tapped, your code is no good any more.\nOrganization: University of Nebraska at Omaha\nDistribution: na\nLines: 18\n\ngeoff@ficus.cs.ucla.edu (Geoffrey Kuenning) writes:\n\n>It always amazes me how quick people are to blame whatever\n>administr''',
    '''FrMmZ tr8ssellkWw#s.znWm}h\t.e2H (T#m R]ssell)\n L?bjeut/ Re? Onue tUppeE, @fzr &&?e zs n@ ugo} $n\n m?re.# OrzUnM>?tkWnd $nzversMt8 gf Nebr?s\n\x02 :t Om9h\x02g DMstr8b@tkWnd n\t# B#nesu 10k f 0eMizk?#g-s.\ns.Mkl\x02.eU] (GeoM?re\n K>ennznz) wrWtesZk & #2t }lw\t0s \x02m?8es me how q{8uo peMple \x02re t] bl\x02me wh$tever& #'''
    
]

plain_2, cipher_2 = [
    '''From: hollasch@kpc.com (Steve Hollasch)\nSubject: Re: Raytracing Colours?\nSummary: Illumination Equations\nOrganization: Kubota Pacific Computer, Inc.\nLines: 44\n\nasecchia@cs.uct.ac.za (Adrian Secchia) writes:\n| When an incident ray (I) strikes an object at point P ...  The reflected\n| ray (R) and the ''',
    '''FrMmZ h]ll\x02sWhi{p&.\n#m (Eteve {fll'sch)i ZHbje-t? ReZ RU\ntr$okn& C&l#?rs?f ?]mmarfI Sll>mzn\x02tW@n ?q@Ity&ns> Or{\x02n#8at]{nZ K>bMt\x02 PUu80k& C&mpMter, dnu.& 1inesa 44k f :sekuhW\x02yos.z&t.}i.y\x02 (AUrHan LeWuh\n9) wrMtes?f | chen ?n kn&>?ent r9\n (\x02) strHkes ?n {bjeut :t pMMnt P ...  The rezleute?& | rIu (R) '''
]

plain_3, cipher_3 = [
    '''From: snichols@adobe.com (Sherri Nichols)\nSubject: Re: Braves Pitching UpdateDIR\nOrganization: Adobe Systems Incorporated\nLines: 13\n\nIn article <1993Apr15.010745.1@acad.drake.edu> sbp002@acad.drake.edu writes:\n>or second starter.  It seems to me that when quality pitchers take the\n>mound, the other ''',
    '''FrMmZ snWuh{lsiIa&be.iMm ($herr# NHchkls)k $>bjektZ Re? ?r$ves P#tghyno $p2\x02teD\x02Rc Org?n8W:tz@n? AE{be ?=stems 2n\n#rpMr\x02teUc L]nesI !3{ # Sn UrtMWle <ZdB3AprL5.o\x02#?45.ZzaH$E.\x02r9?e.e1Hz sbp>i90\tk\x02U.:ra8e.e?H wr\ntes?f k@r seWfnL st?rter.  ?t seems t# me th9t when qH?lHty p]tWhers t\x02oe the# cm]Hn?, the'''
]

plain_4, cipher_4 = [
    '''From: art@cs.UAlberta.CA (Art Mulder)\nSubject: comp.windows.x: Getting more performance out of X.  FAQ\nSummary: This posting contains a list of suggestions about what you can do to get the best performance out of X on your workstation -- without buying more hardware.\nKeywords: FAQ speed X\nNntp-Posti''',
    '''FrMmZ UrtkWs./AlbertI.CA (Art \nfl?er)# $@bjeMt: H{mp.wMn\x02Mws.xZ GettWnz m{re per0urm?n\ne #?t @0 X.  FAQc S{mm$r\n/ Thzs p@stWnz WfntIyns } lust Mi s@?gest]{ns \x02b??t wh\x02t #@? oIn a& t# {et the best perifrm'nce k?t @y X Mn \n]Hr w{rMstIty&n \no wMthf]t buf8nz m?re h9rEwUre.f Ke@wfrLs' FAQ spee' X> NntpoP'''   
]

plain_5, cipher_5 = [
    '''From: joslin@pogo.isp.pitt.edu (David Joslin)\nSubject: Apology to Jim Meritt (Was: Silence is concurance)\nDistribution: usa\nOrganization: Intelligent Systems Program\nLines: 39\n\nm23364@mwunix.mitre.org (James Meritt) writes:\n>}So stop dodging the question.  What is hypocritical about my\n>}criticizing''',
    '''FrMmZ j]slMnypf]W.>sp.putt.e2H (D?vHd Jksl8n)f !?bjeut? Ap@l{u= tW J>m Herztt (??su S]lenWe Ms -Mnu8r\x02nWe)& DkstrybftuMnZ @s'\n OrW$nM#9tM]nZ :ntell#]ent 1]stems Pr#{r\x02m& 1inesa 3}f { ma3364kmw8nMx.m8tre.fr] (J?mes cerutt) wrztesZ& i}L{ st@p \x02MEyMn- the q@estkWn.  kh\tt zs h\npfMrit]W\x02l :bMHt m#f z2gry'''
]


# In[ ]:


print(pd.DataFrame({'plain':list(plain_1)[:len(cipher_1)], 'cipher':list(cipher_1)}).T)


# There is an issue, as some people have pointed out in discussion, some extra spaces are present after newline chars. So if you dont know the key for '\n' you cannot clean it right now.  So assuming that the only issue, we can atleast clean data for these pairs, ignore next char in cipher when you encounter '\n' in plain. Now the pair does seem to align pretty well.
# 
# Now I tried to crack the code, but unfortunately could not figure out. So I tried another approach. Some chars are left unchanged from level 2-> 3, but those which are changed are not 1 to 1, maybe there are some hidden state involved. 
# 
# If anyone does figure out the cipher, do let me know.
# 
# For the aligned and cleaned pairs, find all the chars which are left unchanged. 

# In[ ]:


plain, cipher = [], []

pairs = [(plain_1, cipher_1), (plain_2, cipher_2), (plain_3, cipher_3), (plain_4, cipher_4), (plain_5, cipher_5)]
for p_temp, c_temp in pairs:
    i1,i2 = 0,0
    while 1:
        p,c = p_temp[i1], c_temp[i2]
        plain.append(p)
        cipher.append(c)
        if p == '\n':
            i2+=1
        i1 += 1 
        i2 += 1
        if i2 == 300:
            break

pd.DataFrame({'plain':list(plain), 'cipher':list(cipher)}).T

possible_maps = collections.defaultdict(list)
for i,j in zip(plain, cipher):
    possible_maps[i].append(j)

sure_map = {}
unsure_map = {}
for i,j in possible_maps.items():
    if (len(set(j)) == 1) and len(j) !=1:
        sure_map[i] = j[0]
    else:
        unsure_map[i] = set(j)
print(len(sure_map), len(unsure_map))


# So this is where magic happens. The sure_map above has all the chars which are left unchanged. In the plaintext and cipher data(after applying level 2 decryption), apply the sure map and replace others with '?'. Then a simple word based tf-idf feature extraction followed by nearest neighbour search on plaintext data should do some non trivial work for us.

# In[ ]:


d = train_data.query("difficulty == 3").copy()
d["trans"] = d["ciphertext"].apply(
    lambda x:''.join([decrypt_map_2.get(k,'?') for k in x])
)
d["trans"] = d["trans"].apply(
    lambda x:''.join([k if k in sure_map else '?' for k in x])
)


# In[ ]:


X_train = [''.join([k if k in sure_map else '?' for k in i]) for i in plaintext_data.data]
y_train = plaintext_data.target

X_test = d["trans"].values
y_test = d["target"].values


# In[ ]:


clf = Pipeline([
    ('vectorizer', sktext.CountVectorizer(lowercase=True, ngram_range = (1,2))),
    ('tfidf', sktext.TfidfTransformer()),
    ('clf', KNeighborsClassifier(n_neighbors = 1))
])

clf.fit(X_train, y_train)


# In[ ]:


print(classification_report(y_train, clf.predict(X_train)))


# In[ ]:


print(classification_report(y_test, clf.predict(X_test)))


# Hint for level 4: Can be solved similary upto an extent, if you are not keen on getting 100 or 99% score.
