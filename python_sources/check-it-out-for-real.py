#!/usr/bin/env python
# coding: utf-8

# Here we explore the text normalization data with a primary focus on dicks.  We demonstrate that you can never have enough dicks.  Over 30 dicks are erected in this kegel notebook.  The strength and capability of a dick is truly impressive because they are solid, inflexible, and fast to finish (look ups).  We will use dicks to find the g-spot (g for google).  Some dicks are very long -- over 10 million in length!  Whether you like long or short dicks... this notebook is for you! 
# 
# 1. (1)  **READ THE TRAIN SHIT FILE**
# 1. (2)  **FIGURE OUT WHAT'S DIFFERENT**
# 1. (3)  **LOOK AT NOT SAME SHIT**
# 1. (4)  **FIGURE OUT WHICH SHIT IS THE SAME**
# 1. (5)  **BUILD SOME DICKS**
# 1. (6)  **FUCKING MISSING SHIT**
# 1. (7)  **LOOK AT THE FUCKING MISSING SHIT**
# 1. (8)  **BUILD SOME MORE DICKS**
# 1. (9)  **FIGURE OUT WHERE NOT SAME SHIT HAS DIFFERENT SHIT**
# 1. (10)  **LOOK AT THAT SHIT**
# 1. (11)  **LOOK AT THAT SHIT AGAIN BUT DIFFERENT**
# 1. (12)  **FIND WHERE THE AFTER PARTY HAS CAPITAL LETTERS AND FIGURE OUT WTF**
# 
# UPDATE! 
# 1. (13)  **OMG KEGEL, G-SPOT. WT
# 
# 
# 1. (13)  **THROW AN ERROR**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import re, string, collections
from fuzzywuzzy import fuzz
from tqdm import tqdm
from IPython.display import display


# 1. **READ THE TRAIN SHIT FILE**

# In[ ]:


# Read en_train.csv  file.
test_df = pd.read_csv(filepath_or_buffer="../input/en_test.csv", encoding="utf-8", dtype={'class':'category'})
train_df = pd.read_csv(filepath_or_buffer="../input/en_train.csv", encoding="utf-8", dtype={'class':'category'})
train_df.head()


# In[ ]:


test_before = test_df['before'].tolist()
train_after = train_df['after'].tolist()
train_class = train_df['class'].tolist()
test_idx = test_df.index.tolist()
train_before = train_df['before'].tolist()
train_idx = train_df.index.tolist()

test_before_DICK = dict(zip(test_idx, test_before))
train_idx_before_DICK = dict(zip(train_idx, train_before))
train_idx_after_DICK = dict(zip(train_idx, train_after))


test_before_str_only = [(x,y) for x,y in zip(test_idx, test_before) if type(y) == type(str())]
test_before_str_only_idxs = [x[0] for x in test_before_str_only]

train_before_str_only = [(x,y) for x,y in zip(train_idx, train_before) if type(y) == type(str())]
train_before_str_only_idxs = [x[0] for x in train_before_str_only]


# In[ ]:


#display(train_df[pd.isnull(train_df['after'])])
#display(test_df[pd.isnull(test_df['before'])])
trb_nan_idx = train_df[pd.isnull(train_df['before'])].index.tolist()

train_df.loc[trb_nan_idx, 'before'] = ' '
train_df.loc[trb_nan_idx, 'after'] = ' ' 

#train_df['before'] = train_df['before'].fillna(' ')
#train_df['after'] = train_df['after'].fillna(' ')
test_df['before'] = test_df['before'].fillna(' ')

display(train_df[pd.isnull(train_df['before'])])


# (2)  **FIGURE OUT WHAT'S DIFFERENT**

# In[ ]:


#np.array(train_df['before'])
arr_after = np.array(train_df['after'])

idx_not_same = list()
for each_after, each_beforeiter in zip(tqdm(arr_after), train_df['before'].iteritems()):
    if each_after != each_beforeiter[1]:
        idx_not_same.append(each_beforeiter[0])

print(str(len(idx_not_same)) + ' NOT SAME out of ' + str(len(train_df)) +' total (' + str(len(idx_not_same) / len(train_df)) + ' %)')


# In[ ]:


#thing = 
counter_notsame_class = collections.Counter(np.array(train_df.loc[idx_not_same, 'class']))
display(pd.DataFrame([(i, str(counter_notsame_class[i] / len(idx_not_same) * 100.0)[:5] + ' %') for i, count in counter_notsame_class.most_common()], columns=['classy', '%_of_notsame']))
#tkeys, tvalues = zip(*thing.items())

display(train_df.loc[idx_not_same[:10]])


# (3)  **LOOK AT NOT SAME SHIT**

# In[ ]:


for each_subdf in train_df.loc[idx_not_same].groupby(by=['class']):
    display(each_subdf[1][:10])


# (4)  **FIGURE OUT WHICH SHIT IS THE SAME**

# In[ ]:


idx_are_same = set(train_df.index) - set(idx_not_same)
print(str(len(idx_are_same)) + ' ARE SAME out of ' + str(len(train_df)) +' total (' + str(len(idx_are_same) / len(train_df)) + ' %)')


# (5)  **BUILD SOME DICKS**

# In[ ]:


# find rows where before is the same but after is different
# find rows where after is the same but before is different
# and class is different?
#build a before dick
train_df_idx = np.array(train_df.index)
arr_before = np.array(train_df['before'])
before_DICK = dict()
for each_idx, each_before in zip(tqdm(train_df_idx), arr_before):
    if each_before in before_DICK:
        before_DICK[each_before].append(each_idx)
    else:
        before_DICK[each_before] = [each_idx]
            


# In[ ]:


# build a after dick
after_DICK = dict()
for each_idx, each_after in zip(tqdm(train_df_idx), arr_after):
    if each_after in after_DICK:
        after_DICK[each_after].append(each_idx)
    else:
        after_DICK[each_after] = [each_idx]


# In[ ]:


# build a classy dick
arr_class = np.array(train_df['class'])
classy_DICK = dict()
for each_idx, each_class in zip(tqdm(train_df_idx), arr_class):
    if each_class in classy_DICK:
        classy_DICK[each_class].append(each_idx)
    else:
        classy_DICK[each_class] = [each_idx]


# **DICKS ERECTION**

# In[ ]:


train_class = train_df['class'].tolist()
tr_sentID = train_df['sentence_id'].tolist()
t_sentID = test_df['sentence_id'].tolist()

before_DICK = dict()
after_DICK = dict()
classy_DICK = dict()
tr_sentID_DICK = dict()
#t_sentID_DICK = dict()

tr_ba_diff = list()

for each_idx, each_before, each_after, each_class, each_tr_sID in zip(tqdm(train_idx), train_before, train_after, train_class, tr_sentID):
    if each_before != each_after:
        tr_ba_diff.append(each_idx)
        
    if each_before in before_DICK:
        before_DICK[each_before].append(each_idx)
    else:
        before_DICK[each_before] = [each_idx]
            
    if each_after in after_DICK:
        after_DICK[each_after].append(each_idx)
    else:
        after_DICK[each_after] = [each_idx]

    if each_class in classy_DICK:
        classy_DICK[each_class].append(each_idx)
    else:
        classy_DICK[each_class] = [each_idx] 
    
    if each_tr_sID in tr_sentID_DICK:
        tr_sentID_DICK[each_tr_sID].append(each_idx)
    else:
        tr_sentID_DICK[each_tr_sID] = [each_idx]
print('i done')        


# (6)  **FUCKING MISSING SHIT**

# In[ ]:


print(len(before_DICK))
print(len(after_DICK))
print(len(classy_DICK))

# fucking missing shit
after_dick_nan_idxs = after_DICK.pop(np.nan, None)
before_dick_nan_idxs = before_DICK.pop(np.nan, None)


# (7)  **LOOK AT THE FUCKING MISSING SHIT... LOOK AT IT!!!**

# In[ ]:


for key,val in classy_DICK.items():
    #classy_letters = classy_DICK.get('LETTERS')
    display(train_df.loc[set(val).intersection(before_dick_nan_idxs)])


# In[ ]:


for key,val in classy_DICK.items():
    #classy_letters = classy_DICK.get('LETTERS')
    display(train_df.loc[set(val).intersection(after_dick_nan_idxs)])


# (8)  **BUILD SOME MORE DICKS**

# In[ ]:


# build index to after dick
idx_after_DICK = dict(zip(np.array(train_df.index), np.array(train_df['after'])))


# (9)  **FIGURE OUT WHERE NOT SAME SHIT HAS DIFFERENT SHIT**

# In[ ]:


# find rows where before is the same but after is different
# find rows where after is the same but before is different
# and class is different?
before_with_different_after = list()
most_common_after_of_before = list()
with tqdm(total=len(before_DICK)) as pbar:
    for key, value in before_DICK.items():
        #unique_afters_of_before = train_df.loc[value, 'after'].unique() #slow as shit
        get_that_shit = [idx_after_DICK.get(x) for x in value]
        cafters, ccounts = zip(*collections.Counter(get_that_shit).most_common())
        most_common_after_of_before.append((key, cafters[0])) #before, after
        unique_afters_of_before = set(get_that_shit)
        pbar.update()
        if len(unique_afters_of_before) != 1:
            before_with_different_after.append((key, value, unique_afters_of_before))

print(len(before_with_different_after))


# In[ ]:


# build me a quick DICK

print(len(most_common_after_of_before))


# (10)  **LOOK AT THAT SHIT**

# In[ ]:


pd.DataFrame(before_with_different_after[:10])


# In[ ]:


the_before_diff_after, some_idxs, _ = zip(*before_with_different_after)
list_after_idxs_not_matching = list()
for each_before, each_listidxs in zip(tqdm(the_before_diff_after), some_idxs):
    after_idxs_not_matching = list()
    seen_afters = set()
    for each_after in each_listidxs:
        some_after = idx_after_DICK.get(each_after)
        if (each_before != some_after) & (some_after not in seen_afters):
            seen_afters.add(some_after)
            after_idxs_not_matching.append(each_after)
    list_after_idxs_not_matching.append(after_idxs_not_matching)


# (11)  **LOOK AT THAT SHIT AGAIN BUT DIFFERENT**

# In[ ]:


print(sum([len(x) for x in list_after_idxs_not_matching]))
keep_min = [min(x) for x in list_after_idxs_not_matching]
#display(train_df.loc[keep_min[:10]])

for key,val in classy_DICK.items():
    #classy_letters = classy_DICK.get('LETTERS')
    display(train_df.loc[set(val).intersection(keep_min)].sort_values(by=['before','after']))


# (12)  **FIND WHERE THE AFTER PARTY HAS CAPITAL LETTERS AND FIGURE OUT WTF**

# In[ ]:


mixedcase_after = list()
allcaps_after = list()
with tqdm(total=len(after_DICK)) as pbar:
    for key, value in after_DICK.items():
        if key.isupper():
            allcaps_after.append((key, value))
        elif key.lower() != key:
            mixedcase_after.append((key, value))
        pbar.update()
print(len(allcaps_after))
print(len(mixedcase_after))        


# In[ ]:


test_df = pd.read_csv(filepath_or_buffer="../input/en_test.csv", encoding="utf-8", dtype={'class':'category'})

UNI_RANGES = [('CJK Ideographs Extension A', 13312, 19893, '3400', '4DB5'),
 ('CJK Ideographs', 19968, 40869, '4E00', '9FA5'),
 ('Hangul Syllables', 44032, 55203, 'AC00', 'D7A3'),
 ('Non-Private Use High Surrogates', 55296, 56191, 'D800', 'DB7F'),
 ('Private Use High Surrogates', 56192, 56319, 'DB80', 'DBFF'),
 ('Low Surrogates', 56320, 57343, 'DC00', 'DFFF'),
 ('The Private Use Area', 57344, 63743, 'E000', 'F8FF')]
NAMED_RANGE = [(13312, 19893), (19968, 40869), (44032, 55203), (55296, 56191), (56192, 56319), (56320, 57343), (57344, 63743)]


test_before = test_df['before'].tolist()
train_after = train_df['after'].tolist()
test_idx = test_df.index.tolist()
train_before = train_df['before'].tolist()
train_idx = train_df.index.tolist()


test_before_DICK = dict(zip(test_idx, test_before))
train_idx_before_DICK = dict(zip(train_idx, train_before))
train_idx_after_DICK = dict(zip(train_idx, train_after))


test_before_str_only = [(x,y) for x,y in zip(test_idx, test_before) if type(y) == type(str())]
test_before_str_only_idxs = [x[0] for x in test_before_str_only]

train_before_str_only = [(x,y) for x,y in zip(train_idx, train_before) if type(y) == type(str())]
train_before_str_only_idxs = [x[0] for x in train_before_str_only]

print(len(train_before_str_only))
print(len(test_before_str_only))


train_crazy = ''.join([x[1] for x in tqdm(train_before_str_only)])
test_crazy = ''.join([x[1] for x in tqdm(test_before_str_only)])
train_crazyset = set(train_crazy)
test_crazyset = set(test_crazy)
print(str(len(train_crazyset)))
print(str(len(test_crazyset)))


train_not_in_test = sorted(list(train_crazyset - test_crazyset))
crazy_ords = [ord(x) for x in train_not_in_test]

weird_ords = [y for y in crazy_ords if not any([True if (y <= x[1]) and (y >= x[0]) else False for x in NAMED_RANGE])]

thingy = pd.DataFrame(weird_ords)
thingy.plot()
#display(pd.DataFrame(np.array(train_not_in_test[:1000]).reshape((50, 20))))


len(weird_ords)
display(pd.DataFrame(np.array([chr(x) for x in weird_ords + [34]*3]).reshape((20, 20))))


t_alpha = [x for x in tqdm(test_before_str_only) if x[1].isalpha()]
t_alnum = [x for x in tqdm(test_before_str_only) if x[1].isalnum() and not x[1].isalpha()]
t_upper = [x for x in tqdm(test_before_str_only) if x[1].isupper()]
t_lower = [x for x in tqdm(test_before_str_only) if x[1].islower()]
t_numer = [x for x in tqdm(test_before_str_only) if x[1].isnumeric()]
t_space_all = [x for x in tqdm(test_before_str_only) if x[1].isspace()]
t_space_any = [x for x in tqdm(test_before_str_only) if any([y.isspace() for y in x[1]])]
t_punct_all = [x for x in tqdm(test_before_str_only) if all([y in string.punctuation for y in x[1]])]
#t_punct_any = [x for x in tqdm(test_before_str_only) if any([y in string.punctuation for y in x[1]])]
t_punct_any = [x for x in tqdm(test_before_str_only) if (any([y in string.punctuation for y in x[1]]) and len(x[1]) != 1)]


tr_alpha = [x for x in tqdm(train_before_str_only) if x[1].isalpha()]
tr_alnum = [x for x in tqdm(train_before_str_only) if x[1].isalnum() and not x[1].isalpha()]
tr_upper = [x for x in tqdm(train_before_str_only) if x[1].isupper()]
tr_lower = [x for x in tqdm(train_before_str_only) if x[1].islower()]
tr_numer = [x for x in tqdm(train_before_str_only) if x[1].isnumeric()]
tr_space_all = [x for x in tqdm(train_before_str_only) if x[1].isspace()]
tr_space_any = [x for x in tqdm(train_before_str_only) if any([y.isspace() for y in x[1]])]
tr_punct_all = [x for x in tqdm(train_before_str_only) if all([y in string.punctuation for y in x[1]])]
tr_punct_any = [x for x in tqdm(train_before_str_only) if (any([y in string.punctuation for y in x[1]]) and len(x[1]) != 1)]

# FOR TEST
thing = [len(x) for x in [t_alpha, t_alnum, t_upper, t_lower, t_numer, t_space_all, t_space_any, t_punct_all, t_punct_any]]
thing2 = list(zip(['t_alpha', 't_alnum', 't_upper', 't_lower', 't_numer', 't_space_all', 't_space_any', 't_punct_all', 't_punct_any'], thing))
display(pd.DataFrame(thing2, columns=['type', 'howmany']))


# FOR TRAIN
thing = [len(x) for x in [tr_alpha, tr_alnum, tr_upper, tr_lower, tr_numer, tr_space_all, tr_space_any, tr_punct_all, tr_punct_any]]
thing2 = list(zip(['tr_alpha', 'tr_alnum', 'tr_upper', 'tr_lower', 'tr_numer', 'tr_space_all', 'tr_space_any', 'tr_punct_all', 'tr_punct_any'], thing))
display(pd.DataFrame(thing2, columns=['type', 'howmany']))


thing = [[y[1] for y in x[:30]] for x in [t_alpha, t_alnum, t_upper, t_lower, t_numer, t_space_all, t_space_any, t_punct_all, t_punct_any]]
thing2 = pd.DataFrame(thing).T
thing2.columns = ['t_alpha', 't_alnum', 't_upper', 't_lower', 't_numer', 't_space_all', 't_space_any', 't_punct_all', 't_punct_any']
display(thing2)
print(sum(thing))
print(len(train_before_str_only))
print(sum(thing))
print(len(test_before_str_only))
tr_punct_any = [x for x in tqdm(train_before_str_only) if (any([y in string.punctuation for y in x[1]]) and len(x[1]) != 1)]


identified_test_idxs = [y[0] for x in [t_alpha, t_alnum, t_upper, t_lower, t_numer, t_space_all, t_space_any, t_punct_all, t_punct_any] for y in x ]
unique_ID_test_idxs = set(identified_test_idxs)
print(len(unique_ID_test_idxs))

mystery_test_idxs = set(test_before_str_only_idxs) - unique_ID_test_idxs
print(len(mystery_test_idxs))

mystery_test_before = [test_before_DICK[x] for x in list(mystery_test_idxs)]
unique_mystery_test_before = list(set(mystery_test_before))
print(len(unique_mystery_test_before))
display(pd.DataFrame(unique_mystery_test_before))
#display(pd.DataFrame(np.array(unique_mystery_test_before[:240]).reshape((30, 8))))

# FOR TEST
#thing = [(x,y) for x,y in t_alpha if not all([z in string.ascii_letters for z in y])]
thing = [y for x,y in t_alpha if not all([z in string.ascii_letters for z in y])]
print(len(thing))
thing2 = list(set(thing))
print(len(thing2))
thing3 = [x for x in thing2 if len(x) == 1]
print(len(thing3))

#display(pd.DataFrame(np.array(thing2[:300]).reshape((30, 10))))
display(pd.DataFrame(np.array(thing3[:400]).reshape((20, 20))))

thing4 = pd.DataFrame(sorted([ord(x) for x in thing3]))

thing4.hist()
thing4.plot()



#FOR TRAIN
thing = [y for x,y in tr_alpha if not all([z in string.ascii_letters for z in y])]
print(len(thing))
thing2 = list(set(thing))
print(len(thing2))
thing3 = [x for x in thing2 if len(x) == 1]
print(len(thing3))

thing_b_a = [(y, train_idx_after_DICK[x]) for x,y in tr_alpha if (not all([z in string.ascii_letters for z in y])) and (len(y) == 1)]
thing_b_a_unique = list(set(thing_b_a))
#thing_b_a = list(zip(thing, thing_after))
#display(pd.DataFrame(np.array(thing2[:300]).reshape((30, 10))))
thingcols = ['b','a'] * 10
display(pd.DataFrame(np.array(thing_b_a_unique[:200]).reshape((20, 20)), columns=thingcols))

thing4 = pd.DataFrame(sorted([ord(x) for x in thing3]))
thing4.hist()
thing4.plot()


#np.array(train_df['before'])
arr_after = np.array(train_df['after'])

idx_not_same = list()
for each_after, each_beforeiter in zip(tqdm(arr_after), train_df['before'].iteritems()):
    if each_after != each_beforeiter[1]:
        idx_not_same.append(each_beforeiter[0])
print(str(len(idx_not_same)) + ' NOT SAME out of ' + str(len(train_df)) +' total (' + str(len(idx_not_same) / len(train_df)) + ' %)')

idx_are_same = set(train_df.index) - set(idx_not_same)
print(str(len(idx_are_same)) + ' ARE SAME out of ' + str(len(train_df)) +' total (' + str(len(idx_are_same) / len(train_df)) + ' %)')


# In[ ]:


# what is the most common "type" of before giving multiple afters? 
#after_wards, after_wards_n = zip(*[(y, len(x.split(' '))) for x in tqdm(train_after) for y in x.split(' ')])
#after_wards = [y for i in tqdm(range(len(train_after))) for y in train_after[i].split(' ')]
after_wards = set()
after_wards_n = list()
for i in tqdm(range(len(train_after))):
    spilt = train_after[i].split(' ')
    after_wards_n.append(len(spilt))
    after_wards.update(spilt)
    #for x in spilt:
        #after_wards.append(x)
after_wards = list(after_wards)
print(len(after_wards))
print(after_wards[:10])
#unique_after_wards = list(set(after_wards))
#print(len(unique_after_wards))


# In[ ]:


print(np.argmax(after_wards_n))
print(max(after_wards_n))


# In[ ]:


superlong = np.argmax(after_wards_n)
display(train_df.loc[superlong])
print(max(after_wards_n))
print(len(train_df['before'].loc[superlong]))
print(len(train_df['after'].loc[superlong]))
print(train_idx_before_DICK[superlong])
print(train_idx_after_DICK[superlong])


# In[ ]:


#display(train_df.loc[list(range(superlong-10, superlong+10))])
display(train_df.loc[tr_sentID_DICK[670765]])

' '.join([train_idx_before_DICK[x] for x in tr_sentID_DICK[670765]])


# (13)  **THROW AN ERROR**

# In[ ]:


THROW AN ERROR

