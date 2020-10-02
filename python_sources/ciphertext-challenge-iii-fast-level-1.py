#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from string import ascii_lowercase, ascii_uppercase
from flashtext import KeywordProcessor

train = pd.read_csv("../input/train.csv", index_col='index', usecols=['index', 'text'])
test = pd.read_csv('../input/test.csv', index_col='ciphertext_id')
sub = pd.read_csv('../input/sample_submission.csv', index_col='ciphertext_id')

def decode_level_1(text, key):
    key = [ord(x) - 97 for x in key]
    i = 0
    def substitute(char):
        nonlocal i
        if char in ascii_lowercase and char != 'z':
            char = chr((ord(char) - 97 - key[i]) % 25 + 97)
            i = (i + 1) % len(key)
        if char in ascii_uppercase and char != 'Z':
            char = chr((ord(char) - 65 - key[i]) % 25 + 65)
            i = (i + 1) % len(key)
        return char
    return ''.join([substitute(x) for x in text])

test1 = test[test["difficulty"] == 1].reset_index()
test1["text"] = test1["ciphertext"].map(lambda x: decode_level_1(x, 'pyle'))
print(test1["text"][0])


# In[ ]:


keyword_processor = KeywordProcessor(case_sensitive=True)
keyword_processor.set_non_word_boundaries(set())

for index, text in tqdm(train.itertuples()):
    if len(text) < 3:
        continue
    keyword_processor.add_keyword(text, index)
print(len(keyword_processor))


# In[ ]:


def good_match(match, text):
    d = (len(text) - len(match)) // 2
    return match == text[d:d+len(match)]

matched, unmatched = 0, 0
for row in tqdm(test1.itertuples()):
    matches0 = keyword_processor.extract_keywords(row.text)
    matches = [x for x in matches0 if good_match(train.loc[x]['text'], row.text)]
    if len(matches) == 1:
        matched += 1
        sub.loc[row.ciphertext_id] = matches[0]
    else:
        unmatched += 1
        print(row.text, matches0, [train.loc[x]['text'] for x in matches0])
print(f"Matched {matched}   Unmatched {unmatched}")

sub.to_csv('submit-level-1.csv')


# I definitely took some code (how to match decoded and plain strings) from [https://www.kaggle.com/tarobxl/cipher-challenge-iii-level-2](https://www.kaggle.com/tarobxl/cipher-challenge-iii-level-2), and changed to improve performance (Thanks to the awesome [flashtext](https://github.com/vi3k6i5/flashtext) library)
# 
# Decoding itself comes from my previos Kernel: [https://www.kaggle.com/elvenmonk/difficulty-1-reverse-engineering-no-ml](https://www.kaggle.com/elvenmonk/difficulty-1-reverse-engineering-no-ml)
# 
# I hope it can be useful to speedup cracking next Difficulty levels!
