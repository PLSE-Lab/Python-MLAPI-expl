import os

os.system("pip install git+https://github.com/LIAAD/yake")

import pandas as pd
from nltk.tokenize import RegexpTokenizer
import yake

def keywords_yake(sample_posts):
    # take keywords for each post & turn them into a text string "sentence"
    simple_kwextractor = yake.KeywordExtractor()

    # create empty list to save our "sentnecs" to
    sentences = []

    for post in sample_posts:
        post_keywords = simple_kwextractor.extract_keywords(post)

        sentence_output = ""
        for word, number in post_keywords:
            sentence_output += word + " "

        sentences.append(sentence_output)
        
    return(sentences)

def tokenizing_after_YAKE(sentences):
    tokenizer = RegexpTokenizer(r'\w+')
    sample_data_tokenized = [w.lower() for w in sentences]
    sample_data_tokenized = [tokenizer.tokenize(i) for i in sample_data_tokenized]
    
    return(sample_data_tokenized)