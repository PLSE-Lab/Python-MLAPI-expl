# %% [code]
import numpy as np
import pandas as pd
from flashtext.keyword import KeywordProcessor
import string

def polite_post_index(forum_posts):
    '''Pass in a list of fourm posts, get
    back the indexes of short, polite ones.'''
    
    polite_indexes = []
    
    # create  custom stop word list to identify polite forum posts
    stop_word_list = ["no problem", "thanks", "thx", "thank", "great",
                      "nice", "interesting", "awesome", "perfect", 
                      "amazing", "well done", "good job"]

    # create a KeywordProcess
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(stop_word_list)

    # test our keyword processor
    for i,post in enumerate(forum_posts):
        post = post.lower().translate(str.maketrans({a:None for a in string.punctuation}))
        
        if len(post) < 100:
            keywords_found = keyword_processor.extract_keywords(post.lower(), span_info=True)
            if keywords_found:
                polite_indexes.append(i)

    return(polite_indexes)