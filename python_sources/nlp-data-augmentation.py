#!/usr/bin/env python
# coding: utf-8

# # Introduction to Data Augmentation in NLP
# 
# This Notebook includes a brief Introduction to Text Augmentation with Nouns and Verbs. This would allow you to boost up your accuracy while performing NLP tasks.
# 
# ### Function and Features.
# - Choose Percentage value to be augmented from selected Noun & Verbs.
# - Unique words are sorted into dictionary. 
# - Words repeated only once in a text plus elimination of 2 letters words for processing list.
# - Random selection from sorted words.
# - Using NLTK, Thesaurus and Spacy Library, checking the most similar word with synonyms and replace it.
# - Input & Output directory is set for processing out all the text file.

# In[ ]:


get_ipython().system('pip install thesaurus')


# # Importing Libraries.

# In[ ]:


import os
import spacy
import random
from thesaurus import Word
import nltk 
from nltk.corpus import wordnet 
import en_core_web_sm
import re


# # Choosing Percentage level
# 
# - The total length of the noun & verb would are calculated. 
# - This value will determine: the number of words to be select at random for a list.
# - The default value is Kept at 50%.

# In[ ]:


percent = 50 #input('enter percentage for 0 to 100:')


# # Word Similarity Check
# 
# - Create a synalter_Noun_Verb(word, al, POS) which take arguments as follows
#     a) word: specific word as a string (present in input data)
#     b) al: List of all synonyms of a word.
#     c) (POS) Part of Speech: Noun or Verb.
#   
#   ## Function:
#      Comparing a word with a list of synonyms
#      ```
#             w1 = wordnet.synset(word+'.'+POS+'.01') 
#             w2 = wordnet.synset(i+'.'+POS+'.01') # n denotes noun 
#             if(max_temp<w1.wup_similarity(w2)):
#             ```
# This function uses NLTK library which allows to check the level of similarity that returns a value. For example: `word = 'rock', a1 = ['stone','earth',..] ` . Value would be like : rock --> stone = 0.67 and rock -->earth = 0.30. Selection of a new word would be done across maximum similar value. Hence, Rock would be replaced by world stone. 
# 
# I have been using only 2 Types of POS - Noun, and Verb.  POS takes in input as `'n' or 'v'` where n stands for noun and v for verbs. Noun and Verb are sorted out in the early stage of the process using Spacy. If the output of `wup_similarity` is NULL then it passes on to Spacy for comparison of words. 
# 
# ```nlp = en_core_web_sm.load()
# for i in a1:
#             j=i.replace(' ', '')
#             tokens = nlp(u''+j)
#             token_main = nlp(u''+word_str)
#   ``` 
#   Above function allows us to process with the word similarity in different way due to returning of NIL value from wordnet processing. Repeating the same process for finding out the best replacement from the context is done by finding out the max value using `token1.similarity(tokens)` which yeilds out float value.
# 
# #### Main aim of this function is to return the most similar word to the given input word.
#   

# In[ ]:


def synalter_Noun_Verb(word,al,POS):
    max_temp = -1
    flag = 0
    for i in a1:
        try:
            w1 = wordnet.synset(word+'.'+POS+'.01') 
            w2 = wordnet.synset(i+'.'+POS+'.01') # n denotes noun 
            if(max_temp<w1.wup_similarity(w2)):
                max_temp=w1.wup_similarity(w2)
                temp_name = i
                flag =1
        except:
            f = 0
            
    if flag == 0:
        max1 = -1.
        nlp = en_core_web_sm.load()
        for i in a1:
            j=i.replace(' ', '')
            tokens = nlp(u''+j)
            token_main = nlp(u''+word_str)
            for token1 in token_main:
                if max1<float(token1.similarity(tokens)):
                    max1 = token1.similarity(tokens)
                    value = i
        max1 = -1.
        return value 
    else:
        return temp_name


# # Text Replacement
# 
# Pseudo Code: 
# - Filters out text files from the Input Folder and create a list of Text Files
# - Loop each file to do the following operation
#     
#    
#     1) Check if File is corrupt or not.
#         1.1) File is Valid: 
#             1.1.1) Split out words
#             1.1.2) Count the total number of Unique words and make a dictionary
#             1.1.3) Filter out unique words and eliminate 2 letters word & numbers.
#             1.1.4) Make a list of Noun and Verb using spacy from the above-filtered list.
#             1.1.5) Random selection from Noun and Verb along with above mention percentage.
#             1.1.6) Loop all the randomly selected words
#                 1.1.6.1) create a word_str variable to store a word.
#                 1.1.6.2) Finding out all the synonyms using the thesaurus and making a list.
#                 1.1.6.3) Checking If a selected word is Noun or Verb and than passing to `synalter_Noun_Verb(word, al, POS)`
#                 1.1.6.4) Replacing the word with the new most similar word
#             1.1.7) Generating the output files at a specific path.
#         1.2) Ignore the File corrupt
#    
#    
#    The process to solve NLP Text Augmentation requires to follow the above procedure. I had created the NLP augmentation version that would help you to increase your accuracy in NLP. This is quite a slow process but when comparing with its results, its worth to give a try.

# In[ ]:


synonyms = [] 
antonyms = []   
all_files = os.listdir("../input/")
txt_files = filter(lambda x: x[-4:] == '.txt', all_files)
print(txt_files)
for i in txt_files:
    textfile = i
    print("Input File: "+ textfile)
    print(" ")
    path = '../input/'+textfile
    exists = os.path.isfile(path)
    if exists: 
        file_open = open(path,"r")
        text = file_open.read()
        output_text = text
        print("Sentence: "+text)
        words = text.split()
        counts = {}
        for word in words:
            if word not in counts:
                counts[word] = 0
            counts[word] += 1
        one_word = []
        for key, value in counts.items():
            if value == 1 and key.isalpha() and len(key)>2:
                one_word.append(key)
        noun = []
        verb = []
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(u''+' '.join(one_word))
        for token in doc:
            if  token.pos_ == 'VERB':
                verb.append(token.text)
            if  token.pos_ == 'NOUN':
                noun.append(token.text)
            
        all_main =verb + noun
        len_all = len(noun)+len(verb)
        final_value = int(len_all * percent /100)
        random.seed(4)
        temp = random.sample(range(0, len_all), final_value)
        for i in temp:
            try:
                word_str = all_main[i]
                w = Word(word_str)
                a1= list(w.synonyms())
                if i<len(verb):
                    change_word=synalter_Noun_Verb(word_str,a1,'v')
                    try:
                        search_word = re.search(r'\b('+word_str+r')\b', output_text)
                        Loc = search_word.start()
                        output_text = output_text[:int(Loc)] + change_word + output_text[int(Loc) + len(word_str):]
                    except:
                        f=0

                else:
                    change_word=synalter_Noun_Verb(word_str,a1,'n')
                    try:
                        search_word = re.search(r'\b('+word_str+r')\b', output_text)
                        Loc = search_word.start()
                        output_text = output_text[:int(Loc)] + change_word + output_text[int(Loc) + len(word_str):]
                    except:
                        f=0

            except:
                f=0
        print('')
        print('Output:')
        print(output_text)
        f = open(textfile, "a")
        f.write(str(output_text))
        print('')


# # Output
# 
# - 50 % of Nouns or Verb Augumentation. 
# 
# ![Output](https://res.cloudinary.com/dykavvkou/image/upload/v1551444587/Screen_Shot_2019-03-01_at_6.18.22_PM_h0qvaa.png)

# # NEXT STEP
# 
# - Translating into multiple language which would generate some kind of variation when compared with orginal text.
# - Checking Grammer of Input/output file
# - Use Fastai IMDB notebook and Improve the score.
