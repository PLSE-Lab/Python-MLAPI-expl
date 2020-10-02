import nltk
import re
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter

text = "Three@ a significant versions r of SNMP#@ have been developed j and deployed. SNMPv1 is the original version of the protocol. More recent versions, SNMPv2c and SNMPv3, feature improvements in performance, flexibility and security."

#1. Text preprocessing  - remove special characters, one letter words and stop words

text = re.sub("[^A-Za-z .]", "", text)
text = re.sub(" [A-Za-z] ", " ", text)

stop_words = set(stopwords.words('english'))
document = text.split(".") #list of sentences

doc_tokenized = {}
for sentence in document:
    doc_tokenized[sentence] = word_tokenize(sentence)
# print(doc_tokenized)


ps = PorterStemmer()

#2. Word frequency table

def word_frequency_overall(doc_tokenized):

    all_words_in_doc = []
    
    for word_list in doc_tokenized.values():
        all_words_in_doc.extend(word_list)
    
    all_words_stemmed = []
    
    for word in all_words_in_doc:
        if word in stop_words:
            continue
        else:
            all_words_stemmed.append(ps.stem(word))
   
    word_frequency = dict(Counter(all_words_stemmed))
    print(word_frequency) 
            
# word_frequency_overall(doc_tokenized)

def word_frequency_per_sentence(doc_tokenized):
    
    frequency_per_sentence = {}
    
    word_processer = lambda x: ps.stem(x).lower()
    processed_docs = {}
    
    for sentence, word_list in doc_tokenized.items():
        processed_docs[sentence] = list(map(word_processer, word_list))
    
    for sentence, word_list in processed_docs.items():
        frequency_per_sentence[sentence] = dict(Counter(word_list))
    
    print(frequency_per_sentence)
    
word_frequency_per_sentence(doc_tokenized)

#3. TF_table - count in sentence/total words in sentence

# def calculate_TF_table(word_frequency_per_sentence):
    
#     tf_table = {}
    
#     for sentence, word_count_per_sentence in word_frequency_per_sentence.items():
#         frequency_per_sentence = {}

#     print(tf_table)
    
# calculate_TF_table(word_frequency_per_sentence)