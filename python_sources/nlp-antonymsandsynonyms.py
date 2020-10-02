# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 11:47:41 2018
@author: uknemani
"""
import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet

#give definition and examples
syn = wordnet.synsets('work')
print(syn[0].definition())
print(syn[0].examples())

#Print Synonyms of given word
synonyms =[]
for syn in wordnet.synsets('activity'):
  for lemma in syn.lemmas():
    synonyms.append(lemma.name())

print(synonyms)

#print Antonyms
antonyms =[]
for syn in wordnet.synsets('easy'):
  for lem in syn.lemmas():
    if lem.antonyms():
      antonyms.append(lem.antonyms()[0].name())
      
print(antonyms)