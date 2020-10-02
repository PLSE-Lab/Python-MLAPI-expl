# It's yours to take from here!

import pandas as pd
import string
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import nltk
from tabulate import tabulate
#import tokenize
#from nltk.stem.snowball import SnowballStemmer
#import geograpy
#from geograpy import places

df_Emails= pd.read_csv("../input/Emails.csv", header=0)



full_text=[]
for raw_email in df_Emails['RawText']:
    raw_email=raw_email.translate(str.maketrans("","")).lower()
    if raw_email !="":
        full_text.append(raw_email)
    
full_text=nltk.tokenize.word_tokenize(str(full_text))



fdist = FreqDist()
for word in  full_text:
    if len(word)>2 and word.isalpha():
        fdist[word]+=1

for sw in stopwords.words("english"):
     if sw in fdist:
          fdist.pop(sw)
#fdist.items().sort()

print ("AFTER removing meaningless words")
fdist=fdist.most_common(50)

print (fdist)