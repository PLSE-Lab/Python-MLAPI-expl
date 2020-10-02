# The data comes both as CSV files and a SQLite database
import numpy as np
import pandas as pd
import re

#read a CSV file
df = pd.read_csv("../input/Emails.csv")
df = df[['Id','ExtractedBodyText']].dropna()

#clean the Emial texts. see https://www.kaggle.com/smarugan/d/kaggle/hillary-clinton-emails/lesson for details
def cleanEmailText(text):
    text = text.replace('\n'," ") #remove line break
    text = re.sub(r"-", " ", text) #replace hypens with space
    text = re.sub(r"\d+/\d+/\d+", "", text) #remove date
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #remove times
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #remove email addresses
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #removes web addresses
    clndoc = ''
    for letter in text:
        if letter.isalpha() or letter==' ':
            clndoc+=letter
    text = ' '.join(word for word in clndoc.split() if len(word)>1)
    return text

docs = df['ExtractedBodyText']
docs = docs.apply(lambda s: cleanEmailText(s))   
docs.head()

doclist = docs.values
stoplist = ['a' 'about' 'above' 'after' 'again' 'against' 'all' 'am' 'an' 'and' 'any'
 'are' "aren't" 'as' 'at' 'be' 'because' 'been' 'before' 'being' 'below'
 'between' 'both' 'but' 'by' 'can' "can't" 'cannot' 'could' "couldn't"
 'did' "didn't" 'do' 'does' "doesn't" 'doing' "don't" 'down' 'during'
 'each' 'few' 'for' 'from' 'further' 'had' "hadn't" 'has' "hasn't" 'have'
 "haven't" 'having' 'he' "he'd" "he'll" "he's" 'her' 'here' "here's" 'hers'
 'herself' 'him' 'himself' 'his' 'how' "how's" 'i' "i'd" "i'll" "i'm"
 "i've" 'if' 'in' 'into' 'is' "isn't" 'it' "it's" 'its' 'itself' "let's"
 'me' 'more' 'most' "mustn't" 'my' 'myself' 'no' 'nor' 'not' 'of' 'off'
 'on' 'once' 'only' 'or' 'other' 'ought' 'our' 'ours','ourselves' 'out'
 'over' 'own' 'same' "shan't" 'she' "she'd" "she'll" "she's" 'should'
 "shouldn't" 'so' 'some' 'such' 'than' 'that' "that's" 'the' 'their'
 'theirs' 'them' 'themselves' 'then' 'there' "there's" 'these' 'they'
 "they'd" "they'll" "they're" "they've" 'this' 'those' 'through' 'to' 'too'
 'under' 'until' 'up' 'very' 'was' "wasn't" 'we' "we'd" "we'll" "we're"
 "we've" 'were' "weren't" 'what' "what's" 'when' "when's" 'where' "where's"
 'which' 'while' 'who' 'will' "who's" 'whom' 'why' "why's" 'with' "won't"
 'would' "wouldn't" 'you' "you'd" "you'll" "you're" "you've" 'your' 'yours'
 'yourself' 'yourselves' 'monday' 'tuesday' 'wednesday' 'thursday' 'friday'
 'saturday' 'sunday' 'us' 'pm' 'also']


from gensim import corpora, models, similarities
import gensim

texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in doclist]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
lda.print_topics(20)