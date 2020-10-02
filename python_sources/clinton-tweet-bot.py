#Assumes the data sets are in "./data/"
#Twitter api keys have been omitted

from collections import defaultdict
from itertools import chain
from twython import Twython
import random
import re
import sys
import sqlite3
import pandas as pd

#Put you twitter api keys here:
app_key = ""
app_secret = ""
oauth_token = ""
oauth_token_secret = ""

twitter = Twython(app_key, app_secret, oauth_token, oauth_token_secret)

def read_sql_data(fileName):
    con = sqlite3.connect('./data/database.sqlite')
    e = pd.read_sql_query("SELECT p.Name Sender, ExtractedBodyText FROM Emails e INNER JOIN Persons p ON e.SenderPersonId=P.Id WHERE p.Name='Hillary Clinton' AND e.ExtractedBodyText != '' ORDER BY RANDOM()"  ,con)
    cs = ""
    for i in range(len(e.ExtractedBodyText)):
        cs += e.ExtractedBodyText[i].encode('utf-8')
    
    file = open(fileName, "w")
    file.write(cs)

class MarkovChain(object):
    def __init__(self, documents, **kwargs):
        self.word_cache = defaultdict(list)
        self.words = self.documents_to_words(documents)
        self.word_size = len(self.words)
        self.wordbase = self.wordbase()
    
    def documents_to_words(self, documents):
        """Returns a list of words used in a given list of documents."""
        words = []
        for document in documents:
            if document:
                words.append(self.tokenize(document))
        return list(chain.from_iterable(words))
    
    def tokenize(self, document):
        # don't want empty spaces
        words = [w.strip() for w in document.split() if w.strip() != '']
        return words

    def yield_trigrams(self):
        if len(self.words) < 3:
            return

        for i in range(len(self.words) - 3):
            yield (self.words[i], self.words[i+1], self.words[i+2])

    def wordbase(self):
        for w1, w2, w3 in self.yield_trigrams():
            self.word_cache[(w1, w2)].append(w3)

    def generate_tweet(self, min_chars=100, max_chars=140):
        seed = random.randint(0, len(self.words) - 3)
        w1, w2 = self.words[seed], self.words[seed + 1]
        tweet = '  '

        # loop until it's a sentence
        while tweet[-2] not in '.!?':
            tweet += (w1 + ' ')
            w1, w2 = w2, random.choice(self.word_cache[(w1, w2)])

        # if it's too short or too long, try again
        if len(tweet) < min_chars or len(tweet) > max_chars:
            tweet = self.generate_tweet()
        return tweet.strip()

def main():
    fileName = sys.argv[1]
    read_sql_data(fileName)
    with open(fileName) as f:
        text = [line for line in f]
    tweet = MarkovChain(text).generate_tweet()
    twitter.update_status(status=tweet)

if __name__ == '__main__':
    main()
