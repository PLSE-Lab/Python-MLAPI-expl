from gensim.models import word2vec
from nltk.corpus import brown
import logging
import os

class MeditationsSentences(object):
    def __init__(self, fname):
        self.fname = fname
 
    def __iter__(self):
        for line in open(self.fname):
            yield line.split()
          
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = MeditationsSentences("../input/pg2680.txt")
model = word2vec.Word2Vec(sentences, size=300, min_count=1)
model.init_sims(replace=True)
model.save("word2vec_gensim.bin")
model = word2vec.Word2Vec.load("word2vec_gensim.bin")
print(dir(model))