#!/usr/bin/env python
# coding: utf-8

# [GitHub repo][1]
# 
# A lot of these cells will not work in the Kaggle environment. You need to download multiple files and I am unaware of a way of doing so here. Download the ipynb file, move the first two cells to their own python files and remove them from your notebook.
# 
# Using the instructions from [this page][2]. You can train a model ([my model][3]) for vectorizing words. This file is used below.
# 
# 
#   [1]: https://github.com/apjansing/Quora-Question-Pairs/
#   [2]: http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
#   [3]: https://drive.google.com/open?id=0B5yf6IhYey8cWU1yNlNjeHY5d2s

# In[ ]:


# a function to test the similarity of the words
# Some of the values are hard-coded, but this is a first draft of my code and I will make something better
## in the near future.
def testQuestions(first, second):
    count = 0.0
    for f in first:
        for s in second:
            try:
                sim = M.similarity(f, s)
                if sim > .45:
                    count += 1.0
                    #print "Similarity between", f, "and", s, "is", M.similarity(f, s)    
            except:
                pass
    try:
        if count/len(first) > .8 or count/len(second) > .8:
            return 1
        return 0
    except:
        return 0


# Here is the code for processing your Wiki data.
# 
# Run it with 
# **python process_wiki.py enwiki-latest-pages-articles.xml.bz2 wiki.en.text**

# In[ ]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# process_wiki.py
import logging
import os.path
import sys
 
from gensim.corpora import WikiCorpus
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")


# Here is the code to creating your model from the processed Wiki data.
# 
# Run it with **python train_word2vec_model.py wiki.en.text wiki.en.word2vec.model**

# In[ ]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_word2vec_model.py
import logging
import os.path
import sys
import multiprocessing
 
from gensim.corpora import  WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
 
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments

    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)

    model.save(outp)


# Now that we've trained a model on a large corpus of text, we need to think of how we want to interrogate the information provided. My first thought was that stopwords do no provide a lot of information. Not to say that stopwords are useless, but removing them may speed things up.

# In[ ]:


from nltk.corpus import stopwords
import nltk
import re
import pandas as pd

## uncomment if you need to download the nltk gives you problems about not having the "english" stopwords corpus
#nltk.download("stopwords")


# In[ ]:


def formatPart(part):
	part = removeSymbols(part)
	part = removeStopWords(part.split(" "))	
	return part.strip()

def removeSymbols(line):
	return re.sub('[^\w]', ' ', line)


def removeStopWords(words):
	line = ""
	for word in words:
		if word not in stopwords.words('english'):
			line = line + " " + word
	return line

def processPart(part, j):
	if j is 3 or j is 4:
		return formatPart(part)
	else:
		return part

def getHeader(noStop):
	with open("train.csv", "r") as F:
		for f in F:
			noStop.write(f)
			break


# In[ ]:


trainingSet = pd.read_csv("train.csv", quotechar='"').as_matrix()
rows = len(trainingSet[:,0])
#L = []
with open("trainNoStopWords.csv", "w") as noStop:
	getHeader(noStop)
	for i in range(rows):
		l = []
		for j in range(len(trainingSet[i,:])):
			l.append(processPart(str(trainingSet[i,j]), j))
		noStop.write(",".join(l) + "\n")


# In[ ]:


# other imports you may need (change your path to the location of your model)
import gensim as gm
import numpy as np
import scipy as sp

M = gm.models.Word2Vec.load("/home/alex/Documents/Wiki dump/wiki.en.word2vec.model")


# Now, using the version of the training data with the stopwords removed, we can test for some similarity between the target classifications and the classifications proved by the cell below.

# In[ ]:


with open("trainNoStopWords.csv", "r") as F:
    correct = 0.0
    total = 0.0
    header = True
    for f in F:
        if header:
            header = False
        else:
            total += 1.0
            parts = f.split(",")
            if testQuestions(parts[3], parts[4]) == int(parts[5]):
                correct += 1.0
                if int(correct) % 10000 is 0:
                    print(correct/total)
    print(correct/total)


# Some sample output of the above cell: 
# 
#     0.626095667418
#     0.628515760033
#     0.626448662532
#     0.626910116762
#     0.627454917364
#     0.627569111048
#     0.628151976884
#     0.627657738235
#     0.627466291116
#     0.627801564481
#     0.627656842886
#     0.628249226468
#     0.627655465431
#     0.627920953722
#     0.627956629129
#     0.62809631857
#     0.628361270768
# 
# We can see that this model isn't all that great, but it is a something to work from.
