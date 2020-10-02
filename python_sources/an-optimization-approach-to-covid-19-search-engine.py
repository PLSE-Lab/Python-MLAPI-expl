#!/usr/bin/env python
# coding: utf-8

# # An optimization approach to the design and analysis of COVID-19 literature search engines

# # Summary of the methodology

# Contributions
# 
# The main contributions of this work are summarized as follows. 
# 
# 1. Framework : A quantitative framework to evaluate the quality of all Covid-19 literature search engines
# 
# 2. Algorithm : An algorithm to mine relevant documents and texts given any query which is designed according to the framework in (1)
# 
# 3. Retrieval : Evaluation using the algorithm in (2)
# 

# Framework
# 
# Given any text, let $f$ be an encoding function such that $f(text)$ is a $R^n$ compact representation of the text. This is also known as feature vector.  
# 
# 
# Given any two feature vectors $f_1, f_2$ for two sets of text, we define a similarity function $s(f_1, f_2) \in R^{[0,1]}$ such that it represents how similar two sets of text are. The more similar two sets of text are, the closer $s(f_1, f_2)$ is to 1.
# 
# Given a set of ground truth data ($T_1$, $T_2$, $p$) where $T_1$ and $T_2$ are text and $p\in R^{[0,1]} $ represents how relevent $T_1$ and $T_2$ are. $p$ is 0 if $T_1$ and $T_2$ are not totally not related and $1$ if they are the same.
# 
# Given $N$ ground truth data ($T^{(k)}_1$, $T^{(k)}_2$, $p^{(k)}$) $_{k = 1 : N}$, our quantitative framework seeks to find feature vector function $f$ and similarity function $s$ such that it minimize the following loss, which is also known as the cross-entropy loss. 
# 
# $$ \min _{s,f} - \frac{1}{N} [\sum_{k=1}^{N} p^{(k)} \log s(f(T^{(k)}_1), T^{(k)}_2))  +  (1- p^{(k)}) \log (1- s(f(T^{(k)}_1), T^{(k)}_2))) ]  $$
# 
# It is noted that under this framework, one may take $f_1$ using the topic model vector from Latent Dirichlet Allocation and take $s_1$ as consine similarity function. That represents one valid solution under this framework.
# 
# Alternatively, one may take $f_2$ as the bag of words vector and use $s_2$ as a weighted sum of indicator function on individual important words. That represents another valid solution under this framework. 
# 
# One may also concatenate $f_1$ and $f_2$ as $f_3 = (f_1, f_2) $ and take average of $s_1$ and $s_2$ to form $s_3(f_3(T_1), f_3(T_2)) = 0.5*(s_1(f_1(T_1), f_1(T_2)) + s_2(f_2(T_1), f_2(T_2)))$. That also represent a valid solution under this framework. 
# 
# The nice thing of this optimization approach is that we can compare all these methods using **one** number, the value of the loss function. 
# 

# Algorithm
# 
# To concretely construct the algorithm, there are 3 different challenges that need to be addressed.
# 
# 1. How do we obtain the ground truth data ? 
# 
# 2. How do we parametrize feature vector $f$ ?
# 
# 3. How do we parametrize similarity function $s$ ? 
# 
# For (1), we randomly sample words from the texts to construct pseudo documents. If the text are originally from the same document, then we consider them to be highly similar and assign a value of 1. If the psedo documents originate from different sources, we assign similarity as 0. We remark that this step can be improved when we have human labelled data.
# 
# For (2), we use a concatenation of bag of words vector, TFIDF vector, LDA vector and LSI vector. We first do some prerpocessing like filtering common words, filtering words of low frequency, and perform stemming. Then we construct LDA, LSI, TFIDF and BOW models from the data. Finally, we apply the model on each text and concatentate the output to form the final feature vector.
# 
# For (3), we use a neural network. We have also tried using random forest. But it turns out that neural network gives better performance than random forest. We also remark that one may do some feature engineering before feeding them feature vector  into the random forest. But we take a simpler approach of letting the neural network figure out the best way to do similarity calcualation.
# 
# The algorithm is summarized as follows. 
# 
# 
# a. Find feature vector generation function $f$ by concatenating the bag of words vector, TFIDF vector, LDA vector and LSI vector. 
# 
# b. Use sampling to generate ground truth data
# 
# c. Find similarity function $s$ by training a neural network.
#  
# 
# 
# 

# Retrieval
# 
# Once we obtain the functions $f$ and $s$, there are multiple ways for us to perform retrieval, depending on the size of the documents available or whether we have user input.
# 
# A simple approach is to do retrieval is simply compute the similarity of the query $Q$ and the documents and find the one that is most relevant as follows. That is the approach that we implemented.
# 
# $$\max_{T} s(f(Q), f(T))$$
# 
# 
# However, we also would like to mention a few other methods that use the same $f$ and $s$ but will be more appropriate depending on how the application is constructed.
# 
# If we have huge amount of documents, we need a fast way for retrieval.
# 
# To speed up computation,  we can build a graph and perform graph search in real time on that pre-constructed graph. We may want to use keyword matches to quickly find the initial nodes that are relevant.
# 
# If we have input from the user who is manually looking through the documents, we would like to use information from his choices in real time to improve accuracy. 
# 
# For example, when the user only clicks on documents set $D_1$ but not document set $D_2$ in any given time, we can use that information to improve accuracy by doing nearest neighbor search.
# 
# $$ \max_{T} \sum_{x\in D_1} s(f(x), f(T))   -  \lambda \sum_{x\in D_2} s(f(x), f(T)) $$ with $\lambda$ as a regularization constant parameter.
# 
# Our evaluation results will be presented in the later section of this notebook. 

# Pros/Cons of the current approach:
# 
# Pros : This approach gives a quantitative framework for us to compare different search engines. The framework is broad enough that it covers multiple standard approaches. Also, with single parameter to optimize, the approach is compatible with other advanced approaches, like hyperparameter search using reinforcment learning. 
# 
# Cons : The machine learning model need significant amount of data to perform well. If there is lack of data, it might be better to use heuristic based method. Also, the sampling based method to obtain ground truth can be noisy. Ideally, we would like to have some human labelled data to supplement that. We also remark that the current version does not include many features(e.g. same features but for bi-grams, same feature set but for the citation references, etc) we would like to have due to limitation of time.

# # Implementation of a Covid-19 literature search engine prototype

# We use IS_FAST mode to generate the output. For more detailed results using more data, please use **IS_FAST = False**

# In[ ]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')

IS_FAST = True


# In[ ]:


# Data cleaning
import json
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora
from gensim.models.phrases import Phrases, Phraser
import glob

class dataCleaningRobot():
	def __init__(self, path, max_count_arg = 10000000):
		self.max_count = max_count_arg
		self.paths_to_files =  glob.glob(path + "/*.json")
		self.path = path

	def getText(self):
		self.getDocDicFromPath()
		self.filterTokens()

	def getDocDicFromPath(self):
		self.dicOfTexts = {}
		mycount = 0
		for filename in self.paths_to_files:
			with open(filename) as f:
				data = json.load(f)

			wordList = []
			paperID = data['paper_id']

			for eachTextBlk in data['abstract'] + data['body_text']:
				wordList += word_tokenize(eachTextBlk['text'])

			self.dicOfTexts[paperID] = wordList
			mycount +=1 
			if mycount > self.max_count:
				break

	def filterTokens(self):
		self.dicOfFilterTexts = {}

		### Token-based filtering   
		newStopWords = set(['preprint', 'copyright', 'doi', 'http', 'licens', 'biorxiv', 'medrxiv'])
		stopWords = set( stopwords.words('english'))

		porter = PorterStemmer()
		self.wordCtFreq = defaultdict(int)

		for eachText in self.dicOfTexts:
			filtered = []
			for word in self.dicOfTexts[eachText] :
				if word.isalpha() and len(word) > 2 :
					token = word.lower()
					if token in stopWords:
						continue
					token = porter.stem(token)
					if token in newStopWords:
						continue

					filtered.append(token)
					self.wordCtFreq[token] += 1

			self.dicOfFilterTexts[eachText] = filtered
		
		### Count-based filtering
		for eachText in self.dicOfFilterTexts:
			filtered = []

			for word in self.dicOfFilterTexts[eachText] :
				if self.wordCtFreq[word] > 10 :
					filtered.append(word)

			self.dicOfFilterTexts[eachText] = filtered

	def getDicCorpus(self):
		texts = [ self.dicOfFilterTexts[eachitem] for eachitem in self.dicOfFilterTexts ] 
		self.dictionary = corpora.Dictionary(texts)
		self.corpus = [self.dictionary.doc2bow(text) for text in texts]

	def getSingleWordCount(self):
		return self.getCountInfo(self.wordCtFreq)

	def getBigramCount(self):
		return self.getCountInfo(self.bigramCtFreq)

	def getCountInfo(self, ctFreqDic):

		word_freq_list = []
		for each_token in ctFreqDic: 
			word_freq_list.append([ctFreqDic[each_token], each_token])
		word_freq_list.sort(reverse= True)

		# print(word_freq_list[0:5])
		wordList = []
		countList = []

		for eachitem in word_freq_list:
			wordList.append(eachitem[1])
			countList.append(eachitem[0])

		return wordList, countList, word_freq_list


	def getBigramData(self):
		texts = [ self.dicOfFilterTexts[eachitem] for eachitem in self.dicOfFilterTexts ] 
		self.bigram = Phrases(texts)
		self.bigram_model = Phraser(self.bigram)
		bigram_texts = [self.bigram_model[self.dicOfFilterTexts[eachitem]] for eachitem in self.dicOfFilterTexts]

		self.bigramCtFreq = defaultdict(int)
		for eachText in bigram_texts:
			for word in eachText :
				tmp_array = word.split('_') 

				### Only extract two words
				if len(tmp_array) > 1 :
					self.bigramCtFreq[word] += 1


		self.bigram_dictionary = corpora.Dictionary(bigram_texts)
		self.bigram_corpus = [self.bigram_dictionary.doc2bow(text) for text in bigram_texts]



# In[ ]:


# Feature generation
from gensim import models
import numpy as np
from gensim.test.utils import datapath
from nltk.stem.porter import PorterStemmer
from gensim.corpora import Dictionary

class featureGenRobot():
	def __init__(self):
		assert(True)

	def getModels(self):
		self.num_lsi_topics = 2
		self.num_lda_topics = 10
		self.tfidf_model = models.TfidfModel(self.corpus)
		self.corpus_tfidf = self.tfidf_model[self.corpus]
		self.lsi_model = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_lsi_topics)  
		self.lda_model = models.LdaModel(self.corpus,id2word=self.dictionary, num_topics=self.num_lda_topics, iterations=1500, passes=20, minimum_probability=0.0)

	def saveModelsDict(self):
		prefix = '/kaggle/working/'
		self.lda_model.save(datapath(prefix + "lda_debug.model"))
		self.lsi_model.save(datapath(prefix + "lsi_debug.model"))
		self.tfidf_model.save(datapath(prefix + "tfidf_model_debug.model"))
		self.dictionary.save_as_text(prefix + "dictionary_debug.txt")

	def loadModelsDict(self):
		prefix = '/kaggle/working/'

		self.num_lsi_topics = 2 
		self.num_lda_topics = 10

		self.tfidf_model = models.TfidfModel.load(prefix + "tfidf_model_debug.model")
		self.lsi_model = models.LsiModel.load(prefix + "lsi_debug.model")
		self.lda_model = models.LdaModel.load(prefix + "lda_debug.model")

		self.dictionary = Dictionary.load_from_text(prefix+ "dictionary_debug.txt")

	def getFeaVec(self, text):
		porter = PorterStemmer()
		myList = text.lower().split()
		myList2 = [ porter.stem(word.lower()) for word in myList ]
		bow_vec = self.dictionary.doc2bow(myList2)
		return self.getFeaVecFromBow(bow_vec)

	def getFeaVecFromBow(self, bow_vec):
		### Computer BOW, TFIDF , LSI, LDA values . models topic division as dense features. 
		vector_tfidf = self.tfidf_model[bow_vec]
		vector_lsi = self.lsi_model[bow_vec]
		vector_lda = self.lda_model[bow_vec]
		
		### Convert LSI, LDA values as dense vectors. 
		lsi_topic = self.num_lsi_topics 
		lda_topic = self.num_lda_topics

		N = lsi_topic + lda_topic
		denseVector = np.zeros(N)
		
		base = 0 
		for i in range(len(vector_lsi)):
			idx = vector_lsi[i][0]
			denseVector[base + idx] = vector_lsi[i][1]

		base = len(vector_lsi)
		for i in range(len(vector_lda)):
			idx = vector_lda[i][0]
			denseVector[base + idx] = vector_lda[i][1]

		### Convert BOW, IFIDF as sparse vectors.
		m1 = len(self.dictionary)
		m2 = len(self.dictionary)
		sparseVec = np.zeros(m1 + m2)
		base = 0 
		for eachitem in bow_vec:
			idx = eachitem[0]
			val = eachitem[1]
			sparseVec[base + idx] = val

		base = m1
		for eachitem in vector_tfidf:
			idx = eachitem[0]
			val = eachitem[1]
			sparseVec[base + idx] = val

		### Convert dense and sparse feature vectors.
		combined = np.concatenate((sparseVec, denseVector))
		#combined = denseVector
		return denseVector, sparseVec, combined

	def genFeaVecMap(self, dictionary_arg, corpus_arg):
		self.dictionary = dictionary_arg
		self.corpus = corpus_arg
		self.getModels()


# In[ ]:


# Example generation
import random
import numpy as np 


class exampleGenRobot():
	def __init__(self, dicOfFilterTexts, dictionary, featureRobot):
		assert(True)
		self.dicOfFilterTexts = dicOfFilterTexts
		self.dictionary = dictionary
		self.featureRobot= featureRobot

	def getIJPair(self):	
		self.ijPairs = []
		N = len(self.dicOfFilterTexts)
		posNum = 100
		negNum = 100
		docRatio = 0.01

		keyList = list(self.dicOfFilterTexts)

		for id_text in self.dicOfFilterTexts:
			wholeText = self.dicOfFilterTexts[id_text]
			sentArr = [ [] for i in range(posNum)  ]
			### Pos eg
			for word in wholeText:
				for i in range(posNum):
					if random.random() < docRatio:
						sentArr[i].append(word)
			
			for i in range(int(posNum/2)):
				bow_vec1 = self.dictionary.doc2bow(sentArr[2*i])
				bow_vec2 = self.dictionary.doc2bow(sentArr[2*i+1])
				denseVector1, sparseVec1, combined1 = self.featureRobot.getFeaVecFromBow(bow_vec1)
				denseVector2, sparseVec2, combined2 = self.featureRobot.getFeaVecFromBow(bow_vec2)
				self.ijPairs.append([combined1, combined2, 1])

			### Neg eg
			for j in range(negNum):

				k = random.choice(keyList)
				while ( k == id_text):
					k = random.choice(keyList)

				negSent = []
				for word in self.dicOfFilterTexts[k]:
					if random.random() < docRatio:
						negSent.append(word)

				bow_vecneg = self.dictionary.doc2bow(negSent)
				bow_vec1 = self.dictionary.doc2bow(sentArr[j])

				denseVector1, sparseVec1, combined1 = self.featureRobot.getFeaVecFromBow(bow_vec1)
				denseVectorneg, sparseVecneg, combinedneg = self.featureRobot.getFeaVecFromBow(bow_vecneg)
				self.ijPairs.append([combined1, combinedneg, 0])
				
		train_X = np.zeros((len(self.ijPairs),  self.ijPairs[0][0].shape[0] + self.ijPairs[0][1].shape[0] ))
		
		y = np.zeros(len(self.ijPairs))

		for i in range(len(self.ijPairs)) :
			train_X[i] = np.concatenate((self.ijPairs[i][0], self.ijPairs[i][1]))
			y[i] = self.ijPairs[i][2]

		return train_X, y 



# In[ ]:


# Model training
from sklearn.model_selection import train_test_split
import tensorflow as tf

class simGenerator():
	def __init__(self,train_eg, train_labels):
		self.train_eg = train_eg
		self.train_labels = train_labels

	def trainModel(self):
		train_examples, test_examples, train_labels, test_labels = train_test_split(self.train_eg, self.train_labels, test_size=0.33, random_state=42)
		train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
		test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
		
		BATCH_SIZE = 4
		SHUFFLE_BUFFER_SIZE = 16

		train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
		test_dataset = test_dataset.batch(BATCH_SIZE)

		print(train_examples.shape[1])

		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Dense(10, input_dim=train_examples.shape[1], activation='relu'))
		model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

		model.compile(optimizer=tf.keras.optimizers.RMSprop(),
			loss=tf.keras.losses.BinaryCrossentropy(),
			metrics=['binary_accuracy'])

		model.fit(train_dataset, epochs=10)
		print(model.evaluate(test_dataset))


		model_json = model.to_json()
		with open("/kaggle/working/kmodel.js", "w") as json_file:
			json_file.write(model_json)

		model.save_weights("/kaggle/working/kmodel.h5")
		print("Saved model to disk")

		self.trained_model = model

		


# In[ ]:


# Retrieval 

import numpy as np

class retrievalMethod():
	def __init__(self, sim_model,dicOfFilterTexts, corpus, featureBot):
		self.dicOfFilterTexts = dicOfFilterTexts
		self.sim_model = sim_model
		self.corpus = corpus
		self.featureBot= featureBot

	def findClosest(self, searchtext, limit=10, offset=0):
		denseVector, sparseVec, combined = self.featureBot.getFeaVec(searchtext)
		texts = [(eachitem, self.dicOfFilterTexts[eachitem]) for eachitem in self.dicOfFilterTexts] 

		relevantList = []
		for i in range(len(self.corpus)):
			each_bow = self.corpus[i]
			denseVector1, sparseVec1, combined1 = self.featureBot.getFeaVecFromBow(each_bow)
			modelVec = np.concatenate((combined1, combined))
			modelVec = modelVec.reshape((1, modelVec.shape[0]))
			prob = self.sim_model.predict(modelVec)
			relevantList.append((prob, i))

		relevantList.sort(reverse=True)

		return [{
		"prob": r[0],
		"paper_id": texts[r[1]][0]
		} for r in relevantList[offset:offset + limit]]



# In[ ]:


# Example usage 

print("Data cleaning")
fileList = "/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json"
if IS_FAST:
    cleaner_bot = dataCleaningRobot(fileList, 70)
else:
    cleaner_bot = dataCleaningRobot(fileList)

cleaner_bot.getText()
cleaner_bot.getDicCorpus()

print("Feature generation")
feat_bot = featureGenRobot()
feat_bot.genFeaVecMap(cleaner_bot.dictionary, cleaner_bot.corpus)
feat_bot.saveModelsDict()

print("Ground truth generation")
fileList = "/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json"
cleaner_bot_small = dataCleaningRobot(fileList, 70)
cleaner_bot_small.getText()
cleaner_bot_small.getDicCorpus()

eg_bot = exampleGenRobot(cleaner_bot_small.dicOfFilterTexts, cleaner_bot.dictionary, feat_bot)
train_eg, train_labels = eg_bot.getIJPair()

print("Model training")
sim_bot = simGenerator(train_eg , train_labels)
model = sim_bot.trainModel()


# In[ ]:


retrieve_bot = retrievalMethod(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)
retrieve_bot.findClosest("vaccine and drug") 


# # Data exploration and visualization

# The library code defined above can also be used to do data exploration and visualization.

# In[ ]:


### Single word count visualization
import glob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np


wordList, countList, word_freq_list = cleaner_bot.getSingleWordCount()

plt.plot(range(len(countList)), countList)  
plt.xlabel("Top i vacab")
plt.ylabel("Word count")
plt.show()



# In[ ]:


offset = 0 
numItems = 10
plt.barh(wordList[offset:offset+ numItems], countList[offset:offset+ numItems])
plt.show()


# In[ ]:


offset = 50 
numItems = 10
plt.barh(wordList[offset:offset+ numItems], countList[offset:offset+ numItems])
plt.show()


# In[ ]:


### Bigram word count visualization

cleaner_bot.getBigramData()
wordList_bi, countList_bi, word_freq_list_bi = cleaner_bot.getBigramCount()
plt.xlabel("Top i bigram")
plt.ylabel("Bigram count")
plt.plot(range(len(countList_bi)), countList_bi)  
plt.show()


# In[ ]:



offset = 0 
numItems = 10
plt.barh(wordList_bi[offset:offset+ numItems], countList_bi[offset:offset+ numItems])
plt.show()


# In[ ]:


offset = 20 
numItems = 10
plt.barh(wordList_bi[offset:offset+ numItems], countList_bi[offset:offset+ numItems])
plt.show()


# In[ ]:


### Word cloud visualization

texts = ""
for eachDoc in cleaner_bot_small.dicOfFilterTexts:
  for eachword in cleaner_bot_small.dicOfFilterTexts[eachDoc]:
    if cleaner_bot_small.wordCtFreq[eachword] > countList[-30] and cleaner_bot_small.wordCtFreq[eachword] < countList[30]:
      texts = texts + " " + eachword


wordcloud = WordCloud(width = 800, height = 800, 
      background_color ='white', 
      min_font_size = 10).generate(texts)

plt.imshow(wordcloud, interpolation='bilinear')
plt.show()


# # Evaluation

# We first test the simplest method using the function $f$ and $s$ to do retrieval, namely, directly use apply $f$ on the query and the documents and use $s$ to find relevant documents. But we found that when we use a small dataset to train the model, the performance is not good. We suspect that the size of the dataset is one of the culpit and we also note that the inherent noise in this unsupervised approach may be hard to deal with if dataset is not big enough. In particular, it gave the same set of document for a variety of query here

# In[ ]:


from IPython.core.display import display, HTML
from jinja2 import Template
import glob

class evaluatorRobot():
	def __init__(self):
		
		self.table_template = Template('''
		<table>
		<thead>
		<tr>
		<th>Title</th>
		<th>Authors</th>
		<th>Abstract</th>
		<th>Paper ID</th>
		</tr>
		</thead>
		<tbody>
		{% for paper in papers %}
		<tr>
		<td>{{ paper.title }}</td>
		<td>{{ paper.authors }}</td>
		<td>
		{% for paragraph in paper.abstract %}
		<p>{{ paragraph }}</p>
		{% endfor %}
		</td>
		<td>{{ paper.paper_id }}</td>
		</tr>
		{% endfor %}
		</tbody>
		</table>
		''')

	def loadPath(self, path):
		self.pathtofiles = path

	def load_paper(self, paper_id):
		matches = glob.glob(self.pathtofiles + "/" + f'{paper_id}.json', recursive=True)
		filename = matches[0]
		with open(filename) as f:
			data = json.load(f)
		return data

	def formatPaper(self, raw_paper):
		paper = self.load_paper(raw_paper['paper_id'])
		authors = [f'{author["first"]} {author["last"]}' for author in paper['metadata']['authors']]
		abstract_paragraphs = [paragraph['text'][:100] + '...' for paragraph in paper['abstract']]
	
		return {
			'title': paper['metadata']['title'],
			'authors': ', '.join(authors),
			'abstract': abstract_paragraphs,
			'paper_id': paper['paper_id'],
			"prob": raw_paper['prob']
		}

	def presentResults(self, results):
		papers = [self.formatPaper(r) for r in results]
		render = self.table_template.render(papers=papers)
		display(HTML(render))


# In[ ]:


retrieve_bot = retrievalMethod(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)
results = retrieve_bot.findClosest("drug and vaccines", limit=5, offset=0)

eval_bot = evaluatorRobot()
eval_bot.loadPath(cleaner_bot.path)
eval_bot.presentResults(results)


# We then try to use another approach to leverage the $f$ and $s$ that we first use key word search, then cosine similarity search, then search using the similarity function learnt. 

# In[ ]:


# Retrieval 

import numpy as np
from scipy import spatial
from nltk.stem.porter import PorterStemmer

class retrievalMethod2():
	def __init__(self, sim_model,dicOfFilterTexts, corpus, featureBot):
		self.dicOfFilterTexts = dicOfFilterTexts
		self.sim_model = sim_model
		self.corpus = corpus
		self.featureBot= featureBot
        
	def findClosestWithSeed(self,  searchtext, limit=10, offset=0):
		# = ["therapeutic"]
		#key_words_2  = ["animal" ,"model"]
		texts = [ [eachitem , self.dicOfFilterTexts[eachitem]] for eachitem in self.dicOfFilterTexts ] 
		relevantList = []
		alreadyFoundDic = {}

		### Key word match first : score range 0.7 to 1 
		key_words_1 = searchtext.split()
		porter = PorterStemmer()

		kw_stem = []
		for word in key_words_1:
			kw_stem.append(porter.stem(word))

		kw_matched_list = []

		index = 0
		for doc_id in self.dicOfFilterTexts:
			N = len(self.dicOfFilterTexts[doc_id])
			count  = 0 
			total_count = 0
			for i in range(N - len(kw_stem)):
				total_count += 1
				if self.dicOfFilterTexts[doc_id][i:i+ len(kw_stem)] == kw_stem:
					count += 1 

			if count > 0:
				kw_matched_list.append([ count*1.0/total_count, count , doc_id, index])

			index += 1 

		kw_matched_list.sort(reverse=True)
		kw_matched_score_list= []
		for j in range(len(kw_matched_list)):
			index = kw_matched_list[j][-1]
			#print("d1", j)
			relevantList.append([1 - (1-0.7)*j/len(kw_matched_list), index])
			kw_matched_score_list.append([1 - (1-0.7)*j/len(kw_matched_list), index])
			alreadyFoundDic[index] = True

		### Cosine distance match : score range 0.3 to 0.7 
		cos_sim_matched_list = []

		for each_already_rel in kw_matched_score_list:
			score, kk = each_already_rel[0],  each_already_rel[1]
			denseVector, sparseVec, combined = self.featureBot.getFeaVecFromBow(self.corpus[kk])

			for i  in range(len(self.corpus )):
				if  i in alreadyFoundDic:
					continue
				each_bow = self.corpus[i]
				denseVector1, sparseVec1, combined1 = self.featureBot.getFeaVecFromBow(each_bow)
				similarity = 1 - spatial.distance.cosine(denseVector, denseVector1)
				if similarity > 0.9 :
					cos_sim_matched_list.append([score*similarity, i ] )


		cos_sim_matched_list.sort(reverse=True)
		for j in range(len(cos_sim_matched_list)):
			index = cos_sim_matched_list[j][-1]
			#print("d2", j)
			relevantList.append([0.7 - (0.7-0.3)*j/len(cos_sim_matched_list), index])
			alreadyFoundDic[index] = True


		### s distance match using doc  : score range 0.1 to 0.3 
		### Cosine distance match : score range 0.3 to 0.7 
		doc_sim_matched_list = []

		for each_already_rel in kw_matched_score_list:
			score, kk = each_already_rel[0],  each_already_rel[1]
			denseVector, sparseVec, combined = self.featureBot.getFeaVecFromBow(self.corpus[kk])
			for i  in range(len(self.corpus )):
				if  i in alreadyFoundDic:
					continue
				each_bow = self.corpus[i]
				denseVector1, sparseVec1, combined1 = self.featureBot.getFeaVecFromBow(each_bow)

				modelVec = np.concatenate((combined1, combined))
				modelVec= modelVec.reshape((1, modelVec.shape[0]))
				prob = self.sim_model.predict(modelVec)
				
				if prob > 0.9:
					doc_sim_matched_list.append([score*prob, i ] )


		doc_sim_matched_list.sort(reverse=True)
		for j in range(len(doc_sim_matched_list)):
			index = doc_sim_matched_list[j][-1]
			#print("d3", j)
			relevantList.append([0.3 - (0.3-0.1)*j/len(doc_sim_matched_list), index])
			alreadyFoundDic[index] = True

		relevantList.sort(reverse = True)

		return [{
			"prob": r[0],
			"paper_id": texts[r[1]][0]
		} for r in relevantList[offset:offset + limit]]




# In[ ]:


retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)
results = retrieve_bot.findClosestWithSeed("vaccine") 

eval_bot = evaluatorRobot()
eval_bot.loadPath(cleaner_bot.path)
eval_bot.presentResults(results)


# In[ ]:


retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)
results = retrieve_bot.findClosestWithSeed("drug") 

eval_bot = evaluatorRobot()
eval_bot.loadPath(cleaner_bot.path)
eval_bot.presentResults(results)


# We further categorize each subquestions in this task and identify important keywords to start the search. Here are what we used. 
# Cure (Drug, vaccine and preventive measures):
# 
# [Keyword: therapeutic]
#   1. Effectiveness of drugs being developed and tried to treat COVID-19 patients. Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication. 
# 
#   4. Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.
# 
# [Keyword: universal vaccine]
#   6. Efforts targeted at a universal coronavirus vaccine. 
# 
# [Keyword: prophylaxis]
#   8. Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers 
# 
# 
# Animal model:
# 
# [Keyword: animal model]
# 
#     3. Exploration of use of best animal models and their predictive value for a human vaccine.
# 
#     7. Efforts to develop animal models and standardize challenge studies
# 
#     10. Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]
# 
# 
# Side effects:
# 
# [Keyword: enhanced disease]
# 
#     2. Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.
# 
#     9. Approaches to evaluate risk for enhanced disease after vaccination
# 
# 
# Distribution method:
# 
# [Keyword: prioritize]
# 
#     10. Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.
# 

# In[ ]:


retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)
results = retrieve_bot.findClosestWithSeed("therapeutic") 

eval_bot = evaluatorRobot()
eval_bot.loadPath(cleaner_bot.path)
eval_bot.presentResults(results)


# In[ ]:


retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)
results = retrieve_bot.findClosestWithSeed("universal vaccine") 

eval_bot = evaluatorRobot()
eval_bot.loadPath(cleaner_bot.path)
eval_bot.presentResults(results)


# In[ ]:


retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)
results = retrieve_bot.findClosestWithSeed("prophylaxis") 

eval_bot = evaluatorRobot()
eval_bot.loadPath(cleaner_bot.path)
eval_bot.presentResults(results)


# In[ ]:


retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)
results = retrieve_bot.findClosestWithSeed("animal model") 

eval_bot = evaluatorRobot()
eval_bot.loadPath(cleaner_bot.path)
eval_bot.presentResults(results)


# In[ ]:


retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)
results = retrieve_bot.findClosestWithSeed("enhanced disease") 

eval_bot = evaluatorRobot()
eval_bot.loadPath(cleaner_bot.path)
eval_bot.presentResults(results)



# In[ ]:


retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)
results = retrieve_bot.findClosestWithSeed("prioritize") 

eval_bot = evaluatorRobot()
eval_bot.loadPath(cleaner_bot.path)
eval_bot.presentResults(results)





# In[ ]:




