from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import csv 
import pandas as pd 
import nltk 
import operator
import re 
import pickle
import time 
import numpy as np 

class spam:
	def __init__(self):
		self.lmtzr = WordNetLemmatizer()
		self.file = self.openFilePandas("../input/spam.csv")
		self.file = self.setup(self.file)	
		self.file = self.preProcessing(self.file)
		self.Train, self.Test = self.createTrainingTest(self.file)

	def openFile(self, fileName):
		f = open(fileName, 'rb')
		reader = csv.reader(f)
		return reader

	def openFilePandas(self, fileName):
		return pd.read_csv(fileName)

	def setup(self, fileName):
		#0 for Spam and 1 for ham
		fileName.v1[fileName.v1=='ham'] = 1
		fileName.v1[fileName.v1=='spam'] = 0
		fileName = fileName.drop('Unnamed: 2', 1)
		fileName = fileName.drop('Unnamed: 3', 1)
		fileName = fileName.drop('Unnamed: 4', 1)
		return fileName

	def saveToDisk(self, fileName, data):
		with open(fileName, 'wb') as handle:
			pickle.dump(data, handle)

	def retrieveFromDisk(self, fileName):
		with open(fileName, 'rb') as handle:
			return pickle.load(handle)

	def analytics(self, data):
		SortedFreqWords = self.findWordFreq(data[data.v1==0])
		#SortedFreqWords = sorted(FreqWords.items(), key=operator.itemgetter(1), reverse=True)
		#SortedFreqWords =dict(SortedFreqWords[0])
		if "COSTS" in SortedFreqWords:
			print ("Cost are", SortedFreqWords["COSTS"])
		#if "PHONE NUMBER" in SortedFreqWords:
		print ("Printing",SortedFreqWords["PHONE NUMBER"])
		
		print (SortedFreqWords[1:20])
		FreqWords = self.findWordFreq(data[data.v1==1])
		SortedFreqWords = sorted(FreqWords.items(), key=operator.itemgetter(1), reverse=True)
		print (SortedFreqWords[1:20])

	def findWordFreq(self, data):
		dictionary = {}
		for index, rows in data.iterrows():
			text = rows['v2']
			for word in text:
				if word not in dictionary:
					dictionary.setdefault(word, 1)
				else:
					dictionary[word] += 1
		return dictionary

	def preProcessing(self, data):
		start = time.time()
		for index, rows in data.iterrows():
			data['v2'][index] = [word.lower() for word in data['v2'][index].split()]
			data['v2'][index] = [re.sub('[^a-z0-9]+', '', word) for word in data['v2'][index]]
			data['v2'][index] = [word for word in data['v2'][index] \
								if not(word.isdigit()==True and len(word)<2)]
			
			for index2, word in enumerate(data['v2'][index]):
				#Finding Phone Numbers
				if word.isdigit() is True and (len(word) > 8 and len(word)<12):
					data['v2'][index][index2] = "PHONE NUMBER"
				#Cost
				elif word.isdigit() is True and (len(word) > 1 and len(word)<5):
					data['v2'][index][index2] = "COSTS"
				#Normalizing texts
				elif word.isalpha() is True:
					data['v2'][index][index2] = self.lmtzr.lemmatize(word)
			data['v2'][index] = [word for word in data['v2'][index] \
										 if word not in stopwords.words('english')]	
		#print data[data.v1==0]
		print ("Time taken for preprocessing: ", (time.time() - start), "seconds")
		return data

	def createTrainingTest(self, data):
		data = data.sample(frac=1)
		#print data.head(20)
		Train = int(0.7 * len(data))
		Training = data[:Train]
		Testing = data[Train:]
		return Training, Testing

	def prepareVector(self, Train, Test):
		database = self.findWordFreq(Train)
		TrainingMatrix = [[0 for i in xrange(len(database))] for j in xrange(len(Train))]
		TestingMatrix = [[0 for i in xrange(len(database))] for j in xrange(len(Test))]
		Train_X, Train_Y = self.fillUpDetails(TrainingMatrix, Train, database)
		Test_X, Test_Y = self.fillUpDetails(TestingMatrix, Test, database)
		return Train_X, Train_Y, Test_X, Test_Y

	def fillUpDetails(self, Matrix, Data, database):
		TrainY = []
		count = 0
		for index, sentence in Data.iterrows():
			TrainY.append(sentence['v1'])
			for word in sentence['v2']:
				if word in database:
					Matrix[count][database[word]] += 1
			count += 1
		Matrix = np.asarray(Matrix)
		TrainY = np.asarray(TrainY)
		return Matrix, TrainY
	
class MachineLearning:
	def __init__(self):
		pass 

	def SVM(self, trainX, trainY, testX, testY):
		print ("******SVM**********")
		from sklearn import svm 
		from sklearn.metrics import confusion_matrix
		#trainY[2], testY[2] = 1,1
		clf = svm.SVC()
		clf.fit(trainX, trainY)
		predict = clf.predict(testX)
		accuracy = 0.0
		for i, j in zip(testY, predict):
			if i==j:
				accuracy += 1
		print ("Accuracy is ", accuracy, len(testY), accuracy/len(testY), accuracy/len(testY)*100,"%")
		print (confusion_matrix(testY, predict))

if __name__ == '__main__':
	spam = spam()
	ML = MachineLearning()
	#print spam.file.head(20)
	#print len(spam.file[spam.file.v1==0]), len(spam.file[spam.file.v1==1])
	#print spam.Train[1:20], spam.Test[1:20]
	#spam.analytics(spam.file)
	#print stopwords.words('english')
	TrainX, TrainY, TestX, TestY = spam.prepareVector(spam.Train , spam.Test)
	print (TrainX.shape, TrainY.shape, TestX.shape, TestY.shape)
	ML.SVM(TrainX, TrainY, TestX, TestY)