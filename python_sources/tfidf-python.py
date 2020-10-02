# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# To Read File
def fileRead(fileNumber):
	if fileNumber == 1:
		return open("/kaggle/input/textfile1.txt", "r")
	if fileNumber == 2:
		return open("/kaggle/input/textfile2.txt", "r")
	if fileNumber == 3:
		return open("/kaggle/input/textfile3.txt", "r")
	if fileNumber == 4:
		return open("/kaggle/input/textfile4.txt", "r")

# To Tokenize
def Splitting(fileName):
	if fileName.mode == "r":
		contents = fileName.read()
	return contents.split()

# To Remove StopWords	
def StopWord(fileName):
	from nltk.corpus import stopwords
	stopWord = set(stopwords.words('english'))
	Content = []
	i = 0
	for word in fileName:
		if word not in stopWord:
			Content.append(word)
			i = i + 1
			
	
	Content = removeCommasAndFullStop(Content)	
	return Content

# Called by StopWord() to remove commas and fullstops
def removeCommasAndFullStop(fileName):
	i = 0
	for word in fileName:
		if (word[-1] == "," or word[-1] == "."):
			word = word[:-1]
			fileName[i] = word
		i = i + 1

	i = 0	
	for word in fileName:
		if word[-1] == ".":
			fileName[i] = word.replace('.', '')
		i = i + 1
	return fileName

# To Remove Stem Words
def Stemming(fileName):
	from nltk.stem import PorterStemmer
	stemWord = PorterStemmer()
	
	total = 0
	for word in fileName:
		total = total + 1
			
	print("Total Words are: ", total)
	count = 0
	i = 0
	for word in fileName:
		if not word[-1] == "e":
			fileName[i] = stemWord.stem(word)
			count = count + 1
		i = i + 1
	print("Total Number of Stems are: ", count)	
	return fileName


# Join List into String
def JoinList(ListName):
	fList = ' '.join(ListName)
	return fList


# TFIDF 
def tf_idf(docList):
	from sklearn.feature_extraction.text import TfidfVectorizer
	tfidf = TfidfVectorizer()
	x = tfidf.fit_transform(docList)
	import pandas as pd
	df_tfidf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
	return df_tfidf.T



# File 1
print("File 1\n")
f1 = fileRead(1)
f1Content = Splitting(f1)
print("Before applying stopword and stemming")
lenf1 = len(f1Content)
print("Total Words are: ", lenf1)
print(f1Content)
list1Content = StopWord(f1Content)
print("\nAfter applying stopword and stemming")
list1Content = Stemming(list1Content)
print(list1Content)


print("\n")
#File 2
print("File 2\n")
f2 = fileRead(2)
f2Content = Splitting(f2)
print("Before applying stopword and stemming")
lenf2 = len(f2Content)
print("Total Words are: ", lenf2)
print(f2Content)
list2Content = StopWord(f2Content)
print("\nAfter applying stopword and stemming")
list2Content = Stemming(list2Content)
print(list2Content)


print("\n")
#File 3
print("File 3\n")
f3 = fileRead(3)
f3Content = Splitting(f3)
print("Before applying stopword and stemming")
lenf3 = len(f3Content)
print("Total Words are: ", lenf3)
print(f3Content)
list3Content = StopWord(f3Content)
print("\nAfter applying stopword and stemming")
list3Content = Stemming(list3Content)
print(list3Content)



print("\n")
#File 4
print("File 4\n")
f4 = fileRead(4)
f4Content = Splitting(f4)
print("Before applying stopword and stemming")
lenf4 = len(f4Content)
print("Total Words are: ", lenf4)
print(f4Content)
list4Content = StopWord(f4Content)
print("\nAfter applying stopword and stemming")
list4Content = Stemming(list4Content)
print(list4Content)


docs = [list1Content, list2Content, list3Content, list4Content]

for i, doc in enumerate(docs):
	docs[i] = JoinList(doc)
	print(docs[i])
	print()

print("\n\nDOCS ~ TFIDF:\n")
tableDF = tf_idf(docs)
print(tableDF)