# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import warnings
import re

# Loading Tools
from sklearn import model_selection 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading Classification Models
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# Function to clean text
def CleanText(text):
	returnText = []
	text = text.lower()
	charSet = 'abcdefghijklmnopqrstuvwxyz -,12345678980'
	for each in text:
		if each in charSet:
			returnText.append(each)
	returnText = ''.join(returnText)
	# Removing extra spaces
	if len(returnText) > 1:
		returnText = re.sub('  ', ' ', returnText)
		if returnText[0] == ' ':
			returnText = returnText[1:]
		if returnText[len(returnText)-1] == ' ':
			returnText = returnText[:len(returnText)-1]
	return returnText

###############################################################################
# Loading data
train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')

# Concatenating all the data into a singel Dataframe
train['TYPE-LABEL'] = ['TRAIN'] * train.shape[0]
test['TYPE-LABEL'] = ['TEST'] * test.shape[0]
data = pd.concat([train, test],ignore_index = True, sort=False)

# Cleaning dataset
data.drop('SentenceId', axis=1, inplace=True)

# Applying a string cleaning function to the data set
data['CleanPhrase'] = data['Phrase'].apply(lambda x: CleanText(x))

# Creating a TF-IDF vector for the text
tfidfVector = TfidfVectorizer(binary=True)
tfidfVector.fit(data['CleanPhrase'].values)
tfidDict = sorted(tfidfVector.vocabulary_)

# Creating Training vector
XTrain = tfidfVector.transform(data.loc[(data['TYPE-LABEL'] == 'TRAIN').tolist(),'CleanPhrase'].values)
XTrain = XTrain.astype('float')
# Encoding Cusine/Target Variable
encoder = LabelEncoder()
yTrain = encoder.fit_transform(data.loc[(data['TYPE-LABEL'] == 'TRAIN').tolist(),'Sentiment'].values)

# Creating Test vector
XTest = tfidfVector.transform(data.loc[(data['TYPE-LABEL'] == 'TEST').tolist(),'CleanPhrase'].values)
XTest = XTest.astype('float')

###############################################################################
# Picking Logistic Regression Classifier
classifier = LogisticRegression()
classifier.fit(XTrain , yTrain)

# Predicting classes
yPredicted = classifier.predict(XTest)
yPredictedSentiment = encoder.inverse_transform(yPredicted)
yPredictedSentiment = map(int, yPredictedSentiment)

# Creating submission file
testID = data.loc[(data['TYPE-LABEL'] == 'TEST').tolist(),'PhraseId'].values
sentimentPredictions = pd.DataFrame({'PhraseId':testID, 'Sentiment':yPredictedSentiment})
sentimentPredictions = sentimentPredictions[['PhraseId','Sentiment']]
sentimentPredictions.to_csv('sample_submission.csv', index = False)
