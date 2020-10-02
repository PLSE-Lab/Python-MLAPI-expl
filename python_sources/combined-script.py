import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from nltk.stem.snowball import SnowballStemmer

data = pd.read_csv('../input/Combined_News_DJIA.csv')
data.head()
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

#########################################################################
#join headlines
trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
	
#########################################################################
#stemming words
trainSet = []
for i in trainheadlines:
	tempSet = [SnowballStemmer('english').stem(j) for j in i.split()]
	trainSet.append(' '.join(x for x in tempSet))
testSet = []
for i in testheadlines:
	tempSet = [SnowballStemmer('english').stem(j) for j in i.split()]
	testSet.append(' '.join(x for x in tempSet))

#########################################################################
#vectorizer = CountVectorizer()
vectorizer = CountVectorizer(ngram_range=(2,2))
#vectorizer = TfidfVectorizer()

trainSetVectorized = vectorizer.fit_transform(trainSet)
testSetVectorized = vectorizer.transform(testSet)

trainDense = trainSetVectorized.toarray()
testDense = testSetVectorized.toarray()


######## Logistic Regression #########
modal = LogisticRegression()
modal = modal.fit(trainSetVectorized, train["Label"])
prediction = modal.predict(testSetVectorized)

pd.crosstab(test["Label"], prediction, rownames=["Actual"], colnames=["Predicted"])
accuracy = accuracy_score(prediction, test['Label'])
print(accuracy)

####### Naive Bayes #######
'''modal = GaussianNB()
modal = modal.fit(trainDense, train["Label"])
prediction = modal.predict(testDense)

pd.crosstab(test["Label"], prediction, rownames=["Actual"], colnames=["Predicted"])
accuracy = accuracy_score(prediction, test['Label'])
print(accuracy)'''

####### Decision Tree #######
modal = DecisionTreeClassifier()
modal = modal.fit(trainSetVectorized, train["Label"])
prediction = modal.predict(testSetVectorized)

pd.crosstab(test["Label"], prediction, rownames=["Actual"], colnames=["Predicted"])
accuracy = accuracy_score(prediction,test['Label'])
print(accuracy)


############ Best words ###############################################
#words = vectorizer.get_feature_names()
#coefficients = modal.coef_.tolist()[0]
#wordCoefficients = pd.DataFrame({'Word' : words, 'Coefficient' : coefficients})
#wordCoefficients = wordCoefficients.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
#
#print(wordCoefficients.head(10))
#print(wordCoefficients.tail(10))