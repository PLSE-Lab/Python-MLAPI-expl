import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report
import pandas as pd
from tkinter.filedialog import askopenfilename
from decimal import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer() 
def stemming_tokenizer(str_input):
	words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
	words = [ps.stem(word) for word in words]
	return words


trainData =pd.read_csv("/kaggle/input/train.csv")
testData = pd.read_csv('/kaggle/input/test.csv')

vectorizer = TfidfVectorizer(min_df = 5,
		                             max_df = 0.8,
		                             sublinear_tf = True,tokenizer=stemming_tokenizer,ngram_range=(1,6),
		                             use_idf = True)
train_vectors = vectorizer.fit_transform(trainData['Content'])
test_vectors = vectorizer.transform(testData['Content'])

classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, trainData['Label'])
prediction_linear =classifier_linear.predict(test_vectors)
report = classification_report(testData['Label'], prediction_linear, output_dict=True)
score =(report['pos']['f1-score']+report['neg']['f1-score'])/2
print(score)

    		