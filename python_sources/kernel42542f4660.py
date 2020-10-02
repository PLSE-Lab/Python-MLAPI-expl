import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif
import nltk
from nltk.corpus import stopwords
import numpy as np
import requests

filename = "/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv"

fields = [] 
rows = [] 

with open(filename, 'r') as csvfile: 
	csvreader = csv.reader(csvfile) 
	fields = csvreader.__next__()

	for row in csvreader: 
		rows.append(row) 
	print("Total no. of rows: %d"%(csvreader.line_num)) 

print('Field names are:' + ', '.join(field for field in fields)) 


description = "description"
label = "fraudulent"
description_data = []
fraudulent_list = []
label_index = 0
description_index = 0
for i in range(len(fields)):
	if fields[i] == description:
		description_index = i
	if fields[i] == label:
		label_index = i

for row in rows:
	#We have some incomplete data records without descriptions and/or labels
	if len(row) >= label_index and len(row) >= description_index:
		fraudulent_list.append(row[label_index])
		description_data.append(row[description_index])

features_train, features_test, labels_train, labels_test = train_test_split(description_data, fraudulent_list, test_size=0.1, random_state=42)
#Convert the job descriptions in the CSV file to a matrix of TF-IDF features
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)
#I only select the top 1 percentile of features
selector = SelectPercentile(f_classif, percentile=1)
selector.fit(features_train_transformed, labels_train)
features_train_transformed = selector.transform(features_train_transformed).toarray()
features_test_transformed  = selector.transform(features_test_transformed).toarray()
#I use a NB classifier to predict spam postings and get an accuracy score in excess of 95%
clf = GaussianNB()
clf.fit(features_train_transformed, labels_train)
pred = clf.predict(features_test_transformed)
print(accuracy_score(labels_test, pred))
