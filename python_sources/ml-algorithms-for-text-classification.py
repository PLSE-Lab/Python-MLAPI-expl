# Text_similaritty dataset 
# ************* Decision tree and Random Forest 90% *****************
# Natural Language Processing
# First We have to Build Corpus either on X description or On Y description
# Then we have t build machine learning algrothms and find out there accuracy by confusion matrix
#Algorithms that are used are as follows ;-
# 1.Naive Bayes
# 2.Logistic Regression
# 3.Decision Tree classification
# 4,Random Forest classification
# Aacuracy measured by confusion matrix 


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/text-similarity/train.csv', delimiter = ',')

# Cleaning the texts
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0, 2100):
    review = re.sub('[^a-zA-Z]', ' ', dataset['description_y'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
 
print(corpus[0:200])
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 577)
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[0:2100, 5].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix for Naive Bayes is : ")
print(cm)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)

# Predicting the Test set results
y_predL = classifier1.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_predL)
print("Confusion matrix for Logistic regression is : ")
print(cm1)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier2.fit(X_train, y_train)

# Predicting the Test set results
y_predD = classifier2.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_predD)
print("Confusion matrix for Decision tree is is : ")
print(cm2)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier3.fit(X_train, y_train)

# Predicting the Test set results
y_predR = classifier3.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_predR)
print("Confusion matrix for Random Forest  is : ")
print(cm3)

#***************************** Prediction of text dataset with random forest************
# as random forest is best for prediction we have to process descrption y of test dataset.
# and pass this corpus to Random Forest classifier

Tdataset = pd.read_csv('../input/text-similarity/test.csv', delimiter = ',')

# ltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 516): 
    review = re.sub('[^a-zA-Z]', ' ', Tdataset['description_y'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 577)
X1 = cv.fit_transform(corpus).toarray()


y_predR = classifier3.predict(X1)

#******************* Showing demo results ********************

Sdataset = pd.read_csv('../input/text-similarity/test.csv', delimiter = ',')

final_op = {'test_ID' : Sdataset['test_id'] , 'description_x' : Sdataset['description_x'],'description_y': Sdataset['description_y'] ,'Same_Security' :y_predR}

final_op=pd.DataFrame(final_op)

print(final_op.iloc[0:10,:].values)
