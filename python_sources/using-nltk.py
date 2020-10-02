# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('../input/train.csv')
dataset = dataset.drop(['id', 'qid1', 'qid2'], axis = 1)
dataset = dataset.fillna('empty')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
question1 = []
for i in range(0, dataset.shape[0]):
    q1 = re.sub('[^a-zA-Z]', ' ', dataset['question1'][i])
    q1 = q1.lower()
    q1 = q1.split()
    ps = PorterStemmer()
    q1 = [ps.stem(word) for word in q1 if not word in set(stopwords.words('english'))]
    q1 = ' '.join(q1)
    question1.append(q1)
question2 = []
for i in range(0, dataset.shape[0]):
    q2 = re.sub('[^a-zA-Z]', ' ', dataset['question2'][i])
    q2 = q2.lower()
    q2 = q2.split()
    ps = PorterStemmer()
    q2 = [ps.stem(word) for word in q2 if not word in set(stopwords.words('english'))]
    q2 = ' '.join(q2)
    question2.append(q2)
    
#Count of similar words
def Score(guess, solution):
    guess = guess.split()
    solution = solution.split()
    c = 0
    for g in guess:
        if g in solution:
            c += 1.0
    if len(guess)+len(solution) != 0:       
        return (c/(len(guess)+len(solution)))
    else:
        return 0.0
        
sw = []    
for i in range(0, dataset.shape[0]):
    sw.append(Score(question1[i],question2[i]))
    
# Creating inputs and outputs
X = np.asarray(sw)
X = X.reshape(len(sw),1)    
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier as GBC
classifier = GBC(learning_rate = 0.05)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Finding log loss
from sklearn.metrics import log_loss
log_loss(y_test,y_pred)

#Reading Test Data
test = pd.read_csv('test.csv')

#Prepaing Dataset

ID = test['test_id']
test = test.drop('test_id',axis = 1)
test = test.fillna('empty')
question1 = []
for i in range(0, test.shape[0]):
    q1 = re.sub('[^a-zA-Z]', ' ', test['question1'][i])
    q1 = q1.lower()
    q1 = q1.split()
    ps = PorterStemmer()
    q1 = [ps.stem(word) for word in q1 if not word in set(stopwords.words('english'))]
    q1 = ' '.join(q1)
    question1.append(q1)
question2 = []
for i in range(0, test.shape[0]):
    q2 = re.sub('[^a-zA-Z]', ' ', test['question2'][i])
    q2 = q2.lower()
    q2 = q2.split()
    ps = PorterStemmer()
    q2 = [ps.stem(word) for word in q2 if not word in set(stopwords.words('english'))]
    q2 = ' '.join(q2)
    question2.append(q2)
sw = []    
for i in range(0, test.shape[0]):
    sw.append(Score(question1[i],question2[i]))
X_submission = np.asarray(sw).reshape(len(sw),1)  

#Predicting Results
y = classifier.predict(X_submission)

CSVFile = np.column_stack((ID,y.T))
outdata = pd.DataFrame(CSVFile, columns = ['test_id', 'is_duplicate'])
outdata['ID'] = outdata['ID'].astype(int)
outdata.to_csv('../output/Quora_Duplicate.csv', index = None)