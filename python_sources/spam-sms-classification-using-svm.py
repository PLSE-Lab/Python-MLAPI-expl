# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#SVM model for SMS classification using SMS Spam COllection Dataset (Kaggle)
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn import svm

email=pd.read_csv("../input/spam.csv",encoding='latin-1')
email=email.rename(columns = {'v1':'label','v2':'message'})
cols=['label','message']
email=email[cols]
email=email.dropna(axis=0, how='any')

#Email preprocessing
num_emails=email["message"].size
def email_processing(raw_email):
    letters_only=re.sub("[^a-zA-Z]"," ",raw_email)
    words=letters_only.lower().split()
    stops=set(stopwords.words("english"))
    m_w=[w for w in words if not w in stops]
    return (" ".join(m_w))


clean_email=[]
for i in range(0,num_emails):
    clean_email.append(email_processing(email["message"][i]))

#Create new dataframe column
email["Processed_Msg"]=clean_email
cols2=["Processed_Msg","label"]
email=email[cols2]

#Create train and test sets
X_train=email["Processed_Msg"][:5000]
Y_train=email["label"][:5000]
X_test=email["Processed_Msg"][5001:5500]
Y_test=email["label"][5001:5500]


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

train_data_features=vectorizer.fit_transform(X_train)
train_data_features=train_data_features.toarray()

test_data_features=vectorizer.transform(X_test)
test_data_features=test_data_features.toarray()

#SVM with linear kernel
clf=svm.SVC(kernel='linear',C=1.0)
print ("Training")
clf.fit(train_data_features,Y_train)

print ("Testing")
predicted=clf.predict(test_data_features)
accuracy=np.mean(predicted==Y_test)
print ("Accuracy: ",accuracy)

#Validation
X=email["Processed_Msg"][5501:5502]
validation_data=vectorizer.transform(X)
validation_data=validation_data.toarray()

print ("SMS: ",X)
classification=clf.predict(validation_data)
print ("Classification: ",classification)

