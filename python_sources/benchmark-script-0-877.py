# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
# All files are in ../input
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
import pandas as pd  # import pandas for data manipulation (although here we will just use it to load the data
import numpy as np # numpy to bring the data into a more favorable format
from sklearn.linear_model import LogisticRegression # the model we will use
from sklearn.metrics import roc_auc_score # the metric we are being tested on (auc)
from sklearn.feature_extraction.text import TfidfVectorizer # thiw will convert the text data to numbers of tfidf (Term Frequency - Inverse Document Frequency

#The model we will use
model=LogisticRegression(C=1., random_state=1) # C is regularization, 
#The tf-idf object (see all parameters here : http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
tfv=TfidfVectorizer(min_df=0, max_features=None, strip_accents='unicode',lowercase =True,
                            analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1,1),
                            use_idf=True,smooth_idf=True, sublinear_tf=True)

#Load train data
train=pd.read_csv("../input/train.csv",encoding='latin1' ) # the encoding here is a bit tricky (e.g latin1)
print ("train shape, ", train.shape)

target=np.array(train.target.values) # get the target variable
text=train.text.values # get the text

# fit and transform the tfidf object
transformed_data=tfv.fit_transform(text)

#load the test data
test=pd.read_csv("../input/test.csv",encoding='latin1')
print ("test shape, ", test.shape)

test_text=test.text.values #get text

#transform text of the test dataset
transformed_test_data=tfv.transform(test_text)

############### validation - split data to train anc cv###############

x_train=transformed_data[:100000]
y_train=target[:100000]

x_cv=transformed_data[100000:]
y_cv=target[100000:]

#fit a model
model.fit(x_train,y_train)
preds=model.predict_proba(x_cv)[:,1] #make predictions an
print (" cv auc ", roc_auc_score(y_cv,preds)) # print auc

# now fit on the whole data
print (" main model ")
model.fit(transformed_data,target)
preds=model.predict_proba(transformed_test_data)[:,1]

#create submission
test["target"]=preds
test.drop("text", inplace=True, axis=1)
test.to_csv("benchmark_sub.csv",index=False,header=True)



