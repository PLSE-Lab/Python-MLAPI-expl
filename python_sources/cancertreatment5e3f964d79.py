#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# 
# <h1>IMPORT LIBRARIES</h1>

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# In[ ]:


train_variants_df = pd.read_csv("../input/training_variants")
test_variants_df = pd.read_csv("../input/test_variants")
train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])


# In[ ]:


gene_group = train_variants_df.groupby("Gene")['Gene'].count()
minimal_occ_genes = gene_group.sort_values(ascending=True)[:10]
print("Genes with maximal occurences\n", gene_group.sort_values(ascending=False)[:10])
print("\nGenes with minimal occurences\n", minimal_occ_genes)


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(train_variants_df.Class,data = train_variants_df)


# 
# <h1>MERGE DATAFRAME</h1>

# In[ ]:


#Merge dataframe by ID number key
train_df = pd.merge(train_text_df,train_variants_df,left_on="ID", right_on="ID")
print(train_df.shape)
train_df.head(3)

test_df = pd.merge(test_text_df,test_variants_df,left_on="ID", right_on="ID")
print(test_df.shape)
test_df.head(3)


# 
# <h1>REMOVE NA VALUES</h1>

# In[ ]:


#This is multi class classification problem and number of classes are total 9. 
#we have to predicat the classes probabalitie for particular Id
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)


# In[ ]:


#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(
    min_df=5, max_features=16000, strip_accents='unicode', lowercase=True,
    analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
    smooth_idf=True, sublinear_tf=True, stop_words = 'english'
)


# In[ ]:


tfidf_vectorizer.fit(train_df['Text'])


# 
# <h1>MODEL EVALUATION</h1>
# 

# In[ ]:



from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score


# In[ ]:


X_train_tfidfmatrix = tfidf_vectorizer.transform(train_df['Text'].values)
X_test_tfidfmatrix = tfidf_vectorizer.transform(test_df['Text'].values)
y_train = train_df['Class'].values


# In[ ]:


def evaluate(X, y, clf=None):
    probas = cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits=5, random_state=8), 
                              n_jobs=-1, method='predict_proba', verbose=2)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(y, probas)))
    print('Accuracy: {}'.format(accuracy_score(y, preds)))


# In[ ]:


#evaluation
evaluate(X_train_tfidfmatrix, y_train, clf=XGBClassifier())


# 
# <h1>TRAINING MODEL</h1>

# In[ ]:


#training
clf = XGBClassifier()
clf.fit(X_train_tfidfmatrix, y_train)


# In[ ]:


import xgboost
xgboost.to_graphviz(clf, num_trees=9)


# 
# <h1>TEST MODEL</h1>

# In[ ]:


#test
y_test_predicted = clf.predict_proba(X_test_tfidfmatrix)


# 
# <h1>KAGGLE SUBMISSION</h1>

# In[ ]:


submission_df = pd.DataFrame(y_test_predicted, columns=['class' + str(c + 1) for c in range(9)])
submission_df['ID'] = test_df['ID'].values


# In[ ]:


submission_df.head()
submission_df.columns


# In[ ]:


submission_df = submission_df[['ID','class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7',
       'class8', 'class9']]
submission_df.head()


# In[ ]:


submission_df["ID"] = pd.to_numeric(submission_df["ID"], errors='coerce')


# In[ ]:


submission_df.to_csv('cancer_treatment3.csv', index=False)

