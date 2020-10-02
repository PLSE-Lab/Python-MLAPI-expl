# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import re
import time
import warnings
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD #This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD)
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE #t-distributed Stochastic Neighbor Embedding. t-SNE is a tool to visualize high-dimensional data.
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier #This estimator implements regularized linear models with stochastic gradient descent (SGD) learning:
from imblearn.over_sampling import SMOTE #This object is an implementation of SMOTE - Synthetic Minority Over-sampling 
from collections import Counter,defaultdict
from scipy.sparse import hstack #Stack sparse matrices horizontally (column wise)
from sklearn.multiclass import OneVsRestClassifier #this strategy consists in fitting one classifier per class
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,train_test_split,GridSearchCV
from sklearn.calibration import CalibratedClassifierCV #Probability calibration with isotonic regression or sigmoid.
from sklearn.naive_bayes import MultinomialNB,GaussianNB
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")\

from mlxtend.classifier import StackingClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

#training var, coma seperated file
data_var = pd.read_csv("/kaggle/input/training_variants")
#training text, seperated by ||
data_text = pd.read_csv("/kaggle/input/training_text", sep="\|\|", engine="python", names=["ID","TEXT"],skiprows=1)

print(data_var.info())
print(data_text.info())


#remove all stop words like a,is,an,th, ....
#collect all of them from nltk library
stop_words = set(stopwords.words('english'))

def data_text_preprocess(total_text, ind, col):
    #remove int val from textdata
    if type(total_text) is not int:
        string = ""
        #replace all special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        #replacing multiple spaces with single space
        total_text = re.sub('\s+', ' ', str(total_text))
        #bring whole text to some lower case scale
        total_text = total_text.lower()
        
        for word in total_text.split():
            #if the word is not a stop word then retain that word from text
            if not word in stop_words:
                string += word + " "
        
        data_text[col][ind] = string

for index, row in data_text.iterrows():
    if type(row['TEXT']) is str:
        data_text_preprocess(row['TEXT'],index, 'TEXT')

print(data_text)

#merge both data
result = pd.merge(data_var,data_text, on="ID", how='left')
result.head()

result[result.isnull().any(axis=1)]

result.loc[result['TEXT'].isnull(),'TEXT'] = result["Gene"]+' '+result['Variation']
result[result.isnull().any(axis=1)]

#split data
#before spliting ensure all the spaces in Gene and Variation column to be replace by '_'
y_true = result['Class'].values
result.Gene = result.Gene.str.replace('\s+','_')
result.Variation = result.Variation.str.replace('\s+','_')

#spliting into train and test
X_train, test_df, y_train, y_test = train_test_split(result, y_true,stratify=y_true,test_size=0.2)

#split into validation and cross validation
train_df, cv_df, y_train,y_cv = train_test_split(X_train,y_train,stratify=y_train,test_size=0.2)

print("Train data",train_df.shape)
print("Test Data ",test_df.shape)
print("Validation data",cv_df.shape)

#look at distribution of data in all sets
train_class_distr = train_df['Class'].value_counts().sort_index()
test_class_distr = test_df['Class'].value_counts().sort_index()
cv_class_distr = cv_df['Class'].value_counts().sort_index()
print(train_class_distr)
print(test_class_distr)
print(cv_class_distr)

#building a random model
test_data_len = test_df.shape[0]
cv_data_len = cv_df.shape[0]

cv_predicted_y = np.zeros((cv_data_len,9))
for i in range(cv_data_len):
    rand_probs = np.random.rand(1,9)
    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log Loss:",log_loss(y_cv,cv_predicted_y,eps=1e-15))


test_predicted_y = np.zeros((test_data_len,9))
for i in range(test_data_len):
    rand_probs = np.random.rand(1,9)
    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log Loss:",log_loss(y_test,test_predicted_y,eps=1e-15))


predicted_y = np.argmax(test_predicted_y,axis=1)
predicted_y

#confusion Mat
C = confusion_matrix(y_test,predicted_y)

labels = [1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(20,7))
sns.heatmap(C,annot=True, cmap="YlGnBu",fmt=".3f",xticklabels=labels,yticklabels=labels)
plt.xlabel("predicted class")
plt.ylabel("Original class")
plt.show()


#evaluating Gene Column

unique_gene = train_df['Gene'].value_counts()
print('Num of Unique Genes:',unique_gene.shape[0])
print(unique_gene.head(10))

s = sum(unique_gene.values)
h = unique_gene.values/s
c = np.cumsum(h)
plt.plot(c,label='Cumulative distribution of Genes')
plt.grid()
plt.legend()
plt.show()


#one hot encoding of Gene feature
gane_vectorizer = CountVectorizer()

train_gene_feature_onehotCoding = gane_vectorizer.fit_transform(train_df['Gene'])
test_gene_feature_onehotCoding = gane_vectorizer.fit_transform(test_df['Gene'])
cv_gene_feature_onehotCoding = gane_vectorizer.fit_transform(cv_df['Gene'])

print(train_gene_feature_onehotCoding.shape)
print(test_gene_feature_onehotCoding.shape)
print(cv_gene_feature_onehotCoding.shape)

gane_vectorizer.get_feature_names()

#Response coding with laplace smoothing

def get_gv_fea_dict(alpha, feature, df):
    value_count = train_df[feature].value_counts()
    
    gv_dict = dict()
    
    for i, denominator in value_count.items():
        vec = []
        for k in range(1,10):
            cls_cnt = train_df.loc[(train_df['Class']==k) & (train_df[feature]==i)]
            vec.append((cls_cnt.shape[0] + alpha*10)/ (denominator + 90*alpha))
        
        gv_dict[i] = vec
    return gv_dict

def get_gv_feature(alpha, feature, df):
    gv_dict = get_gv_fea_dict(alpha, feature, df)
    
    value_count = train_df[feature].value_counts()
    gv_fea = []
    
    for index, row in df.iterrows():
        if row[feature] in dict(value_count).keys():
            gv_fea.append(gv_dict[row[feature]])
        else:
            gv_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])
    return gv_fea


#response coding of gene feature
#alpha is used for laplace smoothing
alpha = 1

train_gene_feature_responseCoding = np.array(get_gv_feature(alpha,"Gene",train_df))
test_gene_feature_responseCoding = np.array(get_gv_feature(alpha,"Gene",test_df))
cv_gene_feature_responseCoding = np.array(get_gv_feature(alpha,"Gene",cv_df))

print(train_gene_feature_responseCoding.shape)
print(test_gene_feature_responseCoding.shape)
print(cv_gene_feature_responseCoding.shape)


#Need a hyperparameter for SGD classifier.
alpha = [10 ** x for x in range(-5,1)]
print(alpha)

#using SGD classifier

cv_log_error_array = []

for i in alpha:
    clf = SGDClassifier(alpha=i,loss='log',random_state=42)
    clf.fit(train_gene_feature_responseCoding,y_train)
    sig_clf = CalibratedClassifierCV(clf,method='sigmoid')
    sig_clf.fit(train_gene_feature_responseCoding,y_train)
    predict_y = sig_clf.predict_proba(cv_gene_feature_responseCoding)
    cv_log_error_array.append(log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))
    print("for alpha val= ",i," The log loss is : ",log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))
    
#use best alpha value to compute log loss
best_alpha = 1.2870578043792422

#values overlapping between train,test, or CV and train
test_coverage =test_df['Gene'].isin(list(set(train_df['Gene']))).shape[0]
cv_coverage =cv_df['Gene'].isin(list(set(train_df['Gene']))).shape[0]

print('1. In test data ',test_coverage, 'out of ',test_df.shape[0],':',(test_coverage/test_df.shape[0])*100)
print('1. In CV data ',cv_coverage, 'out of ',cv_df.shape[0],':',(cv_coverage/cv_df.shape[0])*100)


#evaluating Variation column
unique_variations = train_df['Variation'].value_counts()
print('Number of Unique Variations: ',unique_variations.shape[0])
print(unique_variations.head(10))


s = sum(unique_variations.values)
h = unique_variations.values/s
c = np.cumsum(h)
plt.plot(c,label='Cumulative distribution of Genes')
plt.grid()
plt.legend()
plt.show()

#one hot encoding of Variation feature
variation_vectorizer = CountVectorizer()

train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])
test_variation_feature_onehotCoding = variation_vectorizer.fit_transform(test_df['Variation'])
cv_variation_feature_onehotCoding = variation_vectorizer.fit_transform(cv_df['Variation'])

print(train_variation_feature_onehotCoding.shape)
print(test_variation_feature_onehotCoding.shape)
print(cv_variation_feature_onehotCoding.shape)

#response coding for Variation feature
alpha = 1

train_variation_feature_responseCoding = np.array(get_gv_feature(alpha,"Variation",train_df))
test_variation_feature_responseCoding = np.array(get_gv_feature(alpha,"Variation",test_df))
cv_variation_feature_responseCoding = np.array(get_gv_feature(alpha,"Variation",cv_df))

train_variation_feature_responseCoding.shape


alpha = [10 ** x for x in range(-5,1)]
print(alpha)

#using SGD classifier

cv_log_error_array = []

for i in alpha:
    clf = SGDClassifier(alpha=i,loss='log',random_state=42)
    clf.fit(train_variation_feature_responseCoding,y_train)
    
    sig_clf = CalibratedClassifierCV(clf,method='sigmoid')
    sig_clf.fit(train_variation_feature_responseCoding,y_train)
    predict_y = sig_clf.predict_proba(cv_variation_feature_responseCoding)
    
    cv_log_error_array.append(log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))
    print("for alpha val= ",i," The log loss is : ",log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))

best_alpha= 1.7902297202262054

#values overlapping between train,test, or CV and train
test_coverage =test_df['Variation'].isin(list(set(train_df['Variation']))).shape[0]
cv_coverage =cv_df['Variation'].isin(list(set(train_df['Variation']))).shape[0]

print('1. In test data ',test_coverage, 'out of ',test_df.shape[0],':',(test_coverage/test_df.shape[0])*100)
print('1. In CV data ',cv_coverage, 'out of ',cv_df.shape[0],':',(cv_coverage/cv_df.shape[0])*100)


#Evaluate Text Column
