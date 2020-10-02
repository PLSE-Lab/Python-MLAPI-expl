#!/usr/bin/env python
# coding: utf-8

# ### Notebook - Table of Content
# 
# 1. [**Importing necessary libraries**](#1.-Importing-necessary-libraries)   
# 2. [**Loading data**](#2.-Loading-data)   
#     2.a [**Reading gene and variation data**](#2.a-Reading-gene-and-variation-data)  
#     2.b [**Reading text data**](#2.b-Reading-text-data)  
# 3. [**Text preprocessing**](#3.-Text-preprocessing)  
#     3.a [**Removing non word characters, whitespaces and stop words**](#3.a-Removing-non-word-characters,-whitespaces-and-stop-words)  
#     3.b [**Merging both the data frames**](#3.b-Merging-both-the-data-frames)  
#     3.c [**Checking for missing values**](#3.c-Checking-for-missing-values)  
# 4. [**Train, validation and test split**](#4.-Train,-validation-and-test-split)  
#     4.a [**Splitting data into a ratio of 64:20:16 for train, validation and test set**](#4.a-Splitting-data-into-a-ratio-of-64:20:16-for-train,-validation-and-test-set)  
#     4.b [**Distribution of ${y}_{i}$  in train, validation and test datasets**](#4.b-Distribution-of-${y}_{i}$-in-train,-validation-and-test-datasets)  
# 5. [**Fitting a Random model**](#5.-Fitting-a-Random-model)  
# 6. [**Univariate Analysis**](#6.-Univariate-Analysis)   
#     6.a [**Univariate Analysis of Gene feature**](#6.a-Univariate-Analysis-of-Gene-feature)  
#     6.a.1 [**Number of uniques genes and their distribution**](#6.a.1-Number-of-uniques-genes-and-their-distribution)  
#     6.a.2 [**Distribution of Gene categories**](#6.a.2-Distribution-of-Gene-categories)  
#     6.a.3 [**Featurization of Gene feature**](#6.a.3-Featurization-of-Gene-feature)    
#     6.a.3.1 [**Featurization of Gene feature using Response coding**](#6.a.3.1-Featurization-of-Gene-feature-using-Response-coding)   
#     6.a.3.2 [**Featurization of Gene feature using One hot encoding**](#6.a.3.2-Featurization-of-Gene-feature-using-One-hot-encoding)   
#     6.a.4 [**Checking significance of Gene feature in predicting ${y}_{i}$**](#6.a.4-Checking-significance-of-Gene-feature-in-predicting-${y}_{i}$)  
#     6.a.4.1 [**Checking through Logistic Regresion model**](#6.a.4.1-Checking-through-Logistic-Regresion-model)   
#     6.a.4.2 [**Checking stability of Gene feature across all the datasets**](#6.a.4.2-Checking-stability-of-Gene-feature-across-all-the-datasets)  
#     6.b [**Univariate Analysis of Variation feature**](#6.b-Univariate-Analysis-of-Variation-feature)  
#     6.b.1 [**Number of uniques variations and their distribution**](#6.b.1-Number-of-uniques-variations-and-their-distribution)  
#     6.b.2 [**Distribution of Variation categories**](#6.b.2-Distribution-of-Variation-categories)  
#     6.b.3 [**Featurization of Variation feature**](#6.b.3-Featurization-of-Variation-feature)    
#     6.b.3.1 [**Featurization of Variation feature using Response coding**](#6.b.3.1-Featurization-of-Variation-feature-using-Response-coding)   
#     6.b.3.2 [**Featurization of Variation feature using one hot encoding**](#6.b.3.2-Featurization-of-Variation-feature-using-one-hot-encoding)   
#     6.b.4 [**Checking significance of Variation feature in predicting ${y}_{i}$**](#6.b.4-Checking-significance-of-Variation-feature-in-predicting-${y}_{i}$)  
#     6.b.4.1 [**Checking through Logistic Regresion model**](#6.b.4.1-Checking-through-Logistic-Regresion-model)   
#     6.b.4.2 [**Checking stability of Variation feature across all the datasets**](#6.b.4.2-Checking-stability-of-Variation-feature-across-all-the-datasets)  
#     6.c [**Univariate Analysis of Text feature**](#6.c-Univariate-Analysis-of-Text-feature)  
#     6.c.1 [**Number of uniques words and their distribution**](#6.c.1-Number-of-uniques-words-and-their-distribution)  
#     6.c.2 [**Featurization of Text feature**](#6.c.2-Featurization-of-Text-feature)    
#     6.c.2.1 [**Featurization of Text feature using Response coding**](#6.c.2.1-Featurization-of-Text-feature-using-Response-coding)   
#     6.c.2.2 [**Featurization of Text feature using one hot encoding**](#6.c.2.2-Featurization-of-Text-feature-using-one-hot-encoding)   
#     6.c.3 [**Checking significance of Text feature in predicting ${y}_{i}$**](#6.c.3-Checking-significance-of-Text-feature-in-predicting-${y}_{i}$)  
#     6.c.3.1 [**Checking through Logistic Regresion model**](#6.c.3.1-Checking-through-Logistic-Regresion-model)   
#     6.c.3.2 [**Checking stability of Text feature across all the datasets**](#6.c.3.2-Checking-stability-of-Text-feature-across-all-the-datasets) 
# 7. [**Classical Machine learning models**](#7.-Classical-Machine-learning-models)  
#     7.1 [**Naive Bayes**](#7.1-Naive-Bayes)  
#     7.2 [**kNN**](#7.2-kNN)  
#     7.3 [**Logistic Regression**](#7.3-Logistic-Regression)  
#     7.4 [**Linear SVM**](#7.4-Linear-SVM)   
#     7.5 [**Random Forest**](#7.5-Random-Forest)  
# 8. [**Stacking the models**](#8.-Stacking-the-models) 
# 9. [**Majority Voting classifier**](#9.-Majority-Voting-classifier)

# ### 1. Importing necessary libraries

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
import re
import math
from collections import Counter, defaultdict
from scipy.sparse import hstack
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


# ### 2. Loading Data
# 
# #### 2.a Reading gene and variation data

# In[ ]:


df = pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip')
print('Dataframe shape: ', df.shape)
print('Features names: ', df.columns.values)
df.head()


# #### 2.b Reading text data

# In[ ]:


df_text =pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/training_text.zip",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
print('Text data shape: ', df_text.shape)
print('Features names : ', df_text.columns.values)
df_text.head()


# ### 3. Text preprocessing
# 
# #### 3.a Removing non word characters, whitespaces and stop words

# In[ ]:


stop_words = set(stopwords.words('english'))
for i,text in enumerate(df_text["TEXT"]):
    if type(df_text["TEXT"][i]) is not str:
        print("no text description available at index : ", i)
    else:
        string = ""
        df_text["TEXT"][i] = str(df_text["TEXT"][i]).lower()
        df_text["TEXT"][i] = re.sub("\W"," ",df_text["TEXT"][i])
        df_text["TEXT"][i] = re.sub('\s+',' ', df_text["TEXT"][i])
        for word in df_text["TEXT"][i].split():
            if not word in stop_words:
                string += word + " "
        df_text["TEXT"][i] = string


# In[ ]:


df["Gene"] = df["Gene"].str.replace('\s+', '_')
df["Variation"] = df["Variation"].str.replace('\s+', '_')


# #### 3.b Merging both the data frames

# In[ ]:


final_df = pd.merge(df, df_text,on='ID', how='left')
final_df.head()


# #### 3.c Checking for missing values

# In[ ]:


final_df[final_df.isna().any(axis=1)]


# In[ ]:


#imputing the missing values
final_df.loc[final_df['TEXT'].isna(),'TEXT'] = final_df['Gene'] +' '+final_df['Variation']


# ### 4. Train, validation and test split
# 
# #### 4.a Splitting data into a ratio of 64:20:16 for train, validation and test set

# In[ ]:


y_label = final_df["Class"].values
X_train, X_test, y_train, y_test = train_test_split(final_df, y_label, stratify=y_label, test_size=0.2)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)


# In[ ]:


print('Training data size :', X_train.shape)
print('test data size :', X_test.shape)
print('Validation data size :', X_cv.shape)


# #### 4.b Distribution of ${y}_{i}$  in train, validation and test datasets

# In[ ]:


def dist_class(df, name):
    sns.countplot(df["Class"])
    plt.title("Bar plot of Class using {} data".format(name))
    print("Frequency of each class in {} data".format(name))
    for i in df["Class"].value_counts().index:
        print("Number of observations in class ", i," is : ",df["Class"].value_counts()[i], "(", np.round((df["Class"].value_counts()[i] / len(df["Class"]))*100,2), "%)")


# In[ ]:


dist_class(X_train,"training")


# In[ ]:


dist_class(X_cv,"validation")


# In[ ]:


dist_class(X_test,"test")


# ### 5. Fitting a Random model

# In[ ]:


#user defined function to plot confusion matrix, precision and recall for a ML model
def plot_confusion_recall_precision(cm):
    labels = [1,2,3,4,5,6,7,8,9]
    print("="*30, "Confusion matrix", "="*30)
    plt.figure(figsize=(16,8))
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    
    precision_matrix =(cm/cm.sum(axis=0))
    print("="*30, "Precision matrix (columm sum=1)", "="*30)
    plt.figure(figsize=(16,8))
    sns.heatmap(precision_matrix, annot=True, cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    
    recall_matrix =(((cm.T)/(cm.sum(axis=1))).T)
    print("="*30, "Recall matrix (row sum=1)", "="*30)
    plt.figure(figsize=(16,8))
    sns.heatmap(recall_matrix, annot=True, cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()


# In[ ]:


test_len = X_test.shape[0]
cv_len = X_cv.shape[0]
cv_y_pred = np.zeros((cv_len,9))
for i in range(cv_len):
    rand_probs = np.random.rand(1,9)
    cv_y_pred[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on validation data using Random Model",log_loss(y_cv,cv_y_pred, eps=1e-15))
test_y_pred = np.zeros((test_len,9))
for i in range(test_len):
    rand_probs = np.random.rand(1,9)
    test_y_pred[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on Test Data using Random Model",log_loss(y_test,test_y_pred, eps=1e-15))
y_pred = np.argmax(test_y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred+1)
plot_confusion_recall_precision(cm)


# ## 6. Univariate Analysis

# In[ ]:


#user defined functions to fet feature representations
def get_column_fea_dict(alpha, column, df):
    freq = X_train[column].value_counts()
    column_dict = dict()
    for i, denominator in freq.items():
        vec = []
        for k in range(1,10):
            subset = X_train.loc[(X_train['Class']==k) & (X_train[column]==i)]
            vec.append((subset.shape[0] + alpha*10)/ (denominator + 90*alpha))
        column_dict[i]=vec
    return column_dict
def get_column_feature(alpha, column, df):
    column_dict = get_column_fea_dict(alpha, column, df)
    freq = X_train[column].value_counts()
    column_fea = []
    for index, row in df.iterrows():
        if row[column] in dict(freq).keys():
            column_fea.append(column_dict[row[column]])
        else:
            column_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])
    return column_fea


# ### 6.a Univariate Analysis of Gene feature
# 
# #### 6.a.1 Number of uniques genes and their distribution

# In[ ]:


print('Number of Unique Genes :', X_train["Gene"].nunique())
freq_genes = X_train['Gene'].value_counts()
print("Top 10 genes with highest frequency")
freq_genes.head(10)


# #### 6.a.2 Distribution of Gene categories

# In[ ]:


plt.figure(figsize = (20,8))
sns.countplot(freq_genes)
plt.xticks(rotation = 90)
plt.xlabel('Index of a Gene based on their decreasing order of frequency')
plt.title('Bar plot of most oftenly occuring Genes')


# #### 6.a.3 Featurization of Gene feature
# 
# Two ways to do this
# - One hot encoding
# - Response coding
# 
# #### 6.a.3.1 Featurization of Gene feature using Response coding

# In[ ]:


alpha = 1
train_gene_feat_resp_coding = np.array(get_column_feature(alpha, "Gene", X_train))
val_gene_feat_resp_coding = np.array(get_column_feature(alpha, "Gene", X_cv))
test_gene_feat_resp_coding = np.array(get_column_feature(alpha, "Gene", X_test))


# In[ ]:


print("shape of training gene feature after response coding :", train_gene_feat_resp_coding.shape)


# #### 6.a.3.2 Featurization of Gene feature using one hot encoding

# In[ ]:


gene_vectorizer = CountVectorizer()
train_gene_feat_onehot_en = gene_vectorizer.fit_transform(X_train['Gene'])
val_gene_feat_onehot_en = gene_vectorizer.transform(X_cv['Gene'])
test_gene_feat_onehot_en = gene_vectorizer.transform(X_test['Gene'])


# In[ ]:


gene_vectorizer.get_feature_names()


# In[ ]:


print("shape of training gene feature after one hot encoding :", train_gene_feat_onehot_en.shape)


# ### 6.a.4 Checking significance of Gene feature in predicting ${y}_{i}$ 
# 
# #### 6.a.4.1 Checking through Logistic Regresion model

# In[ ]:


alpha = [10 ** x for x in range(-5, 1)]
val_log_loss_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(train_gene_feat_onehot_en, y_train)
    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
    calib_clf.fit(train_gene_feat_onehot_en, y_train)
    y_pred = calib_clf.predict_proba(val_gene_feat_onehot_en)
    val_log_loss_array.append(log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
    print('For alpha = ', i, "The log loss is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
plt.plot(alpha, val_log_loss_array)
for i, logloss in enumerate(np.round(val_log_loss_array,3)):
    plt.annotate((alpha[i],np.round(logloss,3)), (alpha[i],logloss))
plt.grid()
plt.title("Validation log loss for different values of alpha")
plt.xlabel("Alpha")
plt.ylabel("Log loss")


# In[ ]:


best_alpha = np.argmin(val_log_loss_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_gene_feat_onehot_en, y_train)
calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
calib_clf.fit(train_gene_feat_onehot_en, y_train)

y_pred = calib_clf.predict_proba(train_gene_feat_onehot_en)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is : ",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(val_gene_feat_onehot_en)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is: ",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(test_gene_feat_onehot_en)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is : ",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))


# #### 6.a.4.2 Checking stability of Gene feature across all the datasets

# In[ ]:


print("Number of observations in test and validation datasets covered by the unique ", X_train["Gene"].nunique(), " genes in train dataset")

test_cover=X_test[X_test['Gene'].isin(list(X_train['Gene'].unique()))].shape[0]
validation_cover=X_cv[X_cv['Gene'].isin(list(X_train['Gene'].unique()))].shape[0]

print('In test data',test_cover, 'out of',X_test.shape[0], ":",(test_cover/X_test.shape[0])*100)
print('2. In cross validation data',validation_cover, 'out of ',X_cv.shape[0],":" ,(validation_cover/X_cv.shape[0])*100)


# ### 6.b Univariate analysis of Variation feature
# 
# #### 6.b.1 Number of uniques variations and their distribution

# In[ ]:


print('Number of Unique Variations :', X_train["Variation"].nunique())
freq_variations = X_train['Variation'].value_counts()
print("Top 10 variations with highest frequency")
freq_variations.head(10)


# #### 6.b.2 Distribution of Variation categories

# In[ ]:


total_variations = sum(freq_variations.values)
fraction = freq_variations.values/total_variations
plt.plot(fraction, label="Histrogram of Variations")
plt.xlabel('Index of a variations based on their decreasing order of frequency')
plt.ylabel('Frequency')
plt.legend()
plt.grid()


# #### 6.b.3 Featurization of Variation feature
# 
# Two ways to do this
# - One hot encoding
# - Response coding
# 
# #### 6.b.3.1 Featurization of Variation feature using Response coding

# In[ ]:


alpha = 1
train_variation_feat_resp_coding = np.array(get_column_feature(alpha, "Variation", X_train))
val_variation_feat_resp_coding = np.array(get_column_feature(alpha, "Variation", X_cv))
test_variation_feat_resp_coding = np.array(get_column_feature(alpha, "Variation", X_test))


# In[ ]:


print("shape of training variation feature after response coding :", train_variation_feat_resp_coding.shape)


# #### 6.b.3.2 Featurization of Variation feature using one hot encoding

# In[ ]:


variation_vectorizer = CountVectorizer()
train_variation_feat_onehot_en = variation_vectorizer.fit_transform(X_train['Variation'])
val_variation_feat_onehot_en = variation_vectorizer.transform(X_cv['Variation'])
test_variation_feat_onehot_en = variation_vectorizer.transform(X_test['Variation'])


# In[ ]:


variation_vectorizer.get_feature_names()


# In[ ]:


print("shape of training varaition feature after one hot encoding :", train_variation_feat_onehot_en.shape)


# ### 6.b.4 Checking significance of Variation feature in predicting ${y}_{i}$ 
# 
# #### 6.b.4.1 Checking through Logistic Regresion model

# In[ ]:


alpha = [10 ** x for x in range(-5, 1)]
val_log_loss_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(train_variation_feat_onehot_en, y_train)
    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
    calib_clf.fit(train_variation_feat_onehot_en, y_train)
    y_pred = calib_clf.predict_proba(val_variation_feat_onehot_en)
    val_log_loss_array.append(log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
    print('For alpha = ', i, "The log loss is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
plt.plot(alpha, val_log_loss_array)
for i, logloss in enumerate(np.round(val_log_loss_array,3)):
    plt.annotate((alpha[i],np.round(logloss,3)), (alpha[i],logloss))
plt.grid()
plt.title("Validation log loss for different values of alpha")
plt.xlabel("Alpha")
plt.ylabel("Log loss")


# In[ ]:


best_alpha = np.argmin(val_log_loss_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_variation_feat_onehot_en, y_train)
calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
calib_clf.fit(train_variation_feat_onehot_en, y_train)

y_pred = calib_clf.predict_proba(train_variation_feat_onehot_en)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(val_variation_feat_onehot_en)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(test_variation_feat_onehot_en)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))


# #### 6.b.4.2 Checking stability of Variation feature across all the datasets

# In[ ]:


print("Number of observations in test and validation datasets covered by the unique ", X_train["Variation"].nunique(), " variations in train dataset")

test_cover=X_test[X_test['Variation'].isin(list(X_train['Variation'].unique()))].shape[0]
validation_cover=X_cv[X_cv['Variation'].isin(list(X_train['Variation'].unique()))].shape[0]

print('In test data',test_cover, 'out of',X_test.shape[0], ":",(test_cover/X_test.shape[0])*100)
print('2. In cross validation data',validation_cover, 'out of ',X_cv.shape[0],":" ,(validation_cover/X_cv.shape[0])*100)


# ### 6.c Univariate Analysis of Text feature
# 
# #### 6.c.1 Number of uniques words and their distribution

# In[ ]:


#using Bag of words approach
text_vectorizer = CountVectorizer(min_df=3)
train_text_feat_onehot_en = text_vectorizer.fit_transform(X_train['TEXT'])
train_text_features= text_vectorizer.get_feature_names()
train_text_feat_counts = train_text_feat_onehot_en.sum(axis=0).A1
text_feat_dict = dict(zip(list(train_text_features),train_text_feat_counts))
print("Total number of unique words in TEXT feature of training data :", len(train_text_features))


# #### 6.c.2 Featurization of Text feature
# #### 6.c.2.1 Featurization of Text feature using Response coding

# In[ ]:


def get_word_count_dictionary(df_cls):
    dic = defaultdict(int)
    for index, row in df_cls.iterrows():
        for word in row['TEXT'].split():
            dic[word] +=1
    return dic


# In[ ]:


def get_text_resp_coding(df):
    text_feat_resp_coding = np.zeros((df.shape[0],9))
    for i in range(0,9):
        row_index = 0
        for index, row in df.iterrows():
            total_prob = 0
            for word in row['TEXT'].split():
                total_prob += math.log(((dic_list[i].get(word,0)+10 )/(total_dic.get(word,0)+90)))
            text_feat_resp_coding[row_index][i] = math.exp(total_prob/len(row['TEXT'].split()))
            row_index += 1
    return text_feat_resp_coding


# In[ ]:


dic_list = []
for i in range(1,10):
    subset_cls = X_train[X_train['Class']==i]
    dic_list.append(get_word_count_dictionary(subset_cls))
total_dic = get_word_count_dictionary(X_train)


# In[ ]:


train_text_feat_resp_coding  = get_text_resp_coding(X_train)
val_text_feat_resp_coding  = get_text_resp_coding(X_cv)
test_text_feat_resp_coding  = get_text_resp_coding(X_test)


# In[ ]:


train_text_feat_resp_coding = (train_text_feat_resp_coding.T/train_text_feat_resp_coding.sum(axis=1)).T
val_text_feat_resp_coding = (val_text_feat_resp_coding.T/val_text_feat_resp_coding.sum(axis=1)).T
test_text_feat_resp_coding = (test_text_feat_resp_coding.T/test_text_feat_resp_coding.sum(axis=1)).T


# #### 6.c.2.2 Featurization of Text feature using one hot encoding

# In[ ]:


train_text_feat_onehot_en = normalize(train_text_feat_onehot_en, axis=0)
test_text_feat_onehot_en = text_vectorizer.transform(X_test['TEXT'])
test_text_feat_onehot_en = normalize(test_text_feat_onehot_en, axis=0)
val_text_feat_onehot_en = text_vectorizer.transform(X_cv['TEXT'])
val_text_feat_onehot_en = normalize(val_text_feat_onehot_en, axis=0)


# In[ ]:


sorted_text_feat_dict = dict(sorted(text_feat_dict.items(), key=lambda x: x[1] , reverse=True))
sorted_text_occur = np.array(list(sorted_text_feat_dict.values()))


# In[ ]:


# Number of words to a given frequency
print(Counter(sorted_text_occur))


# ### 6.c.3 Checking significance of Text feature in predicting ${y}_{i}$ 
# 
# #### 6.c.3.1 Checking through Logistic Regresion model

# In[ ]:


alpha = [10 ** x for x in range(-5, 1)]
val_log_loss_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(train_text_feat_onehot_en, y_train)
    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
    calib_clf.fit(train_text_feat_onehot_en, y_train)
    y_pred = calib_clf.predict_proba(val_text_feat_onehot_en)
    val_log_loss_array.append(log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
    print('For alpha = ', i, "The log loss is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
plt.plot(alpha, val_log_loss_array)
for i, logloss in enumerate(np.round(val_log_loss_array,3)):
    plt.annotate((alpha[i],np.round(logloss,3)), (alpha[i],logloss))
plt.grid()
plt.title("Validation log loss for different values of alpha")
plt.xlabel("Alpha")
plt.ylabel("Log loss")


# In[ ]:


best_alpha = np.argmin(val_log_loss_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_text_feat_onehot_en, y_train)
calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
calib_clf.fit(train_text_feat_onehot_en, y_train)

y_pred = calib_clf.predict_proba(train_text_feat_onehot_en)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(val_text_feat_onehot_en)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(test_text_feat_onehot_en)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))


# #### 6.c.3.2 Checking stability of Text feature across all the datasets

# In[ ]:


def get_common_word(df):
    text_vectorizer = CountVectorizer(min_df=3)
    df_text_feat_onehot_en = text_vectorizer.fit_transform(df['TEXT'])
    df_text_features = text_vectorizer.get_feature_names()

    df_text_feat_counts = df_text_feat_onehot_en.sum(axis=0).A1
    df_text_fea_dict = dict(zip(list(df_text_features),df_text_feat_counts))
    df_len = len(set(df_text_features))
    common_words_len = len(set(train_text_features) & set(df_text_features))
    return df_len,common_words_len


# In[ ]:


cv_len,common_words_len = get_common_word(X_cv)
print(np.round((common_words_len/cv_len)*100, 3), "% of word of validation appeared in train data")
test_len,common_words_len = get_common_word(X_test)
print(np.round((common_words_len/test_len)*100, 3), "% of word of test data appeared in train data")


# ## 7. Classical Machine learning models

# In[ ]:


#user defined function to calculate confusion matrix, precision and recall and also to plot
def predict_and_plot_confusion_recall_precision(X_train, y_train,X_test, y_test, classifier):
    classifier.fit(X_train, y_train)
    calib_clf = CalibratedClassifierCV(classifier, method="sigmoid")
    calib_clf.fit(X_train, y_train)
    y_pred = calib_clf.predict(X_test)

    # for calculating log_loss we willl provide the array of probabilities belongs to each class
    print("Log loss :",log_loss(y_test, calib_clf.predict_proba(X_test)))
    # calculating the number of data points that are misclassified
    print("Number of mis-classified points :", np.count_nonzero((y_pred- y_test))/y_test.shape[0])
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_recall_precision(cm)


# In[ ]:


#user defined function to calculate log loss
def calculate_log_loss(X_train, y_train,X_test, y_test, classifier):
    classifier.fit(X_train, y_train)
    calib_clf = CalibratedClassifierCV(classifier, method="sigmoid")
    calib_clf.fit(X_train, y_train)
    calib_clf_probs = calib_clf.predict_proba(X_test)
    return log_loss(y_test, calib_clf_probs, eps=1e-15)


# In[ ]:


# user defined function to get important feature 
def get_impfeature_names(indices, text, gene, var, no_features):
    gene_count_vectorizer = CountVectorizer()
    var_count_vectorizer = CountVectorizer()
    text_count_vectorizer = CountVectorizer(min_df=3)
    
    gene_vec_onehot = gene_count_vectorizer.fit(X_train['Gene'])
    var_vec_onehot  = var_count_vectorizer.fit(X_train['Variation'])
    text_vec_onehot = text_count_vectorizer.fit(X_train['TEXT'])
    
    feat1_len = len(gene_count_vectorizer.get_feature_names())
    feat2_len = len(var_count_vectorizer.get_feature_names())
    
    word_present = 0
    for i,v in enumerate(indices):
        if (v < feat1_len):
            word = gene_count_vectorizer.get_feature_names()[v]
            flag = True if word == gene else False
            if flag:
                word_present += 1
                print(i, "Gene feature [{}] present in test data point [{}]".format(word,flag))
        elif (v < feat1_len+feat2_len):
            word = var_count_vectorizer.get_feature_names()[v-(feat1_len)]
            flag = True if word == var else False
            if flag:
                word_present += 1
                print(i, "variation feature [{}] present in test data point [{}]".format(word,flag))
        else:
            word = text_count_vectorizer.get_feature_names()[v-(feat1_len+feat2_len)]
            flag = True if word in text.split() else False
            if flag:
                word_present += 1
                print(i, "Text feature [{}] present in test data point [{}]".format(word,flag))

    print("Out of the top ",no_features," features ", word_present, "are present in query point")


# ### Using all the three features together

# In[ ]:


train_gene_and_var_onehot_en = hstack((train_gene_feat_onehot_en,train_variation_feat_onehot_en))
val_gene_and_var_onehot_en = hstack((val_gene_feat_onehot_en,val_variation_feat_onehot_en))
test_gene_and_var_onehot_en = hstack((test_gene_feat_onehot_en,test_variation_feat_onehot_en))


# In[ ]:


X_train_onehotCoding = hstack((train_gene_and_var_onehot_en, train_text_feat_onehot_en)).tocsr()
y_train = np.array(X_train['Class'])
X_test_onehotCoding = hstack((test_gene_and_var_onehot_en, test_text_feat_onehot_en)).tocsr()
y_test = np.array(X_test['Class'])
X_cv_onehotCoding = hstack((val_gene_and_var_onehot_en, val_text_feat_onehot_en)).tocsr()
y_cv = np.array(X_cv['Class'])


# In[ ]:


train_gene_and_var_responseCoding = np.hstack((train_gene_feat_resp_coding,train_variation_feat_resp_coding))
test_gene_and_var_responseCoding = np.hstack((test_gene_feat_resp_coding,test_variation_feat_resp_coding))
val_gene_and_var_responseCoding = np.hstack((val_gene_feat_resp_coding,val_variation_feat_resp_coding))

X_train_responseCoding = np.hstack((train_gene_and_var_responseCoding, train_text_feat_resp_coding))
X_test_responseCoding = np.hstack((test_gene_and_var_responseCoding, test_text_feat_resp_coding))
X_cv_responseCoding = np.hstack((val_gene_and_var_responseCoding, val_text_feat_resp_coding))


# In[ ]:


print("Overview about one hot encoding features : ")
print("Size of one hot encoded train data : ", X_train_onehotCoding.shape)
print("Size of one hot encoded test data : ", X_test_onehotCoding.shape)
print("Size of one hot encoded validation data : ", X_cv_onehotCoding.shape)


# In[ ]:


print(" Overview about response coded features :")
print("Size of response coded train data : ", X_train_responseCoding.shape)
print("Size of response coded test data : ", X_test_responseCoding.shape)
print("Size of response coded validation data ", X_cv_responseCoding.shape)


# ## Base Line Model
# 
# ### 7.1 Naive Bayes 
# 
# #### 7.1.1 Hyperparameter tuning

# In[ ]:


alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]
val_log_loss_array = []
for i in alpha:
    print("for alpha =", i)
    clf = MultinomialNB(alpha=i)
    clf.fit(X_train_onehotCoding, y_train)
    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
    calib_clf.fit(X_train_onehotCoding, y_train)
    calib_clf_probs = calib_clf.predict_proba(X_cv_onehotCoding)
    val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))
    print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 
plt.plot(np.log10(alpha), val_log_loss_array)
for i, logloss in enumerate(np.round(val_log_loss_array,3)):
    plt.annotate((alpha[i],str(logloss)), (np.log10(alpha[i]),logloss))
plt.grid()
plt.xticks(np.log10(alpha))
plt.title("Validation log loss for different values of alpha")
plt.xlabel("Alpha")
plt.ylabel("Log loss")

best_alpha = np.argmin(val_log_loss_array)
clf = MultinomialNB(alpha=i)
clf.fit(X_train_onehotCoding, y_train)
calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
calib_clf.fit(X_train_onehotCoding, y_train)

y_pred = calib_clf.predict_proba(X_train_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_cv_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_test_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))


# #### 7.1.2 Testing the model agianst best hyperparameters 

# In[ ]:


best_alpha = np.argmin(val_log_loss_array)
clf = MultinomialNB(alpha=alpha[best_alpha])
predict_and_plot_confusion_recall_precision(X_train_onehotCoding, y_train,X_cv_onehotCoding, y_cv, clf)


# #### 7.1.3 Feature importance check with correctly classified point

# In[ ]:


random_index = 1
no_feature = 100
pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))
print("Actual Class :", y_test[random_index])
indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]
print("="*50)
get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)


# #### 7.1.4 Feature importance check with incorrectly classified point

# In[ ]:


random_index = 100
no_feature = 100
pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))
print("Actual Class :", y_test[random_index])
indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]
print("="*50)
get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)


# ### 7.2 kNN 
# 
# #### 7.2.1 Hyperparameter tuning 

# In[ ]:


alpha = [5, 11, 15, 21, 31, 41, 51, 99]
val_log_loss_array = []
for i in alpha:
    print("for alpha =", i)
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train_responseCoding, y_train)
    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
    calib_clf.fit(X_train_responseCoding, y_train)
    calib_clf_probs = calib_clf.predict_proba(X_cv_responseCoding)
    val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))
    print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 
plt.plot(alpha, val_log_loss_array)
for i, logloss in enumerate(np.round(val_log_loss_array,3)):
    plt.annotate((alpha[i],str(logloss)), (alpha[i],logloss))
plt.grid()
plt.xticks(alpha)
plt.title("Validation log loss for different values of alpha")
plt.xlabel("Alpha")
plt.ylabel("Log loss")

best_alpha = np.argmin(val_log_loss_array)
clf = KNeighborsClassifier(n_neighbors=i)
clf.fit(X_train_responseCoding, y_train)
calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
calib_clf.fit(X_train_responseCoding, y_train)

y_pred = calib_clf.predict_proba(X_train_responseCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_cv_responseCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_test_responseCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))


# #### 7.2.2 Testing the model agianst best hyperparameters 

# In[ ]:


best_alpha = np.argmin(val_log_loss_array)
clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])
predict_and_plot_confusion_recall_precision(X_train_responseCoding, y_train,X_cv_responseCoding, y_cv, clf)


# #### 7.2.3 Feature importance check with correctly classified point

# In[ ]:


random_index = 1
pred_cls = calib_clf.predict(X_test_responseCoding[random_index].reshape(1,-1))
print("Predicted Class :", pred_cls[0])
print("Actual Class :", y_test[random_index])
nearest_neighbors = clf.kneighbors(X_test_responseCoding[random_index].reshape(1, -1), alpha[best_alpha])
print("The ",alpha[best_alpha]," nearest neighbours of the random test point belongs to classes",y_train[nearest_neighbors[1][0]])
print("Fequency of nearest points :",Counter(y_train[nearest_neighbors[1][0]]))


# #### 7.2.4 Feature importance check with incorrectly classified point

# In[ ]:


random_index = 100
pred_cls = calib_clf.predict(X_test_responseCoding[random_index].reshape(1,-1))
print("Predicted Class :", pred_cls[0])
print("Actual Class :", y_test[random_index])
nearest_neighbors = clf.kneighbors(X_test_responseCoding[random_index].reshape(1, -1), alpha[best_alpha])
print("The ",alpha[best_alpha]," nearest neighbours of the random test point belongs to classes",y_train[nearest_neighbors[1][0]])
print("Fequency of nearest points :",Counter(y_train[nearest_neighbors[1][0]]))


# ## 7.3 Logistic Regression
# 
# ### 7.3.a LR with class balancing
# 
# #### 7.3.a.1 Hyperparameter tuning

# In[ ]:


alpha = [10 ** x for x in range(-6, 3)]
val_log_loss_array = []
for i in alpha:
    print("for alpha =", i)
    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(X_train_onehotCoding, y_train)
    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
    calib_clf.fit(X_train_onehotCoding, y_train)
    calib_clf_probs = calib_clf.predict_proba(X_cv_onehotCoding)
    val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))
    print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 
plt.plot(alpha, val_log_loss_array)
for i, logloss in enumerate(np.round(val_log_loss_array,3)):
    plt.annotate((alpha[i],str(logloss)), (alpha[i],logloss))
plt.grid()
plt.xticks(alpha)
plt.title("Validation log loss for different values of alpha")
plt.xlabel("Alpha")
plt.ylabel("Log loss")

best_alpha = np.argmin(val_log_loss_array)
clf = SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(X_train_onehotCoding, y_train)
calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
calib_clf.fit(X_train_onehotCoding, y_train)

y_pred = calib_clf.predict_proba(X_train_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_cv_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_test_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))


# #### 7.3.a.2 Testing the model agianst best hyperparameters 

# In[ ]:


best_alpha = np.argmin(val_log_loss_array)
clf = SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
predict_and_plot_confusion_recall_precision(X_train_onehotCoding, y_train,X_cv_onehotCoding, y_cv, clf)


# #### 7.3.a.3 Feature importance check with correctly classified point

# In[ ]:


random_index = 1
no_feature = 500
pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))
print("Actual Class :", y_test[random_index])
indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]
print("="*50)
get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)


# #### 7.3.a.4 Feature importance check with incorrectly classified point

# In[ ]:


random_index = 100
no_feature = 500
pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))
print("Actual Class :", y_test[random_index])
indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]
print("="*50)
get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)


# ### 7.3.b LR without Class balancing

# In[ ]:


alpha = [10 ** x for x in range(-6, 1)]
val_log_loss_array = []
for i in alpha:
    print("for alpha =", i)
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(X_train_onehotCoding, y_train)
    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
    calib_clf.fit(X_train_onehotCoding, y_train)
    calib_clf_probs = calib_clf.predict_proba(X_cv_onehotCoding)
    val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))
    print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 
plt.plot(alpha, val_log_loss_array)
for i, logloss in enumerate(np.round(val_log_loss_array,3)):
    plt.annotate((alpha[i],str(logloss)), (alpha[i],logloss))
plt.grid()
plt.xticks(alpha)
plt.title("Validation log loss for different values of alpha")
plt.xlabel("Alpha")
plt.ylabel("Log loss")

best_alpha = np.argmin(val_log_loss_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(X_train_onehotCoding, y_train)
calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
calib_clf.fit(X_train_onehotCoding, y_train)

y_pred = calib_clf.predict_proba(X_train_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_cv_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_test_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))


# #### 7.3.b.1 Testing the model agianst best hyperparameters 

# In[ ]:


best_alpha = np.argmin(val_log_loss_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
predict_and_plot_confusion_recall_precision(X_train_onehotCoding, y_train,X_cv_onehotCoding, y_cv, clf)


# #### 7.3.b.2 Feature importance check with correctly classified point

# In[ ]:


random_index = 1
no_feature = 500
pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))
print("Actual Class :", y_test[random_index])
indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]
print("="*50)
get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)


# #### 7.3.b.3 Feature importance check with incorrectly classified point

# In[ ]:


random_index = 100
no_feature = 500
pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))
print("Actual Class :", y_test[random_index])
indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]
print("="*50)
get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)


# ### 7.4 Linear SVM
# 
# #### 7.4.1 Hyperparameter tuning

# In[ ]:


alpha = [10 ** x for x in range(-5, 3)]
val_log_loss_array = []
for i in alpha:
    print("for alpha =", i)
    clf = SGDClassifier(class_weight='balanced',alpha=i, penalty='l2', loss='hinge', random_state=42)
    clf.fit(X_train_onehotCoding, y_train)
    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
    calib_clf.fit(X_train_onehotCoding, y_train)
    calib_clf_probs = calib_clf.predict_proba(X_cv_onehotCoding)
    val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))
    print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 
plt.plot(alpha, val_log_loss_array)
for i, logloss in enumerate(np.round(val_log_loss_array,3)):
    plt.annotate((alpha[i],str(logloss)), (alpha[i],logloss))
plt.grid()
plt.xticks(alpha)
plt.title("Validation log loss for different values of alpha")
plt.xlabel("Alpha")
plt.ylabel("Log loss")

best_alpha = np.argmin(val_log_loss_array)
clf = SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)
clf.fit(X_train_onehotCoding, y_train)
calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
calib_clf.fit(X_train_onehotCoding, y_train)

y_pred = calib_clf.predict_proba(X_train_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_cv_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_test_onehotCoding)
print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))


# #### 7.4.2 Testing the model agianst best hyperparameters 

# In[ ]:


best_alpha = np.argmin(val_log_loss_array)
clf = SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)
predict_and_plot_confusion_recall_precision(X_train_onehotCoding, y_train,X_cv_onehotCoding, y_cv, clf)


# #### 7.4.3 Feature importance check with correctly classified point

# In[ ]:


random_index = 1
no_feature = 500
pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))
print("Actual Class :", y_test[random_index])
indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]
print("="*50)
get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)


# #### 7.4.4 Feature importance check with incorrectly classified point

# In[ ]:


random_index = 100
no_feature = 500
pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))
print("Actual Class :", y_test[random_index])
indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]
print("="*50)
get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)


# ## 7.5 Random Forest
# 
# ### 7.5.a.1 Hyperparamter tuning (with One hot encoding)

# In[ ]:


alpha = [100,200,500,1000,2000]
max_depth = [5, 10]
val_log_loss_array = []
for i in alpha:
    for j in max_depth:
        print("for n_estimators =", i,"and max depth = ", j)
        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42)
        clf.fit(X_train_onehotCoding, y_train)
        calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
        calib_clf.fit(X_train_onehotCoding, y_train)
        calib_clf_probs = calib_clf.predict_proba(X_cv_onehotCoding)
        val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))
        print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 

best_alpha = np.argmin(val_log_loss_array)
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42)
clf.fit(X_train_onehotCoding, y_train)
calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
calib_clf.fit(X_train_onehotCoding, y_train)


# In[ ]:


y_pred = calib_clf.predict_proba(X_train_onehotCoding)
print('For the best alpha of alpha = ', alpha[int(best_alpha/2)], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_cv_onehotCoding)
print('For the best alpha of alpha = ', alpha[int(best_alpha/2)], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_test_onehotCoding)
print('For the best alpha of alpha = ', alpha[int(best_alpha/2)], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))


# #### 7.5.a.2 Testing the model agianst best hyperparameters using one hot encoding

# In[ ]:


clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42)
predict_and_plot_confusion_recall_precision(X_train_onehotCoding, y_train,X_cv_onehotCoding, y_cv, clf)


# #### 7.5.a.3 Feature importance check with correctly classified point

# In[ ]:


random_index = 1
no_feature = 100
pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))
print("Actual Class :", y_test[random_index])
indices=np.argsort(-clf.feature_importances_)
print("="*50)
get_impfeature_names(indices[:no_feature], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)


# #### 7.5.a.4 Feature importance check with incorrectly classified point

# In[ ]:


random_index = 100
no_feature = 100
pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))
print("Actual Class :", y_test[random_index])
indices=np.argsort(-clf.feature_importances_)
print("="*50)
get_impfeature_names(indices[:no_feature], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)


# ### 7.5.b.1 Hyper paramter tuning (with Response coding)

# In[ ]:


alpha = [10,50,100,200,500,1000]
max_depth = [2,3,5,10]
val_log_loss_array = []
for i in alpha:
    for j in max_depth:
        print("for n_estimators =", i,"and max depth = ", j)
        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42)
        clf.fit(X_train_responseCoding, y_train)
        calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
        calib_clf.fit(X_train_responseCoding, y_train)
        calib_clf_probs = calib_clf.predict_proba(X_cv_responseCoding)
        val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))
        print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 

best_alpha = np.argmin(val_log_loss_array)
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_depth=max_depth[int(best_alpha%4)], random_state=42)
clf.fit(X_train_responseCoding, y_train)
calib_clf = CalibratedClassifierCV(clf, method="sigmoid")
calib_clf.fit(X_train_responseCoding, y_train)

y_pred = calib_clf.predict_proba(X_train_responseCoding)
print('For the best alpha of alpha = ', alpha[int(best_alpha/4)], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_cv_responseCoding)
print('For the best alpha of alpha = ', alpha[int(best_alpha/4)], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))
y_pred = calib_clf.predict_proba(X_test_responseCoding)
print('For the best alpha of alpha = ', alpha[int(best_alpha/4)], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))


# #### 7.5.b.2 Testing the model agianst best hyperparameters using response coding

# In[ ]:


clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_depth=max_depth[int(best_alpha%4)], random_state=42)
predict_and_plot_confusion_recall_precision(X_train_responseCoding, y_train,X_cv_responseCoding, y_cv, clf)


# #### 7.5.b.3 Feature importance check with correctly classified point

# In[ ]:


random_index = 1
pred_cls = calib_clf.predict(X_test_responseCoding[random_index].reshape(1,-1))
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_responseCoding[random_index].reshape(1,-1)),4))
print("Actual Class :", y_test[random_index])
indices = np.argsort(-clf.feature_importances_)
print("="*50)
for i in indices:
    if i<9:
        print("Gene is important feature")
    elif i<18:
        print("Variation is important feature")
    else:
        print("Text is important feature")


# #### 7.5.b.4 Feature importance check with incorrectly classified point

# In[ ]:


random_index = 100
pred_cls = calib_clf.predict(X_test_responseCoding[random_index].reshape(1,-1))
print("Predicted Class :", pred_cls[0])
print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_responseCoding[random_index].reshape(1,-1)),4))
print("Actual Class :", y_test[random_index])
indices = np.argsort(-clf.feature_importances_)
print("="*50)
for i in indices:
    if i<9:
        print("Gene is important feature")
    elif i<18:
        print("Variation is important feature")
    else:
        print("Text is important feature")


# ## 8. Stacking the models
# 
# ### 8.1 Testing the model for best hyperparameters 

# In[ ]:


clf1 = SGDClassifier(alpha=0.001, penalty='l2', loss='log', class_weight='balanced', random_state=42)
clf1.fit(X_train_onehotCoding, y_train)
calib_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")

clf2 = SGDClassifier(alpha=1, penalty='l2', loss='hinge', class_weight='balanced', random_state=42)
clf2.fit(X_train_onehotCoding, y_train)
calib_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")


clf3 = MultinomialNB(alpha=0.001)
clf3.fit(X_train_onehotCoding, y_train)
calib_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")

calib_clf1.fit(X_train_onehotCoding, y_train)
print("Logistic Regression :  Log Loss: %0.2f" % (log_loss(y_cv, calib_clf1.predict_proba(X_cv_onehotCoding))))
calib_clf2.fit(X_train_onehotCoding, y_train)
print("SVM : Log Loss: %0.2f" % (log_loss(y_cv, calib_clf2.predict_proba(X_cv_onehotCoding))))
calib_clf3.fit(X_train_onehotCoding, y_train)
print("Naive Bayes : Log Loss: %0.2f" % (log_loss(y_cv, calib_clf3.predict_proba(X_cv_onehotCoding))))
print("="*50)
alpha = [0.0001,0.001,0.01,0.1,1,10] 
best_alpha = 999
for i in alpha:
    lr = LogisticRegression(C=i)
    stack_clf = StackingClassifier(classifiers=[calib_clf1, calib_clf2, calib_clf3], meta_classifier=lr, use_probas=True)
    stack_clf.fit(X_train_onehotCoding, y_train)
    print("Stacking Classifer : for the value of alpha: %f Log Loss: %0.3f" % (i, log_loss(y_cv, stack_clf.predict_proba(X_cv_onehotCoding))))
    logloss =log_loss(y_cv, stack_clf.predict_proba(X_cv_onehotCoding))
    if best_alpha > logloss:
        best_alpha = logloss


# ### 8.2 Testing the model against best hyperparameters 

# In[ ]:


lr = LogisticRegression(C=best_alpha)
stack_clf = StackingClassifier(classifiers=[calib_clf1, calib_clf2, calib_clf3], meta_classifier=lr, use_probas=True)
stack_clf.fit(X_train_onehotCoding, y_train)

logloss = log_loss(y_train, stack_clf.predict_proba(X_train_onehotCoding))
print("Log loss of training data using the stacking classifier :",logloss)

logloss = log_loss(y_cv, stack_clf.predict_proba(X_cv_onehotCoding))
print("Log loss of validation data using the stacking classifier :",logloss)

logloss = log_loss(y_test, stack_clf.predict_proba(X_test_onehotCoding))
print("Log loss of test data using the stacking classifier :",logloss)

print("Number of missclassified point :", np.count_nonzero((stack_clf.predict(X_test_onehotCoding)- y_test))/y_test.shape[0])
cm = confusion_matrix(y_test, stack_clf.predict(X_test_onehotCoding))
plot_confusion_recall_precision(cm)


# ### 9. Majority Voting classifier

# In[ ]:


voting_clf = VotingClassifier(estimators=[('lr', calib_clf1), ('svc', calib_clf2), ('rf', calib_clf3)], voting='soft')
voting_clf.fit(X_train_onehotCoding, y_train)
print("Log loss (train) on the VotingClassifier :", log_loss(y_train, voting_clf.predict_proba(X_train_onehotCoding)))
print("Log loss (CV) on the VotingClassifier :", log_loss(y_cv, voting_clf.predict_proba(X_cv_onehotCoding)))
print("Log loss (test) on the VotingClassifier :", log_loss(y_test, voting_clf.predict_proba(X_test_onehotCoding)))
print("Number of missclassified point :", np.count_nonzero((voting_clf.predict(X_test_onehotCoding)- y_test))/y_test.shape[0])
cm = confusion_matrix(y_test, voting_clf.predict(X_test_onehotCoding))
plot_confusion_recall_precision(cm)

