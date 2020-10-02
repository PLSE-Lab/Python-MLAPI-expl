#!/usr/bin/env python
# coding: utf-8

# **Objective:**
#  
#  This kernel is made as a walkthrough for basic NLP analysis and modelling.
#  
# **Table Of Contents:**
# * Importing packages and modules
# * Text data preprocessing
# * Basic EDA with WordCloud and Plotly
# * Machine Learning:
#     * Logistic Regression
#     * Support Vector Machine
#     * Multinomial Naive Bayes
#     * Decision Tree Classifier
#     * AdaBoost Classifier   
#     * Random Forest Classifier
#     * XGBoost Classifier
#     * XGBClassifier Hyperparameter tuning
# * Deep Learning:
#     * Text data preprocessing
#     * Bidirectional LSTM with own Embeddings
#     * Bidirectional LSTM with GloVe

# **If you like the kernel, please upvote it**. Your upvotes will motivate me to code more. Happy Kaggling !

# In[ ]:


import numpy as np 
import pandas as pd
import plotly.plotly as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,f1_score,roc_curve,make_scorer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import os
import scikitplot as skplt
import seaborn as sns
import time
print(os.listdir("../input"))
stopwords=set(stopwords.words('english'))
stemmer=SnowballStemmer('english')
seed=5


# In[ ]:


data=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
sub=pd.read_csv('../input/sample_submission.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


target_count=data.target.value_counts()
trace1=go.Bar(x=target_count.index,
             y=target_count.values,
             name='Target Counts',
             marker=dict(color='rgba(0,255,255,0.5)',
                         line=dict(color='rgb(0,0,0)',width=0.5)),
             text=['Sincere Questions','Insincere Questions'])
layout=go.Layout(title='Bar plot of target counts',
                xaxis=dict(title='Target'),
                yaxis=dict(title='Number of questions'))
plt_data=[trace1]
fig=dict(data=plt_data,layout=layout)
iplot(fig)


# The data is highly imbalanced. For the baseline model, even if predict always 0, we would be correct (1225312/1306122)*100 = 93.81 percent of time.
# 
# Lets clean the data.

# In[ ]:


target1=data[data['target']==1]
target0=data[data['target']==0]
sampled_size=target1.shape[0]
sampled_target0=target0.sample(sampled_size,random_state=seed)
new_data=pd.concat([target1,sampled_target0],axis=0)
#Shuffling the data
new_data=new_data.sample(frac=1,random_state=seed).reset_index(drop=True)


# In[ ]:


def filter_text(text):
    tokenized_words=word_tokenize(text)
    filtered_words=[word.lower() for word in tokenized_words if ((word.lower() not in string.punctuation) &
                                                                 (word.lower() not in stopwords))]
    stemmed_words=[stemmer.stem(word) for word in filtered_words]
    return ' '.join(filtered_words)


# Preprocessing for traditional NLP: This includes removing stop words and punctuation, followed by stemming.

# In[ ]:


trad_data=new_data.copy()
trad_data['question_text']=trad_data['question_text'].apply(lambda x: filter_text(x))


# In[ ]:


def plot_wordcloud(text,max_font_size=40,max_words=100):
    plt.figure(figsize=(10,5))
    wordcloud=WordCloud(max_font_size=max_font_size,max_words=max_words,random_state=seed)
    plot=wordcloud.generate(text)
    plt.imshow(plot)
    plt.axis('off')
    plt.show()
    


# In[ ]:


target0=trad_data[trad_data['target']==0].reset_index(drop=True)
target1=trad_data[trad_data['target']==1].reset_index(drop=True)

target0_text=''
target1_text=''
for i in range(target0.shape[0]):
    target0_text+=target0.question_text[i]
for i in range(target1.shape[0]):
    target1_text+=target1.question_text[i]


# In[ ]:


plot_wordcloud(target0_text)


# In[ ]:


plot_wordcloud(target1_text,max_words=200)


# In[ ]:


def plot_top_ngrams(text,ngrams=(1,1),top=10,max_features=10000,color='rgba(0,255,255,0.5)'):
    cv=CountVectorizer(ngram_range=ngrams,max_features=max_features)
    trans_text=cv.fit_transform(text)
    col_sum=trans_text.sum(axis=0)
    word_index=[(word,col_sum[0,idx]) for word,idx in cv.vocabulary_.items()]
    sorted_word_index=sorted(word_index,key=lambda x:x[1],reverse=True)
    top_words_index=sorted_word_index[:top]
    top_words=[element[0] for element in top_words_index]
    counts=[element[1] for element in top_words_index]
    trace1=go.Bar(x=top_words,
                 y=counts,
                 marker=dict(color=color,
                             line=dict(color='rgb(0,0,0)',width=0.5)))
    layout=go.Layout(title='Top ngrams',
                    xaxis=dict(title='Ngrams'),
                    yaxis=dict(title='Counts of words'))
    plot_data=[trace1]
    fig=dict(data=plot_data,layout=layout)
    iplot(fig)
    
    
    
    


# In[ ]:


plot_top_ngrams(target1.question_text,ngrams=(1,1),top=30,color='rgba(128,0,0,0.5)')


# We see that most of the insincere questions start with 'Why'. 

# In[ ]:


plot_top_ngrams(target1.question_text,ngrams=(2,2),top=30)


# In[ ]:


plot_top_ngrams(target1.question_text,ngrams=(3,3),top=30,color='rgba(128,128,128,0.5)')


# In[ ]:


plot_top_ngrams(target0.question_text,ngrams=(1,1),top=30,color='rgba(128,0,0,0.5)')


# We see that most of the sincere questions start with 'What'.

# In[ ]:


plot_top_ngrams(target0.question_text,ngrams=(2,2),top=30)


# In[ ]:


plot_top_ngrams(target0.question_text,ngrams=(3,3),top=30,color='rgba(128,128,128,0.5)')


# In[ ]:


mini_df=trad_data.sample(25000,random_state=seed)
X=mini_df['question_text']
Y=mini_df['target']

train_X,val_X,train_y,val_y=train_test_split(X,Y,test_size=0.2,random_state=seed)


# In[ ]:


cv=CountVectorizer(ngram_range=(1,3),analyzer='word')
train_X_cv=cv.fit_transform(train_X.values)
val_X_cv=cv.transform(val_X.values)


# In[ ]:


tsvd=TruncatedSVD(n_components=50,random_state=seed)
train_X_svd=tsvd.fit_transform(train_X_cv)
val_X_svd=tsvd.transform(val_X_cv)
tsne=TSNE(n_components=2,random_state=seed)
train_X_tsne=tsne.fit_transform(train_X_svd)


# In[ ]:


df=pd.DataFrame()
df['tsne1']=pd.Series(train_X_tsne[:,0])
df['tsne2']=pd.Series(train_X_tsne[:,1])
df['target']=train_y
sns.scatterplot(df['tsne1'],df['tsne2'],hue='target',data=df)
plt.show()


# Evidently, we can say from TSNE plot that the two classes are not trivially differentiaable.

# In[ ]:


def get_model(model,train_X,train_y,val_X):
    model.fit(train_X,train_y)
    pred_probs=model.predict_proba(val_X)
    pred_train=model.predict(train_X)
    pred_val=model.predict(val_X)
    score_train=f1_score(train_y,pred_train)
    score_val=f1_score(val_y,pred_val)
    return pred_probs,pred_train,pred_val,score_train,score_val

def get_confusion_matrix(val_y,pred,title):
    cm=confusion_matrix(val_y,pred)
    plt.figure(figsize=(10,5))
    sns.heatmap(cm,annot=True)
    plt.title(title)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()
    
def get_roc_curve(val_y,pred_probs,title):
    plt.title(title)
    skplt.metrics.plot_roc(val_y,pred_probs)
    
    


# In[ ]:


models=[LogisticRegression(random_state=seed),MultinomialNB(),DecisionTreeClassifier(random_state=seed),
        AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=100,learning_rate=0.1,random_state=seed),
        RandomForestClassifier(n_estimators=100,max_depth=3,random_state=seed),
        XGBClassifier(random_state=seed)]
model_names=['LR','Multinomial NB','DTC','ABC','RFC','XGBC']


# In[ ]:


pred_probs={}
pred_train={}
pred_val={}
score_train={}
score_val={}

for i in range(len(models)):
    pred_probs[model_names[i]],pred_train[model_names[i]],pred_val[model_names[i]],    score_train[model_names[i]],score_val[model_names[i]]=get_model(models[i],train_X_cv,train_y,val_X_cv)
    


scl=StandardScaler()                                                     
train_X_scl_cv=scl.fit_transform(train_X_svd)
val_X_scl_cv=scl.transform(val_X_svd)
pred_probs['SVC'],pred_train['SVC'],pred_val['SVC'],score_train['SVC'],score_val['SVC']=get_model(SVC(probability=True,random_state=seed),
                                                      train_X_scl_cv,train_y,val_X_scl_cv)


# In[ ]:


trace1=go.Bar(x=list(score_train.keys()),
              y=list(score_train.values()),
             name='Training Score with CV',
             marker=dict(color='rgba(0,255,0,0.5)',
                        line=dict(color='rgb(0,0,0)',width=1.5)))
trace2=go.Bar(x=list(score_val.keys()),
              y=list(score_val.values()),
             name='Validation Score with CV',
             marker=dict(color='rgba(255,255,0,0.5)',
                        line=dict(color='rgb(0,0,0)',width=1.5)))

layout=go.Layout(barmode='group',
                title='Scores of Different Models')
plot_data=[trace1,trace2]
fig=dict(data=plot_data,layout=layout)
iplot(fig)


# Clearly, LogisticRegression, Multinomial NB and DecisionTree Classifier overfit the data. AdaBoost, RandomForest and XGB Classifiers do not overfit but their scores are low compared to the former 3 models. Regularization and Hyperparameter tuning can surely help to not overfit in case of initial 3 models and can help the latter 3 models to improve their scores,respectively.

# In[ ]:


for model,probs in pred_probs.items():
    get_roc_curve(val_y,probs,model)
    


# I don't know what is the problem with roc curve for LR. It is printed at the bottom with title ROC Curves. Multinomial NB and Decision Tree Classifier seem to perform best. 

# In[ ]:


for model,pred in pred_val.items():
    get_confusion_matrix(val_y,pred,model)


# From the confusion matrix of Multinomial NB and Decision Tree Classifier, we see that MNB has higher misclassification of 'Sincere Questions' and lower misclassification of 'Insincere Questions' as compared to DTC.
# DTC has large misclassification of 'Insincere Questions'. This is not at all desired. Practically speaking, it is better to misclassify true 'Sincere Question' than to misclassify true 'Insincere Question'. In case of classification of true 'Insincere Question', MNB performs best while RFC performs worst. 

# In[ ]:


tfv=TfidfVectorizer(ngram_range=(1,3),analyzer='word',min_df=3)
train_X_tfv=tfv.fit_transform(train_X.values)
val_X_tfv=tfv.transform(val_X.values)
tsvd_tfv=TruncatedSVD(n_components=50,random_state=seed)
train_X_svd_tfv=tsvd_tfv.fit_transform(train_X_tfv)
val_X_svd_tfv=tsvd_tfv.transform(val_X_tfv)
tsne_tfv=TSNE(n_components=2,random_state=seed)
train_X_tsne_tfv=tsne_tfv.fit_transform(train_X_svd_tfv)


df=pd.DataFrame()
df['tsne1']=pd.Series(train_X_tsne_tfv[:,0])
df['tsne2']=pd.Series(train_X_tsne_tfv[:,1])
df['target']=train_y
sns.scatterplot(df['tsne1'],df['tsne2'],hue='target',data=df)
plt.show()


# In[ ]:


pred_probs_tfv={}
pred_train_tfv={}
pred_val_tfv={}
score_train_tfv={}
score_val_tfv={}

for i in range(len(models)):
    pred_probs_tfv[model_names[i]],pred_train_tfv[model_names[i]],pred_val_tfv[model_names[i]],    score_train_tfv[model_names[i]],score_val_tfv[model_names[i]]=get_model(models[i],train_X_tfv,train_y,val_X_tfv)

scl=StandardScaler()
train_X_scl_tfv=scl.fit_transform(train_X_svd_tfv)
val_X_scl_tfv=scl.transform(val_X_svd_tfv)
pred_probs_tfv['SVC'],pred_train_tfv['SVC'],pred_val_tfv['SVC'],score_train_tfv['SVC'],score_val_tfv['SVC']=get_model(SVC(probability=True,random_state=seed),
                                                      train_X_scl_tfv,train_y,val_X_scl_tfv)


# In[ ]:



trace1=go.Bar(x=list(score_train_tfv.keys()),
              y=list(score_train_tfv.values()),
             name='Training Score with TFV',
             marker=dict(color='rgba(0,255,0,0.5)',
                        line=dict(color='rgb(0,0,0)',width=1.5)))
trace2=go.Bar(x=list(score_val_tfv.keys()),
              y=list(score_val_tfv.values()),
             name='Validation Score with TFV',
             marker=dict(color='rgba(255,255,0,0.5)',
                        line=dict(color='rgb(0,0,0)',width=1.5)))

layout=go.Layout(barmode='group',
                title='Scores of Different Models')
plot_data=[trace1,trace2]
fig=dict(data=plot_data,layout=layout)
iplot(fig)


# With the use of TfidfVectorizer, overfitting is not observed in LR and Multinomial NB models unlike during usage of CountVectorizer. Overfitting of DTC still remains an issue. Other models aren't overfitting. It looks like Multinomial NB is the best model. But we haven't yet tuned hyperparameters of tree based models. In the following graphs, we compare the training scores of models with CountVectorizer vs with TfidfVectorizer. Same is done for validation scores as well.

# In[ ]:


trace1=go.Bar(x=list(score_train.keys()),
              y=list(score_train.values()),
             name='Training Score with CV',
             marker=dict(color='rgba(0,0,255,0.5)',
                        line=dict(color='rgb(0,0,0)',width=1.5)))
trace2=go.Bar(x=list(score_train_tfv.keys()),
              y=list(score_train_tfv.values()),
             name='Training Score with TFV',
             marker=dict(color='rgba(255,0,0,0.5)',
                        line=dict(color='rgb(0,0,0)',width=1.5)))

layout=go.Layout(barmode='group',
                title='Training Scores of Different Models')
plot_data=[trace1,trace2]
fig=dict(data=plot_data,layout=layout)
iplot(fig)


# When switched from CountVectorizer to TfidfVectorizer, the training scores of LR and Multinomial NB have decreased, while that of other models it has increased. 

# In[ ]:


trace1=go.Bar(x=list(score_val.keys()),
              y=list(score_val.values()),
             name='Validation Score with CV',
             marker=dict(color='rgba(0,0,255,0.5)',
                        line=dict(color='rgb(0,0,0)',width=1.5)))
trace2=go.Bar(x=list(score_val_tfv.keys()),
              y=list(score_val_tfv.values()),
             name='Validation Score with TFV',
             marker=dict(color='rgba(255,0,0,0.5)',
                        line=dict(color='rgb(0,0,0)',width=1.5)))

layout=go.Layout(barmode='group',
                title='Validation Scores of Different Models')
plot_data=[trace1,trace2]
fig=dict(data=plot_data,layout=layout)
iplot(fig)


# When switched from CountVectorizer to TfidfVectorizer, there is either a slight decrease or increase in validation scores. Increase in the validation score of RFC is huge.  

# In[ ]:


for model,probs in pred_probs_tfv.items():
    get_roc_curve(val_y,probs,model)


# From ROC Curves, it looks like Multinomial NB and DTC are best models. Let us check their confusion matrices.

# In[ ]:


for model,pred in pred_val_tfv.items():
    get_confusion_matrix(val_y,pred,model)
    


# Classification of true 'Insincere Questions' is highest in Multinomial NB and least in RFC. Again, Multinomial NB with TfidfVectorizer seems to be the best model.

# Let us try to tune the hyperparameters of XGBClassifier. Following codes were computationally expensive, so I just commented them.

# In[ ]:


#start=time.time()
#params={'n_estimators':[100,300,500],
#       'learning_rate':[0.01,0.05,0.1],
#       'subsample':[0.8]}

#model=XGBClassifier(random_state=seed)
#score=make_scorer(f1_score)
#grid=GridSearchCV(model,params,cv=3,scoring=score)
#grid.fit(train_X_tfv,train_y)

#end=time.time()
#print('Total time taken: ' + str(end-start))

#Total time taken: 371.8829896450043


# In[ ]:


#print(grid.best_params_)
#print(grid.best_score_)
#xgb1=grid.best_estimator_
#xgb1.fit(train_X_tfv,train_y)
#pred1=xgb1.predict(val_X_tfv)
#score1=f1_score(val_y,pred1)
#print(score1)

#{'learning_rate': 0.1, 'n_estimators': 500, 'subsample': 0.8}
#0.8230178094880998
#0.8253164556962025


# Little bit of tuning has increased our score by 4.5%. The best part is, it is not even overfitting.

# In[ ]:


#start=time.time()
#params={'n_estimators':[500,800,1000],
#       'learning_rate':[0.1,0.15,0.2],
#       'subsample':[0.8],
#       'max_depth':[3,5,7],
#       'gamma':[0,10]}

#model=XGBClassifier(random_state=seed)
#score=make_scorer(f1_score)
#grid2=GridSearchCV(model,params,cv=3,scoring=score)
#grid2.fit(train_X_tfv,train_y)

#end=time.time()
#print('Total time taken: ' + str(end-start))

#0.8397122528906257
#{'gamma': 0, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 800, 'subsample': 0.8}
#Total time taken: 8327.329408407211


# In[ ]:


#print(grid2.best_score_)
#print(grid2.best_params_)
#xgb2=grid2.best_estimator_
#xgb2.fit(train_X_tfv,train_y)
#pred2=xgb2.predict(val_X_tfv)
#score2=f1_score(val_y,pred2)
#print(score2)

#0.8397122528906257
#{'gamma': 0, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 800, 'subsample': 0.8}
#0.8515138946495231


# Thus, by further parameter tuning we have increased XGBClassifier score by 7%. You can try out more out of it. 

# **NLP with Deep Learning**

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Embedding,SpatialDropout1D,Dropout,CuDNNLSTM,Bidirectional
from keras.utils import to_categorical
from keras import backend
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback,EarlyStopping,ReduceLROnPlateau


# In the pre-processing for deeplearning, I am removing only the punctuations. Stop words might be helpful. You can even try the model by keeping punctuations. 

# In[ ]:


def filter_for_nn(text):
    tokenized_word=word_tokenize(text)
    filtered_words=[word.lower() for word in tokenized_word if word not in string.punctuation]
    return ' '.join(filtered_words)


# In[ ]:


new_data['question_text']=new_data['question_text'].apply(lambda x: filter_for_nn(x))


# In[ ]:



X=new_data.question_text
Y=new_data.target

train_X,val_X,train_y,val_y=train_test_split(X,Y,test_size=0.01,random_state=seed)


# Converting text to sequences

# In[ ]:


token=Tokenizer()
token.fit_on_texts(train_X.values)
train_seq=token.texts_to_sequences(train_X.values)
val_seq=token.texts_to_sequences(val_X.values)


# Padding the sequences

# In[ ]:


maxlen=100
train_pad=pad_sequences(train_seq,maxlen=maxlen)
val_pad=pad_sequences(val_seq,maxlen=maxlen)


# In[ ]:


train_y=to_categorical(train_y.values)
val_y=to_categorical(val_y.values)


# In[ ]:


vocabulary=token.word_index


# Getting our own Embeddings.

# In[ ]:


embedding_len=10
model=Sequential()
model.add(Embedding(len(vocabulary)+1,embedding_len,input_length=maxlen))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(CuDNNLSTM(100)))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.8,seed=seed))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.8,seed=seed))
model.add(Dense(2,activation='softmax'))

model.compile(loss='binary_crossentropy',optimizer='adam')


# Using Callbacks. Due to computation cost, I am running the model only for 5 epochs. So, callbacks are rendered ineffective. For better performance, run for about 100 epochs

# In[ ]:


est=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
rlr=ReduceLROnPlateau(monitor='val_loss',patience=10,min_lr=0.0001,factor=0.1)
call_back=[est,rlr]


# In[ ]:


result1=model.fit(train_pad,train_y,epochs=5,batch_size=64,validation_data=[val_pad,val_y],callbacks=call_back)


# In[ ]:


pred1_proba=model.predict(val_pad)
pred1=np.argmax(pred1_proba,axis=1)
score1=f1_score(np.argmax(val_y,axis=1),pred1)
print(score1)

train_pred1_proba=model.predict(train_pad)
train_pred1=np.argmax(train_pred1_proba,axis=1)
train_score1=f1_score(np.argmax(train_y,axis=1),train_pred1)
print(train_score1)


# Simple Bidirectional LSTM with own embeddings with only 5 epochs has worked better than XGBClassifier! But this model used significantly more data than XGBClassifier. But we can see that this model is overfitting. You can train it for around 100 epochs. We have set patience=10 in callbacks and so they are definitely not effective in 5 epochs. During 100 epochs, they will help in not overfitting. You can reduce the learning rate of Adam optimizer to reduce overfitting aswell.

# Taking Embeddings from GloVe

# In[ ]:


Embedding_file='../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr):return word,np.asarray(arr,dtype='float32')
embeddings_index=dict(get_coefs(*o.split(" ")) for o in open(Embedding_file))


# In[ ]:


embedding_stack=np.stack(embeddings_index.values())
emb_mean,emb_std=embedding_stack.mean(),embedding_stack.std()
embedding_matrix=np.random.normal(emb_mean,emb_std,(len(vocabulary)+1,embedding_stack.shape[1]))
for word,i in vocabulary.items():
    embedding_vector=embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector


# In[ ]:



model2=Sequential()
model2.add(Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],weights=[embedding_matrix],input_length=maxlen))
model2.add(SpatialDropout1D(0.3))
model2.add(Bidirectional(CuDNNLSTM(100)))
model2.add(Dense(512,activation='relu'))
model2.add(Dropout(0.8,seed=seed))
model2.add(Dense(512,activation='relu'))
model2.add(Dropout(0.8,seed=seed))
model2.add(Dense(2,activation='softmax'))

model2.compile(loss='binary_crossentropy',optimizer='adam')


# In[ ]:


result2=model2.fit(train_pad,train_y,epochs=5,batch_size=64,validation_data=[val_pad,val_y],callbacks=call_back)


# In[ ]:


pred2_proba=model2.predict(val_pad)
pred2=np.argmax(pred2_proba,axis=1)
score2=f1_score(np.argmax(val_y,axis=1),pred2)
print(score2)

train_pred2_proba=model2.predict(train_pad)
train_pred2=np.argmax(train_pred2_proba,axis=1)
train_score2=f1_score(np.argmax(train_y,axis=1),train_pred2)
print(train_score2)


# Using GloVe Embeddings on the same model has increased the score significantly. Here again, the model is overfitting but you can train for higher epochs with callbacks and reduced learning rate and playing around with dropout values to ensure that the model doesn't overfit. Thus, we see that neural networks are extremely powerful. **If you liked the kernel, please upvote it**
# 
# The important thing you can try out is to optimize the hyperparameters of neural network to get incredible scores without overfitting. You can do it for traditional models as well. Happy Kaggling !!
