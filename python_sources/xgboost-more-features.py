# library(ggplot2)
# library(readr)
# library(data.table)
# library(dtplyr)
# library(topicmodels)
# library(tidytext)
# library(randomForest)
# library(tm)
# library(stringr)
# library(syuzhet)
# library(SnowballC)
# library(h2o)
# library(xgboost)
# library(Matrix)

# h2o.init(nthreads = -1)

# ts1 <- fread("../input/train.csv", select=c("id","question1","question2","is_duplicate"), nrow = 10000)

# print("Some question cleanup")
# # It is important to remove "\n" -- it appears to cause a parsing error when converting to an H2OFrame
# ts1[,":="(question1=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question1),
#          question2=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question2))]
# ts1[,":="(question1=gsub("  ", " ", question1),
#          question2=gsub("  ", " ", question2))]

# print("get list of unique questions")
# # Using only questions from the training set because the test set has 'questions' that are fake
# questions <- as.data.table(rbind(ts1[,.(question=question1)], ts1[,.(question=question2)]))
# #questions <- unique(questions)
# questions.hex <- as.h2o(questions, destination_frame = "questions.hex", col.types=c("String"))
# #print(questions.hex)

# STOP_WORDS = c("ax","i","you","edu","s","t","m","subject","can","lines","re","what",
#               "there","all","we","one","the","a","an","of","or","in","for","by","on",
#               "but","is","in","a","not","with","as","was","if","they","are","this","and","it","have",
#               "from","at","my","be","by","not","that","to","from","com","org","like","likes","so")

# tokenize <- function(sentences, stop.words = STOP_WORDS) {
#   tokenized <- h2o.tokenize(sentences, "\\\\W+")
#   # convert to lower case
#   tokenized.lower <- h2o.tolower(tokenized)
#   # remove short words (less than 2 characters)
#   tokenized.lengths <- h2o.nchar(tokenized.lower)
#   tokenized.filtered <- tokenized.lower[is.na(tokenized.lengths) || tokenized.lengths >= 2,]
#   # remove words that contain numbers
#   tokenized.words <- tokenized.lower[h2o.grep("[0-9]", tokenized.lower, invert = TRUE, output.logical = TRUE),]
#   # remove stop words
#   tokenized.words[is.na(tokenized.words) || (! tokenized.words %in% STOP_WORDS),]
# }
# print("Break questions into sequence of words")
# words <- tokenize(questions.hex$question)

# #print(words)

# print("Build word2vec model")
# vectors <- 20 # Only 10 vectors to save time & memory
# w2v.model <- h2o.word2vec(words
#                           , model_id = "w2v_model"
#                           , vec_size = vectors
#                           , min_word_freq = 5
#                           , window_size = 5
#                           , init_learning_rate = 0.025
#                           , sent_sample_rate = 0
#                           , epochs = 1) # only a one epoch to save time

# h2o.rm('questions.hex') # no longer needed
# #print(words)

# # print("Sanity check - find synonyms for the word 'water'")
# # print(h2o.findSynonyms(w2v.model, "water", count = 5))

# print("Get vectors for each question")
# question_all.vecs <- h2o.transform(w2v.model, words, aggregate_method = "AVERAGE")
# #print(question_all.vecs)

# print("Convert to data.table & merge results")
# # Could do the rest of these steps in H2O but I'm a data.table addict
# question_all.vecs <- as.data.table(question_all.vecs)
# questions_all <- cbind(questions, question_all.vecs)
# ts1 <- merge(ts1, questions_all, by.x="question1", by.y="question", all.x=TRUE, sort=FALSE)
# ts1 <- merge(ts1, questions_all, by.x="question2", by.y="question", all.x=TRUE, sort=FALSE)
# #q1_vec * q2_vec
# colnames(ts1)[5:ncol(ts1)] <- c(paste0("q1vec", 1:vectors), paste0("q2vec", 1:vectors))
# # print(head(ts1))

# # #xgboost
# # ts1 = ts1[1:1000,]
# # ts2 = ts2[1:1000,]
# # ts3 = ts3[1:1000,]
# # plus = ts2[,5:14]+ts3[,5:14]
# # new.train = cbind(ts1$is_duplicate, plus)
# # colnames(new.train)[1] <- "is_dup"
# # print(dim(new.train))
# # colnames(new.train)
# new.train = ts1[,c(4:44)]
# temp1 = ts1[,c(5:24)]
# temp2 = ts1[,c(25:44)]

# new.temp = temp1-temp2
# colnames(new.temp)[1:ncol(new.temp)] <- paste0("sub", 1:vectors)
# new.train = cbind(new.train, new.temp)

# new.temp = temp1+temp2
# colnames(new.temp)[1:ncol(new.temp)] <- paste0("add", 1:vectors)
# new.train = cbind(new.train, new.temp)

# new.temp = temp1*temp2
# colnames(new.temp)[1:ncol(new.temp)] <- paste0("mut", 1:vectors)
# new.train = cbind(new.train, new.temp)

# new.temp = temp1/temp2
# colnames(new.temp)[1:ncol(new.temp)] <- paste0("div", 1:vectors)
# new.train = cbind(new.train, new.temp)

# new.train = na.omit(new.train)
# new.train[,1] = lapply(new.train[,1],as.factor)
# # print(new.train)

# #split
# set.seed(5243)
# s <- sample(1:nrow(new.train), 0.8*nrow(new.train), replace = FALSE)
# training <- new.train[s,]
# validating <- new.train[-s,]
# # head(training)
# # head(validating)

# #dmatrix
# d.training.data <- sparse.model.matrix(is_duplicate~.-1,data=training)
# d.validating.data <- sparse.model.matrix(is_duplicate~.-1,data=validating)

# dtrain <- xgb.DMatrix(data = d.training.data, label= training$is_duplicate==1)
# dvalid <- xgb.DMatrix(data = d.validating.data, label= validating$is_duplicate==1)

# # #train
# # params['objective'] = 'binary:logistic'
# # params['eval_metric'] = 'logloss'
# # params['eta'] = 0.02
# # params['max_depth'] = 4
# # bst <- xgb.train(data=dtrain2, max_depth=2, eta=1, nthread = 2, nrounds=2, watchlist=watchlist, objective = "binary:logistic")
# # bstSparse <- xgboost(data = train$data, label = train$label, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
# # xgb.train(data=dtrain, booster = "gblinear", max_depth=2, nthread = 2, nrounds=2, watchlist=watchlist, eval_metric = "error", eval_metric = "logloss", objective = "binary:logistic")
# watchlist <- list(train=dtrain, valid=dvalid)
# bst <- xgb.train(data = dtrain, max_depth = 6, eta = 0.06, nthreads = 8, nrounds = 400, eval_metric = "logloss", watchlist = watchlist, objective = "binary:logistic")
# # bst <- xgboost(data = d.training.data,label= training$is_duplicate==1, max_depth = 4, eta = 0.02, nthreads = 2, nrounds = 100, eval_metric = "logloss", objective = "binary:logistic")

# importance_matrix <- xgb.importance(model = bst)
# print(importance_matrix)
# xgb.plot.importance(importance_matrix = importance_matrix)
# xgb.plot.tree(model = bst)

# # #pred
# # pred <- predict(bst, d.validating.data)
# # print(head(pred))

# # logLoss = function(pred, actual){
# #   -1*mean(log(pred[model.matrix(~ actual) - pred > 0]))
# # }
# # logLoss(pred, validating$is_dup)



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split

df_train = pd.read_csv('../input/train.csv', nrows = 1000)
df_test = pd.read_csv('../input/test.csv', nrows = 1000)

# explore
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

# word match share
stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

train_word_match = df_train.apply(word_match_share, axis=1, raw=True)

# tfidf
# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)

# model
x_train = pd.DataFrame()
x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match
# x_train['1'] = tfidf_train_word_match + train_word_match
# x_train['2'] = tfidf_train_word_match - train_word_match
# x_train['3'] = tfidf_train_word_match * train_word_match
# x_train['4'] = tfidf_train_word_match / train_word_match

x_test = pd.DataFrame()
x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)

y_train = df_train['is_duplicate'].values

pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]
# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=5243)#4242


# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval = 10)

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test

print(sub.head())
#sub.to_csv('simple_xgb.csv', index=False)