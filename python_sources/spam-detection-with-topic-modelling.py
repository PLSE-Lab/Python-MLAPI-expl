#!/usr/bin/env python
# coding: utf-8

# # Spam Detection with Topic Modelling

# "Spam" is a term used to refer to unwanted electronic messages that one receives, usually through an email or text message. Due to them being unwanted, it would be helpful to implement machine learning to identify spam, so as to prevent them being presented to a user of an electronic device and disturbing them. Though spam messages are usually quite easy for one who receives them to identify, the features that identify a spam message as such may actually be quite difficult to aritculate, and therefore make it difficult to employ a machine learning model for this task.
# 
# Let's try and employ topic modelling to represent messages in a dataset of SMS messages by their subject matter and feed this into a machine learning model to serve as a spam detector.

# ## Data Exploration and Preprocessing

# Let's begin by loading in the packages necessary for our analysis as well as identify where the dataset file is located.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.matutils import Sparse2Corpus
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, confusion_matrix
import os
print(os.listdir("../input"))


# Let's now load the data into a pandas dataframe and take a peek at its contents.

# In[ ]:


data = pd.read_csv("../input/spam.csv", encoding="latin-1")
data.head()


# As we can see, we have 5 columns of data, though the contents of not all these columns is clear. Column 'v1' clearly corresponds to the SMS message label, and 'v2' corresponds to the message content, but the contents of the other columns is unclear. It is worth noting that the "ham" label refers to non-spam messages. Let's view some statistical information about our dataset.

# In[ ]:


data.describe()


# Viewing the above dataset description, we can see that we have a fairly small dataset with only 5,572 samples. We can also see that the unidentified columns are missing so much data as to be unhelpful for developing a machine learning model. We will therefore drop these columns and rename the remaining columns to something more clear.

# In[ ]:


data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
data.rename(columns=dict(v1="Class", v2="Message"), inplace=True)


# Let's also separate our data into a train and test set, where 90% of the dataset will be allocated for training, and the rest allocated for testing a machine learning model.

# In[ ]:


limit = int(0.9*len(data))
train = data.loc[:limit]
test = data.loc[limit:]


# We are left with only one feature in our dataset, the message content itself. We must find a way to extract features from these messages so as to be able to train a machine learning model that can identify a spam message. One thing we can do is extract the subject matter of these messages using topic modelling. A topic model is an unsupervised machine learning model that can be trained on a corpus (a collection of documents) to identify the topics within those documents. In this context, a topic is usually defined as a probability distribution of all terms in the corpus vocabulary appearing for that given topic.
# 
# One popular topic model is latent Dirichlet allocation (LDA). Training an LDA model on our training data, we can use this model to represent messages as topic weight vectors, where the value in each element of the topic weight vector corresponds to the weight in which that given message exhibits the associated topic. One caveat with using LDA however is that one must preemptively set the number of topics believed to be exhibited in a corpus prior to training an LDA model on it. Using the topic modelling package Gensim, we will train an LDA model on the training set, and use this LDA model to transform the train and test set into topic weight vectors. We will use the Gensim default of 100 topics for this task.
# 
# Since more meaning can sometimes be derived from words when taken together instead of individually (i.e. 'see you later' as opposed to the individual terms 'see', 'you' and 'later') we will take into consideration not just individual terms, but collections of terms (also known as n-grams). We will take into consideration n-grams with n equal to 1, 2, and 3 (also known as unigrams, bigrams, and trigrams respectively). All English stop words (i.e. the, and, etc.) will be ignored, offering no useful information with regards to the subject matter of these messages. For simplicity, when using the LDA model to transform messages into topic vectors, any topics that have a probability less than 0.01 of being exhibited by that message will automatically have their topic weight set to zero for that message.
# 
# Let's now build this topic model, and use it to transform our train and test data, taking a peek at the modified training data.

# In[ ]:


vectorizer = CountVectorizer(stop_words="english", 
                             ngram_range=(1, 3))
term_document_matrix = vectorizer.fit_transform(train.Message)
corpus = Sparse2Corpus(term_document_matrix, documents_columns=False)
id2word = {value: key for key, value in vectorizer.vocabulary_.items()}
model = LdaModel(corpus=corpus, 
                 id2word=id2word, 
                 random_state=0)

def transform_messages_to_topic_vectors(data, vectorizer, model):
    num_topics = model.num_topics
    message_topics = pd.DataFrame(index=map(str, range(num_topics)))    
    for index, message in data.Message.iteritems():
        message_transformed = (vectorizer.transform([message]))
        message_corpus = (Sparse2Corpus(message_transformed, documents_columns=False))
        topics_framework = dict.fromkeys(map(str, range(0, num_topics)))
        topics_specific = dict(list(model[message_corpus])[0])
        for key in topics_specific:
            topics_framework[str(key)] = topics_specific[key]
        message_topics[index] = pd.Series(topics_framework)
    message_topics = message_topics.T.fillna(0)
    data = data.join(message_topics)
    data.drop("Message", axis=1, inplace=True)
    data = pd.get_dummies(data, drop_first=True)
    data.rename(columns=dict(Class_spam="Spam"), inplace=True)
    return data

train = transform_messages_to_topic_vectors(train, vectorizer, model)
test = transform_messages_to_topic_vectors(test, vectorizer, model)
train.head()


# Viewing our transformed training data, we can see that the topic weight vectors representing the messages is sparse. This is due to messages seeming to represent most of the 100 LDA model topics with a low probability. Let's view the distribution of how many messages exhibit a given number of topics.

# In[ ]:


topics = list(train.columns)
topics.remove("Spam")
ax = sns.countplot(train[topics].astype(bool).sum(axis=1), color="C0")
ax.set_xlabel("Number of Exhibited Topics")
ax.set_ylabel("Number of Messages")
fig = plt.gcf()
fig.set_size_inches(16, 4)


# As we can see, though we have messages that only exhibit one topic, or in some cases zero (due to all topic exhibitions being of a low probability), we can see that many messages discuss multiple topics. Let's conversely view how many messages exhibit each particular LDA model topic.

# In[ ]:


ax = train[topics].astype(bool).sum(axis=0).plot.bar(color="C0")
ax.set_xlabel("Topic")
ax.set_ylabel("Number of Messages")
fig = plt.gcf()
fig.set_size_inches(24, 8)


# Though as expected there is variability in topic prevalence, we can see that topic 50 seems to be the most prevalent in the training set. The best way to infer the subject matter of a topic is to view the most prevalent words within that topic. Let's view the top 10 terms for topic 50.

# In[ ]:


print("The most prevalent ngrams in the most prevalent topic are:\n- {}".format('\n- '.join([id2word[term[0]] for term in model.get_topic_terms(50)])))


# Viewing the most prevalent ngrams in topic 50, we can see that there seems to be a lot of apologies being made in the training set. Though there are of course spam and ham messages present in the dataset, it would be worthwhile to see the distribution between these two classes in the training set, so as to possibly take this into consideration when training a spam detection machine learning model.

# In[ ]:


X_train = train.drop("Spam", axis=1)
Y_train = train.Spam
X_test = test.drop("Spam", axis=1)
Y_test = test.Spam
sns.countplot(Y_train, color="C0")


# As we can see, there are much more ham then spam messages in our training set. This class imbalance is a common problem in machine learning, and will be addressed when training the spam detectors. Due to this class imbalance, let's just confirm that there are spam messages in the test set and not all spam messages were accidentally placed into the training set.

# In[ ]:


print("Number of spam messages in the test set: {}"      .format(test["Spam"].value_counts()[1]))


# Thankfully, there are a number of spam messages in the test set. Let's finish preprocessing our dataset by standardizing the values so that when building a machine learning model we don't unintentionally lend more weight to some features over others. We will fit a standardizer to the training data, and transform the train and test data based on this scaler.

# In[ ]:


scaler = StandardScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train)
X_test[X_test.columns] = scaler.transform(X_test)


# ## Building a Spam Detector with Machine Learning

# Let's now use machine learning to identify whether or not a message is spam given its topic weights. We will train several different machine learning models for this task, and identify the model that performs best on the test data as our selected spam detector.

# ### Linear Support Vector Machine

# We have quite a small dataset to work with. Support vector machines (SVMs) have been known to perform well with small datasets. Though the two classes are most likely not linearly-separable, let's begin by training a support vector machine with a linear kernel and see what results we get. SVMs have a tuning parameter 'C' that determines the regularization strength within the model. Let's identify which tuning parameter is best for this dataset using cross validation. Since SVMs being trained for this cross validation step will not reach convergence to save time, we will suppress all ConvergenceWarnings for the following cell. We will also ensure that the SVM is aware of the class imbalance in the dataset by setting its 'class_weight' to 'balanced'. This parameter will be set for every implemented individual machine learning model.

# In[ ]:


get_ipython().run_cell_magic('capture', '--no-stdout', '\nparam_grid = dict(C=np.logspace(-3, 3, 7))\nbest_C_linear = GridSearchCV(LinearSVC(class_weight="balanced", random_state=0), \n                             param_grid, \n                             cv=5).fit(X_train, Y_train)\\\n                                  .best_params_["C"]\nprint("The best value for the tuning parameter \'C\' is {}.".format(best_C_linear))')


# It seems that the best value for 'C' is 10. Let's now train a linear SVM using this tuning parameter and sharply increase the bound on the maximum number of training iterations in order to allow the model to reach convergence.

# In[ ]:


svm_linear = LinearSVC(C=best_C_linear, 
                       random_state=0, 
                       max_iter=1e6).fit(X_train, Y_train)


# ### Naive Bayes Classifiers

# Naive Bayes classifiers have been known to perform well in the domain of spam detection. Let's also train a Gaussian Naive Bayes classifier for this task.

# In[ ]:


priors = list(train.Spam.value_counts().div(train.Spam.value_counts().sum()))
nb = GaussianNB(priors).fit(X_train, Y_train)


# ### SVM with 'RBF' Kernel

# As previously mentioned, SVMs seem to perform well on small datasets, but training an SVM with a linear kernel with the hope that the data classes are linearly separable was probably a naive assumption to make. Let's again perform cross-validation to select the hyperparameters to use for an SVM for this task, but instead using an SVM with a RBF kernel for a more flexible approach. We will also fine-tune the 'gamma' parameter for this RBF SVM approach.

# In[ ]:


get_ipython().run_cell_magic('capture', '--no-stdout', '\nparam_grid["gamma"] = ["auto", "scale"]\nbest_svm_rbf_params = GridSearchCV(SVC(class_weight="balanced", \n                                       random_state=0, \n                                       max_iter=1000), \n                                   param_grid, \n                                   cv=5).fit(X_train, Y_train)\\\n                                        .best_params_\nprint("The best value for the tuning parameter \'C\' is {}."\\\n      .format(best_svm_rbf_params["C"]))\nprint("The best value for \'gamma\' is {}.".format(best_svm_rbf_params["gamma"]))')


# Using the identifed hyperparameters, let's train an SVM with an RBF kernel on the training data.

# In[ ]:


svm_rbf = SVC(C=best_svm_rbf_params["C"], 
              gamma=best_svm_rbf_params["gamma"], 
              random_state=0, 
              max_iter=-1).fit(X_train, Y_train)


# ### Ensemble Voting Classifier

# To finish out the list of possible spam detector candidates, we will also train a voting classifier consisting of an ensemble of all aforementioned machine learning models, which will identify a message as the class the majority of the models identify it as being.

# In[ ]:


classifiers = [("svm_linear", svm_linear), 
               ("nb", nb), 
               ("svm_rbf", svm_rbf)]
ensemble = VotingClassifier(classifiers).fit(X_train, Y_train)


# ## Testing Our Spam Detectors

# Let's now view how these models perform on an unseen test set of messages.

# In[ ]:


svm_linear_score = svm_linear.score(X_test, Y_test)
nb_score = nb.score(X_test, Y_test)
svm_rbf_score = svm_rbf.score(X_test, Y_test)
ensemble_score = ensemble.score(X_test, Y_test)

print("The linear SVM has a test accuracy score of {:.3f}.".format(svm_linear_score))
print("The Gaussian naive Bayes classifer has a test accuracy score of {:.3f}."      .format(nb_score))
print("The SVM with an RBF kernel has a test accuracy score of {:.3f}."      .format(svm_rbf_score))
print("The ensemble voting classifier has a test accuracy score of {:.3f}."      .format(ensemble_score))


# As we can see, except for the naive Bayes classifier, all machine learning models perform with a high test accuracy, each being over 90%. However, due to the class imbalance in the dataset, these results may not be as successful as we initially think. Let's build a dummy classifier that predicts every message presented to it as being ham, and view it's test score.

# In[ ]:


dummy_constant = DummyClassifier("constant", 
                                 random_state=0, 
                                 constant=0).fit(X_train, Y_train)
dummy_constant_score = dummy_constant.score(X_test, Y_test)

print("The dummy constant classifier has a test accuracy score of {:.3f}."      .format(dummy_constant_score))


# As we can see, this naive model scored 87% accuracy on the test set, so the test accuracy scores of our machine learning models are not as impressive as we initially thought. However, one thing we can do to get a better grasp of our models' performance is to use a metric other than accuracy to evaluate them. One metric useful for evaluating the performance of machine learning models with imbalanced data is the average precision score. Let's view the average precision score achieved by the dummy classifier and our trained machine learning models.

# In[ ]:


dummy_constant_preds = dummy_constant.predict(X_test)
svm_linear_preds = svm_linear.predict(X_test)
nb_preds = nb.predict(X_test)
svm_rbf_preds = svm_rbf.predict(X_test)
ensemble_preds = ensemble.predict(X_test)

dummy_constant_score = average_precision_score(Y_test, dummy_constant_preds)
svm_linear_score = average_precision_score(Y_test, svm_linear_preds)
nb_score = average_precision_score(Y_test, nb_preds)
svm_rbf_score = average_precision_score(Y_test, svm_rbf_preds)
ensemble_score = average_precision_score(Y_test, ensemble_preds)

print("The dummy constant classifier has a test average precision score of {:.3f}."      .format(dummy_constant_score))
print("The linear SVM has a test average precision score of {:.3f}.".format(svm_linear_score))
print("The Gaussian naive Bayes classifer has a test average precision score of {:.3f}."      .format(nb_score))
print("The SVM with an RBF kernel has a test average precision score of {:.3f}."      .format(svm_rbf_score))
print("The ensemble voting classifier has a test average precision score of {:.3f}."      .format(ensemble_score))


# Viewing these results, we gather a much better appreciation for the performance of these machine learning models. Though all performed better than the dummy classifier with regards to their average precision score, the naive Bayes classifier still performed the worst. The SVM with an RBF kernel performed the best out of all individual machine learning models with an average precision score of 0.584, but applying an emsemble of all these machine learning models had the highest overall average precision score of 0.625. Let's view a confusion matrix of the test predictions made by this voting classifier to gather additional insight into its performance.

# In[ ]:


confusion = pd.DataFrame(confusion_matrix(Y_test, ensemble_preds))
confusion = confusion.div(confusion.sum().sum())
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
ax = sns.heatmap(confusion, vmin=0, vmax=1, annot=True, fmt=".0%")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.set_ticks((0, .25, .5, .75, 1))
ax.collections[0].colorbar.set_ticklabels(("0%", "25%", "50%", "75%", "100%"))


# Despite this satisfactory performance, around 2% of the messages in the test set were identifed as spam when they actually weren't. Since we don't want anyone to miss out on messages important to them, further improvements to this spam detector should focus on minimizing the number of false positives that it predicts.

# ## Final Remarks

# Using the topic model LDA, we were able to adequately transform a dataset of SMS messages in a manner suitable to build a spam detector using machine learning. Implementing a variety of different machine learning frameworks, it was found that using an ensemble of all these methods had the highest performance, achieving a test accuracy of 0.943 and an average precision score of 0.625. Though these results are satisfactory, further improvements to this approach should focus on minimizing the number of false positives made. This can be achieved by perhaps refining the LDA topic model or spam detectors used for this task.
