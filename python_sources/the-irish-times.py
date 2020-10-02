#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize         

from matplotlib import pyplot as plt

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier as RFClassi

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier as SGDC


from sklearn.feature_extraction.text import TfidfVectorizer as TVec
from sklearn.feature_extraction.text import CountVectorizer as CVec
from sklearn.preprocessing import MinMaxScaler as mmScaler
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


# In[ ]:


path = r'../input/ireland-historical-news/irishtimes-date-text.csv'
df = pd.read_csv(path)

category_counts = df.headline_category.value_counts()
print("No of classes are: ", len(category_counts))
print(category_counts)
selected_category_counts = category_counts[category_counts > 3000].index.tolist()
df_small = df.loc[df['headline_category'].isin(selected_category_counts)]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
f, ax = plt.subplots(figsize=(30,30))
category_counts = category_counts.sort_values(ascending=False)
plt.barh(category_counts.index, category_counts)
plt.show()
#print(category_counts, category_counts.index)


# Now, we see there are 156 classes, many of which have counts even lessser than 20 and extremely specific titles. Not only will we rarely encounter such titles as a group, they'll also make our classification very difficult. 
# Another thing to note is that the news tag is the most common (obviously) with 574774 samples. This might cause an imbalance in the classification later.
# For making our problem easier, let's only use the classes with a count > 3000, which gives us 49 classes.

# In[ ]:


stratSplit = StratifiedShuffleSplit(n_splits=3, test_size=0.25)
tr_idx, te_idx = next(stratSplit.split(np.zeros(len(df_small)),df_small['headline_category']))


# Evaluation of any model should provide an accurate estimation of it's performance on data similar to the one used for training. While randomly splitting it in a 75%-25% ratio is very common, it might give a test set without all the classes or worse, a training set without all the classes. Moreover, the distribution of all classes might not be proportionate to the original datatset and lead to some biasing. This calls for a stratified split, which mimics the percentage of samples for each class in each split.
# 
# A better judgement of the model's accuracy can also be found out by using k folds, where k-1 folds (or subsets) of the dataset are used for training and 1 fold for testing. The process is repeated k times and an analysis of the score for each iteration, such as mean or variance, gives us an understanding of how our model will perform on unseen data and whether it is biased or not.
# 
# sklearn's StratifiedShuffleSplit provides train/test indices to split the data. With our imbalanced data, it is better to use this so as to let the model train on each class just as well. The number of folds or splits (k) can be set to create k different models and estimate behavior of the model under different scenarios. Here, I've used only 2 for the sake of simplicity but it's advisable to use more.

# In[ ]:


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# In[ ]:


def getSplit(te_idx, tr_idx):
    vec = CVec(ngram_range=(1,3), stop_words='english', tokenizer=LemmaTokenizer())
    lsa = TruncatedSVD(20, algorithm='arpack')
    mmS = mmScaler(feature_range=(0,1))

    countVec = vec.fit_transform(df_small.iloc[tr_idx]['headline_text'])
    countVec = countVec.astype(float)
    #print(len(countVec))
    dtm_lsa = lsa.fit_transform(countVec)
    X_train = mmS.fit_transform(dtm_lsa)

    countVec = vec.transform(df_small.iloc[te_idx]['headline_text'])
    countVec = countVec.astype(float)
    dtm_lsa = lsa.transform(countVec)
    X_test  = mmS.transform(dtm_lsa)

    x_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    x_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 

    enc = LabelEncoder()
    enc.fit(df_small.iloc[:]['headline_category'].astype(str))
    y_train = enc.transform(df_small.iloc[tr_idx]['headline_category'].astype(str))
    y_test = enc.transform(df_small.iloc[te_idx]['headline_category'].astype(str))

    y_train_c = np_utils.to_categorical(y_train)
    y_test_c = np_utils.to_categorical(y_test)

    return (X_train, y_train, X_test, y_test)


# To extract information from the text, we use a countvectorizer that uses n_grams upto 3 words and removes all stop words. Another option for a vectorizer is the TfIdfVectorizer which uses the term frequency-inverse document frequency as a metric instead of count. A lemmatizing class is passed as an argument to the vectorizer to reduce complex words to their basic form.
# 
# Now, the countvec will create a lot of features, as we have used ngrams, for feature extraction. So, it'll be helpful to do some dimensionality reduction by using single value decomposition. TruncatedSVD is a transformer that is very helpful for latent semantic analysis (To know more about LSA, check out insert link here).
# 
# We reduce the whopping number of features () to a smaller 20. Now this is helpful for two reasons. Reducing dimensionality has not only reduced the complexity of the problem and the time taken to train the model by giving it a smaller number of features, it has also taken care of features that were correlated, hence saving the time needed for correlation analysis.

# The final step is to fix the range of the fetaures using the MinMaxScaler and divide the dataset into training and test sets. Another point to keep in mind is whie transforming the input, we use fit_transform on the training and only transform on the testing set. If the entire dataset is used to transform the training set, information about the test set may leak into the training set. As for the transformation of testing set, it must rely only on the calculations of the training test, as the test rows are supposed to be unseen.

# In[ ]:


rfc = RFClassi(n_estimators=20)
mNB = MultinomialNB(alpha=.5)
gNB = GaussianNB()
bNB = BernoulliNB(alpha=.2)

sgdC = SGDC(n_jobs=-1, max_iter=1000, eta0=0.001)
gsCV_sgdClassifier = GridSearchCV(sgdC, {'loss':['hinge', 'squared_hinge',  'modified_huber', 'perceptron'], 
                                         'class_weight':['balanced',None], 'shuffle':[True, False], 'learning_rate':
                                        ['optimal', 'adaptive']})

models = [rfc, mNB, gNB, bNB, gsCV_sgdClassifier]


# For choosing a model, there are a ton of options to choose from. While NaiveBayes is used very commonly for text classification, decision trees also offer great performance. 
# 
# Here, I've used multiple models to compare and judge on accuracies. RandomForestClassifier uses a number of decision trees to create an ensemble model that controls overfitting and class imbalances. With a huge number of samples for some classes and few for others, this is a problem the model could very well run into. 

# In[ ]:


for model in models:    
    print("For model: ", model)
    acc = 0.0
    for tr_idx, te_idx in stratSplit.split(np.zeros(len(df_small)),df_small['headline_category']):
        (X_train, y_train, X_test, y_test) = getSplit(tr_idx, te_idx)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc += accuracy_score(y_test, y_pred)
    print("Classification Report is:\n", cr(y_test, y_pred))    
    print("Accuracy is: ", acc/3.0, "\n------------------------------------------------------------------------------------\n")
    


# At first glance, the BernoulliNB and MultinomialNB models seem to give great accuracies but closer inspection reveals they have actually cheated by very conveniently classifying all the samples(MultinomialNB) or most of the samples (BernoulliNB) as news, since it is the majority class and has 42% samples. The report shows that the class imbalance has got to them and affected their precision and recall scores. If we had only seen the accuracy of the model, we might not have been able to make this observation, but a classwise score calculation helps us here. The GaussianNB fares better in this aspect as it's precision and recall scores are better and it has actually classified samples into more than one class, but again 11.4% isn't a good score at all.
# 
# The RBF has done considerably better by accurately classifying 48.7% of the samples and without classifying all the samples as one class.
# 
# Choosing SGDClassifier effectively means we're choosing a linear model, and it is interesting to see how the performance will be affected when we consider this low variance model.

# In[ ]:


print(gsCV_sgdClassifier.best_params_, gsCV_sgdClassifier.best_score_)


# SGD Classifier works by fitting the data onto the equation of a line, as it is a SVM model. But since it's a linear model, it may not perform exceptionally well. To get the best accuracy, we use gridSearchCV and try out different combinations of loss and class_weight. It leads us to a model which gives an accuracy of 42.9%, which is very good for a linear model. We see that the best performing classifier is non weighted with modified_huber as the loss function and shuffles the training data before each epoch. It's commendable that it doesn't fall into the class imbalance trap by classifying all the samples as 'news'. 
