#!/usr/bin/env python
# coding: utf-8

# Below is a notebook that examines 4 datasets: Adam Savage, Neil deGrasse Tyson, Five Thirty Eight, and Kim Kardashian. The goal is to demonstrate three different classification tools and determine who tweets more like Kim Kardashian, Adam Savage or Neil deGrasse Tyson.
# 
# The overall findings that are suggested by running the three classifications (though be it on a limited number of sets) and examining the jaccard score, the brier loss score, and the matthews coefficient indicates that Adam Savage tweets more like Kim Kardashian than Neil deGrasse Tyson.
# 
# The libraries chosen for classification are Passive Aggressive, Ridge, and Linear SVC. All perform well and are remarkably fast for how accurate they are on just a few hundred tweets from each person.
# 
# An advantage that Passive Aggressive classifiers have over Ridge and Linear SVC is the ability to do partial fits, which takes previous models and allows you to update them. This batch processing can be extended to fit an entire corpus of text rather than through individual testing.
# 
# :: Future kernels will feature this and scaling capability ::
# 
# But to feed the machine properly, I've found in my own testing that the Robust Scaler tends to be the most effective at giving the classifiers good data.
# 
# Jaccard similarity score will be used instead of accuracy or f1 as several papers have indicated it is better for testing textual similarity than either f1 or accuracy.
# 
# Later, I'll construct a matthews coefficient scorer from a confusion matrix which in playing on a larger scale, yields greater sensitivity to differences between authors.
# 
# Finally, I'll include a brier score to determine not just the binary outcomes of predictions, but also as a way to measure the strength of the predictor.

# In[ ]:


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


# In[ ]:


from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_similarity_score, make_scorer, brier_score_loss, confusion_matrix

import nltk
import re
from itertools import combinations
from collections import defaultdict, namedtuple
import seaborn as sns


# In[ ]:


## Load Datasets

df_kim = pd.read_csv("../input/KimKardashianTweets.csv")
df_neil = pd.read_csv("../input/NeildeGrasseTysonTweets.csv")
df_fte = pd.read_csv("../input/FiveThirtyEightTweets.csv")
df_adam = pd.read_csv("../input/AdamSavageTweets.csv")


# Data cleaning require removing URLs, though because of the language of twitter, I will only remove the hashtag.
# 
# Popular removals include pic.twitter, t.co, instagr.am, and https? urls

# In[ ]:


####### Regular expressions for removal
at = r'@'
hashtag = r'#'
bitly = r'bit\.ly.*\s?'
instagram = r'instagr\.am.*\s?'
url = r'https?:.*\s?'
tweeturl = r't\.co.*\s?'
pic = r'pic\.twitter\.com.+\s?'


# In[ ]:


def munger(data):
    for index, row in data.iterrows():
        text = row['text']
        text = re.sub("@","",text)
        text = re.sub("#","",text)
        text = re.sub("bit\.ly.*\s?","",text)
        text = re.sub("instagr\.am.*\s?","",text)
        text = re.sub("https?:.*\s?","",text)
        text = re.sub("t\.co.*\s?","",text)
        text = re.sub("pic\.twitter\.com\S*\s?","",text)
        #### set_value is considered the new preferred way of setting values
        #### It is also extremely fast when used with iterrows()
        data.set_value(index,"text",text)
   
    return data


# In[ ]:


### Due to memory and CPU limits, it's often required to sample the dataframes and run in batches rather than
##### run a single model with all the tweets at once. This will come in handy when working with all of NASA's tweets.
def sample_dfs(num_of_samples,num_of_dfs,df,random_state):
    sampled_dfs = []
    for i in range(num_of_dfs):
        sampled_df = df.sample(num_of_samples,random_state=random_state*i)
        munged_df = munger(sampled_df)
        sampled_dfs.append(munged_df)
    return sampled_dfs

def munge_dfs(df):
    hashtag = re.compile(r"#")
    at = re.compile(r"\.?@.+\s?")
    for index, row in df.iterrows():
        text = row["text"]
        text = hashtag.sub("",text)
        text = at.sub("",text)
        df.set_value(index,"text",text)


# In[ ]:


### Scorers since Ridge, Linear SVC, and Passive Aggressive do not have predict_proba
def score_decision_function_model(X_test, y_test,model, class_one,class_two):
    predicted_proba_y = model.decision_function(X_test)
    predicted_proba_y = (predicted_proba_y - predicted_proba_y.min()) / (
        predicted_proba_y.max() - predicted_proba_y.min())

    predicted_y = model.predict(X_test)

    type_count = count_matrix(y_test, predicted_y, class_one, class_two)

    # Brier predicts the QUALITY of the prediction
    q_score = brier_score_loss(y_test, predicted_proba_y[:, class_two])

    # Jaccard predicts the VALUE of the prediction
    jaccard_score = jaccard_similarity_score(y_test, predicted_y)

    return jaccard_score, q_score, type_count


# In[ ]:


### Included are the calculations for accuracy and f1 scores should you feel inclined to use those instead.
def count_matrix(y_true, y_pred, class_one, class_two):
    matrix = confusion_matrix(y_true, y_pred, [class_one, class_two])
    TP = matrix[1, 1]
    TN = matrix[0, 0]
    FP = matrix[1, 0]
    FN = matrix[0, 1]

    numerator = TP * TN - FP * FN
    denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    denominator = np.sqrt(denominator)

    matthews_coef = np.divide(numerator, denominator)

    ## Accuracy
    # numerator = TP + TN
    # denominator = TP + FP + TN + FN
    # correct = numerator / denominator

    ## F1-score
    # precision = np.divide(TP, (TP + FP))
    # recall = np.divide(TP, (TP + FN))
    # numerator = 2 * precision * recall
    # denominator = precision + recall
    # f1 = np.divide(numerator,denominator)

    return matthews_coef


# In[ ]:


### Build the testing dataframes - pass lists
def build_and_type(dfs,classes):
    assert len(dfs) == len(classes)
    for i in range(len(classes)):
        dfs[i]['class'] = classes[i]
    
    returned_df = dfs.pop()
    for df in dfs:
        returned_df = returned_df.append(df)
    
    return returned_df


# I use a named tuple and a default_dict to store the results. While the order of a default_dict(list) is not maintained, referencing it is both more transparent and faster when storing many results. Additionally, adding values to a default dict which does not have a key is easier than checking if a key is there, adding one if it does not, or if it does exist, pass to another conditional to add the value to a list.
# 
# Default_dicts are amazing that way. Cleaner, fast, easy.
# 
# A named tuple is used to pass the results because of the ease of refactoring code.
# 
# If a list were passed, then using indexing could become quite messy quite quickly if new scores were added or subtracted. Additionally, named tuples are fast and offer greater transparency for what value is going where, again adding ease to refactoring while being faster than a list.
# 
# Best of both worlds in my opinion.

# In[ ]:


def store_and_score(results, scores):
    scores['model_primary_jaccard'].append(results.m_primary_jaccard)
    scores['model_primary_q'].append(results.m_primary_q)
    scores['model_primary_count'].append(results.m_primary_count)
    
    scores['model_contrast_jaccard'].append(results.m_contrast_jaccard)
    scores['model_contrast_q'].append(results.m_contrast_q)
    scores['model_contrast_count'].append(results.m_contrast_count)

    scores['model_control_jaccard'].append(results.m_control_jaccard)
    scores['model_control_q'].append(results.m_control_q)
    scores['model_control_count'].append(results.m_control_count)    
    
    scores['primary_jaccard'].append(results.primary_jaccard)
    scores['primary_q'].append(results.primary_q)
    scores['primary_count'].append(results.primary_count)
    
    scores['contrast_jaccard'].append(results.contrast_jaccard)
    scores['contrast_q'].append(results.contrast_q)
    scores['contrast_count'].append(results.contrast_count)
    
    scores['control_jaccard'].append(results.control_jaccard)
    scores['control_q'].append(results.control_q)
    scores['control_count'].append(results.control_count)


# In[ ]:


scorer = make_scorer(jaccard_similarity_score)


# The following is a multi-threadable class to build and run for decision function classifiers. These are, but are not limited to, LinearSVC, PassiveAggressive, RidgeClassifier, Perceptron.
# 
# It is important to note that SVC's linear kernel does predict_proba when probability is enabled.
# 
# ## Scoring
# 
# The set up to this might seem a bit strange at first. After all, why not just count up what's confused and what's correctly predicted from the main run? For starters, it enables much better scores and more precise scores. Second, by making predictions based only on two datasets instead of 4 while using the generalized model, it more accurately approximates human perception of text where we use our own generalized models which then cross-compare two sets of tweets to sort them into similar piles.
# 
# *However* this can lead to implicit biases in the general model by overloading it with too many sets of tweets that are highly similar and one that is not. It is very recommended to amend the class to include a fifth "noise" dataset which could take anything, like the KDNuggets dataset, as input.
# 
# There are three kinds of scores: jaccard (which I highly suggest reading the wikipedia page on), brier loss, and the matthews coefficient.
# 
# The way that all these scores are set up is 
# 
# ### Interpretation
# Jaccard coefficient in the SKLearn package means that the higher it is, the more accurate/less similar two texts are and can range from 0 to 1.
# 
# In a similar vein, the geometric scorer, the matthews coefficient, is also where the higher means more dissimilarity and lower means more similarity. It ranges from -1 to 1. The other advantage that the matthews coefficient has is that it scales well. So rather than weighting a harmonic mean like f1 according to various sample size differences, it scales automatically while retaining a fair amount of sensitivity.
# 
# This is important if one wishes to increase the noise in the sample, has mismatched datasets, or wants to build different kinds of models with lopsided sample sizes.
# 
# Finally, the Brier score is one that takes into account the strength of the prediction and can vary between 0 and 1. It is the square of the difference between perfect prediction and the strength of each prediction. So a score of 0, since it is a difference measure, means that it has 100% strength behind each prediction, while 0 means it has no strength behind each prediction.
# 
# Between the three measures above, iterated tests over multiple n-gram ranges is more easily evaluated too.
# # Control / Experimental design
# For true comparison of who tweets more like whom, you take two at a time as primary and contrast and the target is the obvious target. Such as, who tweets more like Kim Kardashian (target), Neil deGrasse Tyson (primary) or Scott Kelly (contrast), with a control of KDNuggets or some other noisy variable that should come out a wash.

# In[ ]:


class terminal_output_decision_fuction_model():

    def __init__(self, ngram, classifier, params, target, primary, contrast, control):
        self.ngram = ngram
        self.params = params
        self.classifier = classifier
        self.stop_words = nltk.corpus.stopwords.words('english')
        self.control = control
        self.primary = primary
        self.contrast = contrast
        self.target = target
        
    #### Vectorization is broken up into two distinct parts. The first operates on all of the data to build a total vocabulary
    def vect_model(self, df, ngram):
        truthvalues = df['class'].values
        df = df.text.astype(str)
        tfidf = TfidfVectorizer(ngram_range=ngram, stop_words=self.stop_words)
        tfidf.fit(df)
        df = tfidf.transform(df)
        return df, truthvalues, tfidf
    
    #### Vectorzation here operates on the smaller test portions of the data using the larger vocabulary
    ###### To not use the larger vocabulary can result in anomalies and bugs when scoring later on.
    def vect_test(self, df, tfidf):
        truthvalues = df['class'].values
        df = df.text.astype(str)
        df = tfidf.transform(df)
        return df, truthvalues
    
    def run(self):
        primary_target = build_and_type([self.primary,self.target],[1,0])
        contrast_target = build_and_type([self.contrast,self.target],[2,0])
        control_target = build_and_type([self.control,self.target],[3,0])
        X = build_and_type([self.control,self.contrast,self.primary,self.target],[3,2,1,0])
        
        X, y, tfidf = self.vect_model(X,self.ngram)
        primary_t_X, primary_t_y = self.vect_test(primary_target,tfidf)
        contrast_t_X, contrast_t_y = self.vect_test(contrast_target,tfidf)
        control_t_X, control_t_y = self.vect_test(control_target,tfidf)
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42)
        
        _, pri_X_test, _, pri_y_test = train_test_split(primary_t_X, primary_t_y,random_state=42)
        _, contrast_X_test, _, contrast_y_test = train_test_split(contrast_t_X, contrast_t_y, random_state = 42)
        _, control_X_test, _, control_y_test = train_test_split(control_t_X, primary_t_y,random_state = 42)
        ### In a script setting, fiddle with n_jobs at 4 to 8 to make it run much much faster with mutlithreading
        gscv = GridSearchCV(self.classifier, self.params, scoring=scorer, n_jobs=-1)
        gscv.fit(X_train,y_train)
        
        print(gscv.best_score_)
        print(gscv.best_params_)
        
        jaccard, q, count = score_decision_function_model(X_test, y_test,gscv.best_estimator_,0,1)
        print("\n<<<<<<>>>>>>\n")
        
        print("Overall Model Scoring of Primary vs Target")
        print("Jaccard score: " + str(jaccard))
        print("Brier Score: " + str(q))
        print("Matthews Score: " + str(count))
        
        jaccard, q, count = score_decision_function_model(X_test, y_test,gscv.best_estimator_,0,2)
        print("\n<<<<<<>>>>>>\n")
        
        print("Overall Model Scoring of Contrast vs Target")
        print("Jaccard score: " + str(jaccard))
        print("Brier Score: " + str(q))
        print("Matthews Score: " + str(count))
        
        jaccard, q, count = score_decision_function_model(pri_X_test, pri_y_test,gscv.best_estimator_,0,1)
        print("\n<<<<<<>>>>>>\n")
        
        print("Scoring of Primary vs Target")
        print("Jaccard score: " + str(jaccard))
        print("Brier Score: " + str(q))
        print("Matthews Score: " + str(count))
        print("\n<<<<<<>>>>>>\n")
        
        jaccard, q, count = score_decision_function_model(contrast_X_test, contrast_y_test,gscv.best_estimator_,0,2)
        print("Scoring of Contrast vs Target")
        print("Jaccard score: " + str(jaccard))
        print("Brier Score: " + str(q))
        print("Matthews Score: " + str(count))
        print("\n<<<<<<>>>>>>\n")
        jaccard, q, count = score_decision_function_model(control_X_test, control_y_test,gscv.best_estimator_,0,3)
        print("Scoring of Control vs Target")
        print("Jaccard score: " + str(jaccard))
        print("Brier Score: " + str(q))
        print("Matthews Score: " + str(count))


# In[ ]:


class graphing_output_decision_fuction_model():

    def __init__(self, ngram, classifier, params, target, primary, contrast, control, scores_dict):
        self.ngram = ngram
        self.params = params
        self.classifier = classifier
        self.stop_words = nltk.corpus.stopwords.words('english')
        self.control = control
        self.primary = primary
        self.contrast = contrast
        self.target = target
        self.scores_dict = scores_dict
        
    #### Vectorization is broken up into two distinct parts. The first operates on all of the data to build a total vocabulary
    def vect_model(self, df, ngram):
        truthvalues = df['class'].values
        df = df.text.astype(str)
        tfidf = TfidfVectorizer(ngram_range=ngram, stop_words=self.stop_words)
        tfidf.fit(df)
        df = tfidf.transform(df)
        return df, truthvalues, tfidf
    
    #### Vectorzation here operates on the smaller test portions of the data using the larger vocabulary
    ###### To not use the larger vocabulary can result in anomalies and bugs when scoring later on.
    def vect_test(self, df, tfidf):
        truthvalues = df['class'].values
        df = df.text.astype(str)
        df = tfidf.transform(df)
        return df, truthvalues
        
    def run(self):
        primary_target = build_and_type([self.primary,self.target],[1,0])
        contrast_target = build_and_type([self.contrast,self.target],[2,0])
        control_target = build_and_type([self.control,self.target],[3,0])
        X = build_and_type([self.control,self.contrast,self.primary,self.target],[3,2,1,0])
        
        X, y, tfidf = self.vect_model(X,self.ngram)
        primary_t_X, primary_t_y = self.vect_test(primary_target,tfidf)
        contrast_t_X, contrast_t_y = self.vect_test(contrast_target,tfidf)
        control_t_X, control_t_y = self.vect_test(control_target,tfidf)
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42)
        
        _, pri_X_test, _, pri_y_test = train_test_split(primary_t_X, primary_t_y,random_state=42)
        _, contrast_X_test, _, contrast_y_test = train_test_split(contrast_t_X, contrast_t_y, random_state = 42)
        _, control_X_test, _, control_y_test = train_test_split(control_t_X, primary_t_y,random_state = 42)
        ### In a script setting, fiddle with n_jobs at 4 to 8 to make it run much much faster with mutlithreading
        gscv = GridSearchCV(self.classifier, self.params, scoring=scorer, n_jobs=-1)
        gscv.fit(X_train,y_train)
        
        print(gscv.best_score_)
        print(gscv.best_params_)
        
        m_primary_jaccard, m_primary_q, m_primary_count = score_decision_function_model(X_test, y_test,gscv.best_estimator_,0,1)
        
        m_contrast_jaccard, m_contrast_q, m_contrast_count = score_decision_function_model(X_test, y_test,gscv.best_estimator_,0,2)

        m_control_jaccard, m_control_q, m_control_count = score_decision_function_model(X_test, y_test,gscv.best_estimator_,0,3)
        
        primary_jaccard, primary_q, primary_count = score_decision_function_model(pri_X_test, pri_y_test,gscv.best_estimator_,0,1)
        
        contrast_jaccard, contrast_q, contrast_count = score_decision_function_model(contrast_X_test, contrast_y_test,gscv.best_estimator_,0,2)

        control_jaccard, control_q, control_count = score_decision_function_model(control_X_test, control_y_test,gscv.best_estimator_,0,3)
        
        scores_tuple = namedtuple("scores_tuple",["m_primary_jaccard", "m_primary_q", "m_primary_count",
                           "m_contrast_jaccard", "m_contrast_q", "m_contrast_count",
                           "m_control_jaccard", "m_control_q", "m_control_count",
                           "primary_jaccard", "primary_q", "primary_count",
                           "contrast_jaccard", "contrast_q", "contrast_count",
                           "control_jaccard", "control_q", "control_count"])
        
        scores = scores_tuple(m_primary_jaccard, m_primary_q, m_primary_count,
                             m_contrast_jaccard, m_contrast_q, m_contrast_count,
                             m_control_jaccard, m_control_q, m_control_count,
                             primary_jaccard, primary_q, primary_count,
                             contrast_jaccard, contrast_q, contrast_count,
                             control_jaccard, control_q, control_count)
        
        store_and_score(scores,self.scores_dict)


# The random number passed to the sampling method really should be a prime number since it is multiplied later on by the number of sampled dataframes desired. To make sure each successive run with different desired samplings, such as multiple batches to slowly build up the scores dictionary, prime numbers are the only way to ensure no repeated samplings.

# In[ ]:


kim = sample_dfs(400,20,df_kim,13)
fte = sample_dfs(400,20,df_fte,13)
adam = sample_dfs(400,20,df_adam,13)
neil = sample_dfs(400,20,df_neil,13)


# The way the two following experiments addresses is who tweets more like Kim Kardashian: Adam Savage or Neil deGrasse Tyson. 
# 
# It uses the Five Thirty Eight tweets as a control/noise variable to help balance the overall simulation against simply favoring all science tweets. It also helps serve as a way to measure the overall results by showing what is presumed a priori to be very dissimilar to Kim Kardashian.
# 
# Included in the overall dataset is 10,000+ tweets that contain the phrase "kdnuggets" in them as a means to generate generalized noise that likely has little to do with any subject.

# In[ ]:


classifier = PassiveAggressiveClassifier()
params = dict(C = [1.0])
terminal_output_decision_fuction_model((1,2),classifier,params,kim[0], adam[0], neil[0], fte[0]).run()


# The set below the scores default dict for easy resetting of a master list of scores for later graphing and analysis.

# In[ ]:


PA_scores = defaultdict(list)


# In[ ]:


classifier = PassiveAggressiveClassifier()
params = dict(C = [1.0])
for i in range(len(adam)):
    graphing_output_decision_fuction_model((1,2),classifier,params,kim[i], neil[i], adam[i], fte[i],PA_scores).run()


# In[ ]:


PA_scores_df = pd.DataFrame.from_dict(PA_scores,orient="columns")
PA_scores_df.head()


# As seen here, the control count can have NaN values. This indicates one of four possibilities where the denominator is 0:
#     (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
# 
# There are no true positives or false positives, no true positives or false negatives, no true negatives or false positives, or no true negatives or false positives.
# 
# Usually this indicates that a control or noise variable is either too small, or has very little in common with the target set. Which one is true can be assessed with the model_control_count. In this case, we'll substitute the count for 1, though plotting it is not the worst case in the world.
# 
# Overall, this is a case where choosing the right control is important. It's meant as a benchmark, a priori, for similarity overall when quantitatively assessing the difference between two datasets.
# 
# Otherwise, we are stuck with qualitative analysis, which is still a valid form of model selection, but restricts interpretability.

# In[ ]:


PA_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']].describe()


# In[ ]:


PA_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']].plot.hist(alpha=0.5,figsize=(10,8),colormap='rainbow',bins=16)


# The histogram nicely shows in both cases, both the overall model and the specific contrast/primary vs target cases, that confusion is higher for the contrast variable than the primary variable.
# 
# Confusion is therefore indicating that it is harder for the classifier to determine the authorship of the tweet, and therefore more similarity. To confirm this, we should also take a look at the jaccard scores.
# 
# Remember, the way that the jaccard scores are used in this case is the intersection of the two sets of predictions, so a score of 1 means that there is perfect similarity between the predictions, not between the sets.
# 
# As such, it is used more as a measure of jaccard distance than jaccard similarity.

# In[ ]:


PA_scores_df[['contrast_jaccard','model_contrast_jaccard','primary_jaccard','model_primary_jaccard']].plot.hist(alpha=0.5,figsize=(8,6),colormap='spring',bins=16)


# Interpretation of these two scores requires balance between the model overall and the specific test scenario between the either the primary or the contrast and the target data. 
# 
# Overall, in the complete model, it shows that the primary data is more easily confused with other data, but this does not necessarily indicate that it is being confused with the target data.  In this case, the contrast data seems to stand out overall as a dataset when combined with all the data.
# 
# However, when we restrict our focus to just predicting off of the primary data and the target data, the jaccard scores bear out what we saw in the count section, that there is greater dissimilarity between the primary and the target, and the contrast and the target.

# In[ ]:


PA_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']].plot.kde(alpha=0.5,colormap='rainbow')


# By beginning to plot kernel densities and the violin plots, we get a better picture overall into the distribution of how tweets become confused, indicating similarity. when examining only the overall model, we see skew in the primary toward similarity with tails going toward dissimilarity, and the opposite with the contrast. So while the means look more dissimilar when plotted and the standard deviations tell one story, the graph shows that the margin of difference may not be as different as previously thought.

# In[ ]:


sns.violinplot(PA_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']])


# In[ ]:


PA_zscores = (PA_scores_df - PA_scores_df.mean()) / PA_scores_df.std()


# In[ ]:


sns.violinplot(PA_zscores[['contrast_count','model_contrast_count','primary_count','model_primary_count']])


# I love the z-score distribution plots because they can be used in multiple cases. Either to detect anomalies between two different tweeters using the same account, or more easily illuminate if there is a subset of tweets that's responsible for the lower overall score. In this case, it's easy to see that the rough distribution of the contrast indicates that there are more dissimilar tweets in the contrast Adam Savage tweets than in the primary Neil tweets, but there is also a more populous style of tweet that is dragging down the overall score. And therefore, the skew overall is positive, while the primary's skew is negative. 
# 
# Now we have finished one of the models, I'd like to see if this bears out across multiple models (since we are using a small subset of data).

# In[ ]:


classifier = RidgeClassifier()
params = dict(alpha = [0.0001])
Ridge_scores = defaultdict(list)
for i in range(len(adam)):
    graphing_output_decision_fuction_model((1,2),classifier,params,kim[i], neil[i], adam[i], fte[i],Ridge_scores).run()
    
Ridge_scores_df = pd.DataFrame.from_dict(Ridge_scores,orient="columns")
Ridge_zscores = (Ridge_scores_df - Ridge_scores_df.mean()) / Ridge_scores_df.std()


# In[ ]:


Ridge_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']].plot.kde(alpha=0.5,colormap='rainbow')


# In my own findings when looking at other classification methods, I've found Ridge Classification to have an advantage in the detection of anomalies. 
# 
# Both overall model curves now show more distorted shapes, this leads me to wonder what in the contrast (Adam) overall that is causing a skew in the overall shape. Since it is in the direct comparison, it leads me to think that there must be either semantics similarities or perhaps some content. But from these graphs it is not possible to draw either conclusion, only hint at further investigation and a possible script to spit out the similar tweets.
# 
# Still, we see that the lump dragging down the mean for the contrast is larger overall and has a lower score overall.

# In[ ]:


sns.violinplot(Ridge_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']])


# In[ ]:


sns.violinplot(Ridge_zscores[['contrast_count','model_contrast_count','primary_count','model_primary_count']])


# In[ ]:


classifier = LinearSVC()
params = dict(C = [1.0,0.1,0.01,0.001])
Linear_scores = defaultdict(list)
for i in range(len(adam)):
    graphing_output_decision_fuction_model((1,2),classifier,params,kim[i], neil[i], adam[i], fte[i],Linear_scores).run()
    
Linear_scores_df = pd.DataFrame.from_dict(Linear_scores,orient="columns")
Linear_zscores = (Linear_scores_df - Linear_scores_df.mean()) / Linear_scores_df.std()


# In[ ]:


Linear_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']].plot.kde(alpha=0.5,colormap='rainbow')


# Now, what we have hinted at in passive aggressive with a lump of similar tweets in the contrast being more similar to the target data is found in the direct tests. There is a sizable lump indicating that there must be tweets that are more highly similar to Kim Kardashian than the normal curve would suggest. Yet, there is a tail from the primary, Neil, set that could be of interest. The overlap between Adam and Neil could be of use in finding textual similarities between the two of them in a future run.

# In[ ]:


sns.violinplot(Linear_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']])


# In[ ]:


sns.violinplot(Linear_zscores[['contrast_count','model_contrast_count','primary_count','model_primary_count']])


# It is weakly confirmed by this data that on the whole, Adam Savage tweets more like Kim Kardashian than Neil deGrasse Tyson.
# 
# But it does not show *why* one tweets more like the other. Possible hypotheses for future investigation could be that the way of presenting ideas between the two are more similar while the content is dissimilar, the other of which is that Adam Savage is more like Kim Kardashian on the whole with his twitter by tweeting more about popular culture things than Tyson. 
# 
# A quick look at the twitter timelines would show that Adam Savage tends to tweet more about places like Comic Con and pop culture events while Tyson tweets more about science. 
# 
# But we cannot be too quick to dismiss the differences in z-scores. While the content of Adam Savage might be bringing down his scores, the z-scores show an overall skew toward being dissimilar. So the semantic differences between his tweets might be greater. Or, it could be the other way around. From these graphs we cannot gain insight into the semantic versus content of the tweets.
# 
# To get insight into these questions, we'll turn to the brier loss scores.
# 
# :: Note on scores ::
# While the brier loss q scores will show the overall strength of the predictions, it is not yet built into the script to analyze the direct comparison predictions that the matthews coefficient measures.

# In[ ]:


### PASSIVE AGGRESSIVE Q
PA_scores_df[['contrast_q','model_contrast_q','primary_q','model_primary_q']].plot.kde(alpha=0.5,colormap='rainbow')


# In[ ]:


### RIDGE Q
Ridge_scores_df[['contrast_q','model_contrast_q','primary_q','model_primary_q']].plot.kde(alpha=0.5,colormap='rainbow')


# In[ ]:


### LINEAR SVC Q
Linear_scores_df[['contrast_q','model_contrast_q','primary_q','model_primary_q']].plot.kde(alpha=0.5,colormap='rainbow')


# With the brier loss score, the lower the score means that the square of the overall probability of the prediction subtracted from one is what is measured. So a low score below 0.1 means that the "loss" in the prediction is very small while a score around .25 indicates almost a 50-50 guess in a binary case. But since we are running 4 different classifications, .25 is still a strong score.
# 
# With these final measurements, we find two things. One is the confirmation of tails indicating more similar tweets than the overall corpus of text. Another good reason to dig into the actual tweets to find why and what these tweets are.
# 
# But overall, we find that in the direct comparison tests, the primary brier score is lower than the contrast score. This indicates that the suspicion that there are semantic similarities between Adam Savage and Kim Kardashian rather than just content based on the strength of the predictions since a semantic similarity would reduce the overall strength of the prediction while a content similarity would be more profound. In both these cases, especially in the Linear SVC case, we find the "hump" of similar tweets, but also an overall trend toward similarity.
# 
# A good addition to this notebook would be to create a method to directly calculate the brier loss score of just the tweets between the target and the interested text.

# In[ ]:




