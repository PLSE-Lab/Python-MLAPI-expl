# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import re, html, unicodedata, time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer#, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels

# nltk
from nltk.stem.snowball import SnowballStemmer, EnglishStemmer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#df = pd.read_csv('../input/Tweets.csv')



class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = EnglishStemmer()
        self._RE_tweet_emojis = re.compile(u'[\U0001F600-\U0001F64F]', re.UNICODE)
        self._RE_links = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')# links url
        self._RE_tweet_accounts = re.compile('@\\w+')
        self._RE_tweet_hashtag = re.compile('#\\w+')
        self._RE_ACRONYMS = re.compile('[A-Z]{2,}')
        self._RE_DATES =re.compile('\d+/\d+/?\d*')
        self._RE_WHITESPACES =re.compile('\s+')
        

    def clean_text(self, tweet):
        ''' tweet: tweet
        '''
        tweet = self._RE_WHITESPACES.sub(' ', tweet)
        tweet = html.unescape(tweet)
        tweet = self._RE_tweet_emojis.sub(lambda m: unicodedata.name(m.group()).upper().replace(' ', ''), tweet)
        # print(tweet)
        __LINK__='URLLINKURL' #'LINK'
        tweet = self._RE_links.sub(__LINK__, tweet)
        __ACCOUNTMENTION__ = 'ACCOUNTMENTION'#'ACCOUNTMENTION'#'ORGANIZATION'
        tweet = self._RE_tweet_accounts.sub(__ACCOUNTMENTION__, tweet)
        __HASHTAG__='HASHTAG'#'HASHTAG'#'HASHTAG'
        tweet = self._RE_tweet_hashtag.sub(__HASHTAG__, tweet)
        # __ACRONYMS__=''#'ACRONYMS'
        # tweet = self._RE_ACRONYMS.sub(__ACRONYMS__, tweet)
        __DATES__='datemention'
        tweet = self._RE_DATES.sub(__DATES__, tweet)
        
        # tweet = " ".join([self.stemmer.stem(token) for token in tweet.split()])
        
        return tweet
    
    def transform(self, X, y=None):
        ''' Sentences as text
        '''
        # print(len(X))
        out = [self.clean_text(x) for x in X]
        # print(len(out))
        return out

    def fit_transform(self, X, y=None):
        return self.transform(X, y)
    
    def fit(self, X, y=None):
        self.transform(X)
        return self

# Any results you write to the current directory are saved as output.
def cleanup_data(df):
    '''
        remove uninformative data
    '''
    # df = df[pd.notnull(df['negativereason'])]
    df=df.fillna('NOREASON')
    return df

def build_lda(X, y=None, n_topics = 8):
    
    best_params = {
         'estimator__random_state':42
        ,'estimator__n_jobs':-1
        ,'estimator__n_topics':n_topics
        ,'estimator__max_iter':5
        ,'estimator__learning_method':'batch'
        ,'estimator__learning_offset':50.0
        ,'tfidf__norm': 'l2'
        ,'tfidf__use_idf': True
        ,'vectorizer__max_df': 1.0
        ,'vectorizer__max_features': None
        ,'vectorizer__ngram_range': (1, 2)
    }
    lda_model = Pipeline([
         ('cleantxt', TextPreprocessor())
        ,('vectorizer', CountVectorizer())
        ,('tfidf', TfidfTransformer())
        ,('estimator', LatentDirichletAllocation())
    ])
    lda_model = lda_model.fit(X, y)
    return lda_model
    

prev_cnf_matrix=None
prev_scoring_values={}
def train_reason_model(X, y, model_name=None):
    best_params = {
         'estimator__alpha': 0.0000001
        ,'estimator__n_iter': 5
        ,'estimator__penalty': 'elasticnet'
        ,'estimator__random_state':42
        ,'estimator__n_jobs':-1
        ,'estimator__class_weight':'balanced'
        # ,'estimator__loss':'log'
        ,'tfidf__norm': 'l2'
        ,'tfidf__use_idf': True
        ,'vectorizer__max_df': 1.0
        ,'vectorizer__max_features': None
        ,'vectorizer__ngram_range': (1, 2)
    }
    reason_model_pipeline = Pipeline([
         ('preprocessor', TextPreprocessor())
        ,('vectorizer', CountVectorizer())
        ,('tfidf', TfidfTransformer())
        ,('estimator', SGDClassifier())
    ])
    reason_model_pipeline.set_params(**best_params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    t0 = time.time()
    reason_model_pipeline.fit(X_train, y_train)
    print("Training done in %0.3fs" % (time.time() - t0))
    # scores
    scores = cross_val_score(reason_model_pipeline, X_test, y_test)
    score = scores.mean()
    global prev_scoring_values
    global prev_cnf_matrix
    
    print("Score:{}".format(score))
    # classification_report
    y_true, y_pred = y_test, reason_model_pipeline.predict(X_test)
    acc_score = accuracy_score(y_true, y_pred)
    print("Accuracy Score", acc_score)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    global prev_cnf_matrix
    if prev_cnf_matrix is not None:
    
        cnf_matrix_diff = cnf_matrix - prev_cnf_matrix
        
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.suptitle('Confusion Matrix', fontsize=14)
        class_names = unique_labels(y_true, y_pred)
        n_classes = len(class_names)
        
        
        fig.subplots_adjust(hspace=0, wspace=0, left=0.27, bottom=0.36)
        
        tick_marks = np.arange(n_classes)
        
        # print(axes)
        
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        
        plt.sca(axes[0,0])
        plt.yticks(tick_marks, class_names, fontsize=8)
        plt.sca(axes[1,1])
        plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
        plt.text(-0.5, .5, 'Difference', horizontalalignment='center',fontsize=12,transform = axes[1,1].transAxes)
        plt.text(0.5, 2.05, 'Regular Data', horizontalalignment='center',fontsize=12,transform = axes[1,1].transAxes)
        plt.text(-0.5, 2.05, "'Enhanced' data", horizontalalignment='center',fontsize=12,transform = axes[1,1].transAxes)
        
        fig.delaxes(axes[1, 0])
        
        ((ax1, ax2), (_, ax3)) = axes
        title = 'Regular data'
        # ax1.set_title(title)
        # ax1.set_xticklabels(class_names)
        # ax1.set_yticklabels(class_names)
        
        normalization = colors.LogNorm()
        im1 = ax1.imshow(prev_cnf_matrix, interpolation='nearest', norm=normalization, cmap=plt.cm.Blues)
        # fig.colorbar(im1, ax=ax1, extend='both')
        # pcm.plot_confusion_matrix(prev_cnf_matrix, classes=class_names,title=title)
        # plt.savefig('confusion_matrix_regular_data.png')

        title = 'LDA enhanced data'
        # ax2.set_title(title)
        im2 = ax2.imshow(cnf_matrix, interpolation='nearest', norm=normalization, cmap=plt.cm.Blues)
        fig.colorbar(im2, ax=ax2, extend='both')
        # pcm.plot_confusion_matrix(cnf_matrix, classes=class_names,title=title)
        # plt.savefig('confusion_matrix_enhanced_data.png')
        
        title = 'Difference'
        # ax3.set_title(title)
        
        normalization = colors.SymLogNorm(linthresh=0.03)
        # normalization = colors.Normalize()
        im3 = ax3.imshow(cnf_matrix_diff, interpolation='nearest',norm=normalization,cmap=plt.cm.RdBu_r)
        fig.colorbar(im3, ax=ax3, extend='both')

        # pcm.plot_confusion_matrix(cnf_matrix_diff, normalize=False, classes=class_names,title=title, cmap=plt.cm.bwr)
        plt.savefig('confusion_matrix_delta.png')
        # plt.show()
        
        prev_cnf_matrix=None
        fig, ax = plt.subplots()
        index = np.arange(2)
        bar_width = 0.35
        opacity = 0.8
        rects1 = plt.bar(index, [prev_scoring_values['Score'],score], bar_width,
                 alpha=opacity,
                 color='b',
                 label='Score')

        rects2 = plt.bar(index + bar_width, [prev_scoring_values['Accuracy Score'], acc_score], bar_width,
                         alpha=opacity,
                         color='g',
                         label='Accuracy Score')
        print([prev_scoring_values['Score'],score])
        print([prev_scoring_values['Accuracy Score'], acc_score])
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                t = "{0:.2f}".format(height)
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, t, ha='center', va='bottom',fontsize=8)
                
        autolabel(rects1)
        autolabel(rects2)

        plt.xlabel('Algorithm')
        plt.ylabel('Scores')
        plt.ylim(0,1)
        plt.title('Increasing performance by incorporating LDA topics')
        plt.xticks(index + bar_width/2, ('Regular data', 'LDA enhanced data'))
        plt.legend()

        plt.tight_layout()
        # plt.show()
        plt.savefig('scores_gains.png')
    else:
        prev_scoring_values['Score'] = score
        prev_scoring_values['Accuracy Score'] = acc_score
        
    prev_cnf_matrix = cnf_matrix

    
    # print(classification_report(y_true, y_pred))
    return reason_model_pipeline

    
df = pd.read_csv('../input/Tweets.csv')
df = cleanup_data(df)

#preprocess
X=df['text'].values
y=df['negativereason'].values

print("Model with regular data")
train_reason_model(X,y, model_name="Negative Reason model")

t0 = time.time()
print("Enhancing X")
# Build the topics words dictionary
n_topics = 8
lda_pipeline_model = build_lda(X, n_topics=n_topics)
n_top_significant_words=10;
tf_feature_names = lda_pipeline_model.get_params()['vectorizer'].get_feature_names()
topic_significant_words=dict()
for topic_idx, topic in enumerate(lda_pipeline_model.get_params()['estimator'].components_):
    topic_significant_words[topic_idx] = " ".join([tf_feature_names[i] for i in topic.argsort()[:-n_top_significant_words - 1:-1]])

#Add the most significant topic words to the document
docs_topic_distr = lda_pipeline_model.transform(X)

enhanced_X=[]
for x, doc_topic_distr in zip(X, docs_topic_distr):
    top_topics = doc_topic_distr.argsort()[:-n_topics-1:-1]
    topic_words_to_be_added=''
    for top_topic_idx in top_topics:
        topic_words_to_be_added += ' '+topic_significant_words[top_topic_idx]
    enhanced_X.append(x.strip()+' '+topic_words_to_be_added)
X=enhanced_X
print("Done enhancing in %0.3fs" % (time.time() - t0))
#Retrain using enhanced X
print()
print()
print("Training model with enhanced data")
train_reason_model(X,y, model_name="Negative Reason model Trained with enhanced X")