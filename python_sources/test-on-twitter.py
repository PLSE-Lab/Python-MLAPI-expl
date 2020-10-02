import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


twitter_data = pd.read_csv('../input/Tweets.csv', header=0)

columns = twitter_data.columns
print(twitter_data['tweet_created'])

col_names = ['airline_sentiment', 'airline', 'text', 'retweet_count']

filtered_data = twitter_data[col_names]

# filtered_data.loc[:,'sentiment'] = filtered_data.airline_sentiment.map({'negative':0,'neutral':2,'positive':4})

# filtered_data = filtered_data.drop(['airline_sentiment'], axis=1)
# filtered_data = filtered_data[filtered_data['retweet_count']<=2]

# plt.figure(1)
# pd.Series(filtered_data['sentiment']).value_counts().plot(kind = "barh" , title = "sentiment")

# airlines = ['Delta', 'Virgin America', 'US Airways', 'United', 'American', 'Southwest']

# is_negative = filtered_data['airline_sentiment'] == 'negative'
# is_delta = filtered_data['airline'] == 'Delta'
# delta_negative = len(filtered_data.loc[(is_negative & is_delta), 'airline'])
# print(delta_negative)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split 
 
# X_clean = twitter_data["text"]
# #X_train, X_test, y_train, y_test = train_test_split(X_clean, Y, test_size=0.33, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X_clean, twitter_data["airline_sentiment"], test_size=0.33, random_state=42)
# vectorizer = TfidfVectorizer(max_df=0.5, max_features=2000, min_df=2, stop_words='english')
# X_train = vectorizer.transform(X_train)
# X_test = vectorizer.transform(X_test)



# # select K best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# word_list = vectorizer.get_feature_names()
# term_doc_mat = vectorizer.fit_transform(X_clean)  
# selector = SelectKBest(chi2, k=10).fit(term_doc_mat, twitter_data["airline_sentiment"])
# informative_words_index = selector.get_support(indices=True)
# labels = [word_list[i] for i in informative_words_index]
# print(labels)
# data = pd.DataFrame(term_doc_mat[:,informative_words_index].todense(), columns=labels)
# data['airline_sentiment'] = twitter_data["airline_sentiment"]
# # print (data.corr())
# # sns.heatmap(data.corr())

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
corpus = [
'This is the first document.',
'This is the second second document.',
'And the third one.',
'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)

def tokenize(s):
    return s.split(' ')

bigram_vectorizer = CountVectorizer(ngram_range=(1, 3),
                                    min_df=1, stop_words='english', tokenizer=tokenize)
analyze = bigram_vectorizer.build_analyzer()
m = analyze('Bi-grams are very cool and useful or do not use them!')
print(m)











