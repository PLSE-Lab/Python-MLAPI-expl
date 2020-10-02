import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn import linear_model, ensemble, tree, naive_bayes, svm, neighbors
import xgboost
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

common_aux = ["evet", "hayir", "gibi", "kadar", "en", "cok", "icin", "ile", "bile", "zaten", "beri", 
            "tamam", "ben" ,"beni", "sen", "seni", "sana", "bana", "o", "ona", "onun", "benim", "senin", 
            "onlar", "biz", "siz", "bizim", "sizin", "kendi", "kendisi", "ki", "bu", "su", "o"]

def data_cleansing(corpus):
    letters_only = re.sub("[^a-zA-Z]", " ", corpus) 
    words = letters_only.lower().split()                            
    return (" ".join(words))    
    
df = pd.read_csv("../input/lyrics.csv", encoding='latin_1')
df["lyrics"] = df["lyrics"].apply(lambda x: data_cleansing(x))

# word cloud
# all_words = df['lyrics'].str.split(expand=True).unstack().value_counts()
# all_words_sa = df['lyrics'].loc[df['author']  == "sabahattin ali"].str.split(expand=True).unstack().value_counts()
# all_words_nhr = df['lyrics'].loc[df['author']  == "nazim hikmet ram"].str.split(expand=True).unstack().value_counts()
# wordcloud = WordCloud(background_color="black", width=3000, height=2500).fit_words(all_words)
# wordcloud_sa = WordCloud(background_color="red", width=3000, height=2500).fit_words(all_words_sa)
# wordcloud_nhr = WordCloud(background_color="green", width=3000, height=2500).fit_words(all_words_nhr)
# plt.imshow(wordcloud)
# plt.imshow(wordcloud_sa)
# plt.imshow(wordcloud_nhr)

##### develop model ######
# vectorize
vectorizer = TfidfVectorizer()
df_features = vectorizer.fit_transform(df["lyrics"])
df_feature_names = vectorizer.get_feature_names()

# map authors
df["author"] = df["author"].map({"sabahattin ali": 0, "nazim hikmet ram": 1})

# train, test split
train_X, test_X, train_Y, test_Y = train_test_split(df_features, df["author"].values, test_size=0.30, random_state=123465)

# model list
CLASSIFIERS = [linear_model.LogisticRegressionCV(),
            neighbors.KNeighborsClassifier(),
            tree.DecisionTreeClassifier(),
            tree.ExtraTreeClassifier(),
            ensemble.RandomForestClassifier(),
            ensemble.GradientBoostingClassifier(),
            xgboost.XGBClassifier(),
            naive_bayes.BernoulliNB()]


for clf in CLASSIFIERS:
    clf.fit(train_X, train_Y)
    prediction = clf.predict(test_X)
    score = accuracy_score(test_Y, prediction)
    print("Accuracy score of {} model is {}".format(clf.__class__.__name__, score))
    

# variable importance
importance = {}
model = ensemble.RandomForestClassifier()
model.fit(train_X, train_Y)
importance_list = model.feature_importances_

for i in range(len(importance_list)):
    importance[df_feature_names[i]] = importance_list[i]

sorted_importance = [(k, importance[k]) for k in sorted(importance, key=importance.get, reverse=True)]

test_X = vectorizer.inverse_transform(test_X)

for i in range(20):
    print(test_X[i], test_Y[i], prediction[i])
