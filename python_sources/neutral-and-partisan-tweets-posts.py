# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import model_selection
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#Setting dataframe max limit of columns in output to 10
pd.set_option('display.max_columns', 10)

#Loading data and printing the columns
data = pd.read_csv("../input/political_social_media.csv", encoding = "ISO-8859-1")
print(data.columns)

#Printing the shape of the data and its description
print(data.shape)
print(data.describe())

#Checking the class labels distribution
classes = data['bias']
print(classes.value_counts())

#Transforming class labels to numerical values
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
print("Neutral encoded to:", 0)
print("Partisan encoded to:", 1)

#Using regular expressions to remove URLs, numbers etc
processed = data['text'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',' ')
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',' ')
processed = processed.str.replace(r'http',' ')
processed = processed.str.replace(r'£|\$', ' ')
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',' ')
processed = processed.str.replace(r'\d+(\.\d+)?', ' ')
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', ' ')
processed = processed.str.lower()

#Removing stop words from text
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words))

#Removing meaningless words from text
processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in ['ûªs', 'û', 'ûªt', 'r', 'en', 'fl', 'p', 'va', 'amp', 'icymi', 'th', 'pm', 'hours', 'u']))

#Creating bag-of-words
all_words = []
for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)

#Printing the total number of words and the 15 most common words
print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(15)))


# use the 1500 most common words as features
word_features = list(all_words.keys())[:1500]

# The find_features function will determine which of the 1500 word features are contained in the text
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

#Unifying text with their respective encoded class labels
messages = list(zip(processed, Y))
print("Testing(Unification of texts and labels):", messages[0])

#Define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

#Calling find_features function for each tweet/message
featuresets = [(find_features(text), label) for (text, label) in messages]

#Splitting the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)
print("Training set:", len(training))
print("Testing set:", len(testing))

#Defining Naive Bayes model for training
model = MultinomialNB()

#Training the model on the training data and calculating accuracy
nltk_model = SklearnClassifier(model)
nltk_model.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Naive Bayes Accuracy: {}".format(accuracy))

#Listing the predicted labels for testing dataset
txt_features, labels = list(zip(*testing))
prediction = nltk_model.classify_many(txt_features)

# Bag of words - Neutral/Partisan tweets/Posts
neut=[];
part=[];
unifyy = list(zip(prediction, txt_features, labels))
for p, t, l in unifyy:
    for key, value in t.items():
        if value==True and l==p==0:
            neut.append(key)
            break
        elif value==True and l==p==1:
            part.append(key)
            break
print("Neutral Words:", neut)
print("Partisan Words:", part)

#Printing a confusion matrix and classification report
print(classification_report(labels, prediction))

#Displaying the false positives and True positives in confusion matrix
df = pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['Neutral','Partisan']],
    columns = [['predicted', 'predicted'], ['Neutral', 'Partisan']])
print(df)

#Generating wordcloud for Neutral and Partisan words
atck = str(neut)
wordCloud = WordCloud(background_color="white").generate(atck)
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.title('US Senators/Representatives - Neutral Tweets/Posts Words')
plt.show()

sprt = str(part)
wordCloud = WordCloud(background_color="white").generate(sprt)
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.title('US Senators/Representatives - Partisan Tweets/Posts Words')
plt.show()
