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
import warnings
from wordcloud import WordCloud
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

#Setting dataframe max limit of columns in output to 5
pd.set_option('display.max_columns', 5)

#Importing data - First 10000 records
data = pd.read_csv("../input/googleplaystore_user_reviews.csv")[:10000]

#Removing NaNs from Translated_Review & Sentiment columns
data= data[pd.notnull(data['Translated_Review'])]
data = data[pd.notnull(data['Sentiment'])]

#Displaying first 10 records
print(data.head(10))

#Droping columns from the data
data = data.drop(['App', 'Sentiment_Polarity', 'Sentiment_Subjectivity'], axis=1)
print(data.head(10))

#Printing the shape of the data and its description
print(data.shape)
print(data.describe())

#Checking the class labels distribution
classes = data['Sentiment']
print(classes.value_counts())

#Transforming class labels to numerical values
#1: Neutral
#2: Positive
#0:Negative
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
print("Displaying encoded Sentiments:",Y[:10])
print("Negative:", 0)
print("Neutral:", 1)
print("Positive:", 2)

#Using regular expressions for filtering expressions etc
processed = data['Translated_Review'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',' ')
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', ' ')
processed = processed.str.replace(r'\d+',' ')
processed = processed.str.lower()

#Removing stop words from reviews
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words))

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

#Using the 3000 words as features
word_features = list(all_words.keys())[:3000]

#The find_features function will determine which of the 3000 word features are contained in the reviews
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

#Unifying reviews with their respective encoded class labels
messages = list(zip(processed, Y))
print("Testing(Unification of reviews and labels):", messages[0])

#Defining a seed for reproducibility and shuffling
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

#Calling find_features function for each review
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

# Bag of words - Negative/Neutral/Positive reviews
neut=[];
neg=[];
pos=[];
unifyy = list(zip(prediction, txt_features, labels))
for p, t, l in unifyy:
    for key, value in t.items():
        if value==True and l==p==0:
            neg.append(key)
            break
        elif value==True and l==p==1:
            neut.append(key)
            break
        elif value==True and l==p==2:
            pos.append(key)
            break
print("Negative Words:", neg)
print("Neutral Words:", neut)
print("Positive Words:", pos)

#Printing a confusion matrix and classification report
print(classification_report(labels, prediction))

#Displaying the false positives and True positives in confusion matrix
df = pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual', 'actual'], ['Negative','Neutral', 'Positive']],
    columns = [['predicted', 'predicted', 'predicted'], ['Negative','Neutral', 'Positive']])
print(df)

#Generating wordcloud for Negative, Neutral, and Positive words
negative = str(neg)
wordCloud = WordCloud(background_color="white").generate(negative)
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews - Words')
plt.show()

neutral = str(neut)
wordCloud = WordCloud(background_color="white").generate(neutral)
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.title('Neutral Reviews - Words')
plt.show()

positive = str(pos)
wordCloud = WordCloud(background_color="white").generate(positive)
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews - Words')
plt.show()