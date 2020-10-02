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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
import warnings
from wordcloud import WordCloud
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

#Setting dataframe max limit of columns in output to 5
pd.set_option('display.max_columns', 5)

#Importing data and displaying first 10 records
data = pd.read_csv('../input/amazon_alexa.tsv',delimiter='\t',encoding='utf-8')
print(data.head(10))

#Displaying the shape and description of the data
print("Shape:", data.shape)
print(data.describe())

#Dropping date and variation columns
data = data.drop(['date', 'variation'], axis=1)
print(data.head(5),"\n")

#Checking the class labels distribution
classes = data['feedback']
print(classes.value_counts(),"\n")

ratings = data['rating']
print(ratings.value_counts())

#Using regular expressions to remove URLs, numbers etc
processed = data['verified_reviews'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',' ')
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',' ')
processed = processed.str.replace(r'http',' ')
processed = processed.str.replace(r'£|\$', ' ')
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',' ')
processed = processed.str.replace(r'\d+(\.\d+)?', ' ')
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

#Using the 1000 words as features
word_features = list(all_words.keys())[:1000]

#The find_features function will determine which of the 1000 word features are contained in the reviews
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

#Unifying reviews with their respective class labels
messages = list(zip(processed, data['feedback']))
print("Testing(Unification of review, feedback, and rating):", messages[0])

#Defining a seed for reproducibility and shuffling
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

#Forming a featureset from reviews and labels
featuresets = [(find_features(text), label) for (text, label) in messages]
print(featuresets[0])

#Splitting the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=1)
print("Training set:", len(training))
print("Testing set:", len(testing))

#Defining Naive Bayes model for training
model = MultinomialNB()

#Training the model on the training data and calculating accuracy
nltk_model = SklearnClassifier(model)
nltk_model.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Naive Bayes Accuracy: {}".format(accuracy))

#Listing the predicted labels for testing dataset and computing error value
txt_features, labels = list(zip(*testing))
prediction = nltk_model.classify_many(txt_features)
print("Mean Absoulte Error:", mean_absolute_error(prediction, labels) *100)

# Bag of words - Negative/Positive reviews w.r.t ratings
neg_1=[];neg_2=[];pos_2=[];pos_3=[];pos_4=[];pos_5=[];
unifyy = list(zip(prediction, txt_features, labels, data['rating']))
for p, t, l, r in unifyy:
    for key, value in t.items():
        if value==True and l==p==0 and r==1:
            neg_1.append(key)
            break
        elif value==True and l==p==0 and r==2:
            neg_2.append(key)
            break
        elif value==True and l==p==1 and r==2:
            pos_2.append(key)
            break
        elif value==True and l==p==1 and r==3:
            pos_3.append(key)
            break
        elif value==True and l==p==1 and r==4:
            pos_4.append(key)
            break
        elif value==True and l==p==1 and r==5:
            pos_5.append(key)
            break
print("Negative review words with 1 rating:", neg_1)
print("Negative review words with 2 rating:", neg_2)
print("Positive review words with 2 rating:", pos_2)
print("Positive review words with 3 rating:", pos_3)
print("Positive review words with 4 rating:", pos_4)
print("Positive review words with 5 rating:", pos_5)

#Printing a confusion matrix and classification report
print(classification_report(labels, prediction))

#Displaying the false positives and True positives in confusion matrix
df = pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0','1']])
print(df)

#Generating wordcloud for reviews
if(len(neg_1) > 0):
    negtv_1 = str(neg_1)
    wordCloud = WordCloud(background_color="white").generate(negtv_1)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative reviews with 1 rating Words')
    plt.show()

if(len(neg_2) > 0):
    negtv_2 = str(neg_2)
    wordCloud = WordCloud(background_color="white").generate(negtv_2)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative reviews with 2 rating Words')
    plt.show()

if(len(pos_2) > 0):
    postv_2 = str(pos_2)
    wordCloud = WordCloud(background_color="white").generate(postv_2)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive reviews with 2 rating Words')
    plt.show()

if(len(pos_3) > 0):
    postv_3 = str(pos_3)
    wordCloud = WordCloud(background_color="white").generate(postv_3)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive reviews with 3 rating Words')
    plt.show()

if(len(pos_4) > 0):
    postv_4 = str(pos_4)
    wordCloud = WordCloud(background_color="white").generate(postv_4)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive reviews with 4 rating Words')
    plt.show()

if(len(pos_5) > 0):
    postv_5 = str(pos_5)
    wordCloud = WordCloud(background_color="white").generate(postv_5)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive reviews with 5 rating Words')
    plt.show()