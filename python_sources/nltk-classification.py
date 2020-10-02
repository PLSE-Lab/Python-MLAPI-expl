
#Importing necessary header files
import nltk
import csv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment import SentimentAnalyzer


#Initialising the sentiment analyser model
sentim_analyzer = SentimentAnalyzer()


#Initialising the porter stemmer model
stemmer = PorterStemmer()

#Opening the file    
my_file = open("../input/Lonely.csv", 'r')

#Converting the file to a list
TrainingList= csv.reader(my_file)
TrainingList = list(TrainingList)  

#Closing the file
my_file.close()


#Importing the stopwords  
stopwords_set = set(stopwords.words("english"))


#Initialising a list to store all tokenized words(Dictionary)
words=[]

#Tokenizing all the words
for item  in TrainingList:
    for word in nltk.word_tokenize(item[1]):   #item[1]=Sentence
        x=stemmer.stem(word.lower())           #Converting all words to lower case and stemming them
        if x not in stopwords_set:             #Filtering according to stopwords list
            words.append(x)


#Initialising lists    
Sample=[]    
FinalList=[]

#Creating a list that checks the presence of words from dictionary for every sentence along with the class label
for x in TrainingList:
    Sample=[({word:word in nltk.word_tokenize(x[1]) for word in words},x[0])]
    FinalList.extend(Sample)
#Sample list item: 
#[({'feel': False, 'lone': False, 'sad': False, 'one': False, 'love': False, 'wish': False, 'could': False, 'talk': False, 'someon': False, 'isol': False, 'everyon': False, 'hate': False, 'want': False, 'die': False, 'hurt': False, 'today': False, 'life': False, 'know': False, 'go': False, 'away': False, "n't": False, 'like': True, 'anyon': False, 'sit': False, 'alon': False, 'bad': False, 'day': True, 'kill': False, 'human': False, 'drink': False, 'cut': False, 'end': False, 'cat': False, 'thought': False, 'suicid': False, 'poor': False, 'celebr': False, 'birthday': False, 'futur': False, 'past': False, 'haunt': False, 'pain': False, 'onli': False, 'real': False, 'thing': False, 'listen': False, 'eat': False, 'happi': False, 'world': False, 'look': False, 'good': True, 'need': False, 'help': False, 'enjoy': False, 'danc': False, 'call': False, 'mom': False, 'flower': False, 'sing': False, 'play': False, 'wa': False, 'awesom': False, 'live': False, 'fullest': False, 'got': False, 'news': False, 'fun': False, 'fantast': False, "'s": False, 'alway': False, 'time': False, 'vibe': False, 'convers': False, 'dad': False, 'plan': False, '100': False, 'year': False, 'never': False, 'happier': False, 'content': False, 'famili': False, 'ca': False, 'wait': False, 'meet': False, 'new': False, 'peopl': False, 'posit': False}, 'not lonely')]


#Training using NaiveBayes classifer 
model = nltk.NaiveBayesClassifier.train(FinalList)

#Example test sentence
test_sentence = "I want to kill myself!"
print('Test Sentence= ',test_sentence)

#Converting sentence to desired format for model to understand
for word in nltk.word_tokenize(test_sentence):   
        word= stemmer.stem(word.lower())           #Converting all words to lower case and stemming them
        
test_details={word:word in nltk.word_tokenize(test_sentence) for word in words}


#Classifying the sentence    
print('Tag=',model.classify(test_details))

