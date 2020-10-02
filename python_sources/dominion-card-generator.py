#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#Make sure Internet is toggled on in settings to retrieve data from Gutenberg

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import nltk
from nltk.corpus import gutenberg
from nltk import word_tokenize
from nltk.collocations import *

#Removes all pre-existing image files
from os import path
f = os.listdir()
for each in f:
    if (("png" in each) or ("jpg" in each)):
        os.remove(each)


# In[ ]:


commonNames = pd.read_csv('../input/commonnames/USCensusNameData - Sheet2.csv')
femaleEnglishNames = commonNames['FirstnameF']
maleEnglishNames = commonNames['FirstnameF.1']
lastEnglishNames = commonNames['Lastname']

#convert lists to lower case for list comprehension later
tempNames = []
for each in femaleEnglishNames:
    tempNames.append(each.lower())
femaleEnglishNames = tempNames

tempNames = []
for each in maleEnglishNames:
    tempNames.append(each.lower())
    #print(each.lower())
maleEnglishNames = tempNames

tempNames = []
for each in lastEnglishNames:
    tempNames.append(each.lower())
lastEnglishNames = tempNames


# In[ ]:


from nltk.corpus import brown
word_list = brown.words()
word_set = set(word_list)


# In[ ]:


#Pulls file from Gutenberg, tokenizes & POS tags
import urllib3
url = "https://www.gutenberg.org/files/16/16-0.txt" #insert url of book here
raw = urlopen(url).read()
response = request.urlopen(url)
raw = response.read().decode('utf8')
start = raw.find("START OF THIS PROJECT GUTENBERG EBOOK")
end = raw.find("END OF THIS PROJECT GUTENBERG EBOOK")
raw = raw[start:end]
tokens = word_tokenize(raw)
text = nltk.Text(tokens)
taggedText = nltk.pos_tag(text)


# In[ ]:


#Remove irrelevant parts of speech
removablePOS = ["CC", "PRON", "PRP", "MD", "VBD", "VB", "DT", "RB", "IN", "PRP$", "WP", "VBP", "CD"]
taggedText = [s for s in taggedText if (len(s[0]) > 3)]
for each in removablePOS:
    taggedText = [s for s in taggedText if s[1] != each]
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

#Import 100 most common nouns in English
#Added numbers and time of day to end
mostFrequentNouns = ["time","year","people","way","day",	"man",	"thing",	"woman",	"life",	"child",	"world",	"school",	"state",	"family",	"student",	"group",	"country",	"problem",	"hand",	"part",	"place",	"case",	"week",	"company",	"system",	"program",	"question",	"work",	"government",	"number",	"night",	"point",	"home",	"water",	"room",	"mother",	"area",	"money",	"story",	"fact",	"month",	"lot",	"right",	"study",	"book",	"eye",	"job",	"word",	"business",	"issue",	"side",	"kind",	"head",	"house",	"service",	"friend",	"father",	"power",	"hour",	"game",	"line",	"end",	"member",	"law",	"car",	"city",	"community",	"name",	"president",	"team",	"minute",	"idea",	"kid",	"body",	"information",	"back",	"parent",	"face",	"others",	"level",	"office",	"door",	"health",	"person",	"art",	"war",	"history",	"party",	"result",	"change",	"morning",	"reason",	"research",	"girl",	"guy",	"moment",	"air",	"teacher",	"force",	"education", "something", "nothing", "anything", "everything", "voice", "manner", "matter", "course", "heart", "thing", "earth","one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "round", "table", "morning", "afternoon", "night", "evening", "when", "dear", "hours", "months", "hour", "year", "month", "towards", "letter", "which", "when", "things", "that"]
frequentTitles = ["mister", "miss", "mrs.", "mr.", "madam", "mademoiselle", "madame", "signor", "sir", "count", "dear"]

#Function repeatedly lowers the minimum number of occurrences the bigram has to appear in the text to be counted
#Once ten eligible bigrams have been found, the function stops. 
minimumFrequency = 50
numValid = 0

while (numValid < 20 and minimumFrequency > 2):
    #print(minimumFrequency)
    finder = BigramCollocationFinder.from_words(taggedText)
    finder.apply_freq_filter(minimumFrequency)
    numValid = len(finder.nbest(bigram_measures.pmi, 20))
    minimumFrequency-=1

candidates = finder.nbest(bigram_measures.pmi, 20)

#Remove any bigrams that:
for each in candidates:
    #print(each)
    if ((each[1][1].startswith("N") == False) #don't end with a noun
        or (each[1][0].lower() in mostFrequentNouns) #end with one of 100 most common nouns + additional common words (from list above)
        or (len(each[1][0]) > 9) #are over nine characters long (to avoid excessive names)
        or (each[0][0].lower() in frequentTitles)
       ):
        #print(each)
        candidates.remove(each)
        
#Check original text to confirm words appear adjacent to each other
current = 0
approved = []

while current < len(candidates):
    confirmed = False
    count = 0
    while ((count < len(text)) and (confirmed == False)):
        if text[count] == candidates[current][0][0]: #later replace with variable, then list
            if text[(count + 1)] == candidates[current][1][0]:
                approved.append((candidates[current][0][0].capitalize()) + " " + (candidates[current][1][0].capitalize()))
                confirmed = True
        count+=1
    current+=1
    
#Remove any bigrams that share a word with a higher ranked bigram
#Also removes names
singlesList = []
pairsList = []

for each in approved:
    #print(each)
    test = each.split()
    if test[0] not in singlesList:
        if test[0].lower() not in maleEnglishNames and test[0].lower() not in femaleEnglishNames and test[0].lower() not in frequentTitles:
            #print(test[0])
            if test[1] not in singlesList:
                if test[1].lower() not in maleEnglishNames and test[0].lower() not in femaleEnglishNames:
                    if test[1].lower() not in mostFrequentNouns:
                        if test[1].lower() in word_set:
                            if test[0].lower() in word_set:
                                #print(each)
                                singlesList.append(test[0])
                                singlesList.append(test[1])
                                pairsList.append(each)
                                #print(pairsList)

#print(pairsList)            

#Take the top five remaining    
twoWordCards = pairsList[:5]
#In the event there are under five eligible bigrams, save this value to create more one word cards
numTwoWordCards = len(twoWordCards)
twoWordCards


# In[ ]:


#Find the five most commonly occurring words
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() 

abridgedText = []

for each in taggedText:
    #remove any words under four characters long, capital words, and non-nouns
    if((len(each[0]) > 4) and (each[0][0].islower()) and each[1].startswith("N")):
        #remove any words whose lemmas are part of 100 most common nouns
        if (lemmatizer.lemmatize(each[0]) not in mostFrequentNouns):
            abridgedText.append(each[0])
#abridgedText =            
fdist = FreqDist(abridgedText)

#Place the ten most commonly occurring words into a list
mostCommon = fdist.most_common(10)

#Remove any words with the same lemmas (especially plurals)
for each in mostCommon:
    for every in mostCommon:        
        if lemmatizer.lemmatize(each[0]) == lemmatizer.lemmatize(every[0]):
            if each != every:
                mostCommon.remove(every)

#Remove any words that already occur in a two word card
for each in mostCommon:
    for every in twoWordCards:
        #print(every)
        if each[0].capitalize() in every:
            #print(each[0])
            #print(every)
            mostCommon.remove(each)
            
#Remove any words that share the same first four characters (when lemmas don't matchup)
shortWords = []

for each in mostCommon:
    short = each[0][:4]
    if short not in shortWords:
        shortWords.append(each[0][:4])
    else:
        mostCommon.remove(each)         

#Take the number of entries necessary to complete a list of ten
mostCommon = mostCommon[:(10-numTwoWordCards)]
oneWordCards = []

for each in mostCommon:
    oneWordCards.append(each[0].capitalize())

#create final cardlist from text
textCardList = oneWordCards + twoWordCards

#Place all kingdom cards into list
df = pd.read_csv('../input/dominion/DominionCardInfo.csv')
dominionCardList = df['Name']
#dominionCardList = dominionCardList.tolist()
textCardList


# In[ ]:


dominionCardSynsets = []

for each in dominionCardList:
    words = each.split()
    for word in words:
        #Remove apostrophe mark for cards such as Philosopher's stone
        word = word.replace("\'", "")
        #Check whether word exists in sysnet prepositions such as 'of' do not
        if wordnet.synsets(word):
            dominionCardSynsets.append([wordnet.synsets(word)[0], word, each])
            #dominionSynsetReference.append(word)


# In[ ]:


from nltk.corpus import wordnet
import random
comparisonData = []
takenCards = []

#Takes a string of text and returns list where each entry is a word in the string
def convertToSynset(cardName):
    cardSynsets = []
    words = cardName.split()
    for word in words:
        try:
            cardSynsets.append(wordnet.synsets(word)[0])
        except:
            pass
    return cardSynsets

for textCard in textCardList:
    mostSimilar = ["", 0]
    textSynset = convertToSynset(textCard)
    for textWord in textSynset:
        for dominionWord in dominionCardSynsets:
            semanticSimilarity = textWord.wup_similarity(dominionWord[0])
            if semanticSimilarity:
                if semanticSimilarity > mostSimilar[1]:
                    mostSimilar[1] = semanticSimilarity
                    mostSimilar[0] = dominionWord[2]
            else:
                semanticSimilarity = ((random.randint(1, 40))/100)
                if semanticSimilarity > mostSimilar[1]:
                    mostSimilar[1] = semanticSimilarity
                    mostSimilar[0] = dominionWord[2]
    result = [textCard, mostSimilar[0], mostSimilar[1]]
    print(result)
    takenCards.append(mostSimilar[0])
    comparisonData.append(result)
#for each in comparisonData:
#    print(each)
    #print(textCard + " --- " + mostSimilar[0])


# In[ ]:


comparisonData


# The following code block removes any cases where multiple word phrases mapped onto the same dominion card. 
# Preference is given to the word phrase with the closest semantic similarity, with tie going to whichever word phrase appears earlier in the list. 

# In[ ]:


#Takes a list of string and returns same list with all duplicates removed
def removeDuplicates(someList):
    newList = []
    for each in someList:
        if each not in newList:
            newList.append(each)
    return newList
#print(takenCards)

takenCards = removeDuplicates(takenCards)
#print(takenCards)

while len(takenCards) < 10:
    #Remove all duplicates with lower similarity scores
    for x in comparisonData:
        for y in comparisonData:
            if ((x[0] != y[0]) and x[1] == y[1]):
                if (x[2] >= y[2]):
                    y[1] = ""
                    y[2] = 0
                else:
                    x[1] = ""
                    x[2] = 0
    #for each in comparisonData:
        #print(each)

    #Find a new card for any words that lost one
    for x in comparisonData:
        if (x[1] == ""):
            textSynset = convertToSynset(x[0])
            for textWord in textSynset:
                for dominionWord in dominionCardSynsets:
                    #print(dominionWord[2])
                    if(dominionWord[2]) not in takenCards:
                        semanticSimilarity = textWord.wup_similarity(dominionWord[0])
                        if semanticSimilarity:
                            if semanticSimilarity > x[2]:
                                x[2] = semanticSimilarity
                                x[1] = dominionWord[2]

    for each in comparisonData:
        #print(each)
        if each[1] not in takenCards:
            takenCards.append(each[1])
    #print(takenCards)
for x in comparisonData:
    print((x[0]) + " --- " + (x[1]))


# The next task is to work out card buffs and costs based on how frequently a word phrase appears in the text
# 
# mostCommon (single words, includes frequency already)
# twoWordCards (bigrams)

# In[ ]:


import statistics

def standardDeviation(numberList):
    deviationList = []
    if len(numberList) > 1:
        standardDeviation = (statistics.stdev(numberList) + 1)
        average = statistics.mean(numberList)
        for x in numberList:
            deviationList.append(round((x - average)/standardDeviation))
        return deviationList
    else:
        for x in numberList:
            deviationList.append(0)
        return deviationList

#Calculate rounded standard deviation for one word cards
appearances = []

for each in mostCommon:
    appearances.append(each[1])
    
oneWordCostChanges = standardDeviation(appearances)

#Calculate rounded standard deviation for two word cards
twoWordFrequencies = []
appearances = []

for x in twoWordCards:
    words = x.split()
    firstWord = words[0]
    secondWord = words[1]
    count = 0
    occurrence = 0
    while count < len(text):
        if firstWord.lower() == text[count] or firstWord == text[count]:
            if (secondWord.lower() == text[(count+1)]) or (secondWord == text[(count+1)]):
                #print(text[count] + " " + text[(count+1)] + str(count))
                occurrence+=1
                #print(words[0] + str(count)):
        count+=1
    total = [x,occurrence]
    #print(total)
    twoWordFrequencies.append(total)
    appearances.append(occurrence)
    
twoWordCostChanges = standardDeviation(appearances)
costChanges = oneWordCostChanges + twoWordCostChanges
costChanges
count = 0
while count < len(comparisonData):
    print(comparisonData[count][0] + " ... "+ comparisonData[count][1] + " ... " + str(costChanges[count]))
    count+=1
#for x in comparisonData:
 #   print((x[0]) + " --- " + (x[1]))


# In[ ]:


wordCount = len(taggedText)
#print(mostCommon)
for each in mostCommon:
    percent = ((each[1]/wordCount)*100)
    print( (each[0]) + ": " + str(percent))


# Code below is related to image creation. Leave aside for now.

# In[ ]:


#Be sure that Internet toggle is turned on in settings for dependencies to successfully install
get_ipython().system('pip install google-api-python-client')
get_ipython().system('pip install python-resize-image')
get_ipython().system('pip install Google-Images-Search')

from apiclient.discovery import build
#what to do when image function is used from two different libraries?
#from IPython.display import Image
from IPython.core.display import HTML 
from IPython.display import Markdown as md
from resizeimage import resizeimage
import PIL
from PIL import Image


# In[ ]:


#!pip install piexif


# In[ ]:


#import piexif
#from IPython.display import Image


# In[ ]:


cardList


# In[ ]:


api_key = "AIzaSyBcRQ5XGx042vZh8itsNhg6QS39WdDq_ZE" #from personal email Google Developers Account
resource = build("customsearch", "v1", developerKey=api_key).cse()

oneWordCards = ['Children', 'Course', 'Moment', 'Night']
base_set = ["cellar", "market", "merchant", "militia", "mine", "moat", "remodel", "bandit", "village", "workshop"]

count = 0
for each in cardList:
    
    #locates url of google image search result
    search_term = each
    search_term = search_term + " clip art"
    result = resource.list(q=search_term, cx='003240139077527252035:bj4glzhiden', searchType='image').execute()
    url = result['items'][0]['link']
    
    #defines filename
    #filename = each + ".jpg"
    filename = "image" + str(count) + ".png"
    #saves url as image
    urllib.request.urlretrieve(url, filename)
    
    #removes exif metadata
   # img = Image.open(filename)
    #piexif.transplant(filename, "remodel.jpg")
    #image = Image.new(image.mode, image.size)
    
    #resize image
   # basewidth = 200
    #wpercent = (basewidth / float(img.size[0]))
    #hsize = int((float(image.size[1]) * float(wpercent)))
    #img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    
    #save image so markdown can consistently pick it up
    #resized_filename = "image" + str(count) + ".png"
    #new_filename = "image" + str(count) + ".png"
    #img.save(new_filename)
    count = count + 1


# In[ ]:


print(PIL.PILLOW_VERSION)


# In[ ]:


#piexif.transplant("Mother.jpg", "Night.jpg")


# In[ ]:


exif_dict = ""

exif_bytes = piexif.dump(exif_dict)
piexif.insert(exif_bytes, filename)


# | Cellar | Market | Merchant | Militia | Mine | Moat | Remodel | Bandit | Village | Workshop |
# | :---: | :---: | :---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
# | ![alt](image0.png) | ![alt](image1.png) | ![alt](image2.png) | ![alt](image3.png) | ![alt](image4.png) | ![alt](image5.png) | ![alt](image6.png) | ![alt](image7.png) | ![alt](image8.png) | ![alt](image9.png) |
# 

# In[ ]:


cardNames = cardNames.values.tolist()


# In[ ]:


print(cardNames)


# In[ ]:


print(dominionSynsetReference)


# In[ ]:


print(dominionCardSynsets[0])


# In[ ]:


from nltk.corpus import wordnet

#Takes a string of text and returns list where each entry is a word in the string
def convertToSynset(cardName):
    cardSynsets = []
    words = cardName.split()
    for word in words:
        cardSynsets.append(wordnet.synsets(word)[0])
    return cardSynsets

test = 'villager'
testx = wordnet.synsets(test)[0]

for textCard in textCardList: 
    mostSimilar = ["", 0]
    textSynset = convertToSynset(textCard)
    for textWord in textSynset:
        for dominionWord in dominionCardSynsets:
            #print(textWord)
            #print(dominionWord[1])
            semanticSimilarity = textWord.wup_similarity(dominionWord[0])
            #print(semanticSimilarity)
            #print(mostSimilar[1])
            if semanticSimilarity:
                if semanticSimilarity > mostSimilar[1]:
                    mostSimilar[1] = semanticSimilarity
                    mostSimilar[0] = dominionWord[2]
    print(textCard + " is most similar to " + mostSimilar[0] + " " + str(mostSimilar[1]))
    #semanticSimilarity = synset.wup_similarity('villager')
    #print(semanticSimilarity)
    #print(synset)
    #for each in synset:
     #   semanticSimilarity = each.wup_similarity('villager')
      #  print(semanticSimilarity)
       # print('\n')
    #words = textCard.split()
    #for word in words:
       # wordx = wordnet.synsets(word)[0]
        #print(wordx)
        #for dominionCard in dominionCardList:
            


# In[ ]:


dominionCardSynsets = []
dominionSynsetReference = []

for each in dominionCardList:
    words = each.split()
    for word in words:
        #Remove apostrophe mark for cards such as Philosopher's stone
        word = word.replace("\'", "")
        #Check whether word exists in sysnet prepositions such as 'of' do not
        if wordnet.synsets(word):
            dominionCardSynsets.append([wordnet.synsets(word)[0], word, each])
            #dominionSynsetReference.append(word)


# In[ ]:


from nltk.corpus import wordnet

for card in textCardList: 
    mostSimilar = ["", 0]
    words = card.split()
    print(words)
    for each in cardNames:
        wordx = wordnet.synsets(word)[0]
        words = word.split
        if len(each.split()) == 1:
            if wordnet.synsets(each):
                card = wordnet.synsets(each)[0]
                semanticSimilarity = wordx.wup_similarity(card)
                if semanticSimilarity:
                    #print(str(word) + " has a " + str((round(wordx.wup_similarity(card) * 100))) + "% similarity to " + each)
                    if semanticSimilarity > mostSimilar[1]:
                        #print(each + " is currently the most semantically similar")
                        mostSimilar[1] = semanticSimilarity
                        mostSimilar[0] = each
                    elif semanticSimilarity == mostSimilar[1]:
                        #print(each + " is tied for most semantically similar")
                        mostSimilar[0] = (mostSimilar[0] + ", " + each)
                #else:
                    #print(each + " has no semantic relationship with " + word)
    print(mostSimilar[0] + " is/are the most semantically similar kingdom card(s) to " + word) 

#for each in words:
 #   word = wordnet.synsets(each)
  #  print(each)
   # print(word)


# In[ ]:


from nltk.corpus import wordnet

for word in cardList: 
    mostSimilar = ["", 0]
    for each in cardNames:
        wordx = wordnet.synsets(word)[0]
        words = word.split
        if len(each.split()) == 1:
            if wordnet.synsets(each):
                card = wordnet.synsets(each)[0]
                semanticSimilarity = wordx.wup_similarity(card)
                if semanticSimilarity:
                    #print(str(word) + " has a " + str((round(wordx.wup_similarity(card) * 100))) + "% similarity to " + each)
                    if semanticSimilarity > mostSimilar[1]:
                        #print(each + " is currently the most semantically similar")
                        mostSimilar[1] = semanticSimilarity
                        mostSimilar[0] = each
                    elif semanticSimilarity == mostSimilar[1]:
                        #print(each + " is tied for most semantically similar")
                        mostSimilar[0] = (mostSimilar[0] + ", " + each)
                #else:
                    #print(each + " has no semantic relationship with " + word)
    print(mostSimilar[0] + " is/are the most semantically similar kingdom card(s) to " + word) 

#for each in words:
 #   word = wordnet.synsets(each)
  #  print(each)
   # print(word)


# In[ ]:


from nltk.corpus import wordnet

word = wordnet.synsets('oasis')[0] 

action = wordnet.synsets('action')[0]
attack = wordnet.synsets('attack')[0]
treasure = wordnet.synsets('treasure')[0]
event = wordnet.synsets('event')[0]
victory = wordnet.synsets('victory')[0]
reaction = wordnet.synsets('reaction')[0]
duration = wordnet.synsets('duration')[0]
ruins = wordnet.synsets('ruins')[0]
shelter = wordnet.synsets('shelter')[0]
reserve = wordnet.synsets('reserve')[0]

print(word)
print(str(word.wup_similarity(action)) + " action")
print(str(word.wup_similarity(attack)) + " attack")
print(str(word.wup_similarity(treasure)) + " treasure")
print(str(word.wup_similarity(event)) + " event")
print(str(word.wup_similarity(victory)) + " victory")
print(str(word.wup_similarity(reaction)) + " reaction")
print(str(word.wup_similarity(duration)) + " duration")
print(str(word.wup_similarity(ruins)) + " ruins")
print(str(word.wup_similarity(shelter)) + " shelter")
print(str(word.wup_similarity(reserve)) + " reserve")


# TASK
# Write a simple script that takes an array of words and for each returns the title of an existing dominion card that is the most semantically related based on Wu-Palmer similarity

# In[ ]:


word = "feodum"
if wordnet.synsets(word):
    print(wordnet.synsets("feodum"))


# testing of wordnet below

# In[ ]:


from nltk.corpus import wordnet
#example output from Alice & Wonderland
#note that string concatenation can be done in Sheets with =char(34)&A1&char(34)
words = ["cheshire cat", "queen", "golden key", "rabbit-hole", "caterpillar", "mad hatter"]
wordTest = ["cardinal", "cavalry", "fair", "fisherman", "supplies", "sanctuary", "mastermind"]
dominionCardTitles = ["Envoy", "Governor", "Prince", "Stash", "Summon", "Garden", "Black Market", 	"Walled Village", 	"Adventurer", 	"Bureaucrat", 	"Cellar", 	"Chancellor", 	"Chapel", 	"Council Room", 	"Feast", 	"Festival", 	"Gardens", 	"Laboratory", 	"Library", 	"Market", 	"Militia", 	"Mine", 	"Moat", 	"Moneylender", 	"Remodel", 	"Smithy", 	"Spy", 	"Thief", 	"Throne Room", 	"Village", 	"Witch", 	"Woodcutter", 	"Workshop", 	"Baron", 	"Bridge", 	"Conspirator", 	"Coppersmith", 	"Courtyard", 	"Duke", 	"Great Hall", 	"Harem", 	"Ironworks", 	"Masquerade", 	"Mining Village", 	"Minion", 	"Nobles", 	"Pawn", 	"Saboteur", 	"Scout", 	"Secret Chamber", 	"Shanty Town", 	"Steward", 	"Swindler", 	"Torturer", 	"Trading Post", 	"Tribute", 	"Upgrade", 	"Wishing Well", 	"Ambassador", 	"Bazaar", 	"Caravan", 	"Cutpurse", 	"Embargo", 	"Explorer", 	"Fishing Village", 	"Ghost Ship", 	"Haven", 	"Island", 	"Lighthouse", 	"Lookout", 	"Merchant Ship", 	"Native Village", 	"Navigator", 	"Outpost", 	"Pearl Diver", 	"Pirate Ship", 	"Salvager", 	"Sea Hag", 	"Smugglers", 	"Tactician", 	"Treasure Map", 	"Treasury", 	"Warehouse", 	"Wharf", 	"Alchemist", 	"Apothecary", 	"Apprentice", 	"Familiar", 	"Golem", 	"Herbalist", 	"Philosopher's Stone", 	"Possession", 	"Scrying Pool", 	"Transmute", 	"University", 	"Vineyard", 	"Bank", 	"Bishop", 	"City", 	"Contraband", 	"Counting House", 	"Expand", 	"Forge", 	"Goons", 	"Grand Market", 	"Hoard", 	"King's Court", 	"Loan", 	"Mint", 	"Monument", 	"Mountebank", 	"Peddler", 	"Quarry", 	"Rabble", 	"Royal Seal", 	"Talisman", 	"Trade Route", 	"Vault", 	"Venture", 	"Watchtower", 	"Worker's Village", 	"Bag of Gold", 	"Diadem", 	"Fairgrounds", 	"Farming Village", 	"Followers", 	"Fortune Teller", 	"Hamlet", 	"Harvest", 	"Horn of Plenty", 	"Horse Traders", 	"Hunting Party", 	"Jester", 	"Menagerie", 	"Princess", 	"Remake", 	"Tournament", 	"Trusty Steed", 	"Young Witch", 	"Border Village", 	"Cache", 	"Cartographer", 	"Crossroads", 	"Develop", 	"Duchess", 	"Embassy", 	"Farmland", 	"Fool's Gold", 	"Haggler", 	"Highway", 	"Ill-Gotten Gains", 	"Inn", 	"Jack of All Trades", 	"Mandarin", 	"Margrave", 	"Noble Brigand", 	"Nomad Camp", 	"Oasis", 	"Oracle", 	"Scheme", 	"Silk Road", 	"Spice Merchant", 	"Stables", 	"Trader", 	"Tunnel", 	"Colony", 	"Copper", 	"Curse", 	"Duchy", 	"Estate", 	"Gold", 	"Platinum", 	"Potion", 	"Province", 	"Silver", 	"Abandoned Mine", 	"Altar", 	"Armory", 	"Band of Misfits", 	"Bandit Camp", 	"Beggar", 	"Catacombs", 	"Count", 	"Counterfeit", 	"Cultist", 	"Dame Anna", 	"Dame Josephine", 	"Dame Molly", 	"Dame Natalie", 	"Dame Sylvia", 	"Death Cart", 	"Feodum", 	"Forager", 	"Fortress", 	"Graverobber", 	"Hermit", 	"Hovel", 	"Hunting Grounds", 	"Ironmonger", 	"Junk Dealer", 	"Madman", 	"Marauder", 	"Market Square", 	"Mercenary", 	"Mystic", 	"Necropolis", 	"Overgrown Estate", 	"Pillage", 	"Poor House", 	"Procession", 	"Rats", 	"Rebuild", 	"Rogue", 	"Ruined Library", 	"Ruined Market", 	"Ruined Village", 	"Sage", 	"Scavenger", 	"Sir Bailey", 	"Sir Destry", 	"Sir Martin", 	"Sir Michael", 	"Sir Vander", 	"Spoils", 	"Squire", 	"Storeroom", 	"Survivors", 	"Urchin", 	"Vagrant", 	"Wandering Minstrel", 	"Advisor", 	"Baker", 	"Butcher", 	"Candlestick Maker", 	"Doctor", 	"Herald", 	"Journeyman", 	"Masterpiece", 	"Merchant Guild", 	"Plaza", 	"Soothsayer", 	"Stonemason", 	"Taxman", 	"Alms", 	"Amulet", 	"Artificer", 	"Ball", 	"Bonfire", 	"Borrow", 	"Bridge Troll", 	"Caravan Guard", 	"Champion", 	"Coin of the Realm", 	"Disciple", 	"Distant Lands", 	"Dungeon", 	"Duplicate", 	"Expedition", 	"Ferry", 	"Fugitive", 	"Gear", 	"Giant", 	"Guide", 	"Haunted Woods", 	"Hero", 	"Hireling", 	"Inheritance", 	"Lost Arts", 	"Lost City", 	"Magpie", 	"Messenger", 	"Miser", 	"Mission", 	"Page", 	"Pathfinding", 	"Peasant", 	"Pilgrimage", 	"Plan", 	"Port", 	"Quest", 	"Raid", 	"Ranger", 	"Ratcatcher", 	"Raze", 	"Relic", 	"Royal Carriage", 	"Save", 	"Scouting Party", 	"Seaway", 	"Soldier", 	"Storyteller", 	"Swamp Hag", 	"Teacher", 	"Trade", 	"Training", 	"Transmogrify", 	"Travelling Fair", 	"Treasure Hunter", 	"Treasure Trove", 	"Warrior", 	"Wine Merchant"]

#Find the dominion card that most closely resembles word from text
for word in words:
    topScore = 0
    test = word.split()
    for each in test:
        print(each)


# These next two code blocks (and following markdown) retrieve and display clip art for the ten kingdom card names provided. 
# 

# TODO: Figure out a way to include the instructions for each card here. 
# Current thought: explore ways to combine images in Python, so that any combination (+x actions/cards/etc) can be concatenated into a single image, saved, and then displayed in the subsequent row in the above markdown. 
# May be necessary to split into two rows of 5 cards each.
