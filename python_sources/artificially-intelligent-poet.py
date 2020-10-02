import pandas as pd
import random
import string
import time
import os
import re

# Create rhyme maps from the previous song data

def get_rhymes(Lasts):
	mapRhymes = {}
	for i in range(len(Lasts)):
		if(len(Lasts[i])>1):
			ending 				=  Lasts[i][len(Lasts[i]) - 3:]
			mapRhymes[ending] 	= mapRhymes.get(ending, []) + [Lasts[i]]
	return(mapRhymes)


# Logger for console output

def log(phrase, millis, flag):
	if(flag == 1):
		millis = int(round(time.time() * 1000))-millis
		print(phrase + str(millis))	
		millis = int(round(time.time() * 1000))
	else:
		print(phrase)
	return(millis)
	
	
# Remove punctuations and other unnesc. clutter

def clean_file(train):
    train = train.split(".")
    train = [x.lower() for x in train]
    train = [re.sub('['+string.punctuation+']', '', x) for x in train]
    return train
    
    
# Create start, central and end maps (dictionary) from the songs

def create_map(train):

	Firsts 	= []
	Lasts 	= []
	Centres = []

	mapFirsts 		= {}
	mapCentres 		= {}
	mapLasts 		= {}
	for i in range(len(train)):
		sentence = train[i]
		sentence = sentence.split(" ")
		if(len(sentence) > 3):
			Firsts.append(sentence[0])
			mapFirsts[sentence[0]] = mapFirsts.get(sentence[0], []) + [sentence[1]]
			for j in range(1,len(sentence)-3):
				Centres.append(sentence[j])
				mapCentres[sentence[j-1]] = mapCentres.get(sentence[j-1], []) + [sentence[j]]
			Lasts.append(sentence[len(sentence)-1])
			mapLasts[sentence[len(sentence)-1]] = mapLasts.get(sentence[len(sentence)-1], []) + [sentence[len(sentence)-2]]
	return([mapFirsts, mapCentres, mapLasts, Firsts, Centres, Lasts])


# Write poems. Magic?

def write_poem(Firsts, Centres, Lasts, mapFirsts, mapCentres, mapLasts, mapRhymes, lines, catches):
    millis = 0
    poem = ""
    for i in range(lines):
        line1 	= random.choice(Firsts)	
        word  	= random.choice(mapFirsts[line1])
        line1   = line1.title()
        line1 	= line1 + " " + word
        for j in range(catches - 2):
            while(True):
                try:
                    word 	= random.choice(mapCentres[word])
                    line1 	= line1 + " " + word
                    break
                except:
                    line1 	= random.choice(Firsts)	
                    word  	= random.choice(mapFirsts[line1])
                    line1 	= line1 + " " + word

        while(True):
            wordl 	= random.choice(Lasts)
            if(len(wordl)>2):
            	word1 	= random.choice(mapLasts[wordl])
            	if(word1 in mapCentres[word]):
            		line1 	= line1 + " " + word1 + " "  + wordl + ",\n"
            		endword = random.choice(mapRhymes[wordl[len(wordl)-3:]])
            		break
        
        log(line1, millis, 0)
        poem += line1 + ",\n"
        
        line1 	= random.choice(Firsts)	
        word  	= random.choice(mapFirsts[line1])
        line1   = line1.title()
        line1 	= line1 + " " + word
        for j in range(catches - 2):
            while(True):
                try:
                	word 	= random.choice(mapCentres[word])
                	line1 	= line1 + " " + word
                	break
                except:
                    line1 	= random.choice(Firsts)	
                    word  	= random.choice(mapFirsts[line1])
                    line1 	= line1 + " " + word
                
        tries = 0
        while(True):
            if(tries>100):
                break
            
            tries += 1
            word1 	= random.choice(mapLasts[endword])
            if(word1 in mapCentres[word]):
                line1 	= line1 + " " + word1 + " "  + endword + ",\n"
                break
        
        log(line1, millis, 0)
        poem += line1 + ",\n"
        print(i)
    
    return poem

# Get train text

millis 			= int(round(time.time() * 1000))
LyricsData		= []
df              = pd.read_csv("../input/Lyrics1.csv")
df              = df.sample(5000)
song            = ['.'.join(x.splitlines()) for x in df["Lyrics"]]
LyricsData      = '.'.join(song)
train           = clean_file(LyricsData)

mapFirsts, mapCentres, mapLasts, Firsts, Centres, Lasts = create_map(train)

# Map with lag 1 (Hope to create a probabilistic model)

millis			= log("Processed document in ", millis, 1)
mapRhymes  		= get_rhymes(Lasts)
millis 			= log("Created rhyme maps for the document in ", millis, 1)
millis 			= log("Now I will start writing!\n", millis, 0)

poem = write_poem(Firsts, Centres, Lasts, mapFirsts, mapCentres, mapLasts, mapRhymes, 14, 5)

f = open('output.txt', 'w')
f.write(poem)  # python will convert \n to os.linesep
f.close() 