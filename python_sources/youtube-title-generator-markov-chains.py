#!/usr/bin/env python
# coding: utf-8

# ## About the data
# So the data basically consists of YouTube titles from a certain channel which (imo) has very very weird titles. The data has been cleaned and I've made a few changes to the raw data, like replacing a few characters which were not utf-8 etc. Any sort of feedback is highly welcomed and appreciated. 
# I got this idea from [Jarvis Johnson's video](https://www.youtube.com/watch?v=KWqaUHFhE8M)
# Enjoy the titles generated. They did make me laugh :)
# You could contact me for the dataset any time!

# So my basic approach is to divide the the text generation into 4 parts : 
# 
# * Read the file(Various titles)
# * Assemble the Chain
# * Generate the New Title
# * Output

# ## 1. Read the file

# In[ ]:


def Read_file(FileName):
    with open(FileName,'r') as file:
        for line in file: #The delimiter for a file is \n by default
            for word in line.replace('\n','').split(' '): #Splits according to space and replaces '\n' by nothing
                yield word #Returns a sequnce/ series of words


# In[ ]:


Words = Read_file('../input/Title_Data.txt')


# ## 2. Assemble the Chain
# So in order to assemble the chain, we create a dictionary which basically keeps track of what the present "state" (word) is and what the next "state" is.

# In[ ]:


from collections import defaultdict
def assemble_chain(Words):
    chain = defaultdict(list) #create a dictionary which maps word to list
    try:
        word,next_word = next(Words),next(Words) #from iterables
        while True:
            chain[word].append(next_word)
            word,next_word = next_word, next(Words)
    except StopIteration:  #Error which arises when no words left
        return chain


# In[ ]:


Chain = assemble_chain(Words)
#print(Chain)


# ## 3. Generate a Random Chain
# You can also use the random_word() with the Chain too, but all titles of my data set began with I or My so I have used them as the first word.The chain is continued by a word from the list of words that has been linked the present word.

# In[ ]:


#Create the random word generator
from random import choice

def random_word(sample): #Could be any list of words cos Im using it only for beginning
    return choice(list(sample)) #Converting the dictionary to a list and then passing to choice
    
#Create a chain based on the first state

def random_title(Sample):
    word = random_word(['I','My'])#random_word(Sample)
    i = 0.0
    while True:
        yield word
        if word in Sample:
            word = random_word(Sample[word]) #Cos we only need the words which have a chance of coming after the given word
            
        else:
            i = i+1
            word = random_word(Sample)
        if(word[-1] =='.' or word[-1]=='!' or word[-1]=='?'):
            yield word
            #print(i)
            break


# ## 4. Generating Output
# I chose to add the full stop at the end because it would help the algorithm identify the end instead of ending at random words.

# In[ ]:


Title = random_title(Chain)
T_l = list(Title)
print(*T_l, sep=' ')

