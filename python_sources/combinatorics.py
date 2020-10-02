#!/usr/bin/env python
# coding: utf-8

# #  Combinatorics

# This is just a trial notebook to solve a simple combinatorics problem. The problem (from Miss Cave) is:
# >Four letters are picked from the word BREAKDOWN.  What is the probability that there is at least one vowel among the letters?
# 
# We need to use a library of useful functions called  ```python itertools``` and the *set* datastructure.
# First we do the setup:

# In[ ]:


import itertools
word = 'BREAKDOWN'


# The next step is to turn our word into a set of letters (I could have just typed it in as a set!).
# The ```print``` statement is added so you can see the result

# In[ ]:


letterset = {letter for letter in word}
print(letterset)


# Next we can use a clever function to get all possible subsets of length 4. The result is not a set so we convert it. The ```print``` function lets us see the result.

# In[ ]:


data = itertools.combinations(letterset, 4)
subsets = set(data)
print(subsets)


# There are a lot of subsets! We don't need to see them all, just to know how many:

# In[ ]:


numberOfSubsets = len(subsets)
print(numberOfSubsets)


# Now let's count how many have vowels. To do this we can use the set intersection function. Here is an illustration:

# In[ ]:


vowels = {'A','E','I','O','U'}
myset = {'B','F','I','Z','A'}
vowelsInSet = myset.intersection(vowels)
print(vowelsInSet)
print(len(vowelsInSet))


# As you can see, we can now count the number of vowels in each subset. Unfortunately, that's not what we need. We just need to know if there *are* any vowels. To do this we can use some more trickery. Use the `min` function to give 0 if the intersection is empty and 1 if not.

# In[ ]:


set1 ={'A','E'}
set2 = {}
print(min(len(set1),1))
print(min(len(set2),1))


# Now we can go through the whole collection of subsets and check if each has any vowels. We make a list of 1s and 0s.
# WE go through the collection with a `for` loop. An added complication is that the objects in the collection are not themselves sets, they are *tuples*. We convert to sets as we go:

# In[ ]:


biglist = [min(len(set(choice).intersection(vowels)),1) for choice in subsets]
print(biglist)


# now add up to get the number with vowels:

# In[ ]:


numberWithVowels = sum(biglist)
print(numberWithVowels)


# Finally, we can print the answer:

# In[ ]:


print(numberWithVowels,'/',numberOfSubsets)


# Actually, we could tidy this all up by making a function:

# In[ ]:


def probabilityOfVowel(word,subsetlength):
    word = word.upper()
    letterset = {letter for letter in word}
    data = itertools.combinations(letterset, subsetlength)
    subsets = set(data)
    numberOfSubsets = len(subsets)
    vowels = {'A','E','I','O','U'}
    numberWithVowels = sum([min(len(set(choice).intersection(vowels)),1) for choice in subsets])
    answer = str(numberWithVowels)+'/'+str(numberOfSubsets)
    return answer


# now we can try the function on different words:

# In[ ]:


print(probabilityOfVowel('BREAKDOWN',4))
print(probabilityOfVowel('PythonJapes',5))


# Finally, a question. What goes wrong with this approach when the word contains repeated letters?
