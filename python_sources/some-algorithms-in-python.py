#!/usr/bin/env python
# coding: utf-8

# ## This kernel includes some algorithm examples written in python.

# ## Table of Contents
# * [1. Loading Libraries](#1) <br>
# * [2. Factorial algorithm](#2) <br>
# * [3. Algorithm that reverses a string](#3) <br>
# * [4. Algorithm that converts minutes to hour:min](#4) <br>
# * [5. Uppercasing the first letters of the words in a text](#5) <br>
# * [6. Word shuffeling](#6) <br>
# * [7. Frequency of letters in a string](#7) <br>
# * [8. Finding missing digit](#8) <br>
# * [9. Array/List rotation](#9) <br>
# * [10.Array pairs](#10) <br>

# <a id="1"></a>
# ## 1.Loading Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id="2"></a>
# ## 2.Factorial Algorithm

# * input will be "n"--> integer
# * output will be n!

# ![image.png](attachment:image.png)

# In[ ]:


def factorial(n):
    """
    Input n ---> Output n!
    """
    fact=1
    for each in range(1,n+1):
        fact = fact * each
    return fact


# In[ ]:


print("5! =", factorial(5))
print("0! =", factorial(0))


# <a id="3"></a>
# ## 3.Algorithm that reverses a string

# * Let say we have a string "world" as input.
# * We will get "dlrow" as output.

# In[ ]:


def reverse(string):
    """
    input "string"---> output "gnirts"
    """
    list_ = []              # creating an empy list
    for each in string:
        list_.append(each)  # adding each letter at the end of list_
    reversed_=list_[::-1]   # reversing the list_
    joined = "".join(reversed_)
    return joined
        


# In[ ]:


reverse("world")


# In[ ]:


reverse("hello world")


# * We can do it with another way: 

# In[ ]:


def reverse2(string):
    reverse = string[::-1]  # slicing method
    return reverse


# In[ ]:


reverse2("good for you")


# <a id="4"></a>
# ## 4.Algorithm that converts minutes to "hours and min"

# * Input is minutes like 156 
# * Output will be hour:min format like "2 hours and 36 minutes"

# In[ ]:


def minutes_to_hours(num):
    hours = num // 60            # hours is quotient
    minutes = num % 60           # minutes is remainder
    return print(hours,"hours and",minutes,"minutes")
    


# In[ ]:


minutes_to_hours(6)
minutes_to_hours(60)
minutes_to_hours(688)
minutes_to_hours(986547)


# * Alternative way:

# In[ ]:


def minutes_to_hours1(num):
    import math
    hours = math.floor(num /60)  # rounding to lower value (for ex: 2.69 --> 2)      
    minutes = num % 60           # minutes is remainder
    return print(str(hours)+":"+str(minutes))   # str(hours)---> converting hours to string


# In[ ]:


minutes_to_hours1(6)
minutes_to_hours1(60)
minutes_to_hours1(688)
minutes_to_hours1(986547)


# <a id="5"></a>
# ## 5.Uppercasing the first letter of the words in a text

# * input as "let us try this"
# * output as "Let Us Try This"

# In[ ]:


a = "hello world"
word1 = a.split()[0][0].upper()+a.split()[0][1::]
word2 = a.split()[1][0].upper()+a.split()[1][1::]
b=[word1, word2]
" ".join(b)


# In[ ]:


def upper_case(text):
    word_count = len(text.split())          # word qty in text
    word_list = []                          # empty list for words that the first letters uppercased
    i = 0
    for each in text:
        while i < word_count:
            new_word = text.split()[i][0].upper() + text.split()[i][1::]      # splitting text into words and uppercasing first letters
            word_list.append(new_word)
            i = i + 1
    return " ".join(word_list)   


# In[ ]:


upper_case("let us try this function")


# * Alternative way:

# In[ ]:


def upper_case1(text):
    words = text.split(" ")
    for i in range(0,len(words)):
        words[i] = words[i][0].upper()+words[i][1::]
    return " ".join(words)


# In[ ]:


upper_case1("let me do it this way")


# In[ ]:


# There is function doing this :) ---> title()
"let me do it this way".title()


# <a id="6"></a>
# ## 6.Word shuffeling

# * input to be 2 strings as "city", "tyic"
# * outputs:
# * if "city" = tyic" --> output true
# * if "city" not equal "tyic" --> output false

# In[ ]:


def word_shuffle(str1, str2):
    for letter in str2:
        if letter not in str1:
            return False
    return True       


# In[ ]:


word_shuffle("city","tciy1")


# <a id="7"></a>
# ## 7.Frequency of letters in a string

# * Input--> Let say we have a string as "klmkqllnmmk"
# * Output ---> "3kk3lll3mmm1q1n"
# * Algorithm will calculate the frequency of each letters and write in front of letters.

# In[ ]:


def freq(string):
    
    freq_str = ""
    i=0
    while i < len(string):
        
        l = string[i]
        liste = [0,l]
        
        for j in range(len(string)):
            if string[j] == l:
                liste[0] += 1   
 
        freq_str += "".join(map(str,liste))
        string = string.replace(l, "")    
    return freq_str

freq("aabderacced")


# ### An other alternative:

# In[ ]:


def letter_freq1(strg):

    # letter count of str
    n = len(strg)   

    # initializing frequency of each letters with zero
    let_qty = ord("z") - ord("a") + 1               # letter quantity in alphabet (abcdefghijklmnopqrstuvwxyz)

    freq = np.zeros(let_qty, dtype = np.int)        # creating n size of aray from zeros as type of integer.
    # freq = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    # there are 26 rows in array and each represent one letter from alphabet starting from a to z.


    # counting the letters in strng in a for loop and increasing the count in each loop by updating the letter's row in freq
    for i in range(n):
        freq[ord(strg[i])-ord("a")] +=1             # updating each letters frequency in rows of freq

    for i in range(n):
        if freq[ord(strg[i])-ord("a")] != 0:
            print(str(freq[ord(strg[i])-ord("a")])+strg[i],end ="")
            freq[ord(strg[i])-ord("a")] = 0         # resetting the frequency to zero not to get it printed double


# In[ ]:


letter_freq1("aabderacced")


# <a id="8"></a>
# ## 8.Finding missing digit

# In[ ]:


strg = "120 / 4 = 3x"

for i in range(10):
    replaced = strg.replace("x", str(i))
    index_equal = strg.index("=")    
    if eval(replaced[:index_equal]) == eval(replaced[index_equal+1:]):
        print("x = ",i)
        
    


# In[ ]:


eval("55") == eval("50+5")


# <a id="9"></a>
# ## 9.Array/List rotation

# 
# * input --> [2,3,4,5],algorithm orders list items: beacuse the first item is 2, output will start 4 because it is in the 2nd index.
# * output --> 4523
# * input --> [4,5,6,7,8,9,10,11,12,13]
# * output --> 89101112134567

# In[ ]:


def array_rotate(list1):                        # example: list = [2,3,4,5]
    n = len(list1)                              # length of list: n = 4
    a = list1[0]                                # zeroth index: a = 2

    result = ""                                 # empty string initialization as result 
    for i in range(a,n):                        
        result = result + str(list1[i])
    for i in range(a):
        result = result + str(list1[i])
    return result
    


# In[ ]:


print("[1,2,3,4,5,6,7]-->",array_rotate([1,2,3,4,5,6,7]))
print("[3,4,5,6,7,9,11,13]-->",array_rotate([3,4,5,6,7,9,11,13]))
print("[5,4,5,6,7,9,11,13]-->",array_rotate([5,4,5,6,7,9,11,13]))


# ## Alternative way:

# In[ ]:


def array_rotate1(list1):
    result = ""
    for each in list1[list1[0]:] + list1[:list1[0]]:
        result = result + str(each)
    return result


# In[ ]:


print("[1,2,3,4,5,6,7]-->",array_rotate1([1,2,3,4,5,6,7]))
print("[3,4,5,6,7,9,11,13]-->",array_rotate1([3,4,5,6,7,9,11,13]))
print("[5,4,5,6,7,9,11,13]-->",array_rotate1([5,4,5,6,7,9,11,13]))


# <a id="10"></a>
# ## 10.Array pairs

# * input = [1,2,2,1,3,4]
# * output = "3,4"
# * input = [5,6,6,5,3,4,4,3]
# * output = "ok"

# In[ ]:


def array_pairs(list1):

    # converting string
    pairs = ""

    for i in range(len(list1)):
        pairs += str(list1[i])
        if i%2 == 1:
            pairs += ","
    pairs = pairs.split(",")

    # find pairs that do not have reverse
    no_reverse = []
    for i in pairs:
        if i[::-1] not in pairs:
            no_reverse.append(i)

    # if there is no reverse pair; print ok
    if no_reverse == []:
        print("Ok, List is composed of pairs!")

    # if there is reverse pair: print

    else:
        result = ""
        for each in no_reverse:
            result = result + str(each)+","
        print(result,"those not have pairs")


# In[ ]:


array_pairs([1,2,2,1,3,4,4,3,7,8,9,0])

