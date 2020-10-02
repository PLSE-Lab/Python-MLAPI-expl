#!/usr/bin/env python
# coding: utf-8

# # Basic Algorithm Scripting

# In this notebook, I will work through the basic algorithm scripting challenges from freeCodeCamp. On freeCodeCamp, the challenges are done in JavaScript, so to get some additional Python practice I will do them here in Python.

# ## Convert Celsius to Fahrenheit

# The algorithm to convert from Celsius to Fahrenheit is the temperature in Celsius times $\frac{9}{5}$, plus $32$.
# 
# You are given a variable `celsisus` representing a temperature in Celsius. Use the variable `fahrenheit` already defined and assign it the Fahrenheit temperature equivalent to the given Celsius temperature.

# In[ ]:


def convertToF(celsius):
    fahrenheit = (9/5) * celsius + 32
    return fahrenheit


# In[ ]:


celsiusTemps = [0, -30, 10, 20, 30]
for temp in celsiusTemps:
    print("Converting {0}C to Fahrenheit:".format(temp), convertToF(temp))


# ## Reverse a string

# Reverse the provided string.

# In[ ]:


def reverseString(str):
    return str[::-1]


# In[ ]:


strings = ["hello", "Howdy", "Greetings from Earth"]
for string in strings:
    print("Reversed string:", reverseString(string))


# ## Factorialize a number

# Return the factorial of the provided integer.
# 
# Recall that given an integer $n$, we define $n! = n \times (n - 1) \times \cdots \times 2 \times 1$. By convention, we define $0! = 1$.

# In[ ]:


def recursive_factorialize(num):
    if num == 0:
        return 1
    else:
        return num * recursive_factorialize(num - 1)
    
def iterative_factorialize(num):
    factorial = 1
    for i in range(1, num + 1):
        factorial *= i
    return factorial


# In[ ]:


factorials = [5, 10, 20, 0]
for num in factorials:
    print("Recursive factorial of", num, "=", recursive_factorialize(num))
    print("Iterative factorial of", num, "=", iterative_factorialize(num))


# ## Find the longest word in a string

# Return the length of the longest word in the provided sentence.

# In[ ]:


def findLongestWordLength(str):
    words = str.split()
    longest = 0
    for word in words:
        if len(word) > longest:
            longest = len(word)
    return longest


# In[ ]:


print(findLongestWordLength("The quick brown fox jumped over the lazy dog"))
print(findLongestWordLength("May the force be with you"))
print(findLongestWordLength("Google do a barrel roll"))
print(findLongestWordLength("What is the average airspeed velocity of an unladen swallow"))
print(findLongestWordLength("What if we try a super-long word such as otorhinolaryngology"))


# ## Return the largest numbers in arrays

# Return an array consisting of the largest number from each provided sub-array.

# In[ ]:


def largestOfFour(arr):
    largest = []
    for ele in arr:
        largest.append(max(ele))
    return largest


# In[ ]:


print(largestOfFour([[4, 5, 1, 3], [13, 27, 18, 26], [32, 35, 37, 39], [1000, 1001, 857, 1]]))
print(largestOfFour([[4, 9, 1, 3], [13, 35, 18, 26], [32, 35, 97, 39], [1000000, 1001, 857, 1]]))
print(largestOfFour([[17, 23, 25, 12], [25, 7, 34, 48], [4, -10, 18, 21], [-72, -3, -17, -10]]))


# ## Confirm the ending

# Check if a string (first argument `s`) ends with the given target string (second argument, `target`). Note that the built-in string method `str.endswith()` performs the same function.

# In[ ]:


def confirmEnding(s, target):
    targetLength = len(target)
    strEnding = s[len(s)-targetLength:]
    return strEnding == target


# In[ ]:


print(confirmEnding("Bastian", "n"))
print(confirmEnding("Congratulation", "on"))
print(confirmEnding("Connor", "n"))
print(confirmEnding("Walking on water and developing software from a specification are easy if both are frozen", "specification"))
print(confirmEnding("He has to give me a new name", "name"))
print(confirmEnding("Open sesame", "same"))
print(confirmEnding("Open sesame", "pen"))
print(confirmEnding("Open sesame", "game"))
print(confirmEnding("If you want to save our world, you must hurry. We dont know how much longer we can withstand the nothing", "mountain"))
print(confirmEnding("Abstraction", "action"))


# ## Repeat a string

# Repeat a given string `s` (first argument) for `num` times (second argument). Return an empty string if `num` is not a positive number.

# In[ ]:


def repeatStringNumTimes(s, num):
    if (num < 0):
        return ""
    else:
        return s*num


# In[ ]:


print(repeatStringNumTimes("*", 3))
print(repeatStringNumTimes("abc", 3))
print(repeatStringNumTimes("abc", 4))
print(repeatStringNumTimes("abc", 1))
print(repeatStringNumTimes("*", 8))
print(repeatStringNumTimes("abc", -2))


# ## Truncate a string

# Truncate a string (first argument) if it is longer than the given maximum string length (second argument). Return the truncated string with a `...` ending.

# In[ ]:


def truncateString(s, maxLength):
    if maxLength >= len(s):
        return s
    else:
        return s[0:maxLength] + "..."


# In[ ]:


print(truncateString("A-tisket a-tasket A green and yellow basket", 8))
print(truncateString("Peter Piper picked a peck of pickled peppers", 11))
print(truncateString("A-tisket a-tasket A green and yellow basket", len("A-tisket a-tasket A green and yellow basket")))
print(truncateString("A-tisket a-tasket A green and yellow basket", len("A-tisket a-tasket A green and yellow basket")+2))
print(truncateString("A-", 1))
print(truncateString("Absolutely Longer", 2))


# ## Finders keepers

# Create a function that looks through an array (first argument) and returns the first element in the array that passes a truth test (second argument). If no element passes the test, return undefined (i.e. don't return anything).

# In[ ]:


def findElement(arr, func):
    for ele in arr:
        if func(ele):
            return ele


# In[ ]:


print(findElement([1, 3, 5, 8, 9, 10], lambda x: x%2 == 0))
print(findElement([1, 3, 5, 9], lambda x: x%2 == 0))


# ## Boo who

# Check if a value is classified as a boolean primitive (`True` or `False`). Return true if the value is a boolean primitive and return false otherwise.

# In[ ]:


def booWho(boo):
    return type(boo) == bool


# In[ ]:


booWhoCheck = [True, False, [1, 2, 3], [1, 2, 3].copy, {"a":1}, 1, "a", "True", "False"]
for boo in booWhoCheck:
    print(booWho(boo))


# ## Title case a sentence

# Return the provided string with the first letter of each word capitalized. Make sure the rest of the word is in lower case. For the purpose of this exercise, you should also capitalize connecting words such as "the" and "of".

# In[ ]:


def titleCase(string):
    words = string.split()
    titleCased = []
    for word in words:
        titleCased.append(word[0].upper() + word[1:].lower())
    return " ".join(titleCased)


# In[ ]:


titleCaseCheck = ["I'm a little tea pot", "sHoRt AnD sToUt", "HERE IS MY HANDLE AND HERE IS MY SPOUT"]
for phrase in titleCaseCheck:
    print(titleCase(phrase))


# ## Slice and splice

# Given two arrays, `arr1` and `arr2`, and an index `n`, copy each element of `arr1` into `arr2` in order. Begin by inserting elements at index `n` of the second array. Return the resulting array. The input arrays should remain the same after the function runs.

# In[ ]:


def frankenSplice(arr1, arr2, n):
    spliced = arr2[0:n]
    spliced.extend(arr1)
    spliced.extend(arr2[n:])
    return spliced


# In[ ]:


frankenSpliceCheck = [([1, 2, 3], [4, 5], 1), ([1, 2], ["a", "b"], 1), (["claw", "tentacle"], ["head", "shoulders", "knees", "toes"], 2)]
for check in frankenSpliceCheck:
    print(frankenSplice(check[0], check[1], check[2]))
    print("Checking for mutations:", check[0], check[1])


# ## Falsy bouncer

# Remove all falsy values from an array. In Python, the following built-in objects are considered false.
# - constants defined to be false: `None` and `False`
# - zero of any numeric type: `0`, `0.0`, `0j`, `Decimal(0)`, `Fraction(0, 1)`
# - empty sequences and collections: `''`, `()`, `[]`, `{}`, `set()`, `range(0)`

# In[ ]:


def bouncer(arr):
    return [item for item in arr if bool(item) == True]


# In[ ]:


bouncerCheck = [[7, "ate", "", False, 9], ["a", "b", "c"], [False, None, 0, (), ""],
               [1, None, 0j, 2, None]]
for check in bouncerCheck:
    print("Before bouncing:", check)
    print("After bouncing:", bouncer(check))


# ## Where do I belong

# Return the lowest index at which a value (second argument) should be inserted into an array (first argument) once it has been sorted. The returned value should be a number.

# In[ ]:


def getIndexToIns(arr, num):
    appended_arr = arr.copy()
    appended_arr.append(num)
    appended_arr.sort()
    return appended_arr.index(num)


# In[ ]:


indexCheck = [([10, 20, 30, 40, 50], 35), ([10, 20, 30, 40, 50], 30),
             ([40, 60], 50), ([3, 10, 5], 3), ([5, 3, 20, 3], 5),
             ([2, 20, 10], 19), ([2, 5, 10], 15), ([], 1)]
for check in indexCheck:
    print("Insertion Index:", getIndexToIns(check[0], check[1]))


# ## Mutations

# Return true if the string in the first element of array contains all of the letters of the string in the second element of the array, ignoring case.

# In[ ]:


def mutation(arr):
    str_1, str_2 = arr[0].lower(), arr[1].lower()
    for char in str_2:
        if char not in str_1:
            return False
    return True


# In[ ]:


mutationCheck = [["hello", "hey"], ["hello", "Hello"], ["zyxwvutsrqponmlkjihgfedcba", "qrstu"],
                ["Mary", "Army"], ["Mary", "Aarmy"], ["Alien", "line"], ["floor", "for"],
                ["hello", "neo"], ["voodoo", "no"]]
for check in mutationCheck:
    print(check, mutation(check))


# ## Chunky monkey

# Write a function that splits an array (first argument) into groups the length of `size` (second argument) and returns them as a two-dimensional array. Note that you should not assume that the length of the starting array is evenly divisible by `size`.

# In[ ]:


def chunkArrayIntoGroups(arr, size):
    chunkedArray = []
    for i in range(0, len(arr), size):
        chunkedArray.append(arr[i:i + size])
    return chunkedArray


# In[ ]:


chunkCheck = [(["a", "b", "c", "d"], 2), ([0, 1, 2, 3, 4, 5], 3), ([0, 1, 2, 3, 4, 5], 2),
             ([0, 1, 2, 3, 4, 5], 4), ([0, 1, 2, 3, 4, 5, 6], 3), ([0, 1, 2, 3, 4, 5, 6, 7, 8], 4),
             ([0, 1, 2, 3, 4, 5, 6, 7, 8], 2)]
for check in chunkCheck:
    print(chunkArrayIntoGroups(check[0], check[1]))

