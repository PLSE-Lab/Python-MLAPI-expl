#!/usr/bin/env python
# coding: utf-8

# # 30 REGULAR EXPRESSION EXERCISES + SOLUTIONS!

# Regex, you never know when they might come in handy. It's one of the "good programmer"'s fundamental yet a few people actually masters them.
# 
# Challenge yourself in 30 exercises, just hide the solution in order not to peek ;)

# In[ ]:


import re


# 1) Write a Python program to check that a string contains only a certain set of characters (in this case a-z, A-Z and 0-9).

# In[ ]:


# Your code here


# In[ ]:


# Solution
def is_allowed_specific_char(string):
    charRe = re.compile(r'[^a-zA-Z0-9.]')
    string = charRe.search(string)
    return not bool(string)

print(is_allowed_specific_char("ABCDEFabcdef123450")) 
print(is_allowed_specific_char("*&%@#!}{"))


# 2) Write a Python program that matches a string that has an a followed by zero or more b's

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = 'ab*?'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("ac"))
print(text_match("abc"))
print(text_match("abbc"))


# 3) Write a Python program that matches a string that has an a followed by one or more b's

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = 'ab+?'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("ab"))
print(text_match("abc"))


# 4) Write a Python program that matches a string that has an a followed by zero or one 'b'

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = 'ab?'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("ab"))
print(text_match("abc"))
print(text_match("abbc"))
print(text_match("aabbc"))


# 5) Write a Python program that matches a string that has an a followed by three 'b'

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = 'ab{3}?'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("abbb"))
print(text_match("aabbbbbc"))


# 6) Write a Python program that matches a string that has an a followed by two to three 'b'.

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = 'ab{2,3}?'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("ab"))
print(text_match("aabbbbbc"))


# 7) Write a Python program to find sequences of lowercase letters joined with a underscore.

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = '^[a-z]+_[a-z]+$'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("aab_cbbbc"))
print(text_match("aab_Abbbc"))
print(text_match("Aaab_abbbc"))


# 8) Write a Python program to find the sequences of one upper case letter followed by lower case letters.

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = '^[a-z]+_[a-z]+$'
        if not re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("aab_cbbbc"))
print(text_match("aab_Abbbc"))
print(text_match("Aaab_abbbc"))


# 9) Write a Python program that matches a string that has an 'a' followed by anything, ending in 'b'.

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = 'a.*?b$'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("aabbbbd"))
print(text_match("aabAbbbc"))
print(text_match("accddbbjjjb"))


# 10) Write a Python program that matches a word at the beginning of a string.

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = '^\w+'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("The quick brown fox jumps over the lazy dog."))
print(text_match(" The quick brown fox jumps over the lazy dog."))


# 11) Write a Python program that matches a word at the end of string, with optional punctuation.

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = '\w+\S*$'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("The quick brown fox jumps over the lazy dog."))
print(text_match("The quick brown fox jumps over the lazy dog. "))
print(text_match("The quick brown fox jumps over the lazy dog "))


# 12) Write a Python program that matches a word containing 'z'

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = '\w*z.\w*'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("The quick brown fox jumps over the lazy dog."))
print(text_match("Python Exercises."))


# 13) Write a Python program that matches a word containing 'z', not at the start or end of the word.

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = '\Bz\B'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("The quick brown fox jumps over the lazy dog."))
print(text_match("Python Exercises."))


# 14) Write a Python program to match a string that contains only upper and lowercase letters, numbers, and underscores.

# In[ ]:


# Your code here


# In[ ]:


# Solution
def text_match(text):
        patterns = '^[a-zA-Z0-9_]*$'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("The quick brown fox jumps over the lazy dog."))
print(text_match("Python_Exercises_1"))


# 15) Write a Python program where a string will start with a specific number. 

# In[ ]:


# Your code here


# In[ ]:


# Solution
def match_num(string):
    text = re.compile(r"^5")
    if text.match(string):
        return True
    else:
        return False
print(match_num('5-2345861'))
print(match_num('6-2345861'))


# 16) Write a Python program to remove leading zeros from an IP address

# In[ ]:


# Your code here


# In[ ]:


# Solution
ip = "216.08.094.196"
string = re.sub('\.[0]*', '.', ip)
print(string)


# 17) Write a Python program to check for a number at the end of a string.

# In[ ]:


# Your code here


# In[ ]:


# Solution
def end_num(string):
    text = re.compile(r".*[0-9]$")
    if text.match(string):
        return True
    else:
        return False

print(end_num('abcdef'))
print(end_num('abcdef6'))


# 18) Write a Python program to search the numbers (0-9) of length between 1 to 3 in a given string. 

# In[ ]:


# Your code here


# In[ ]:


# Solution
results = re.finditer(r"([0-9]{1,3})", "Exercises number 1, 12, 13, and 345 are important")
print("Number of length 1 to 3")
for n in results:
     print(n.group(0))


# 19) Write a Python program to search some literals strings in a string. Go to the editor
# Sample text : 'The quick brown fox jumps over the lazy dog.'
# Searched words : 'fox', 'dog', 'horse'

# In[ ]:


# Your code here


# In[ ]:


# Solution
patterns = [ 'fox', 'dog', 'horse' ]
text = 'The quick brown fox jumps over the lazy dog.'
for pattern in patterns:
    print('Searching for "%s" in "%s" ->' % (pattern, text),)
    if re.search(pattern,  text):
        print('Matched!')
    else:
        print('Not Matched!')


# 20) Write a Python program to search a literals string in a string and also find the location within the original string where the pattern occurs
# 
# Sample text : 'The quick brown fox jumps over the lazy dog.'
# Searched words : 'fox'

# In[ ]:


# Your code here


# In[ ]:


# Solution
pattern = 'fox'
text = 'The quick brown fox jumps over the lazy dog.'
match = re.search(pattern, text)
s = match.start()
e = match.end()
print('Found "%s" in "%s" from %d to %d ' % (match.re.pattern, match.string, s, e))


# 21) Write a Python program to find the substrings within a string.
# 
# Sample text :
# 
# 'Python exercises, PHP exercises, C# exercises'
# 
# Pattern :
# 
# 'exercises'
# 
# Note: There are two instances of exercises in the input string.

# In[ ]:


# Your code here


# In[ ]:


# Solution
text = 'Python exercises, PHP exercises, C# exercises'
pattern = 'exercises'
for match in re.findall(pattern, text):
    print('Found "%s"' % match)


# 22) Write a Python program to find the occurrence and position of the substrings within a string.

# In[ ]:


# Your code here


# In[ ]:


# Solution
text = 'Python exercises, PHP exercises, C# exercises'
pattern = 'exercises'
for match in re.finditer(pattern, text):
    s = match.start()
    e = match.end()
    print('Found "%s" at %d:%d' % (text[s:e], s, e))


# 23) Write a Python program to replace whitespaces with an underscore and vice versa.

# In[ ]:


# Your code here


# In[ ]:


# Solution
text = 'Python Exercises'
text =text.replace (" ", "_")
print(text)
text =text.replace ("_", " ")
print(text)


# 24) Write a Python program to extract year, month and date from a an url. 

# In[ ]:


# Your code here


# In[ ]:


# Solution
def extract_date(url):
        return re.findall(r'/(\d{4})/(\d{1,2})/(\d{1,2})/', url)
url1= "https://www.washingtonpost.com/news/football-insider/wp/2016/09/02/odell-beckhams-fame-rests-on-one-stupid-little-ball-josh-norman-tells-author/"
print(extract_date(url1))


# 25) Write a Python program to convert a date of yyyy-mm-dd format to dd-mm-yyyy format.

# In[ ]:


# Your code here


# In[ ]:


# Solution
def change_date_format(dt):
        return re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', '\\3-\\2-\\1', dt)
dt1 = "2026-01-02"
print("Original date in YYY-MM-DD Format: ",dt1)
print("New date in DD-MM-YYYY Format: ",change_date_format(dt1))


# 26) Write a Python program to match if two words from a list of words starting with letter 'P'.

# In[ ]:


# Your code here


# In[ ]:


# Solution
def change_date_format(dt):
        return re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', '\\3-\\2-\\1', dt)
dt1 = "2026-01-02"
print("Original date in YYY-MM-DD Format: ",dt1)
print("New date in DD-MM-YYYY Format: ",change_date_format(dt1))


# 27) Write a Python program to separate and print the numbers of a given string.

# In[ ]:


# Your code here


# In[ ]:


# Solution
# Sample string.
text = "Ten 10, Twenty 20, Thirty 30"
result = re.split("\D+", text)
# Print results.
for element in result:
    print(element)


# 28) Write a Python program to find all words starting with 'a' or 'e' in a given string.

# In[ ]:


# Your code here


# In[ ]:


# Solution
# Input.
text = "The following example creates an ArrayList with a capacity of 50 elements. Four elements are then added to the ArrayList and the ArrayList is trimmed accordingly."
#find all the words starting with 'a' or 'e'
list = re.findall("[ae]\w+", text)
# Print result.
print(list)
	


# 29) Write a Python program to separate and print the numbers and their position of a given string.

# In[ ]:


# Your code here


# In[ ]:


# Solution
# Input.
text = "The following example creates an ArrayList with a capacity of 50 elements. Four elements are then added to the ArrayList and the ArrayList is trimmed accordingly."

for m in re.finditer("\d+", text):
    print(m.group(0))
    print("Index position:", m.start())
	


# 30) Write a Python program to abbreviate 'Road' as 'Rd.' in a given string.

# In[ ]:


# Your code here


# In[ ]:


# Solution
street = '21 Ramkrishna Road'
print(re.sub('Road$', 'Rd.', street))

