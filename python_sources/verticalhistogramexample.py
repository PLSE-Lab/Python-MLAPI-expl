#!/usr/bin/env python
# coding: utf-8

# Asked by Travis:  Printing Vertical histogram
# The code accepts a string of various characters and white spaces. The only thing that matters is the Capital Letters. The purpose is to count the number of capital letters and print them in alphabetical order as a VERTICAL histogram where a '*' is equal to a single count.
# Example: "XXY YY ZZZ123ZZZ AAA BB C"
# Produces (roughly this):
# ```
#             *
#             *
#             *
# *         * *
# * *   * * *
# * * * * * *
# A B C X Y Z
# ```
# Here is my code currently.  Any guidance would be appreciated. Thank you.

# In[ ]:


string = "XXY YY ZZZ123ZZZ AAA BB C"
string = sorted(string)
letters = {}
for char in string:
    if char.isupper() and char.isalpha():
        if char in letters:
            letters[char] += 1
        else:
            letters[char] = 1
while any(value > 0 for key, value in letters.items()):
    for key, value in letters.items():
        max_len = max(letters.values())
        if value == max_len:
            # PROBLEM PRINTING HISTOGRAM VERTICALLY
            letters[key] -= 1
print(' '.join([k for k in letters.keys()]))


# Let's begin by confirming that the distribution is calculated properly:

# In[ ]:


string = "XXY YY ZZZ123ZZZ AAA BB C"
string = sorted(string)
letters = {}
for char in string:
    if char.isupper() and char.isalpha():
        if char in letters:
            letters[char] += 1
        else:
            letters[char] = 1
print(letters)


# Yes!  So the problem is only to print a vertical histogram.
# Let's only think about the very first line.  What is the correct thing to print on the first line?  Well since Z has the most characters you should print a "*" for it.  To indent properly, you should print spaces for the other characters.
# #### For the first line, what is a good algorithm?
# How about "Scan through each letter.  If its count is >= 6 then print "*", otherwise print " "

# In[ ]:


this_row = list()
for key, value in letters.items():
    if value >=6:
        this_row.append('*')
    else:
        this_row.append(' ')
print(''.join(this_row))


# #### Great!  It works for the top row.
# Let's convert the above into a function then count down from 6 to 1.
# Finally, print the "keys" which are the characters:

# In[ ]:


def build_a_row(distribution_dict, height):
    this_row = list()
    for key, value in distribution_dict.items():
        if value >= height:
            this_row.append('*')
        else:
            this_row.append(' ')
    return ''.join(this_row)

for count in range(max(letters.values()),0,-1):
    print(build_a_row(letters, count))

print(''.join(letters.keys()))


# In[ ]:


max([x for x in letters.values()])


# In[ ]:




