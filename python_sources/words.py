import sqlite3
import matplotlib.pyplot as plt
from collections import Counter

conn = sqlite3.connect('../input/database.sqlite')

c = conn.cursor()

word_dict = {}

def word_split(text):
    word = ""
    if len(text) == 0:
        return word_dict
    for char in text:
        if char.isalpha() or char == "'":
            word += char
        else:
            lower_word = word.lower()
            if word == "":
                return word_split(text[text.index(char)+1:])
            #if char == "'" and (text[text.index(char)+1]).isalpha():
                #continue
            elif lower_word in word_dict:
                word_dict[lower_word] += 1
            else:
                word_dict[lower_word] = 1
            return word_split(text[text.index(char)+1:])

comment_list = []

for comment in c.execute('SELECT body FROM May2015 ORDER BY ups LIMIT 5'):
    comment_list.append(comment)

print (comment_list)