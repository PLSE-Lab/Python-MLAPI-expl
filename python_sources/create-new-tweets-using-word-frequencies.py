"""
Read the content of each tweets, and manage to create a new tweet
according to the frequence of each words in the text.
For each words, we know what are the next possible words and their frequencies.
"""
import pandas as pd
from collections import defaultdict
from nltk.tokenize import word_tokenize
from random import random

class WordLink:
    """
    Data structure to remember what word can follow the
    particular word given.
    """
    def __init__(self, word):
        self.word = word # The given word
        self.link = defaultdict(int) # Key is a following word, value is how many times it happened
        self.start_count = 0 # Does that word can start a sentence ?
        self.end_count = 0 # Does that word can end a sentence ?
        self.sum_possibilities = 0 # How much words / ends there have been after this word ?

    def add_link(self, word):
        self.link[word] += 1
        self.sum_possibilities += 1

    def incr_start(self):
        self.start_count += 1

    def incr_end(self):
        self.end_count += 1
        self.sum_possibilities += 1

    def __str__(self):
        return "{}:\n{} starts\n{} ends\n{} next_possibilities".format(
                self.word, self.start_count, self.end_count, self.sum_possibilities)


class CreateSentence:
    """
    Manipulates WordLink to create a sentence.
    """
    def __init__(self):
        self.link_words = dict()
        self.sum_starts = 0 # How much differents starts there are ?

    def add_link(self, previous_word, next_word):
        self.add_word(previous_word)
        self.link_words[previous_word].add_link(next_word)

    def add_start(self, starting_word):
        self.add_word(starting_word)
        self.link_words[starting_word].incr_start()
        self.sum_starts += 1

    def add_end(self, ending_word):
        self.add_word(ending_word)
        self.link_words[ending_word].incr_end()

    def add_word(self, word):
        if (word not in self.link_words):
            self.link_words[word] = WordLink(word)

    def choose_start(self):
        r = random()
        tot = 0
        for word_link in self.link_words.values():
            tot += word_link.start_count / self.sum_starts
            if (tot >= r):
                return word_link.word

        return None # Error

    def choose_next(self, previous_word):
        r = random()
        tot = 0
        word_link = self.link_words[previous_word]
        for count, next_word in enumerate(word_link.link):
            tot += count / word_link.sum_possibilities
            if tot >= r:
                return next_word

        return None # End the sentence

    def sentence(self):
        sentence = create_sentence.choose_start()
        next_word = create_sentence.choose_next(sentence)
        while(next_word is not None):
            sentence += " " + next_word
            next_word = create_sentence.choose_next(next_word)

        return sentence


df = pd.read_csv("../input/trump-tweets/trumptweets.csv", sep=",", engine="python")
content = df["content"]

create_sentence = CreateSentence()
for cont in content:
    words = word_tokenize(cont, language="english")
    words = [w.lower() for w in words]

    previous_word = words[0]
    create_sentence.add_start(previous_word)
    for w in words[1:]:
        create_sentence.add_link(previous_word, w)
        previous_word = w

    create_sentence.add_end(previous_word)


print("A really simple and dumb tweet's Trump generator:")
for i in range(15):
    print(create_sentence.sentence())