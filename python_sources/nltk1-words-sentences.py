import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

data = "All work and no play makes jack dull boy. This video is about NLTK programming."

print(word_tokenize(data))

print(sent_tokenize(data))