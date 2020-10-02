'''
Oct17-2018
Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for text search using nltk concordance
Results:
Input dataset: seekingalpha.com roche transcript


'''
import sys
import nltk
from nltk.corpus import PlaintextCorpusReader


corp = PlaintextCorpusReader("../input/", "roche.txt")
text = nltk.Text(corp.words())



#match = text.concordance('Herceptin')
##concord = text.concordance('Herceptin', 300, sys.maxsize)


'''write Herceptin concordance to file'''
# Open the file
fileconcord = open('containing-tag-Herceptin.txt', 'w')
# Save old stdout stream
tmpout = sys.stdout
# Redirect all "print" calls to that file
sys.stdout = fileconcord
# Init the method
text.concordance("Herceptin", 250, sys.maxsize)
# Close file
fileconcord.close()
# Reset stdout in case you need something else to print
sys.stdout = tmpout

