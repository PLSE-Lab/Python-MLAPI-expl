import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import string


nltk.download_shell()

messages = pd.read_csv('SMS Spam Collection Dataset/spam', sep='\t', names=['label','message'])
messages.groupby('label').describe()
messages['length'] = messages['messages'].apply(len)

nopunc = [c for c in mess if c not in string.punctuation]

stopwords.words('english')

nopunc = ''.join(nopunc)

