import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation."""

sentences=nltk.sent_tokenize(paragraph)
lemma=WordNetLemmatizer()

for i in range(len(sentences)):
    words=nltk.word_tokenize(sentences[i])
    words=[lemma.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i]=' '.join(words)
    
print(sentences)