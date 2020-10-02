import pandas as pd
#import csv
from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.corpus import stopwords
import gensim

tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
# Create p_stemmer of class PorterStemmer
#p_stemmer = PorterStemmer()

df9 = pd.read_csv("../input/papers.csv")

texts = []
################################### Remove numbers and single letter words
# loop through document list
for i in df9['paper_text']:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if (not i in en_stop and not str(i).isdigit() and len(str(i)) > 2 )]
    
    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stopped_tokens)
    
df9['Cleaned_PaperText'] = pd.Series(texts, index = df9.index)
        
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
df9['Corpus'] = pd.Series(corpus, index = df9.index)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=30, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=2, num_words=4))

model = ldamodel
#save model
#ldamodel.save('../output/model.atmodel')
#Load model
#model = gensim.models.ldamodel.LdaModel.load('../output/model.atmodel')

###Running the model
#model[df1.Corpus[100]]
#t = model.get_document_topics(df1.Corpus[300], minimum_probability=0.5)
Top_words = []
for index,row in df9.iterrows():
    words = []
    topics = model.get_document_topics(row['Corpus'])
    for topic in topics:
        t1 = model.show_topic(k[0])
        for i in t1:
            words.append(i[0])
            #cnt+=1
            #if cnt == :
            #    break 
    Top_words.append(words)
df9['Top_Topic_words'] = pd.Series(Top_words, index = df1.index)