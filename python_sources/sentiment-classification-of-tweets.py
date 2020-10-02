#!/usr/bin/env python
# coding: utf-8

# # Tweets Airlines Sentiments

# In[2]:


# Before we begin, we supress deprecation warnings resulting from nltk on Kaggle
import warnings
import gensim
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# # Sentiment classification on tweets about airlines
# 
# This notebook describes an attempt to classify tweets by sentiment. It describes the initial data exploration, as well as implementation of a classifier.

# ## What is in the dataset?
# 
# It's always good to start by exploring the data that we have available. To do this we load the raw csv file using [Pandas][1] and check what the columns are.
# 
#   [1]: http://pandas.pydata.org/

# In[ ]:


import pandas as pd
tweets = pd.read_csv("../input/Tweets.csv")
list(tweets.columns.values)


# We want to be able to determine the sentiment of a tweet without any other information but the tweet text itself, hence the 'text' column is our focus. Using the text we are going to try and predict 'airline_sentiment'.
# 
# First we take a look at what a typical record looks like.

# In[ ]:


tweets.head()


# Now lets take a look at what sentiments have been found.

# In[ ]:


sentiment_counts = tweets.airline_sentiment.value_counts()
number_of_tweets = tweets.tweet_id.count()
print(sentiment_counts)


# In[ ]:


dff = tweets.groupby(["airline", "airline_sentiment" ]).count()["name"]
dff


# In[ ]:


df_companySentiment = dff.to_frame().reset_index()
df_companySentiment.columns = ["airline", "airline_sentiment", "count"]
df_companySentiment

#df2 = dff.pivot('airline', 'airline_sentiment')
#df2


# In[ ]:


df2 = df_companySentiment
df2.index = df2['airline']
del df2['airline']
df2


# In[ ]:


dff


# In[ ]:


df3 = dff.pivot('airline', 'airline_sentiment')
df3


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.style
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.style
from matplotlib.pyplot import subplots
matplotlib.style.use('ggplot')

fig, ax = subplots()
my_colors =['darksalmon', 'papayawhip', 'cornflowerblue']
df2.plot(kind='bar', stacked=True, ax=ax, color=my_colors, figsize=(12, 7), width=0.8)
ax.legend(["Negative", "Neutral", "Positive"])
plt.title("Tweets Sentiments Analysis Airlines, 2017")
plt.show()


# It turns out that our dataset is unbalanced with significantly more negative than positive tweets. We will focus on the issue of identifying negative tweets, and hence treat neutral and positive as one class. It's good to keep in mind that, while a terrible classifier, if we always guessed a tweet was negative we'd be right 62.7% of the time (9178 of 14640). That clearly wouldn't be a very useful classifier, but worth to remember.

# # What characterizes text of different sentiments?
# 
# While we still haven't decided what classification method to use, it's useful to get an idea of how the different texts look. This might be an "old school" approach in the age of deep learning, but lets indulge ourselves nevertheless. 
# 
# To explore the data we apply some crude preprocessing. We will tokenize and lemmatize using [Python NLTK][1], and transform to lower case. As words mostly matter in context we'll look at bi-grams instead of just individual tokens.
# 
# As a way to simplify later inspection of results we will store all processing of data together with it's original form. This means we will extend the Pandas dataframe into which we imported the raw data with new columns as we go along.
# 
# ### Preprocessing
# Note that we remove the first two tokens as they always contain "@ airline_name". We begin by defining our normalization function.
# 
# 
#   [1]: http://www.nltk.org/

# In[ ]:


import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas


# In[ ]:


normalizer("I recently wrote some texts.")


# In[ ]:


pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
tweets['normalized_tweet'] = tweets.text.apply(normalizer)
tweets[['text','normalized_tweet']].head()


# In[ ]:


from nltk import ngrams
def ngrams(input_list):
    #onegrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams
tweets['grams'] = tweets.normalized_tweet.apply(ngrams)
tweets[['grams']].head()


# And now some counting.

# In[ ]:


import collections
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt


# In[ ]:


tweets[(tweets.airline_sentiment == 'negative')][['grams']].apply(count_words)['grams'].most_common(20)


# We can already tell there's a pattern here. Sentences like "cancelled flight", "late flight", "booking problems",  "delayed flight" stand out clearly. Lets check the positive tweets.

# In[ ]:


tweets[(tweets.airline_sentiment == 'positive')][['grams']].apply(count_words)['grams'].most_common(20)


# ### Some useful functions may use

# In[1]:


# some references:

class Voc:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = ["PAD", "UNK"] # might be changed
        self.n_words = 10000 + 2 # might be changed

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    def remove_punctuation(self, sentence):
        sentence = self.unicodeToAscii(sentence)
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
        sentence = re.sub(r"\s+", r" ", sentence).strip()
        return sentence

    def fit(self, train_df, train_df_no_label, USE_Word2Vec=True):
        print("Voc fitting...")
        
        # tokenize
        tokens = []
        sentences = []
        
        for sequence in train_df["seq"]:
            token = sequence.strip(" ").split(" ")
            tokens += token
            sentences.append(token)


        for sequence in train_df_no_label["seq"]:
            token = sequence.strip(" ").split(" ")
            tokens += token
            sentences.append(token)

        # Using Word2Vec
        if USE_Word2Vec:
            dim = 100
            print("Word2Vec fitting")
            model = Word2Vec(sentences, size=dim, window=5, min_count=20, workers=20, iter=20)
            print("Word2Vec fitting finished....")
            # gensim index2word 
            self.index2word += model.wv.index2word
            self.n_words = len(self.index2word)
            # build up numpy embedding matrix
            embedding_matrix = [None] * len(self.index2word) # init to vocab length
            embedding_matrix[0] = np.random.normal(size=(dim,))
            embedding_matrix[1] = np.random.normal(size=(dim,))
            # plug in embedding
            for i in range(2, len(self.index2word)):
                embedding_matrix[i] = model.wv[self.index2word[i]]
                self.word2index[self.index2word[i]] = i
            
            # 
            self.embedding_matrix = np.array(embedding_matrix)
            return
        else:
            # Counter
            counter = Counter(tokens)
            voc_list = counter.most_common(10000)

            for i, (voc, freq) in enumerate(voc_list):
                self.word2index[voc] = i+2
                self.index2word[i+2] = voc
                self.word2count[voc] = freq

def print_to_csv(y_, filename):
    d = {"id":[i for i in range(len(y_))],"label":list(map(lambda x: str(x), y_))}
    df = pd.DataFrame(data=d)
    df.to_csv(filename, index=False)


class BOW():
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=10000)

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    def remove_punctuation(self, sentence):
        sentence = self.unicodeToAscii(sentence)
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
        sentence = re.sub(r"\s+", r" ", sentence).strip()
        return sentence

    def fit(self, train_df, train_df_no_label):
        # prepare copus
    
        corpus = list(map(lambda x: self.remove_punctuation(x), train_df['seq']))
        corpus += list(map(lambda x: self.remove_punctuation(x), train_df_no_label['seq']))
        print("BOW fitting")
        self.vectorizer.fit(corpus)
        self.dim = len(self.vectorizer.get_feature_names())
        print("BOW fitting done")
        return self

    def batch_generator(self, df, batch_size, shuffle=True, training=True):
         # (B, Dimension)
        N = df.shape[0]
        df_matrix = df.as_matrix()
        
        if shuffle == True:
            random_permutation = np.random.permutation(N)
            
            # shuffle
            X = df_matrix[random_permutation, 1]
            y = df_matrix[random_permutation, 0].astype(int) # 0 is label's index
        else:
            X = df_matrix[:, 1]
            y = df_matrix[:, 0].astype(int)
        #
        quotient = X.shape[0] // batch_size
        remainder = X.shape[0] - batch_size * quotient

        for i in range(quotient):
            batch = {}
            batch_X = self.vectorizer.transform(X[i*batch_size:(i+1)*batch_size]).toarray()
            batch['X'] = Variable(torch.from_numpy(batch_X)).float()
            if training:
                batch_y = y[i*batch_size:(i+1)*batch_size]
                batch['y'] = Variable(torch.from_numpy(batch_y))
            else:
                batch['y'] = None
            batch['lengths'] = None
            yield batch
            
        if remainder > 0: 
            batch = {}
            batch_X = self.vectorizer.transform(X[-remainder:]).toarray()
            batch['X'] = Variable(torch.from_numpy(batch_X)).float()
            if training:
                batch_y = y[-remainder:]
                batch['y'] = Variable(torch.from_numpy(batch_y))
            else:
                batch['y'] = None
            batch['lengths'] = None
            yield batch

class Preprocess:
    '''
        Preprocess raw data
    '''
    def __init__(self):
        self.regex_remove_punc = re.compile('[%s]' % re.escape(string.punctuation))
        pass
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, sentence):
        sentence = self.unicodeToAscii(sentence.strip())
        #sentence = self.unicodeToAscii(sentence.lower().strip())
        # remove punctuation
        if False:
            sentence = self.regex_remove_punc.sub('', sentence)
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
        sentence = re.sub(r"\s+", r" ", sentence).strip()
        return sentence

    def remove_punctuation(self, sentence):
        sentence = self.regex_remove_punc.sub('', sentence)
        return sentence

    def read_txt(self, train_filename, test_filename, train_filename_no_label):
        train_df = None
        test_df = None
        train_df_no_label = None
        
        if train_filename is not None:
            train_df = pd.read_csv(train_filename, header=None, names=["label", "seq"], sep="\+\+\+\$\+\+\+",
                                  engine="python")
            # remove puncuation
            #train_df["seq"] = train_df["seq"].apply(lambda seq: self.normalizeString(seq))
            
        
        if test_filename is not None:
            with open(test_filename, "r") as f:
                reader = csv.reader(f, delimiter=",")
                rows = [[row[0], ",".join(row[1:])] for row in reader]
                test_df = pd.DataFrame(rows[1:], columns=rows[0]) # first row is column name
            # remove puncuation
            #test_df["text"] = test_df["text"].apply(lambda seq: self.normalizeString(seq))
        if train_filename_no_label is not None:
            train_df_no_label = pd.read_csv(train_filename_no_label, sep="\n", header=None, names=["seq"])
            train_df_no_label.insert(loc=0, column="nan", value=0)
            # remove puncuation
            #train_df_no_label["seq"] = train_df_no_label["seq"].apply(lambda seq: self.normalizeString(seq))
        
        return train_df, test_df, train_df_no_label

class Sample_Encode:
    '''
        Transform 
    '''
    def __init__(self, voc):
        self.voc = voc

    def sentence_to_index(self, sentence):
        encoded = list(map(lambda token: self.voc.word2index[token] if token in self.voc.word2index             else UNK_token, sentence))
        return encoded

    def pad_batch(self, index_batch):
        '''
            Return padded list with size (B, Max_length)
        '''
        return list(itertools.zip_longest(*index_batch, fillvalue=PAD_token))

    def batch_to_Variable(self, sentence_batch, training=True):
        '''
            Input: a numpy of sentence
            ex. ["i am a", "jim l o "]

            Output: a torch Variable and sentence lengths
        '''
        # split sentence
        sentence_batch = sentence_batch.tolist()
        
        # apply
        for training_sample in sentence_batch:
            # split training sentence
            training_sample[1] = training_sample[1].strip(" ").split(" ")

        # encode batch
        index_label_batch = [(training_sample[0], self.sentence_to_index(training_sample[1]))             for training_sample in sentence_batch]

        # sort sentence batch (in order to fit torch pack_pad_sequence)
        #index_label_batch.sort(key=lambda x: len(x[1]), reverse=True) 
        
        # index batch
        index_batch = [training_sample[1] for training_sample in index_label_batch]
        label_batch = [training_sample[0] for training_sample in index_label_batch]

        # record batch's length
        lengths = [len(indexes) for indexes in index_batch]

        # padded batch
        padded_batch = self.pad_batch(index_batch)

        # transform to Variable
        if training:
            pad_var = Variable(torch.LongTensor(padded_batch), volatile=False)
        else:
            pad_var = Variable(torch.LongTensor(padded_batch), volatile=True)

        # label
        if training:
            label_var = Variable(torch.LongTensor(label_batch), volatile=False)
        else:
            label_var = None

        
        return pad_var, label_var, lengths
    
    def generator(self, df, batch_size, shuffle=False, training=True):
        '''
        Return sample batch Variable
            batch['X'] is (T, B)
        '''
        df_matrix = df.as_matrix()
        if shuffle == True:
            random_permutation = np.random.permutation(len(df['seq']))
            
            # shuffle
            df_matrix = df_matrix[random_permutation]
        #
        quotient = df.shape[0] // batch_size
        remainder = df.shape[0] - batch_size * quotient

        for i in range(quotient):
            batch = {}
            X, y, lengths = self.batch_to_Variable(df_matrix[i*batch_size:(i+1)*batch_size], training)
            batch['X'] = X
            batch['y'] = y
            batch['lengths'] = lengths
            yield batch
            
        if remainder > 0: 
            batch = {}
            X, y, lengths = self.batch_to_Variable(df_matrix[-remainder:],training)
            batch['X'] = X
            batch['y'] = y
            batch['lengths'] = lengths
            yield batch

def trim(text_list, threshold=2):
    result = []
    for _, text in enumerate(text_list):
        grouping = []
        for _, g in itertools.groupby(text):
            grouping.append(list(g))
        r = ''.join([g[0] if len(g)<threshold else g[0]*threshold for g in grouping])
        result.append(r)
    return result

def token_counter(corpus):
    tokenizer = Tokenizer(num_words=None,filters="\n")
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
 
stemmer = gensim.parsing.porter.PorterStemmer()
def preprocess(string, use_stem = True):
    string = string.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt")    .replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")    .replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")    .replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")    .replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")    .replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")    .replace("couldn ' t","couldnt")
    for same_char in re.findall(r'((\w)\2{2,})', string):
        string = string.replace(same_char[0], same_char[1])
    for digit in re.findall(r'\d+', string):
        string = string.replace(digit, "1")
    for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
        if punct[0:2] =="..":
            string = string.replace(punct, "...")
        else:
            string = string.replace(punct, punct[0])
    return string

def getFrequencyDict(lines):
    freq = {}
    for s in lines:
        for w in s:
            if w in freq: freq[w] += 1
            else:         freq[w] = 1
    return freq

def initializeCmap(lines):
    print('  Initializing conversion map...')
    cmap = {}
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            cmap[w] = w
    print('    Conversion map size:', len(cmap))
    return cmap

def convertAccents(lines, cmap):
    print('  Converting accents...')
    for i, s in enumerate(lines):
        s = [(''.join(c for c in udata.normalize('NFD', w) if udata.category(c) != 'Mn')) for w in s]
        for j, w in enumerate(s):
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s
    clist = 'abcdefghijklmnopqrstuvwxyz0123456789.!?'
    for i, s in enumerate(lines):
        s = [''.join([c for c in w if c in clist]) for w in s]
        for j, w in enumerate(s):
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertPunctuations(lines, cmap):
    print('  Converting punctuations...')
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            excCnt, queCnt, dotCnt = w.count('!'), w.count('?'), w.count('.')
            if queCnt:        s[j] = '_?'
            elif excCnt >= 5: s[j] = '_!!!'
            elif excCnt >= 3: s[j] = '_!!'
            elif excCnt >= 1: s[j] = '_!'
            elif dotCnt >= 2: s[j] = '_...'
            elif dotCnt >= 1: s[j] = '_.'
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertNotWords(lines, cmap):
    print('  Converting not words...')
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if w[0] == '_': continue
            if w == '2':        s[j] = 'to'
            elif w.isnumeric(): s[j] = '_n'
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertTailDuplicates(lines, cmap):
    print('  Converting tail duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            w = re.sub(r'(([a-z])\2{2,})$', r'\g<2>\g<2>', w)
            s[j] = re.sub(r'(([a-cg-kmnp-ru-z])\2+)$', r'\g<2>', w)
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertHeadDuplicates(lines, cmap):
    print('  Converting head duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            s[j] = re.sub(r'^(([a-km-z])\2+)', r'\g<2>', w)
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertInlineDuplicates(lines, cmap, minfreq=64):
    print('  Converting inline duplicates...')
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            w = re.sub(r'(([a-z])\2{2,})', r'\g<2>\g<2>', w)
            s[j] = re.sub(r'(([ahjkquvwxyz])\2+)', r'\g<2>', w)  # repeated 2+ times, impossible
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if freq[w] > minfreq: continue
            if w == 'too': continue
            w1 = re.sub(r'(([a-z])\2+)', r'\g<2>', w) # repeated 2+ times, replace by 1
            f0, f1 = freq.get(w,0), freq.get(w1,0)
            fm = max(f0, f1)
            if fm == f0:   pass
            else:          s[j] = w1;
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertSlang(lines, cmap):
    print('  Converting slang...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if w == 'u': lines[i][j] = 'you'
            if w == 'dis': lines[i][j] = 'this'
            if w == 'dat': lines[i][j] = 'that'
            if w == 'luv': lines[i][j] = 'love'
            w1 = re.sub(r'in$', r'ing', w)
            w2 = re.sub(r'n$', r'ing', w)
            f0, f1, f2 = freq.get(w,0), freq.get(w1,0), freq.get(w2,0)
            fm = max(f0, f1, f2)
            if fm == f0:   pass
            elif fm == f1: s[j] = w1;
            else:          s[j] = w2;
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertSingular(lines, cmap, minfreq=512):
    print('  Converting singular form...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if freq[w] > minfreq: continue
            w1 = re.sub(r's$', r'', w)
            w2 = re.sub(r'es$', r'', w)
            w3 = re.sub(r'ies$', r'y', w)
            f0, f1, f2, f3 = freq.get(w,0), freq.get(w1,0), freq.get(w2,0), freq.get(w3,0)
            fm = max(f0, f1, f2, f3)
            if fm == f0:   pass
            elif fm == f1: s[j] = w1;
            elif fm == f2: s[j] = w2;
            else:          s[j] = w3;
            cmap[original_lines[i][j]] = s[j]
    lines[i] = s

def convertRareWords(lines, cmap, min_count=16):
    print('  Converting rare words...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if freq[w] < min_count: s[j] = '_r'
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertCommonWords(lines, cmap):
    print('  Converting common words...')
    #beverbs = set('is was are were am s'.split())
    #articles = set('a an the'.split())
    #preps = set('to for of in at on by'.split())

    for i, s in enumerate(lines):
        #s = [word if word not in beverbs else '_b' for word in s]
        #s = [word if word not in articles else '_a' for word in s]
        #s = [word if word not in preps else '_p' for word in s]
        lines[i] = s

def convertPadding(lines, maxlen=38):
    print('  Padding...')
    for i, s in enumerate(lines):
        lines[i] = [w for w in s if w]
    for i, s in enumerate(lines):
        lines[i] = s[:maxlen]

def preprocessLines(lines):
    global original_lines
    original_lines = lines[:]
    cmap = initializeCmap(original_lines)
    convertAccents(lines, cmap)
    convertPunctuations(lines, cmap)
    convertNotWords(lines, cmap)
    convertTailDuplicates(lines, cmap)
    convertHeadDuplicates(lines, cmap)
    convertInlineDuplicates(lines, cmap)
    convertSlang(lines, cmap)
    convertSingular(lines, cmap)
    convertRareWords(lines, cmap)
    convertCommonWords(lines, cmap)
    convertPadding(lines)
    return lines, cmap

def readData(path, label=True):
    print('  Loading', path+'...')
    _lines, _labels = [], []
    with open(path, 'r', encoding='utf_8') as f:
        for line in f:
            if label:
                _labels.append(int(line[0]))
                line = line[10:-1]
            else:
                line = line[:-1]
            _lines.append(line.split())
    if label: return _lines, _labels
    else:     return _lines

def padLines(lines, value, maxlen):
    maxlinelen = 0
    for i, s in enumerate(lines):
        maxlinelen = max(len(s), maxlinelen)
    maxlinelen = max(maxlinelen, maxlen)
    for i, s in enumerate(lines):
        lines[i] = (['_r'] * max(0, maxlinelen - len(s)) + s)[-maxlen:]
    return lines

def getDictionary(lines):
    _dict = {}
    for s in lines:
        for w in s:
            if w not in _dict:
                _dict[w] = len(_dict) + 1
    return _dict

def transformByDictionary(lines, dictionary):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in dictionary: lines[i][j] = dictionary[w]
            else:               lines[i][j] = dictionary['']

def transformByConversionMap(lines, cmap, iter=2):
    cmapRefine(cmap)
    for it in range(iter):
        for i, s in enumerate(lines):
            s0 = []
            for j, w in enumerate(s):
                if w in cmap and w[0] != '_':
                    s0 = s0 + cmap[w].split()
                elif w[0] == '_':
                    s0 = s0 + [w]
            lines[i] = [w for w in s0 if w]

def transformByWord2Vec(lines, w2v):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in w2v.wv:
                lines[i][j] = w2v.wv[w]
            else:
                lines[i][j] = w2v.wv['_r']

def readTestData(path):
    print('  Loading', path + '...')
    _lines = []
    with open(path, 'r', encoding='utf_8') as f:
        for i, line in enumerate(f):
            if i:
                start = int(np.log10(max(1, i-1))) + 2
                _lines.append(line[start:].split())
    return _lines

def savePrediction(y, path, id_start=0):
    pd.DataFrame([[i+id_start, int(y[i])] for i in range(y.shape[0])],
                 columns=['id', 'label']).to_csv(path, index=False)

def savePreprocessCorpus(lines, path):
    with open(path, 'w', encoding='utf_8') as f:
        for line in lines:
            f.write(' '.join(line) + '\n')

def savePreprocessCmap(cmap, path):
    with open(path, 'wb') as f:
        pickle.dump(cmap, f)

def loadPreprocessCmap(path):
    print('  Loading', path + '...')
    with open(path, 'rb') as f:
        cmap = pickle.load(f)
    return cmap

def loadPreprocessCorpus(path):
    print('  Loading', path + '...')
    lines = []
    with open(path, 'r', encoding='utf_8') as f:
        for line in f:
            lines.append(line.split())
    return lines

def removePunctuations(lines):
    rs = {'_!', '_!!', '_!!!', '_.', '_...', '_?'}
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in rs:
                s[j] = ''
        lines[i] = [w for w in x if w]

def removeDuplicatedLines(lines):
    lineset = set({})
    for line in lines:
        lineset.add(' '.join(line))
    for i, line in enumerate(lineset):
        lines[i] = line.split()
    del lines[-(len(lines)-len(lineset)):]
    return lineset

def shuffleData(lines, labels):
    for i, s in enumerate(lines):
        lines[i] = (s, labels[i])
    np.random.shuffle(lines)
    for i, s in enumerate(lines):
        labels[i] = s[1]
        lines[i] = s[0]

def cmapRefine(cmap):
    cmap['teh'] = cmap['da'] = cmap['tha'] = 'the'
    cmap['evar'] = 'ever'
    cmap['likes'] = cmap['liked'] = cmap['lk'] = 'like'
    cmap['wierd'] = 'weird'
    cmap['kool'] = 'cool'
    cmap['yess'] = 'yes'
    cmap['pleasee'] = 'please'
    cmap['soo'] = 'so'
    cmap['noo'] = 'no'
    cmap['lovee'] = cmap['loove'] = cmap['looove'] = cmap['loooove'] = cmap['looooove']         = cmap['loooooove'] = cmap['loves'] = cmap['loved'] = cmap['wuv']         = cmap['loovee'] = cmap['lurve'] = cmap['lov'] = cmap['luvs'] = 'love'
    cmap['lovelove'] = 'love love'
    cmap['lovelovelove'] = 'love love love'
    cmap['ilove'] = 'i love'
    cmap['liek'] = cmap['lyk'] = cmap['lik'] = cmap['lke'] = cmap['likee'] = 'like'
    cmap['mee'] = 'me'
    cmap['hooo'] = 'hoo'
    cmap['sooon'] = cmap['soooon'] = 'soon'
    cmap['goodd'] = cmap['gud'] = 'good'
    cmap['bedd'] = 'bed'
    cmap['badd'] = 'bad'
    cmap['sadd'] = 'sad'
    cmap['madd'] = 'mad'
    cmap['redd'] = 'red'
    cmap['tiredd'] = 'tired'
    cmap['boredd'] = 'bored'
    cmap['godd'] = 'god'
    cmap['xdd'] = 'xd'
    cmap['itt'] = 'it'
    cmap['lul'] = cmap['lool'] = 'lol'
    cmap['sista'] = 'sister'
    cmap['w00t'] = 'woot'
    cmap['srsly'] = 'seriously'
    cmap['4ever'] = cmap['4eva'] = 'forever'
    cmap['neva'] = 'never'
    cmap['2day'] = 'today'
    cmap['homee'] = 'home'
    cmap['hatee'] = 'hate'
    cmap['heree'] = 'here'
    cmap['cutee'] = 'cute'
    cmap['lemme'] = 'let me'
    cmap['mrng'] = 'morning'
    cmap['gd'] = 'good'
    cmap['thx'] = cmap['thnx'] = cmap['thanx'] = cmap['thankx'] = cmap['thnk'] = 'thanks'
    cmap['jaja'] = cmap['jajaja'] = cmap['jajajaja'] = 'haha'
    cmap['eff'] = cmap['fk'] = cmap['fuk'] = cmap['fuc'] = 'fuck'
    cmap['2moro'] = cmap['2mrow'] = cmap['2morow'] = cmap['2morrow']         = cmap['2morro'] = cmap['2mrw'] = cmap['2moz'] = 'tomorrow'
    cmap['babee'] = 'babe'
    cmap['theree'] = 'there'
    cmap['thee'] = 'the'
    cmap['woho'] = cmap['wohoo'] = 'woo hoo'
    cmap['2gether'] = 'together'
    cmap['2nite'] = cmap['2night'] = 'tonight'
    cmap['nite'] = 'night'
    cmap['dnt'] = 'dont'
    cmap['rly'] = 'really'
    cmap['gt'] = 'get'
    cmap['lat'] = 'late'
    cmap['dam'] = 'damn'
    cmap['4ward'] = 'forward'
    cmap['4give'] = 'forgive'
    cmap['b4'] = 'before'
    cmap['tho'] = 'though'
    cmap['kno'] = 'know'
    cmap['grl'] = 'girl'
    cmap['boi'] = 'boy'
    cmap['wrk'] = 'work'
    cmap['jst'] = 'just'
    cmap['geting'] = 'getting'
    cmap['4get'] = 'forget'
    cmap['4got'] = 'forgot'
    cmap['4real'] = 'for real'
    cmap['2go'] = 'to go'
    cmap['2b'] = 'to be'
    cmap['gr8'] = cmap['gr8t'] = cmap['gr88'] = 'great'
    cmap['str8'] = 'straight'
    cmap['twiter'] = 'twitter'
    cmap['iloveyou'] = 'i love you'
    cmap['loveyou'] = cmap['loveya'] = cmap['loveu'] = 'love you'
    cmap['xoxox'] = cmap['xox'] = cmap['xoxoxo'] = cmap['xoxoxox']         = cmap['xoxoxoxo'] = cmap['xoxoxoxoxo'] = 'xoxo'
    cmap['cuz'] = cmap['bcuz'] = cmap['becuz'] = 'because'
    cmap['iz'] = 'is'
    cmap['aint'] = 'am not'
    cmap['fav'] = 'favorite'
    cmap['ppl'] = 'people'
    cmap['mah'] = 'my'
    cmap['r8'] = 'rate'
    cmap['l8'] = 'late'
    cmap['w8'] = 'wait'
    cmap['m8'] = 'mate'
    cmap['h8'] = 'hate'
    cmap['l8ter'] = cmap['l8tr'] = cmap['l8r'] = 'later'
    cmap['cnt'] = 'cant'
    cmap['fone'] = cmap['phonee'] = 'phone'
    cmap['f1'] = 'fONE'
    cmap['xboxe3'] = 'eTHREE'
    cmap['jammin'] = 'jamming'
    cmap['onee'] = 'one'
    cmap['1st'] = 'first'
    cmap['2nd'] = 'second'
    cmap['3rd'] = 'third'
    cmap['inet'] = 'internet'
    cmap['recomend'] = 'recommend'
    cmap['ah1n1'] = cmap['h1n1'] = 'hONEnONE'
    cmap['any1'] = 'anyone'
    cmap['every1'] = cmap['evry1'] = 'everyone'
    cmap['some1'] = cmap['sum1'] = 'someone'
    cmap['no1'] = 'no one'
    cmap['4u'] = 'for you'
    cmap['4me'] = 'for me'
    cmap['2u'] = 'to you'
    cmap['yu'] = 'you'
    cmap['yr'] = cmap['yrs'] = cmap['years'] = 'year'
    cmap['hr'] = cmap['hrs'] = cmap['hours'] = 'hour'
    cmap['min'] = cmap['mins'] = cmap['minutes'] = 'minute'
    cmap['go2'] = cmap['goto'] = 'go to'
    for key, value in cmap.items():
        if not key.isalpha():
            if key[-1:] == 'k':
                cmap[key] = '_n'
            if key[-2:]=='st' or key[-2:]=='nd' or key[-2:]=='rd' or key[-2:]=='th':
                cmap[key] = '_ord'
            if key[-2:]=='am' or key[-2:]=='pm' or key[-3:]=='min' or key[-4:]=='mins'                     or key[-2:]=='hr' or key[-3:]=='hrs' or key[-1:]=='h'                     or key[-4:]=='hour' or key[-5:]=='hours'                    or key[-2:]=='yr' or key[-3:]=='yrs'                    or key[-3:]=='day' or key[-4:]=='days'                    or key[-3:]=='wks':
                cmap[key] = '_time'
def preprocessTestingData(path):
    print('Loading testing data...')
    lines = readTestData(path)

    cmap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/cmap.pkl')
    w2v_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/word2vec.pkl')
    cmap = loadPreprocessCmap(cmap_path)
    transformByConversionMap(lines, cmap)
    
    lines = padLines(lines, '_', maxlen)
    w2v = Word2Vec.load(w2v_path)
    transformByWord2Vec(lines, w2v)
    return lines

def preprocessTrainingData(label_path, nolabel_path, retrain=False, punctuation=True):
    print('Loading training data...')
    if retrain:
        preprocess(label_path, nolabel_path)

    lines, labels = readData(label_path)
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/corpus.txt')
    cmap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/cmap.pkl')
    w2v_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/word2vec.pkl')
    lines = readData(corpus_path, label=False)[:len(lines)]
    shuffleData(lines, labels)
    labels = np.array(labels)

    cmap = loadPreprocessCmap(cmap_path)
    transformByConversionMap(lines, cmap)
    if not punctuation:
        removePunctuations(lines)

    lines = padLines(lines, '_', maxlen)
    w2v = Word2Vec.load(w2v_path)
    transformByWord2Vec(lines, w2v)
    return lines, labels

def preprocess(label_path, nolabel_path):
    print('Preprocessing...')
    labeled_lines, labels = readData(label_path)
    nolabel_lines = readData(nolabel_path, label=False)
    lines = labeled_lines + nolabel_lines

    lines, cmap = preprocessLines(lines)
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/corpus.txt')
    cmap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/cmap.pkl')
    w2v_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/word2vec.pkl')
    savePreprocessCorpus(lines, corpus_path)
    savePreprocessCmap(cmap, cmap_path)

    transformByConversionMap(lines, cmap)
    removeDuplicatedLines(lines)

    print('Training word2vec...')
    model = Word2Vec(lines, size=256, min_count=16, iter=16, workers=16)
    model.save(w2v_path)


# Some more good looking patterns here. We can however see that with 3-grams clear patterns are rare. "great customer service" occurs 12 times in 2362 positive responses, which really doesn't say much in general. 
# 
# Satisfied that our data looks possible to work with begin to construct our first classifier.
# 
# # First Classifier
# Lets start simple with a bag-of-words Support-Vector-Machine (SVM) classifier. Bag-of-words means that we represent each sentence by the unique words in it. To make this representation useful for our SVM classifier we transform each sentence into a vector. The vector is of the same length as our vocabulary, i.e. the list of all words observed in our training data, with each word representing an entry in the vector. If a particular word is present, that entry in the vector is 1, otherwise 0.
# 
# To create these vectors we use the CountVectorizer from [sklearn][1]. 
# 
# 
#   [1]: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# ## Preparing the data

# In[ ]:


import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2))


# In[ ]:


vectorized_data = count_vectorizer.fit_transform(tweets.text)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))


# In[ ]:


def sentiment2target(sentiment):
    return {
        'negative': 0,
        'neutral': 1,
        'positive' : 2
    }[sentiment]
targets = tweets.airline_sentiment.apply(sentiment2target)


# To check performance of our classifier we want to split our data in to train and test.

# In[ ]:


from sklearn.model_selection import train_test_split
data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.4, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]


# ## Fitting a classifier
# 
# We're now ready to fit a classifier to our data. We'll spend more time on hyper parameter tuning later, so for now we just pick some reasonable guesses.

# In[ ]:


from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf.fit(data_train, targets_train)


# ## Evaluation of results

# In[ ]:


clf.score(data_test, targets_test)


# It's most likely possible to achieve a higher score with more tuning, or a more advanced approach. Lets check on how it does on a couple of sentences.

# In[ ]:


sentences = count_vectorizer.transform([
    "What a great airline, the trip was a pleasure!",
    "My issue was quickly resolved after calling customer support. Thanks!",
    "What the hell! My flight was cancelled again. This sucks!",
    "Service was awful. I'll never fly with you again.",
    "You fuckers lost my luggage. Never again!",
    "I have mixed feelings about airlines. I don't know what I think.",
    ""
])
clf.predict_proba(sentences)


# So while results aren't very impressive overall, we can see that it's doing a good job on these obvious sentences. 
# 
# ## What is hard for the classifier?
# 
# It's interesting to know which sentences are hard. To find out, lets apply the classifier to all our test sentences and sort by the marginal probability.

# Here are some of the hardest sentences.

# In[ ]:


predictions_on_test_data = clf.predict_proba(data_test)
index = np.transpose(np.array([range(0,len(predictions_on_test_data))]))
indexed_predictions = np.concatenate((predictions_on_test_data, index), axis=1).tolist()


# In[ ]:


def margin(p):
    top2 = p.argsort()[::-1]
    return abs(p[top2[0]]-p[top2[1]])
margin = sorted(list(map(lambda p : [margin(np.array(p[0:3])),p[3]], indexed_predictions)), key=lambda p : p[0])
list(map(lambda p : tweets.iloc[data_test_index[int(p[1])].toarray()[0][0]].text, margin[0:10]))


# and their probability distributions?

# In[ ]:


list(map(lambda p : predictions_on_test_data[int(p[1])], margin[0:10]))


# How about the easiest sentences?

# In[ ]:


list(map(lambda p : tweets.iloc[data_test_index[int(p[1])].toarray()[0][0]].text, margin[-10:]))


# and their probability distributions?

# In[ ]:


list(map(lambda p : predictions_on_test_data[int(p[1])], margin[-10:]))


# Looks like all of the easiest sentences are negative. What is the distribution of certainty across all sentences?

# In[ ]:


import matplotlib.pyplot as plt
marginal_probs = list(map(lambda p : p[0], margin))
n, bins, patches = plt.hist(marginal_probs, 25, facecolor='blue', alpha=0.75)
plt.title('Marginal confidence histogram - All data')
plt.ylabel('Count')
plt.xlabel('Marginal probability [abs(p_positive - p_negative)]')
plt.show()


# Lets break it down by positive and negative sentiment to see if one is harder than the other.
# 
# ### Positive data

# In[ ]:


positive_test_data = list(filter(lambda row : row[0]==2, hstack((targets_test[:,None], data_test)).toarray()))
positive_probs = clf.predict_proba(list(map(lambda r : r[1:], positive_test_data)))
marginal_positive_probs = list(map(lambda p : abs(p[0]-p[1]), positive_probs))
n, bins, patches = plt.hist(marginal_positive_probs, 25, facecolor='green', alpha=0.75)
plt.title('Marginal confidence histogram - Positive data')
plt.ylabel('Count')
plt.xlabel('Marginal probability')
plt.show()


# ### Neutral data

# In[ ]:


positive_test_data = list(filter(lambda row : row[0]==1, hstack((targets_test[:,None], data_test)).toarray()))
positive_probs = clf.predict_proba(list(map(lambda r : r[1:], positive_test_data)))
marginal_positive_probs = list(map(lambda p : abs(p[0]-p[1]), positive_probs))
n, bins, patches = plt.hist(marginal_positive_probs, 25, facecolor='yellow', alpha=0.75)
plt.title('Marginal confidence histogram - Neutral data')
plt.ylabel('Count')
plt.xlabel('Marginal probability')
plt.show()


# ### Negative data

# In[ ]:


negative_test_data = list(filter(lambda row : row[0]==0, hstack((targets_test[:,None], data_test)).toarray()))
negative_probs = clf.predict_proba(list(map(lambda r : r[1:], negative_test_data)))
marginal_negative_probs = list(map(lambda p : abs(p[0]-p[1]), negative_probs))
n, bins, patches = plt.hist(marginal_negative_probs, 25, facecolor='red', alpha=0.75)
plt.title('Marginal confidence histogram - Negative data')
plt.ylabel('Count')
plt.xlabel('Marginal probability')
plt.show()


# Clearly the positive data is much harder for the classifier. This makes sense since there's a lot less of it. An important challenge in building a classifier will then be how to handle positive data.

# # In Progress
# # Second classifier - Convolutional Neural Network
# 
# We're going to build a classifier based on convolutional neural networks.  A good resource for learning about Deep Learning (and machine learning in general) is [Christopher Olah's blog][1]. The convolution neural network approach in particular is explained nicely in [this post][2] by WildML. Finally I recommend [this paper][3] by Yoon Kim, then at NYU. I'll leave these resources to explain the theory behind our approach, and instead focus on getting a working implementation.
# 
# 
#   [1]: http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/
#   [2]: http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
#   [3]: https://arxiv.org/pdf/1408.5882.pdf

# # Word Embeddings
# Word embeddings, or vector representations of words, are critical to building a CNN classifier. The vector representations of words are what will build up our input matrix. These vector space models represent words in a vector space such that similar words are mapped to nearby points. This representation rests on the [Distributional Hypothesis][1], i.e. assumption that words that appear in similar contexts share semantic meaning. We will use gensim to train word embeddings from our corpus.
# 
# 
#   [1]: https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis

# In[ ]:


# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 7           # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


# In[ ]:


from gensim.models import word2vec
model = word2vec.Word2Vec(tweets.normalized_tweet, workers=num_workers,                           size=num_features, min_count = min_word_count,                           window = context, sample = downsampling)
model.init_sims(replace=True)


# In[ ]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
X = model[model.wv.vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
plt.rcParams["figure.figsize"] = (20,20)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
labels = list(model.wv.vocab.keys())
for label, x, y in zip(labels, X_tsne[:, 0], X_tsne[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-1, -1),
        textcoords='offset points', ha='right', va='bottom')

plt.show()


# In[ ]:




