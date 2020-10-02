# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import re
import gc
import time
import numpy as np
import pandas as pd
from gensim import corpora
from gensim import models
import keras.backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input, CuDNNLSTM, CuDNNGRU, Dense, Bidirectional, Embedding
from keras.layers import concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import BatchNormalization, SpatialDropout1D, Dropout, Permute, Multiply, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback



# create an array for replace some contractions
replace_patterns = [
('(A|a)in\'t', 'is not'),
('(C|c)an\'t', 'can not'),
('(H|h)ow\'s', 'how is'),
('(H|h)ow\'d', 'how did'), 
('(H|h)ow\'d\'y', 'how do you'),
('(H|h)ere\'s', 'here is'),
('(I|i)t\'s', 'it is'),
('(I|i)\'m', 'i am'),
('f(u\*|\*c)k', 'fuck'),    
('(L|l)et\'s', 'let us'),
('(M|m)a\'am', 'madam'),
('sh\*t', 'shit'),
('(S|s)han\'t', 'shall not'), 
('(S|s)ha\'n\'t', 'shall not'), 
('(S|s)o\'s', 'so as'),    
('(T|t)his\'s', 'this is'),
('(T|t)here\'s', 'there is'),
('(W|w)on\'t', 'will not'),
('(W|w)hat\'s', 'what is'),    
('(W|w)hatis', 'what is'),
('(W|w)hen\'s', 'when is'), 
('(W|w)here\'d', 'where did'), 
('(W|w)here\'s', 'where is'), 
('(W|w)ho\'s', 'who is'), 
('(W|w)hy\'s', 'why is'),     
('(Y|y)\'all', 'you all'), 
('o\'clock', 'of the clock'),
('\'cause', 'because'),  
('(\w+)\'ve', '\g<1> have'),
('(\w+)\'ll', '\g<1> will'),
('(\w+)n\'t', '\g<1> not'),
('(\w+)\'re', '\g<1> are'),
('(\w+)\'d', '\g<1> would')]
# function for clean contractions
def clean_contraction(x):    
    x = str(x)    
    for punct in "��":
        if punct in x:
            x = x.replace(punct, "'") 
    for pattern, repl in replace_patterns:
        if re.search(pattern,x):
            x = re.sub(pattern, repl, x)
    return x


# unify the expression of time
check_pm = re.compile(r'[0-9]+p[.]?m[.]?')
check_PM = re.compile(r'[0-9]+P[.]?M[.]?')
check_am = re.compile(r'[0-9]+a[.]?m[.]?')
check_AM = re.compile(r'[0-9]+A[.]?M$[.]?')
# write the function 
def fix_time(x):  
    x = str(x)
    if re.search(check_pm, x):
        x = re.sub('p.m', ' PM', x)
    if re.search(check_PM, x):
        x = re.sub('P.M', ' PM', x)
    if re.search(check_am, x):
        x = re.sub('a.m', ' AM', x)
    if re.search(check_AM, x):
        x = re.sub('A.M', ' AM', x)       
    return x



# fix duplication of letters
goood = re.compile(r'g+(o)\1{2,}(d)+') # replace gooodddd by good
check_duplicate = re.compile(r'\w*(\S)\1{2,}\w*') # replace words such as fantasticccccc by fantastic
# fix duplications and clean some puncs
def clean_punc(x):
    x = str(x)
    if re.search(goood,x): # we can treat goood and goooood in the same way
        x = re.sub(goood, 'good', x)
    if re.findall(check_duplicate,x): # we replace other duplicate characters
        x = re.sub(r'(\D)\1{2,}', r'\1', x)
    if re.search('(\[.*math).+(math\])',x): # dealing with math functions(borrowed from kaggle)
        x = re.sub('(\[.*math).+(math\])', '[latex formula]', x)
    if "'s " in x:
        x = x.replace("'s "," ")
    if "'" in x:
        x = x.replace("'", '')
    if "_" in x:
        x = x.replace("_", ' and ')
    return x


# we fix common wrong spellings in our specific document context
mispell_dict = {    'colour':'color',
                    'centre':'center',
                    'didnt':'did not',
                    'Didnt':'Did not',
                    'Doesnt':'Does not',
                    'Couldnt':'Could not',
                    'doesnt':'does not',
                    'isnt':'is not',
                    'shouldnt':'should not',
                    'flavour':'flavor',
                    'flavours':'flavors',
                    'wasnt':'was not',
                    'cancelled':'canceled',
                    'neighbourhood':'neighborhood',
                    'neighbour':'neighbor',
                    'theatre':'theater',
                    'grey':'gray',
                    'favourites':'favorites',
                    'favourite':'favorite',
                    'flavoured':'flavored',
                    'acknowledgement':'acknowledgment',
                    'judgement':'judgment',
                    'speciality':'specialty',
                    'favour':'favor',
                    'colours':'colors',
                    'coloured':'colored',
                    'theatres':'theaters',
                    'behaviour':'behavior',
                    'travelling':'traveling',
                    'colouring':'coloring',
                    'labelled':'labeled',
                    'cancelling':'canceling',
                    'waitedand': 'waited and',
                    'whisky':'Whisky',
                    'tastey':'tasty',
                    'goodbut': 'good but',
                    'sushis':'sushi',
                    'disapoointed': 'disappointed',
                    'disapointed':'disappointed',
                    'disapointment':'disappointment',
                    'Amzing':'Amazing',
                    'bAd':'bad',
                    'fantastics':'fatastic',
                    'flavuorful':'flavorful',
                    'infomation':'information',
                    'informaiton':'information',
                    'eveeyone':'everyone',
                    'Hsppy':'Happy',
                    'waygu':'wagyu',
                    'unflavorful':'untasty',
                    'fiancé':'fiance',
                    'jalapeño':'jalapeno',
                    'jalapeños':'jalapenos',
                    'sautéed':'sauteed',
                    'Café':'Cafe',
                    'café':'cafe',
                    'entrée':'entree',
                    'brûlée':'brulee',
                    'entrées':'entrees',
                    'Montréal':'Montreal',
                    'crème':'creme',
                    'Jalapeño':'jalapeno',
                    'crêpe':'crepe',
                    'Crêpe':'Crepe',
                    'Flavortown': 'Flavor Town',
                    '\u200b': ' ',
                    'fck':'fuck',
                    'wi-fi':'wifi',
                    'ayce':'all you can eat',
                    'appriceiate':'appriciate',
                    'worest':'worst'}
def correct_spelling(x):
    x = str(x)
    for word in mispell_dict.keys():
        if word in x:
            x = x.replace(word, mispell_dict[word])
    return x

# seperate words, numbers and some unremoved punctuations such as ,.?!
def seperate_word(x):
    for pattern, repl in [('[\W]',lambda p:' '+p.group()+' '),('[0-9]{1,}',lambda p:' '+p.group()+' ')]:
        if re.search(pattern,x):   
            x = re.sub(pattern, repl, x)
    return x
    
    
    
# read the train data and test data
train_df = pd.read_csv('../input/assessment2-dataset/train_data.csv')
label_df = pd.read_csv('../input/assessment2-dataset/train_label.csv')
test_df = pd.read_csv('../input/assessment2-dataset/test_data.csv')


# apply above defined function to complete the preprocessing
# please note that we have to make the test data having the same look as train data for making predictions
# however, we will not use anything from test data when train the model
train_df['text'] = train_df['text'].str.lower().map(clean_contraction).map(clean_punc).map(fix_time).map(correct_spelling).map(seperate_word)
test_df['text'] = test_df['text'].str.lower().map(clean_contraction).map(clean_punc).map(fix_time).map(correct_spelling).map(seperate_word)


# ================================== END of preprocessing ===========================================




# setting some the parameters
maxlen = 280    # max number of words in each review
max_words = 100000  # we only keep the most frequent 100000 words in vocab


train_X = train_df["text"].values   # put training data into an array
test_X = test_df["text"].values    # put testing data into an array
tokenizer = Tokenizer(num_words=max_words, filters='\t\n\r')
tokenizer.fit_on_texts(list(train_X)+list(test_X)) # tokenize the training data and build corpus


# convert words to corresponding index according to the oder of words in vocab
train_X = tokenizer.texts_to_sequences(train_X) 
test_X = tokenizer.texts_to_sequences(test_X) 






#========================================= tfidf ========================================================

# tranform the vocab dictionary to {index:word} format
id_to_word = {}
for word in tokenizer.word_index:
    id_to_word.update({tokenizer.word_index[word]:word})

# using the function from genism to build a vocab dictionary
word_list = pd.concat([train_df["text"], test_df["text"]], ignore_index=True).str.split()
dictionary = corpora.Dictionary(word_list)

# create the bag of word model looks like [[(0,f),(1,f),(2,f)],[(0,f),(1,f),(2,f)]] where f represents term frequency
string_bow = word_list.map(dictionary.doc2bow)
tfidf = models.TfidfModel(list(string_bow))

# function for converting a tup to a dictionary
def tup_to_dict(x):
    dic = {}
    for i in x:
        dic.update({i[0]:i[1]}) 
    return dic
id_to_tfidf_tup = [tfidf[id_to_fre] for id_to_fre in string_bow]
id_to_tfidf_dict = [tup_to_dict(i) for i in id_to_tfidf_tup]
train_dict = id_to_tfidf_dict[:len(train_X)]
test_dict = id_to_tfidf_dict[len(train_X):]


word_to_id = dictionary.token2id  # convert tokenized word to corresponding index according to vocab

# write a function to get corresponding index in tfidf dict for each token
def get_id(token_id):
    word = [id_to_word[i] for i in token_id] # converting token ids back to token words
    # check whether these token words exist in the dictionary build for tfidf, replace by 0 if not exist
    tfidf_id = [word_to_id.get(i) if word_to_id.get(i) is not None else 0 for i in word] 
    return tfidf_id    

# apply above function
# replace tokens in each reviews by corresponding id in the tfidf dict
train_tfidf_id = [get_id(i) for i in train_X]
test_tfidf_id = [get_id(i) for i in test_X]

# a function to get corresponding tfidf value given token id
def get_tfidf(id_tf,tid):
    return[id_tf.get(i,0) for i in tid]

# replace each token id by corresponding tfidf value for each review
train_tfidf = [get_tfidf(train_dict[i],tid) for i,tid in enumerate(train_tfidf_id)]
test_tfidf = [get_tfidf(test_dict[i],tid) for i,tid in enumerate(test_tfidf_id)]

#========================================= END ====================================================






del id_to_word, word_list, dictionary, string_bow, tfidf, id_to_tfidf_tup, id_to_tfidf_dict, train_dict, test_dict, word_to_id, train_tfidf_id, test_tfidf_id
# delete variables that will not be used further and collect memory
gc.collect()



# fix the length of each review and padding those shorter sentences with 0s
train_tfidf = pad_sequences(train_tfidf, maxlen=maxlen, dtype='float64')
test_tfidf = pad_sequences(test_tfidf, maxlen=maxlen, dtype='float64')
train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)
train_y = label_df['label'].values
# Now all the features required had been extracted
# the next step is getting the embedding matrix





# pre-trained embedding files
# remember to download the embedding files
glove = '../input/glove.840B.300d.txt'
paragram =  '../input/paragram_300_sl999.txt'
wiki_news = '../input/wiki-news-300d-1M.vec'





# ======================================= embedding ==============================================

# function for loading embedding files
def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')    
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding="utf8", errors='ignore') if len(o)>100)   
    return embeddings_index


embedding_matrix = []
for embed_path in [glove,paragram,wiki_news]:
    embed = load_embed(embed_path)
    word_index = tokenizer.word_index
    nb_words = min(max_words, len(word_index))
    embedding_vec = np.zeros((nb_words, 300))    
    for word, i in word_index.items():
        if i >= max_words: continue   # only consider the most 100000 frequent words
            # embedding the word if it can be easily found in pre-trained file
        if embed.get(word) is not None: 
            embedding_vec[i] = embed.get(word)
            # embedding the word if the upper case version of it can be found
        elif embed.get(word.upper()) is not None: 
            embedding_vec[i] = embed.get(word.upper())
            # embedding the word if the capitalized version can be found
        elif embed.get(word.capitalize()) is not None: 
            embedding_vec[i] = embed.get(word.capitalize())
        else: # otherwise, we use the vector of the word 'something'
            embedding_vec[i] = embed.get('something')        
    embedding_matrix.append(embedding_vec)    
    del embed
    gc.collect()
    
# concate all 3 embedding vectors
embedding = np.concatenate(embedding_matrix,axis=1)   
del embedding_matrix
gc.collect()

# https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go
from gensim.models import KeyedVectors
# load the 4th pre-trained embeddings (googleNews embeddings)
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'
ggle = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

words = ggle.index2word
embed = {}  # change ggle to a dictionary
for i,word in enumerate(words):
    embed[word] = i

embedding_vec = np.zeros((nb_words, 300))    
for word, i in word_index.items():
    if i >= max_words: continue            
    if embed.get(word) is not None: embedding_vec[i] = ggle[word]
            # embedding the word if the upper case version of it can be found
    elif embed.get(word.upper()) is not None: embedding_vec[i] = ggle[word.upper()]
            # embedding the word if the capitalized version can be found
    elif embed.get(word.capitalize()) is not None: embedding_vec[i] = ggle[word.capitalize()]
    else: embedding_vec[i] = ggle['something']             
        
ggle_embed = embedding_vec

# concate the preovious embedding vectors with the 4th one
embedding = np.concatenate([embedding, ggle_embed], axis=1)

#============================================== END ==============================================


#========================== build model and training=============================


# this is the cylical learning that we will use in our model
# referenced from: https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py
class CyclicLR(Callback):
    def __init__(self, base_lr=0.0001, max_lr=0.002, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)        
    def on_train_begin(self, logs={}):
        logs = logs or {}
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())                    
    def on_batch_end(self, epoch, logs=None):        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)        
        K.set_value(self.model.optimizer.lr, self.clr())      



# manipulate the label from 1-5 to 0-4
train_y = train_y - 1
train_y


# cutting the training data and label for validation
train_x, val_x = train_X[:600000], train_X[600000:]
train_idf, val_idf = train_tfidf[:600000], train_tfidf[600000:]
train_y1, val_y = train_y[:600000], train_y[600000:]


# setting the parameters
maxlen=280
max_words = 100000 # embedding size or vocab size
EMBEDDING_DIM = 1200

#=================================== training ============================================
word = Input(shape=(maxlen,))    
embed = Embedding(max_words, EMBEDDING_DIM, trainable=False, input_length=maxlen, weights=[embedding])(word)

tfidf = Input(shape=(maxlen,)) 
output2 = RepeatVector(900)(tfidf)  # repeating the tfidf input 900 times make=ing it the same dimension as embeddings
output2 = Permute((2, 1), input_shape=(900, maxlen))(output2)  # switch the axis of matrix
output2 = Multiply()([output2, embed]) # multiply embedding outputs and repeated tfidf
output2 = GlobalMaxPooling1D()(output2) # pooling
output2 = Dense(256, activation='relu')(output2) # pass it to a regular feed forward dense layer

output1 = SpatialDropout1D(0.2)(embed)    
output1 = Bidirectional(CuDNNLSTM(300, return_sequences=True))(output1)
output1 = Bidirectional(CuDNNLSTM(300, return_sequences=True))(output1)
avg_pool = GlobalAveragePooling1D()(output1)
max_pool = GlobalMaxPooling1D()(output1)
output1 = concatenate([avg_pool, max_pool])    
output1 = BatchNormalization()(output1)
output1 = Dense(256, activation='relu')(output1)
output1 = concatenate([output1, output2])  # concate the output1(from LSTM) and output2(from tfidf)
output1 = Dense(128, activation='relu')(output1) 
output1 = Dense(5, activation='softmax')(output1)

model = Model(inputs=[word,tfidf], outputs=outputs) 
adam = optimizers.Adam(clipvalue=3.5) #Gradients will be clipped when their absolute value exceeds this value
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['acc'])    


# fit your model
clr = CyclicLR(base_lr=1e-4, max_lr=2e-3, step_size=np.ceil(1.5 * train_X.shape[0]/512))

# if you want to add validation or change some parameters feel free to alter the model.fit
model.fit([train_X,train_tfidf], train_y, batch_size=512, epochs=3, verbose=1, callbacks=[clr])
# validation_data = ([val_x,val_idf],val_y)

#======================================= end ==================================================

