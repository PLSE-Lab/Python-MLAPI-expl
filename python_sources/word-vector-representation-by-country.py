#### This script trains skipgram models on the statements of different countries.
#### The trained vectors are then used to visualize some most frequent words for each country and to find the words that are the closest to a given.
#### The script is largely inspired by the this tutorial https://www.tensorflow.org/versions/master/tutorials/word2vec

## TODO 
# add RNN
# concat all sentences to only array of words when training RNN to generate text

import numpy as np
import tensorflow as tf
import csv
import string
import collections
from nltk.corpus import stopwords
from nltk import tokenize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re


#### Helper function 
## Load data to sentences collection by country
def collect_sent_country():
    country_collect ={}
    with open('../input/un-general-debates.csv', newline='') as file:
        read = csv.DictReader(file, delimiter=',', quotechar='"')
        for row in read:
            if row['country'] in country_collect:
                country_collect[row['country']]+= ' '+row['text']
            else:
                country_collect[row['country']] = row['text']
    return country_collect
 
## Strip punctuations and split sentences into words 
def sent_split_words(countrylist,country_collect):
    collect_return=[]
    length_return = []
    for c in countrylist:
        sentences = tokenize.sent_tokenize(country_collect[c])
        word_collection = []
        for s in sentences:
            words = re.sub('(?<=\w)([!?.,;\'])', r' \1',s).replace('\ufeff',' ').replace('\n',' ').lower().split()
            words.append('<EOS>')
            word_collection.append(words)
        collect_return.append(word_collection)
    return collect_return
    

def tokens_create(data):
    inverse_tokens = []
    tokens = {}
    data =  [w for s in data for w in s]
    idx= 0 
    count = collections.Counter(data).most_common()
    for w in data:
        if w not in tokens:
            tokens[w] = idx
            inverse_tokens.append(w)
            idx += 1
    assert all(w in tokens for w in data) # check all words ar included
    return tokens, inverse_tokens, count

## Generate batches
def batch_generate(data, tokens, batch_size,windows_size,number_skip):
    center = []
    target = []
    for _ in range(batch_size//number_skip):
        sent = data[np.random.randint(0,len(data))] # pick a sentence
        while windows_size>=len(sent)-windows_size:
            sent = data[np.random.randint(0,len(data))] 
        exclude_indx = []
        center_indx = np.random.randint(windows_size,len(sent)-windows_size) # pick a center word that allows full window
        for _ in range(number_skip):
            span = list(range(center_indx-windows_size,center_indx+windows_size+1))
            exclude_indx.append(center_indx)
            target_indx = center_indx
            while target_indx in exclude_indx:
                target_indx = np.random.choice(span)
            center.append(tokens[sent[center_indx]])
            target.append(tokens[sent[target_indx]])  
    center = np.array(center)
    target = np.reshape(np.array(target),(-1,1))
    return center,target
    
## Filter for stop words    
def frequent_non_stop_word(count,number): 
    stop_words = stopwords.words("english")
    word_return = []
    freq_return = []
    i = 1
    while len(word_return)<number:
        if count[i][0] not in stop_words:
            word_return.append(count[i][0])
            freq_return.append(count[i][1])
        i+=1
    return word_return, freq_return


#### Skipgram w/ noise-contrastive loss
class Skipgram:
    def __init__(self, data, batch_size = 132,embed_size = 128,neg_samp =64,token_size=5000):
            
            self.batch_size = batch_size
            self.data = data
            self.tokens,self.inverse_tokens,self.frequent_w = tokens_create(data)
            
            vocab_size = len(list(self.tokens.keys()))
            self.center = tf.placeholder(tf.int32,shape=[batch_size]) # center word
            self.target = tf.placeholder(tf.int32,shape = [batch_size,1]) # context words
            embed_matrix = tf.Variable(tf.random_uniform([vocab_size,embed_size],-1.0,1.0))
            embedding = tf.nn.embedding_lookup(embed_matrix,self.center)
            weights = tf.Variable(tf.truncated_normal([vocab_size,embed_size],stddev=1.0/embed_size**0.5))
            biases = tf.Variable(tf.zeros([vocab_size]))
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights = weights, biases = biases, labels = self.target, inputs = embedding, num_sampled = neg_samp, num_classes = vocab_size))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(self.loss)
            
            norm = tf.sqrt(tf.reduce_sum(tf.square(embed_matrix), 1, keep_dims=True))
            self.embed_matrix_norm  = embed_matrix / norm
            
    def train(self,train_round = 100001,windows_size= 3,number_skip= 2):
        loss_ar = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for r in range(train_round):
                center_batch, target_batch = batch_generate(data=self.data, tokens = self.tokens, 
                batch_size= self.batch_size, windows_size= windows_size, number_skip= number_skip)
                
                cur_loss,_ = sess.run([self.loss,self.optimizer],feed_dict = {
                self.center:center_batch,
                self.target:target_batch})
                
                loss_ar.append(cur_loss)
                
                if r%10000 == 0:
                    print('Round %d completed'% r)
        
            trained_embedding = sess.run(self.embed_matrix_norm)
        return trained_embedding,loss_ar
        
    def gettokens(self):
        return self.tokens,self.inverse_tokens,self.frequent_w
        
    
#### Visualization    
def plot_loss(loss,filename):
    plt.figure(figsize=(18, 18))
    plt.plot(loss)
    plt.savefig(filename)
    
def plot_embedding(tokens,embedding,wordlist,wordsize,filename,title):
    wordlist_indx = [tokens[w] for w in wordlist]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(embedding[wordlist_indx, :])
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(wordlist):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y , color=['yellow'], s=wordsize[i])
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.title(title)
    plt.savefig(filename)

#### Nearest words    
def nearest_word(target,topn,embedding,tokens,inverse_tokens,country):
    dist = np.matmul(embedding[tokens[target]],embedding.T)
    ranking = -dist.argsort()[:topn]
    print("Nearest "+str(topn)+" words to "+target+" for "+country+" :", end='')
    for r in ranking:
        print(inverse_tokens[r]+' ', end='')
    print('')
    
def analogy(target_vec,topn,embedding,tokens,inverse_tokens):
    target = embedding[tokens[target_vec[2]]]-embedding[tokens[target_vec[0]]]+embedding[tokens[target_vec[1]]]
    dist = np.matmul(target,embedding.T)
    ranking = -dist.argsort()[:topn]
    print(target_vec[0]+" - "+target_vec[1]+" = "+target_vec[2]+" - :", end='')
    for r in ranking:
        print(inverse_tokens[r]+' ', end='')
    print('')
    
    
def main():
    country_collect = collect_sent_country()
    data_fr,data_us, data_ch, data_ru = sent_split_words(['FRA','USA','CHN','RUS'],country_collect)
    FR = Skipgram(data_fr)
    tokens,inverse_tokens, w_frequency = FR.gettokens()
    embedding, loss = FR.train()
    word_to_plot, word_frequency = frequent_non_stop_word(count= w_frequency,number=150)
    plot_embedding(tokens=tokens,embedding=embedding,wordlist=word_to_plot,wordsize=word_frequency,filename='embed_fr.png',title='France-150 most frequent non stop words')
    #nearest_word('negotiations',15,embedding,tokens,inverse_tokens,"France")
    #analogy(['peace','war','cooperation'],10,embedding,tokens,inverse_tokens)
    #US = Skipgram(data_us)
    #tokens,inverse_tokens, w_frequency = US.gettokens()
    #embedding, loss = US.train()
    #word_to_plot, word_frequency = frequent_non_stop_word(count= w_frequency,number=150)
    #plot_embedding(tokens=tokens,embedding=embedding,wordlist=word_to_plot,wordsize=word_frequency,filename='embed_us.png',title='USA-150 most frequent non stop words')
    #nearest_word('negotiations',15,embedding,tokens,inverse_tokens,"USA")
    #analogy(['peace','war','cooperation'],10,embedding,tokens,inverse_tokens)
    #CH = Skipgram(data_ch)
    #tokens,inverse_tokens, w_frequency = CH.gettokens()
    #embedding, loss= CH.train()
    #word_to_plot, word_frequency = frequent_non_stop_word(count= w_frequency,number=150)
    #plot_embedding(tokens=tokens,embedding=embedding,wordlist=word_to_plot,wordsize=word_frequency,filename='embed_ch.png',title='China-150 most frequent non stop words')
    #nearest_word('negotiations',15,embedding,tokens,inverse_tokens,"China")
    #analogy(['peace','war','cooperation'],10,embedding,tokens,inverse_tokens)
    #RU = Skipgram(data_ru)
    #tokens,inverse_tokens, w_frequency = RU.gettokens()
    #embedding, loss = RU.train()
    #word_to_plot, word_frequency = frequent_non_stop_word(count= w_frequency,number=150)
    #plot_embedding(tokens=tokens,embedding=embedding,wordlist=word_to_plot,wordsize=word_frequency,filename='embed_ru.png',title='Russia-150 most frequent non stop words')
    #nearest_word('negotiations',15,embedding,tokens,inverse_tokens,"Russia")
    #analogy(['peace','war','cooperation'],10,embedding,tokens,inverse_tokens)
if __name__ == "__main__":
    main()
    


            
    