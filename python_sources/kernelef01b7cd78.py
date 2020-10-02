

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import nltk 
from nltk.tokenize import word_tokenize 
from collections import defaultdict

import os

def compute_divergence(common_words,artist1,artist2):
   
    arr1,arr2 = [],[]
    for word in common_words:
        arr1.append(artist1[word])
        arr2.append(artist2[word])
    
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    sum1 = np.sum(arr1)
    sum2 = np.sum(arr2)
    
    arr1 = arr1/sum1
    arr2 = arr2/sum2

    kl1 = (arr1*np.log(arr1/arr2)).sum()
    kl2 = (arr2*np.log(arr2/arr1)).sum()
    a=(kl1+kl2)/2
    s= np.exp(-a/0.5)
    return(s)


artist = pd.read_csv('../input/artist-data-pop/artist_data_pop.csv',encoding='ISO-8859-1')

with open('../input/contentfree/content-free.txt', 'r') as file:
    CFW = file.read().replace('\n', '')
content_free_tokens = word_tokenize(CFW)

no_of_artists = artist.shape[0]
temporal_distance = np.zeros((no_of_artists, no_of_artists))
divergence_matrix = np.zeros((no_of_artists, no_of_artists))

artist_corpus = {}

artist_weighted_dict = dict(zip(artist.artist, artist['unlist(Weighted_year)']))

for(idx,row) in artist.iterrows():
    artist_vocab = {}
    for word in content_free_tokens:
        if(row.lyrics_corp.count(word) > 0):
            artist_vocab[word] = row.lyrics_corp.count(word)
    artist_corpus[row.artist] = artist_vocab

artist_name = list(artist_corpus)
artist_corr = defaultdict(dict)
temporal_corr = defaultdict(dict)

file_artist = "../input/artist_file_50.csv"
csv = open(file_artist,"w")

columnTitleRow = "Artist1, Artist2, Similarity, TemporalDistance\n"
csv.write(columnTitleRow)

cnt=0
for idx,value in enumerate(artist_name):
    artist_1 = artist_name[idx]
    artist_1_vocab = artist_corpus[artist_1]
    Artist1 = artist_1
    
    for val in np.arange(idx,len(artist_name)):
        artist_2 = artist_name[val]
        artist_2_vocab = artist_corpus[artist_2]
        Artist2 = artist_2
        common_words = set(list(artist_1_vocab)).intersection(list(artist_2_vocab)) 
        artist_corr[artist_1][artist_2] = compute_divergence(common_words,artist_1_vocab,artist_2_vocab)
        temporal_corr[artist_1][artist_2] = abs(artist_weighted_dict[artist_1] - artist_weighted_dict[artist_2])
        Similarity = artist_corr[artist_1][artist_2]
        TemporalDistance = temporal_corr[artist_1][artist_2]
       
        row = Artist1 + "," + Artist2 + "," + str(Similarity) + "," + str(TemporalDistance) + "\n"
        csv.write(row)
        cnt=cnt+1

print(cnt)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))




artistss = pd.read_csv('../input/artist_file_50.csv',delimiter=',',encoding='utf-8-sig')
type(artistss)
create_download_link(artistss)
list(artistss.columns.values)


def remove_bom(filename):
    fp = open(filename, 'rbU')
    if fp.read(2) != b'\xfe\xff':
        fp.seek(0, 0)
    return fp
abc = pd.read_table(remove_bom('../input/artist_file_50.csv'))


abc.to_csv('csv_to_submit.csv', index = False)
