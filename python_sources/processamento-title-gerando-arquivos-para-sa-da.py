import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def get_tokens_ids(sentence):
  tokens = word_tokenize(sentence)
  tokens_ids = list(range(1,len(tokens)+1))

  for idx, token in enumerate(tokens):
    if token in stop_words:
      tokens_ids.remove(idx+1)

  return tokens_ids


def row_to_sentences(row):
  sentences = []
  sentences.append(row['title'])
  #sentences.extend(sent_tokenize(row['content']))

  return sentences


def main():
  folderInput = '../input/fakenewsonlytitlelower/'
  folderOutput = './'
  filename = 'titles'
  texts = {}

  with open(folderInput+'titles_1_perc.csv', 'r') as csvfile:
    reader =  csv.DictReader(csvfile)
    for row in reader:
      texts[row['id']] = {'sentences':[], 'tokens':[], 'type':row['type']}
      texts[row['id']]['sentences'] = row_to_sentences(row)
      for sentence in texts[row['id']]['sentences']:
        texts[row['id']]['tokens'].append(get_tokens_ids(sentence))
  
  with open(folderOutput+filename+'_sentences.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["txt_id","sentence_id","type","sentence_txt"])

    for txt_id, text in texts.items():
      for sentence_id, sentence_txt in enumerate(text['sentences']):
        txtid = int(float(txt_id))
        writer.writerow([txtid,sentence_id,text['type'],sentence_txt])

  with open(folderOutput+'all_pos_tokens_'+filename+'.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)

    for txt_id, text in texts.items():
      for sentence_id, tokens in enumerate(text['tokens']):
        for token in tokens:
          txtid = int(float(txt_id))
          string = str(txtid) + "_" + str(sentence_id) + "_" + str(token) 
          writer.writerow([string])

if __name__ == "__main__":
  main()