# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import math

from sklearn.preprocessing import MultiLabelBinarizer

import pickle
from nltk.tokenize import word_tokenize

# Any results you write to the current directory are saved as output.
class FootballCorpus:
    def split_train_test_sets(self, rate):
        rand_indexes = np.random.rand(len(self.x)) < rate                
        
        self.x_train = self.x[rand_indexes]
        self.y_train = self.y[rand_indexes]
        
        self.x_test = self.x[~rand_indexes]
        self.y_test = self.y[~rand_indexes]
        
    def save_to_pkl(self, dir):
        with open(os.path.join(dir, 'football_events_x_train.pkl'), 'wb') as handle:
            pickle.dump(self.x_train, handle)
            
        with open(os.path.join(dir, 'football_events_y_train.pkl'), 'wb') as handle:
            pickle.dump(self.y_train, handle)
            
        with open(os.path.join(dir, 'football_events_x_test.pkl'), 'wb') as handle:
            pickle.dump(self.x_test, handle)
            
        with open(os.path.join(dir, 'football_events_y_test.pkl'), 'wb') as handle:
            pickle.dump(self.y_test, handle)
    


''' This class is responsible for preparing the corpus for event classifiers'''
class FootballEventsCorpus(FootballCorpus):
    
    def __init__(self, football_events_corpus):
        self.football_events_corpus = football_events_corpus
        self.football_events_3mins_prior_goal = None
        self.last_index = []
        self.last_index_minute = []
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
    def clean_older_events(self, event):
        i = 0
        if len(self.last_index_minute) == 0:
            return
        while i < len(self.last_index_minute) and \
        self.last_index_minute[i] < event['time']-3:
            i += 1
        
        if i >= len(self.last_index_minute):
            self.last_index_minute = []
            self.last_index = []
        elif i > 0:
            self.last_index_minute = self.last_index_minute[i:]
            self.last_index = self.last_index[i:]
        
    def stack_new_event(self, event, index):
        self.clean_older_events(event)
        self.last_index.append(index)
        self.last_index_minute.append(event['time'])
        
    def gen_x_and_y(self):
        self.x = self.football_events_3mins_prior_goal['text']
        self.y = []
        for index, event in self.football_events_3mins_prior_goal.iterrows():
            if not math.isnan(event['event_type2']):
                self.y.append([int(event['event_type']), int(event['event_type2'])])
            else:
                self.y.append([int(event['event_type'])]) 
        
        self.y = MultiLabelBinarizer().fit_transform(self.y)
        
    def filter_3mins_prior_goal(self):
        for index, event in self.football_events_corpus.iterrows():
            if event['is_goal'] == 1:
                self.stack_new_event(event, index)
                if self.football_events_3mins_prior_goal is None:
                    self.football_events_3mins_prior_goal = self.football_events_corpus.iloc[self.last_index[0]:index+1]
                else:
                    self.football_events_3mins_prior_goal = self.football_events_3mins_prior_goal.append(self.football_events_corpus.iloc[self.last_index[0]:index+1])
            else:
                self.stack_new_event(event, index)
        
        self.gen_x_and_y()
                        

''' This class is responsible for preparing the corpus for NER (player names)'''
class FootballNERCorpus(FootballCorpus):
    
    def __init__(self, football_events_corpus):
        self.football_events_corpus = football_events_corpus
        self.x_list = []
        self.y_list = []
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
    
    def annotate_named_entity(self, text, entity, labels):
        begin_search = 0
        
        ind_ne = text.lower().find(entity.lower(), begin_search)     

        while ind_ne >= 0:
            word_ind_ne = len(word_tokenize(text[:ind_ne]))
            num_words = len(word_tokenize(entity))
            
            for i in range(num_words):
                labels[word_ind_ne+i] = 1
            
            begin_search = ind_ne + len(entity)
            ind_ne = text.lower().find(entity.lower(), begin_search)

            
        return labels
    
    def annotate_named_entities(self, event):
        player_names = [event[field] for field in ['player', 'player2', 'player_in', 'player_out'] \
                        if isinstance(event[field], str)]        
        
        
        text = event['text']
            
        self.x_list.append(word_tokenize(text))
        
        labels = [0] * len(word_tokenize(text))
        
        for player in player_names:
            labels = self.annotate_named_entity(text, player, labels)
            
            
        self.y_list.append(labels)
        
        
    def ne_annotate_corpus(self):
        for index, event in self.football_events_corpus.iterrows():
            self.annotate_named_entities(event)
            
        self.x = np.array(self.x_list)
        self.y = np.array(self.y_list)
                
    def save_to_pkl(self, dir):
        with open(os.path.join(dir, 'football_ner_x_train.pkl'), 'wb') as handle:
            pickle.dump(self.x_train, handle)
            
        with open(os.path.join(dir, 'football_ner_y_train.pkl'), 'wb') as handle:
            pickle.dump(self.y_train, handle)
            
        with open(os.path.join(dir, 'football_ner_x_test.pkl'), 'wb') as handle:
            pickle.dump(self.x_test, handle)
            
        with open(os.path.join(dir, 'football_ner_y_test.pkl'), 'wb') as handle:
            pickle.dump(self.y_test, handle)
            

path = '../input/events.csv'

football_events_corpus = pd.read_csv(open(path), engine='c')

football_events_corpus = football_events_corpus.iloc[:2000]

rate = 0.8

event_corpus = FootballEventsCorpus(football_events_corpus)



event_corpus.filter_3mins_prior_goal()

event_corpus.split_train_test_sets(rate)

event_corpus.save_to_pkl('.')    

ner_corpus = FootballNERCorpus(football_events_corpus)

ner_corpus.ne_annotate_corpus()

ner_corpus.split_train_test_sets(rate)

ner_corpus.save_to_pkl('.')
