# %% [code]
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os
import time
import spacy
from match import Match
import gensim


def read_data(path):
    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
    sha2file = {}
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if re.search('.json$', filename):
                sha, _ = filename.split('.')
                sha2file[sha] = os.path.join(dirname, filename)
    print('Total of json files in CORD-l9:', len(sha2file))
    return sha2file


def preprocess_corpus(df, model_name="en_core_sci_sm"):
    nlp = spacy.load(model_name)
    preprocessed_text = []
    author2doc = {}
    docind = []
    start = time.time()
    # extract all entities in corpus:
    entvocab = set([entity.text for index, doc in df.iterrows() for entity in nlp(doc['abstract']).ents
                    if len(entity.text.split(' ')) > 1])
    matcher = Match()
    matcher.matchinit_from_list(entvocab)

    i = 0
    for index, row in df.iterrows():
        doc_tokenized = " ".join([tok.lemma_ for tok in nlp(row['abstract']) if (not tok.is_stop and
                                                                               not tok.is_punct and
                                                                               not tok.like_num)])
        # find multiwords expresions
        doc_tokenized = matcher.match(doc_tokenized)
        preprocessed_text.append(doc_tokenized.split(' '))

        docind.append(index)
        for author in row['authors']:
            if author not in author2doc:
                author2doc[author] = []
            author2doc[author].append(i)
        i += 1
    end = time.time()
    print('Preprocessing time: ', round((end - start) / 60, 2))
    return preprocessed_text, author2doc, docind


def display_topics(model, no_top_words):
    feature_names = model.id2word
    topic_dict = {}
    for topic_idx, topic in enumerate(model.get_topics()):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


def display_author_topics(model, no_top_authors):
    author_vecs = [model.get_author_topics(author, minimum_probability=0.0) for author in model.id2author.values()]
    topic_dict = {}
    for author_idx, topics in enumerate(author_vecs):
        for topic in topics:
            author_name = model.id2author[author_idx]
            if "Topic %d authors" % (topic[0]) not in topic_dict:
                topic_dict["Topic %d authors" % (topic[0])] = []
                topic_dict["Topic %d weights" % (topic[0])] = []
            topic_dict["Topic %d authors" % (topic[0])].append(author_name)
            topic_dict["Topic %d weights" % (topic[0])].append(topic[1])
    # convert to numpy
    for topic in topic_dict:
        topic_dict[topic] = np.array(topic_dict[topic])
    # sort each topic
    for topic in range(0, model.num_topics):
        sorted_idx = np.argsort(-topic_dict['Topic %d weights' % topic])
        topic_dict["Topic %d authors" % (topic)] = topic_dict["Topic %d authors" % (topic)][sorted_idx]
        topic_dict["Topic %d weights" % (topic)] = topic_dict["Topic %d weights" % (topic)][sorted_idx]
    df = pd.DataFrame(topic_dict)
    return df.head(no_top_authors)


def raw_vocab(df, model_name="en_core_sci_sm", remove=False):
    nlp = spacy.load(model_name)
    vocab = set()
    for index, row in df.iterrows():
        if remove:
            vocab.update(set([tok.lemma_ for tok in nlp(row['abstract']) if (not tok.is_stop and
                                                                             not tok.is_punct and
                                                                             not tok.like_num)]))
        else:
            vocab.update(set([tok.lemma_ for tok in nlp(row['abstract'])]))
    return vocab


def filter_covid(df_mdata):
    synonyms = ['coronavirus 2019', 'coronavirus disease 19', 'cov2', 'cov-2', 'covid', 'ncov 2019', '2019ncov',
                '2019-ncov', '2019 ncov', 'novel coronavirus', 'sarscov2', 'sars-cov-2', 'sars cov 2',
                'severe acute respiratory syndrome coronavirus 2', 'wuhan coronavirus', 'wuhan pneumonia',
                'wuhan virus']

    # Create a filter with 'False' values
    index_list = list(df_mdata.index.values)
    filter = pd.Series([False] * len(index_list))
    filter.index = index_list

    # Update the filter for each synonym
    for s in synonyms:
        # Check if a synonym is in title or abstract
        filter = filter | df_mdata.title.str.lower().str.contains(s) | df_mdata.abstract.str.lower().str.contains(s)
    df_mdata = df_mdata[filter]
    return df_mdata


def save_doctopics_numpy(fname, model, corpus):
    num_topics = model.num_topics
    num_docs = len(corpus)
    all_topics = model.get_document_topics(corpus, minimum_probability=0.0)
    all_topics = gensim.matutils.corpus2dense(all_topics, num_terms=num_topics, num_docs=num_docs)
    all_topics = all_topics.T
    np.save(fname, all_topics)
    
    
if __name__ == '__main__':
    vocab = ['virus transmission', 'incubation']
    sentence = "How long is the incubation after the virus transmission"
    matcher = Match()
    matcher.matchinit_from_list(vocab)
    matched_sentence = matcher.match(sentence)
    print(matched_sentence)