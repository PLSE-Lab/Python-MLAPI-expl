# Last updated: 2019-05-20

import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
style.use('fivethirtyeight')
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)

import plotly.offline as pyo
import plotly.graph_objs as go
pyo.init_notebook_mode()

import nltk

from textblob import TextBlob

import gensim
from gensim.models.word2vec import Word2Vec
from gensim import corpora


class AspectOpinionExtractor:
    
    def __init__(self,
                 df = None,
                 id_column = None,
                 review_column = None):
        self.df = df
        self.id_column = id_column
        self.review_column = review_column
    
    @staticmethod
    def aspect_extractor(sentence: str) -> list:
        '''
        extract aspects from a sentence
        '''
        sentence_blob = TextBlob(sentence)

        #extract noun phrases
        np = sentence_blob.noun_phrases

        #extract nouns
        n1 = [n for n,t in sentence_blob.tags if t in ['NN', 'NNS', 'NNP', 'NNPS']]

        #delete nouns that are actually part of the noun phrases
        n2 = []

        if len(n1) == 1:
            n2 = n1

        if len(n1) > 1:
            for i in range(len(n1)):
                if i == 0 and ' '.join([n1[i].lower(), n1[i+1].lower()]) not in np:
                    n2.append(n1[i])
                if i > 0 and i < len(n1) - 1 and ' '.join([n1[i-1].lower(), n1[i].lower()]) not in np and ' '.join([n1[i].lower(), n1[i+1].lower()]) not in np:
                    n2.append(n1[i])
                if i == len(n1) - 1 and ' '.join([n1[i-1].lower(), n1[i].lower()]) not in np:
                    n2.append(n1[i])

        #merge the results
        candidate_aspects = np + n2

        return candidate_aspects
     

    @staticmethod
    def valid(x: int, sentence_blob) -> bool:
        '''
        check if the index is valid
        '''
        return True if x >= 0 and x < len(sentence_blob.tags) else False 
    

    @staticmethod
    def merge_two_dicts(dict1, dict2) -> dict:
        '''
        {'a': 'apple', 'b': 'boy'}
        {'a': 'air', 'c': 'cat'}
        ==> {'a': 'apple air', 'b': 'boy', 'c': 'cat'}
        '''
        for k in dict2:
            if k not in dict1:
                dict1[k] = dict2[k]
            elif k in dict1:
                dict1[k] = ' '.join([dict1[k], dict2[k]])
        return dict1


    @staticmethod
    def has_aspect(dict_string, aspect) -> bool:
        aspects = list(eval(dict_string).keys())
        return True if aspect.lower() in aspects else False


    @staticmethod
    def has_opinion(dict_string, aspect, opinion) -> bool:
        opinions = eval(dict_string)[aspect].lower().split()
        return True if opinion.lower() in opinions else False


    def is_be(self, x: int, sentence_blob) -> bool:
        '''
        check if the word is one of ['am','is','are','was','were']
        '''
        return True if self.valid(x, sentence_blob) and sentence_blob.words[x].lower() in ['am','is','are','was','were'] else False

    
    def add_word(self,
                 index: int, 
                 word_list: list, 
                 sentence_blob):
        '''
        Add the adverbs and ajectives
        '''
        if self.valid(index, sentence_blob):
            if sentence_blob.tags[index][1] in ['RB','RBR','RBS','JJ','JJR','JJS']:
                word_list.append(sentence_blob.tags[index][0])


    def extract_attributes_pref(self,
                                aspect: str, 
                                first_word_index: int, 
                                sentence_blob) -> str:
        '''
        first_word_index: the index of the first word in the aspect in the sentence

        extract the attributes that are right before the aspect in the sentence;
        we look at no more than 2 words before;
        e.g. "very nice coffee" --> {'coffee': 'very nice'}
             "weather was perfect, the coffee ..." --> {'coffee': ''}
             "friendly staff" --> {'staff': 'friendly'}
        '''
        pref_words = []
        
        self.add_word(first_word_index - 1, pref_words, sentence_blob)
        
        #only when there's a adj/adv one index before the aspect will we look at the work 2 indexes before
        if len(pref_words) > 0:
            self.add_word(first_word_index - 2, pref_words, sentence_blob)
        
        return ' '.join(pref_words)


    def extract_attributes_suff(self,
                                aspect: str, 
                                ca_last_index: int, 
                                sentence_blob) -> str:
        '''
        last_word_index: the index of the last word in the aspect in the sentence

        extract the attributes that are right after the aspect in the sentence;
        we look at no more than 2 words after;
        e.g. "coffee was very nice" --> {'coffee': 'very nice'}
             "We had coffee. The delicious desert...." --> {'coffee': ''}
             "The cafe is beautiful. The coffee.." --> {'cafe': 'beautiful'}
             "the bed was not comfortable" --> {'bed': 'notcomfortable'}
        ''' 
        suff_words = []
        
        #the adj/adv must come after the "be noun"
        if self.is_be(ca_last_index + 1, sentence_blob):
            self.add_word(ca_last_index + 2, suff_words, sentence_blob)
        
        #only when there's a qualified adj/adv 2 index after the aspect will we look at the work 3 indexes after
        if len(suff_words) > 0:
            self.add_word(ca_last_index + 3, suff_words, sentence_blob)
        
        attr = ' '.join(suff_words)
        if len(suff_words) == 2 and suff_words[0] in ['not', "n't"]:
            attr = ''.join(suff_words)
        
        return attr


    def opinion_extractor(self,
                          candidate_aspects: list,
                          sentence_blob) -> dict:
        '''
        extract the aspects and opinions associated with them in a sentence; returns a dictionary
        '''
        candidate_aspects_dict = {}
        aspect_opinion_dict = {}
        
        if len(candidate_aspects) > 0:
            for a in candidate_aspects:
                try:
                    candidate_aspects_dict[a] = {}
                    first_word_index = sentence_blob.words.index(a.split()[0])
                    last_word_index = sentence_blob.words.index(a.split()[-1])
                    aspect_opinion_dict[a] = self.extract_attributes_pref(a, first_word_index, sentence_blob)
                    aspect_opinion_dict[a] = ' '.join([aspect_opinion_dict[a], self.extract_attributes_suff(a, last_word_index, sentence_blob)])
                except:
                    pass

        return aspect_opinion_dict


    def aspect_opinion_for_one_comment(self, sentence: str) -> dict:
        '''
        end to end: extract aspects and opinions for one sentence
        '''
        sentence = str(sentence).lower()
        sentence_blob = TextBlob(sentence)
        candidate_aspects = self.aspect_extractor(sentence)
        aspect_opinion_dict = self.opinion_extractor(candidate_aspects, sentence_blob)

        return aspect_opinion_dict


    def aspect_opinon_for_all_comments(self, report_interval = 500):
        '''
        extract aspects and opinions for all the comments in a pandas dataframe;
        returns a pandas dataframe with id, reviews and the string version of the aspect_opinion_dict
        '''
        df = self.df.copy()
        id_column = self.id_column
        review_column = self.review_column

        begin = datetime.datetime.now()
        print("Aspect Opinion Extractor: {}".format(begin))
        aspect_opinion_dict_all = {}
        df_small = df[[id_column, review_column]].copy()
        df_small['aspects_opinions'] = ""
        for i in range(len(df)):
            d = self.aspect_opinion_for_one_comment(df_small.loc[i, review_column])
            aspect_opinion_dict_all = self.merge_two_dicts(aspect_opinion_dict_all, d)
            df_small.loc[i, 'aspects_opinions'] = str(d)
            if i%report_interval == 0 and i > 0:
                time_pass = (datetime.datetime.now() - begin).seconds / 60
                print("{:.2} min later: finished {:.2%}".format(time_pass, i/len(df)))
        
        aspects_opinions_df = pd.DataFrame({'aspects' : list(aspect_opinion_dict_all.keys()), 'opinions' : list(aspect_opinion_dict_all.values())})
        aspects_opinions_df['opinions'] = aspects_opinions_df.opinions.str.strip()
        aspects_opinions_df['pop'] =  aspects_opinions_df['opinions'].apply(len)
        aspects_opinions_df = aspects_opinions_df[aspects_opinions_df['opinions'] != ""].sort_values(by = ['pop'], ascending = False).drop('pop', 1)

        self.df_with_ao = df_small
        self.aspects_opinions_df = aspects_opinions_df
    

    def single_aspect_view(self, aspect: str, num_top_words = 10, change_figsize = True, xticks_rotation = 45):
        '''
        plot popular opinions around an aspect;
        For example, we are interested in what people say about "staff",
        We pick the top n popular words people used to describe staff and calcualte among those who have expressed opinion towards "staff", how many percentage of them used certain words;
        the output will be a bar chart
        '''
        style.use('fivethirtyeight')

        if change_figsize:
            plt.rcParams['figure.figsize'] = (num_top_words, 5)
        plt.xticks(rotation = xticks_rotation)
        plt.rcParams['xtick.labelsize'] = 20.0

        ao_df = self.aspects_opinions_df.copy()
        df = self.df_with_ao.copy()
        opinion_top_words = pd.Series(ao_df.set_index("aspects").loc[aspect,'opinions'].split()).value_counts()[:num_top_words].index.tolist()
        comments_contain_aspect = df[df['aspects_opinions'].apply(self.has_aspect, aspect = aspect)]
        perc = []
        for o in opinion_top_words:
            comments_contain_aspect_and_opinion = comments_contain_aspect[comments_contain_aspect['aspects_opinions'].apply(self.has_opinion, aspect = aspect, opinion = o)]
            perc.append(round(len(comments_contain_aspect_and_opinion) / len(comments_contain_aspect), 2) * 100)
        aspect_plot = pd.DataFrame({'opinion': opinion_top_words, "% of people use this word": perc})
        ax = sns.barplot(x = 'opinion', y = "% of people use this word", data = aspect_plot)
        ax.set_title("What do people say about {} ? (total {} mentions)".format(aspect, len(comments_contain_aspect)))
        ax.set(xlabel = "")


    def popular_aspects_view(self):
        '''
        quick plot:
        single_aspect_view for the top 9 aspects
        '''
        style.use('fivethirtyeight')
        plt.rcParams['figure.figsize'] = (30, 30)
        plt.rcParams['xtick.labelsize'] = 20.0
        
        self.top_aspects = self.aspects_opinions_df.aspects[:9].values

        for i in range(1, 10):
            plt.subplot(3, 3, i)
            self.single_aspect_view(self.top_aspects[i-1], num_top_words = 6, change_figsize = False, xticks_rotation = 30)

        plt.show()


class SentimentScore:

    def __init__(self,
                 df = None,
                 id_column = None,
                 review_column = None):
        self.df = df
        self.id_column = id_column
        self.review_column = review_column

    
    @staticmethod
    def sentiment_for_one_comment(ctm: str) -> float:
        '''
        calcualte sentiment score for one comment ==> the mean of (polarity * subjectivity) for each sentence (if the sentence has a non-zero polarity)
        '''
        try:
            ctm_blob = TextBlob(ctm)
            sentiment_scores = [s.sentiment.polarity * s.sentiment.subjectivity for s in ctm_blob.sentences if s.sentiment.polarity != 0]
            result = sum(sentiment_scores)/len(sentiment_scores) if len(sentiment_scores) > 0 else 0
        except:
            result = 0
        return result

    
    @staticmethod
    def return_negative_comments(ctm: str) -> str:
        '''
        return all the sentences in a comment that have negative sentiment scores
        '''
        try:
            ctm_blob = TextBlob(ctm)
            negative_sentences = [str(s) for s in ctm_blob.sentences if s.sentiment.polarity < 0]
            if len(negative_sentences) > 0:
                return " ".join(negative_sentences)
        except:
            pass
 
    
    def overall_sentiment(self):
        '''
        Plot the histogram of the sentiment scores for all the comments;
        '''
        df = self.df.copy()
        review_column = self.review_column

        sentiment_scores = df[self.review_column].apply(self.sentiment_for_one_comment)
        self.sentiment_scores_all = sentiment_scores
        print("Average sentiment score: {}".format(sentiment_scores.mean()))
        print("{}% of the comments are positive, {}% of the comments are negative".format(round(100 * sum(sentiment_scores>=0)/len(sentiment_scores), 2), 
                                                                                          round((100 * sum(sentiment_scores<0)/len(sentiment_scores)), 2)))
        plt.figure(figsize = (5, 5))
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        ax = plt.hist(sentiment_scores)
        ax = plt.title('sentiment scores of all comments')


    def sentiment_for_one_aspect(self, aspect, aspects_opinions_df) -> float:
        '''
        return the average sentiment score for an aspect
        '''
        opinions = aspects_opinions_df.where(aspects_opinions_df.aspects == aspect).dropna()['opinions'].values[0].split()
        opinions_polarities = [TextBlob(i).sentences[0].sentiment.polarity for i in opinions if TextBlob(i).sentences[0].sentiment.polarity != 0]
        return sum(opinions_polarities)/len(opinions_polarities)
    

    def aspects_radar_plot(self, aspects, aspects_opinions_df):
        '''
        plot the sentiment score radar chart for designated aspects
        '''
        sentiment_scores = [self.sentiment_for_one_aspect(i, aspects_opinions_df) for i in aspects]
        data = [go.Scatterpolar(
            r = sentiment_scores,
            theta = aspects,
            fill = 'toself'
        )]
        layout = go.Layout(
          polar = dict(
            radialaxis = dict(
              visible = True,
              range = [min(sentiment_scores) - 0.2, max(sentiment_scores)]
            )
          ),
          showlegend = False
        )
        fig = go.Figure(data=data, layout=layout)
        pyo.iplot(fig, filename = "radar/basic")

    
    def return_all_negative_sentences(self) -> list: 
        df = self.df.copy()
        review_column = self.review_column

        negatives = list(df[review_column].apply(self.return_negative_comments).dropna())
        return negatives


class TopicModeling:
    '''
    Topic Modeling on a string (the whole document)
    '''
    def __init__(self,
                 text: str,
                 num_topics: int = 3
                 ):
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sentences = sent_tokenizer.tokenize(text) # the document
        self.num_topics = num_topics # number of topics


    def get_sentences_subset(self):
        '''
        preprocess the document; the output is a list of list;
        The function decomposes the whole document into each sentence and decompose each sentence into each word;
        For each sentence, we only keep the nouns, ajectives, adverbs and verbs
        '''
        n = len(self.sentences)
        shortened_sentences = np.arange(n).tolist()
        key_features = ['NN','NNS','JJ','VB','VBD','VBG','VBN','VBP','VBZ','RB']
        for i in range(n):
            summary = []
            words = nltk.word_tokenize(self.sentences[i])
            word_tag = nltk.pos_tag(words)
            for token in word_tag:
                if (token[1] in key_features):
                    summary.append(token[0])
            shortened_sentences[i]=summary
        self.sentences_clean = shortened_sentences

    
    def get_dictionary(self):
        '''
        create gensim dictionary
        '''
        w2v_model = gensim.models.Word2Vec(self.sentences_clean, min_count = 1)
        dictionary = corpora.Dictionary(self.sentences_clean)
        self.dictionary = dictionary

    def LDA(self):
        '''
        run lda model
        '''
        term_matrix = [self.dictionary.doc2bow(sentence) for sentence in self.sentences_clean]
        lda = gensim.models.ldamodel.LdaModel
        ldamodel = lda(term_matrix, num_topics=self.num_topics, id2word =self.dictionary, passes=50)
        return(ldamodel.print_topics())

    def show_topics(self):
        '''
        getting things together
        '''
        self.get_sentences_subset()
        self.get_dictionary()
        return (self.LDA())


class NegativeCommentsExplore:
    def __init__(self,
                 df,
                 id_column,
                 review_column,
                 ):
        self.df = df
        self.id_column = id_column
        self.review_column = review_column
        self.a = AspectOpinionExtractor(df, id_column, review_column)
        self.s = SentimentScore(df, id_column, review_column)

    def return_all_negative_sentences(self) -> list:
        '''
        return all the negative sentences in a list
        '''
        df = self.df.copy()
        review_column = self.review_column

        negatives = list(df[review_column].apply(self.s.return_negative_comments).dropna())
        self.all_negative_comments = negatives

    def negative_comments_by_aspects(self) -> dict:
        '''
        return a dict ==> keys are the aspects; values are the sentences associated with the aspects
        '''
        result = {}
        for n in self.all_negative_comments:
            aspects = self.a.aspect_extractor(n)
            for a in aspects:
                if a in result:
                    result[a].append(n)
                else:
                    result[a] = [n]
        self.negative_comments_by_aspects_dict = result

    def negative_comments_view(self):
        '''
        barplot on the numbers of negative sentences of each aspect
        '''
        self.return_all_negative_sentences()
        self.negative_comments_by_aspects()
        count_df = self.negative_comments_by_aspects_dict.copy()
        for a in count_df:
            count_df[a] = len(count_df[a])
        count_df = pd.DataFrame({'aspect': list(count_df.keys()), 'numbers_of_negative_sentences': list(count_df.values())}).sort_values("numbers_of_negative_sentences", ascending = False)
        plt.rc('xtick', labelsize=12) 
        plt.rc('ytick', labelsize=12)
        plt.figure(figsize = (15, 5))
        plt.xticks(rotation = 90)
        ax = sns.barplot(x = 'aspect', y = "numbers_of_negative_sentences", data = count_df[:20])
        ax.set_title("Negative comments...")
        plt.show()

    def investigate(self, aspect: str, num_topics: int = 3, topic_modeling = False):
        '''
        investigate negative comments around an aspects;
        print the result of topic modeling on the negative sentences that mention the aspect;
        return the negative sentences that mention the aspect
        '''
        if topic_modeling:
            aspect_corpus = " ".join(self.negative_comments_by_aspects_dict[aspect])
            self.t = TopicModeling(aspect_corpus, num_topics)
            print(self.t.show_topics())
        return(self.negative_comments_by_aspects_dict[aspect])


class ReviewMiner:

    def __init__(self,
                 df,
                 id_column,
                 review_column,
                 time_column):
        '''
        Getting everthing together --- running all the classes above, ploting all necessary charts and stores important tables


        '''

        self.time_column = time_column

        self.AspectOpinionExtractor = AspectOpinionExtractor(df, id_column, review_column)
        self.SentimentScore = SentimentScore(df, id_column, review_column)
        self.NegativeCommentsExplore = NegativeCommentsExplore(df, id_column, review_column)
        
        print("===============================================================")
        self.AspectOpinionExtractor.aspect_opinon_for_all_comments()
        print("===============================================================")
        print("Finished extracting the aspects. Original dataframe with the aspects and opinions in each comment: df_with_ao; The dataframe with all the aspects and relative opinions: aspects_opinions_df")
        self.AspectOpinionExtractor.popular_aspects_view()
        self.top_aspects = self.AspectOpinionExtractor.top_aspects
        
        print("\n===============================================================")
        print("Sentiment Analysis: {}".format(datetime.datetime.now()))
        print("===============================================================")
        self.SentimentScore.overall_sentiment()

        print("\n===============================================================")
        print("Negative Comments Explore: {}".format(datetime.datetime.now()))
        print("===============================================================")
        print("all the negative comments are in all_negative_comments; negative comments by aspects are in negative_comments_by_aspects_dict")
        self.NegativeCommentsExplore.negative_comments_view()

        ## Save important intermediate outputs to the class itself & Calculate other helpful tables

        #Dataframe with columns: ["aspects", "opinions"]
        self.aspects_opinions_df = self.AspectOpinionExtractor.aspects_opinions_df
        #Dataframe with columns: [id, reviews, "aspects_opinions", time]
        self.df_with_aspects_opinions = self.AspectOpinionExtractor.df_with_ao.join(pd.DataFrame(df[time_column]))
        #all negative comments
        self.all_negative_comments = self.NegativeCommentsExplore.all_negative_comments
        
        #Dataframe wtih columns: [id, time, aspect, sentence, sentiment_score]
        df_id = []
        df_aspect = []
        df_sentence = []
        df_time = []
        for i in self.df_with_aspects_opinions.index:
            aspects_opinions_dict = eval(self.df_with_aspects_opinions.loc[i, 'aspects_opinions'])
            for a in aspects_opinions_dict:
                df_id.append(self.df_with_aspects_opinions.loc[i, id_column])
                df_aspect.append(a)
                df_sentence.append(self.extract_setence_of_aspect(a, self.df_with_aspects_opinions.loc[i, review_column]))
                df_time.append(self.df_with_aspects_opinions.loc[i, time_column])
        
        #Dataframe with columns [id, time, aspect, sentence, sentiment_score, positive, negative] (each row is one aspect in one sentence)
        self.aspect_sentence_df = pd.DataFrame({id_column: df_id, time_column: df_time, 'aspect': df_aspect, 'sentence': df_sentence})
        self.aspect_sentence_df['sentiment_score'] = self.aspect_sentence_df.sentence.apply(self.SentimentScore.sentiment_for_one_comment).apply(round, ndigits = 2)
        self.aspect_sentence_df['positive'] = self.aspect_sentence_df.sentiment_score.apply(lambda x: 1 if x>=0 else 0)
        self.aspect_sentence_df['negative'] = self.aspect_sentence_df.sentiment_score.apply(lambda x: 1 if x<0 else 0)

        #Dataframe with columns: [time, sentiment_score, positive, negative] (each row is one comment)
        self.sentiment_time_df = pd.DataFrame(self.SentimentScore.sentiment_scores_all).join(df[time_column]).rename({review_column: 'sentiment_score'}, axis = 1)
        self.sentiment_time_df['positive'] = self.sentiment_time_df.sentiment_score.apply(lambda x: 1 if x>=0 else 0)
        self.sentiment_time_df['negative'] = self.sentiment_time_df.sentiment_score.apply(lambda x: 1 if x<0 else 0)
        
        print("\n===============================================================")
        print("Time series: {}".format(datetime.datetime.now()))
        print("===============================================================")

        #sentiment score time series analysis --- including the sentiment for comments and for top aspects

        plt.rc('xtick', labelsize=12) 
        plt.rc('ytick', labelsize=12)
        plt.rcParams['figure.figsize'] = (20, 10)
        plt.rcParams['legend.fontsize'] = 18.0
        plt.rcParams['axes.titlesize'] = 18.0

        self.time_series_plot_df_1 = self.sentiment_time_df.drop("sentiment_score", axis = 1).groupby(self.time_column).sum()
        ax = self.time_series_plot_df_1.plot(kind = "bar", stacked = True)
        ax.set_title("Numbers of positive and negative comments along with time")
        
        self.time_series_plot_df_2 = self.aspect_sentence_df[self.aspect_sentence_df.aspect.isin(self.top_aspects[:6])].groupby([self.time_column,"aspect"]).sum().unstack()['positive']
        ax = self.time_series_plot_df_2.plot()
        ax.set_title("Numbers of positive comments for top aspects")

        self.time_series_plot_df_3 = self.aspect_sentence_df[self.aspect_sentence_df.aspect.isin(self.top_aspects[:6])].groupby([self.time_column,"aspect"]).sum().unstack()['negative']
        ax = self.time_series_plot_df_3.plot()
        ax.set_title("Numbers of negative comments for top aspects")


    @staticmethod
    def extract_setence_of_aspect(aspect: str, full_comment: str) -> str:
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        try:
            sentences = sent_tokenizer.tokenize(full_comment)
            result = ""
            for s in sentences:
                if aspect in s.lower():
                    result = s
        except:
            result = ""
        return result

    def investigate(self, aspect: str, topic_modeling = False):
        '''
        investigate one aspect, including topic modeling on the negative sentences, popular opinion extraction and sentiment changes with time
        '''
        self.AspectOpinionExtractor.single_aspect_view(aspect)
        negative_comments = self.NegativeCommentsExplore.investigate(aspect, topic_modeling = topic_modeling)
        plt.rcParams['xtick.labelsize'] = 12.0
        ax = pd.DataFrame({'time': self.aspect_sentence_df.groupby([self.time_column,"aspect"]).sum().unstack()['positive'][aspect].index.values, 
                           'positive': self.aspect_sentence_df.groupby([self.time_column,"aspect"]).sum().unstack()['positive'][aspect].values, 
                           'negative': self.aspect_sentence_df.groupby([self.time_column,"aspect"]).sum().unstack()['negative'][aspect].values})\
               .set_index("time")\
               .plot()
        ax.set_title("Numbers of positive and negative comments for this aspect")
        return negative_comments

    def aspect_sentiment_radar_plot(self, aspects: list):
        self.SentimentScore.aspects_radar_plot(aspects, self.AspectOpinionExtractor.aspects_opinions_df)


