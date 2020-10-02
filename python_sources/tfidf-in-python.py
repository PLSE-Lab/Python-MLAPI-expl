#important libraries
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import math

#random texts to work on
text1 = """
If you like tuna and tomato sauce - try combining the two.
It's really not as bad as it sounds.
If the Easter Bunny and the Tooth Fairy had babies would they take
your teeth and leave chocolate for you?
"""

#Preprocessing
    def remove_string_special_characters(s):
        """
        This function removes special characters from within a string
    
        parameters:
        s(str):single input string.
    
        return:
        stripped(str):A string with special characters removed
        """
    
        #replace special character with ' '
        stripped = re.sub('[^\w\s]', '', s)
        stripped = re.sub('_', '', stripped)
    
        #change any whitespace to one space
        stripped = re.sub('\s+', ' ', stripped)
    
        #remove start and end white space
        stripped = stripped.strip()
    
        return stripped
        
#Function for creating the documents
    def get_doc(sent):
        """
        this function splits the text into sentences and 
        considering each sentence as a document, calculates the 
        total word count of each.
        """
        doc_info = []
        i=0
        for sent in text_sents_clean:
            i+=1
            count = count_words(sent)
            temp = {'doc_id' : i, 'doc_length' : count}
            doc_info.append(temp)
        return doc_info
        
#The next two functions are apre-requisite to calculate TF and IDF score:
    def count_words(sent):
        """This function returns the
        total no. of words in the input text.
        """
        count = 0
        words = word_tokenize(sent)
        for word in words:
            count+=1
        return count
    
    def create_freq_dict(sents):
        """
        This function creates a frequency dictionary
        for each word in each document.
        """
        i = 0
        freqDict_list = []
        for sent in sents:
            i+=1
            freq_dict = {}
            words = word_tokenize(sent)
            for word in words:
                word = word.lower()
                if word in freq_dict:
                    freq_dict[word] +=1
                else:
                    freq_dict[word] = 1
                temp = {'doc_id' : i, 'freq_dict' : freq_dict}
            freqDict_list.append(temp)
        return freqDict_list
            
    #The function to get TF and IDF score:
    def computeTF(doc_info,freqDict_lists):
        """
        tf = (frequency of the term in the doc/total no. of terms in the document)
        """
        TF_scores = []
        for tempDict in freqDict_list:
            id = tempDict['doc_id']
            for k in tempDict['freq_dict']:
                temp = {'doc_id' : id,
                        'TF_score' : tempDict['freq_dict'][k]/doc_info[id-1]['doc_length'],
                        'key' : k}
                TF_scores.append(temp)
        return TF_scores
            
    def computeIDF(doc_info, freqDoc_list):
        """
        idf = ln(total no. of docs/no. of docs with term in it)
        """
        IDF_scores = []
        counter = 0
        for dict in freqDict_list:
            counter+=1
            for k in dict['freq_dict'].keys():
                count = sum([k in tempDict['freq_dict'] for tempDict in freqDict_list])
                temp = {'doc_id' : counter, 'IDF_score' : math.log(len(doc_info)/count), 'key' : k}
                
                IDF_scores.append(temp)
        
        return IDF_scores

    #Now, we compute TF-IDF
    
    def computeTFIDF(TF_scores, IDF_scores):
        TFIDF_scores = []
        for j in IDF_scores:
            for i in TF_scores:
                if j ['key'] == i['key'] and j['doc_id'] == i['doc_id']:
                    temp = {'doc_id' : j['doc_id'], 
                            'TFIDF_score' : j['IDF_score']*i['TF_score'],
                            'key' : i['key']}
            TFIDF_scores.append(temp)
        return TFIDF_scores
        
        
#Before computing the word frequency, we must clean the data -
#remove punctuation and special characters.
    
    text_sents = sent_tokenize(text1)
    text_sents_clean = [remove_string_special_characters(s) for s in text_sents]
    doc_info = get_doc(text_sents_clean)
    
    freqDict_list = create_freq_dict(text_sents_clean)
    TF_scores = computeTF(doc_info, freqDict_list)
    IDF_scores = computeIDF(doc_info, freqDict_list)